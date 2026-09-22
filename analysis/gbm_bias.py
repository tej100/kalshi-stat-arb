"""Why did the GBM baseline lose in 2024? Decomposes the model error, not just the P&L.

Run from repo root:  python analysis/gbm_bias.py
Findings are written up in GBM_BIAS.md.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from kalshi_arb import pipeline, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import density, buckets, smoothing, kalshi_pmf
from kalshi_arb.strategy import signals, backtest, metrics, diagnostics as dg

chain = pipeline.build_chain(verbose=False)
k = ksrc.load_kalshi()
pm = {m: density.build_pmf_table(chain, k, method=m) for m in ("bl", "gbm")}

print("=" * 78)
print("1. THE BIAS IS SYSTEMATIC, NOT A 2024 ACCIDENT")
print("=" * 78)
print("   GBM minus BL bucket probability, grouped by where the bucket sits vs SPOT.")
print("   (both models use the SAME fitted smile; GBM collapses it to one ATM number)")
rows = []
for y in config.YEARS:
    spot = chain[chain.quote.dt.year == y].drop_duplicates("quote").set_index("quote")["spot"]
    for d in pm["bl"][y].index:
        if d not in spot.index:
            continue
        S = spot.loc[d]
        for b in pm["bl"][y].columns:
            a, c = pm["bl"][y].loc[d, b], pm["gbm"][y].loc[d, b]
            if np.isnan(a) or np.isnan(c):
                continue
            lo, hi = buckets.bucket_bounds(b)
            mid = 0.5 * (lo + hi)
            rows.append(dict(year=y, day=d, bucket=b, S=S, mid=mid, rel=(mid - S) / S, bl=a, gbm=c, diff=c - a))
D = pd.DataFrame(rows)
D["band"] = pd.cut(D.rel, [-1, -0.20, -0.10, -0.05, -0.02, 0.02, 0.05, 1],
                   labels=["<-20%", "-20..-10%", "-10..-5%", "-5..-2%", "+-2% (at spot)", "+2..+5%", ">+5%"])
t = D.groupby("band", observed=True).agg(n=("diff", "size"), mean_gbm_minus_bl=("diff", "mean"),
                                        mean_bl=("bl", "mean"), mean_gbm=("gbm", "mean"))
print((t * np.array([1, 100, 100, 100])).round(2).to_string())
print("   (probabilities in cents; positive = GBM assigns MORE probability than BL)")
print()
print("   by year, mean GBM-BL for buckets 2-10% BELOW spot (cents):")
sub = D[(D.rel < -0.02) & (D.rel > -0.10)]
print("  ", (sub.groupby("year")["diff"].mean() * 100).round(2).to_dict())
print("   share of bucket-days sitting BELOW spot, by year:")
print("  ", (D.assign(below=D.rel < 0).groupby("year")["below"].mean() * 100).round(0).to_dict(), "%")

print()
print("=" * 78)
print("2. WHY: THE SMILE SLOPE TERM GBM THROWS AWAY")
print("=" * 78)
print("   With a smile, the BL density is the lognormal density TIMES a correction that")
print("   depends on the skew. For strikes below the forward the skew term is negative,")
print("   so a flat-ATM-vol lognormal OVERSTATES probability there. Check the inputs:")
d = pd.Timestamp("2024-11-19")
day = chain[chain.quote == d]
sm = smoothing.fit_daily_smile(day)
F, S = day.forward.iloc[0], day.underlying.iloc[0]
atm = smoothing.eval_smile(sm, np.array([float(F)]))[0]
print(f"   {d.date()}: spot {S:.0f}, forward {F:.0f}, T={day['T'].iloc[0]*365:.0f} days")
print(f"   ATM vol used by GBM: {atm*100:.2f}%")
print(f"{'strike':>8} {'smile IV':>9} {'vs ATM':>8} {'local sigma*sqrtT':>18}")
for K in (5000, 5200, 5400, 5600, 5800, 5900, 6000):
    iv = smoothing.eval_smile(sm, np.array([float(K)]))[0]
    print(f"{K:>8} {iv*100:>8.2f}% {(iv-atm)*100:>+7.2f} {iv*np.sqrt(day['T'].iloc[0])*100:>17.2f}%")
print("   -> the smile is far ABOVE the ATM vol at strikes below spot (a steep equity skew).")
print("      GBM prices every bucket with the single ATM number, ignoring both the level")
print("      difference and the slope, and the slope term is what suppresses BL's density there.")

print()
print("=" * 78)
print("3. WHAT THAT DID TO THE 2024 BOOK")
print("=" * 78)
for mth in ("bl", "gbm"):
    tr = signals.generate(pm[mth], k, 2024, "both")
    m = backtest.run(tr, pm[mth], k, 2024)
    E = dg.episodes(m, 2024)
    spot = chain[chain.quote.dt.year == 2024].drop_duplicates("quote").set_index("quote")["spot"]
    E["S"] = E.entry_day.map(spot)
    E["mid"] = E.bucket.map(lambda b: np.mean(buckets.bucket_bounds(b)))
    E["below"] = E.mid < E.S
    buy = E[E.side > 0]
    print(f"  {mth.upper()} 2024: {len(E)} positions | net ${E.actual.sum():+.2f}")
    print(f"     BUYS: {len(buy)} | of these {int(buy.below.sum())} were buckets BELOW spot at entry "
          f"| their P&L ${buy[buy.below].actual.sum():+.2f}")
    print(f"     SELLS: {len(E)-len(buy)} | P&L ${E[E.side<0].actual.sum():+.2f}")
print()
print("   SPX 2024 path: start", round(chain[chain.quote.dt.year == 2024].sort_values('quote').spot.iloc[0]),
      "-> last", round(chain[chain.quote.dt.year == 2024].sort_values('quote').spot.iloc[-1]),
      "-> settle", config.SPX_YEAR_END_CLOSE[2024])
print("   top bucket:", list(k[2024]['price'].columns)[-1], "-> SPX finished above EVERY bucket,")
print("   so every bucket was 'below spot' by year end and every long in one expired worthless.")

print()
print("=" * 78)
print("4. IS IT REALLY THE MODEL, OR JUST VARIANCE? (forecast quality at the daily level)")
print("=" * 78)
for y in config.YEARS:
    close = config.SPX_YEAR_END_CLOSE[y]
    out = {b: 1.0 if buckets.bucket_bounds(b)[0] <= close <= buckets.bucket_bounds(b)[1] else 0.0
           for b in k[y]["price"].columns}
    o = pd.Series(out)
    r = {}
    for mth in ("bl", "gbm"):
        tab = pm[mth][y]
        m = tab.notna()
        br = (((tab - o) ** 2).where(m)).stack().mean() * 1000
        # calibration on the bucket that actually won
        win = [b for b in tab.columns if o[b] == 1.0]
        mean_on_winner = tab[win[0]].mean() if win else np.nan
        r[mth] = (br, mean_on_winner)
    print(f"  {y}: Brier x1000  BL {r['bl'][0]:.2f}  GBM {r['gbm'][0]:.2f}   |  "
          f"mean prob on the WINNING bucket: BL {r['bl'][1] if not np.isnan(r['bl'][1]) else float('nan'):.3f}  "
          f"GBM {r['gbm'][1] if not np.isnan(r['gbm'][1]) else float('nan'):.3f}"
          + ("   (2024: no bucket won)" if y == 2024 else ""))
