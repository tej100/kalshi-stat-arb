"""Cointegration / lead-lag analysis of the Kalshi vs options-implied bucket
probabilities, and a P&L attribution (convergence vs settlement).

Answers: is the strategy a pairs/convergence trade, and which venue leads?
Run from repo root:  python analysis/cointegration.py
Findings are written up in DUAL_TRADING_ANALYSIS.md.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from kalshi_arb import pipeline, run, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import density, kalshi_pmf
from kalshi_arb.strategy import metrics

ACTIVE_PROB = 0.03      # a bucket is "active" on a day if either side prices it > 3%
MIN_ACTIVE_DAYS = 20    # only analyze buckets that are near-the-money enough to trade


def half_life(s):
    """Mean-reversion half-life (days) from an AR(1) fit; inf if not reverting."""
    s = s.dropna()
    if len(s) < 15:
        return np.nan
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    beta = np.polyfit(lag.values, ds.values, 1)[0]      # ds = a + beta*level
    return np.inf if beta >= 0 else -np.log(2) / np.log(1 + beta)


def half_life_iv(s):
    """Half-life robust to bid-ask bounce in the Kalshi mid.

    Bounce is measurement noise e_t on the observed spread. In the plain AR(1)
    fit it sits in both the regressor s_t and the change s_{t+1} - s_t, which
    pulls the slope toward -1 and makes reversion look faster than it is. Using
    s_{t-1} as an instrument for s_t removes it, since white noise at t is
    uncorrelated with the spread one day earlier:
        phi = cov(s_{t+1}, s_{t-1}) / cov(s_t, s_{t-1}).
    """
    s = s.dropna()
    if len(s) < 15:
        return np.nan
    a, b, c = s.shift(-1), s, s.shift(1)
    v = a.notna() & c.notna()
    den = np.cov(b[v], c[v])[0, 1]
    if abs(den) < 1e-12:
        return np.nan
    phi = np.cov(a[v], c[v])[0, 1] / den
    return np.inf if not 0 < phi < 1 else -np.log(2) / np.log(phi)


def adf_p(s):
    s = s.dropna()
    if len(s) < 15 or s.std() < 1e-6:
        return np.nan
    try:
        return adfuller(s.values, autolag="AIC")[1]
    except Exception:
        return np.nan


def main():
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()
    model = density.build_pmf_table(chain, k, method="bl")          # options-implied PMF
    market = {y: kalshi_pmf.market_pmf(k, y) for y in k}            # Kalshi mid PMF

    # ---- 1. spread stationarity + half-life, per active bucket ----
    rows, dk, dm, sp, dk2, gd, gb = [], [], [], [], [], [], []
    for y in k:
        mdl, mkt = model[y], market[y]
        common = mdl.index.intersection(mkt.index)
        for b in mdl.columns:
            mm, dd = mkt.loc[common, b], mdl.loc[common, b]
            spread = (mm - dd)                                       # Kalshi - model
            active = (mm.fillna(0) > ACTIVE_PROB) | (dd.fillna(0) > ACTIVE_PROB)
            if active.sum() < MIN_ACTIVE_DAYS or spread.dropna().shape[0] < 20:
                continue
            rows.append(dict(year=y, bucket=b, n=int(spread.dropna().shape[0]),
                             mean_spread=spread.mean(), adf_p=adf_p(spread),
                             half_life=half_life(spread), half_life_iv=half_life_iv(spread)))
            # pooled next-day changes vs current spread, for the direction test
            v = spread.notna() & mm.diff().shift(-1).notna() & dd.diff().shift(-1).notna() & active
            sp += spread[v].tolist()
            dk += mm.diff().shift(-1)[v].tolist()
            dm += dd.diff().shift(-1)[v].tolist()
            # the same change one day later (t+1 -> t+2) shares no endpoint with the
            # spread at t, so bid-ask bounce in the Kalshi mid cannot produce it
            dk2 += mm.diff().shift(-2)[v].tolist()
            gd += [d.toordinal() for d in spread[v].index]          # cluster: date
            gb += [f"{y}:{b}"] * int(v.sum())                        # cluster: bucket

    df = pd.DataFrame(rows)
    df["stationary"] = df["adf_p"] < 0.05
    fin = df[np.isfinite(df["half_life"])]
    print("=== 1. Spread stationarity & mean reversion (active buckets) ===")
    print(f"buckets analyzed:            {len(df)}")
    print(f"spread stationary (ADF<5%):  {df['stationary'].mean()*100:.0f}%")
    print(f"mean spread (Kalshi-model):  {df['mean_spread'].mean():+.4f}  "
          f"(|median| {df['mean_spread'].abs().median():.4f}) -> reverts to ~0")
    print(f"half-life (days):            median {fin['half_life'].median():.1f}  "
          f"[{fin['half_life'].quantile(.25):.1f}-{fin['half_life'].quantile(.75):.1f} IQR]  (plain AR(1))")
    fiv = df[np.isfinite(df["half_life_iv"])]
    print(f"half-life, bounce-robust:    median {fiv['half_life_iv'].median():.1f}  "
          f"[{fiv['half_life_iv'].quantile(.25):.1f}-{fiv['half_life_iv'].quantile(.75):.1f} IQR]  "
          f"({len(fiv)} of {len(df)} buckets finite)")

    # ---- 2. direction of adjustment (which venue error-corrects) ----
    sp, dk, dm = np.array(sp), np.array(dk), np.array(dm)
    gd = np.array(gd); gb = pd.factorize(np.array(gb))[0]
    # Standard errors clustered by DATE: every bucket on a day comes from one
    # density and one Kalshi snapshot, so errors are correlated within a date.
    # Two-way (date x bucket) clustering is reported as a check.
    def fit(y, x, ok=slice(None)):
        X = sm.add_constant(x[ok])
        return (sm.OLS(y[ok], X).fit(cov_type="cluster", cov_kwds={"groups": gd[ok]}),
                sm.OLS(y[ok], X).fit(cov_type="cluster",
                                     cov_kwds={"groups": np.column_stack([gd[ok], gb[ok]])}))
    rk, rk2 = fit(dk, sp)
    rm, rm2 = fit(dm, sp)
    bk, bm = rk.params[1], rm.params[1]
    print("\n=== 2. Direction of adjustment (error correction), spread = Kalshi - model ===")
    print(f"n bucket-days:               {len(sp)}")
    print(f"d(Kalshi)_next ~ spread:     coef {bk:+.3f}  t {rk.tvalues[1]:+.1f}  "
          f"(negative => Kalshi corrects toward model)")
    print(f"d(model)_next  ~ spread:     coef {bm:+.3f}  t {rm.tvalues[1]:+.1f}  "
          f"(positive => options drift toward Kalshi)")
    print(f"  (t-stats clustered by date; two-way date x bucket: Kalshi {rk2.tvalues[1]:+.1f}, "
          f"model {rm2.tvalues[1]:+.1f})")
    print(f"share of correction by KALSHI leg: {abs(bk)/(abs(bk)+abs(bm))*100:.0f}%")
    dk2 = np.array(dk2); ok = ~np.isnan(dk2)
    r2, r22 = fit(dk2, sp, ok)
    print(f"d(Kalshi) t+1->t+2 ~ spread: coef {r2.params[1]:+.3f}  t {r2.tvalues[1]:+.1f}  "
          f"(two-way {r22.tvalues[1]:+.1f})  n {ok.sum()}  dates {len(np.unique(gd[ok]))}  "
          f"(no shared endpoint: bounce-free)")

    # ---- 3. P&L attribution: carry vs trading path vs settlement ----
    # The daily mark-to-market path is NOT all "convergence": it also contains the
    # carry on posted collateral (Kalshi interest, paid only from 2024-10-10, minus
    # the cost of capital at the option-implied rate), and it books a contract's
    # drift toward its realised 0/1 outcome before the settlement step arrives. So
    # three pieces are separated here, and the lead-lag question is answered by the
    # regression in section 2, not by this split.
    print("\n=== 3. P&L attribution (BL, both-side): carry / trading path / settlement ===")
    _, results, _ = run.full(chain=chain, kalshi=k, verbose=False)
    for y in config.YEARS:
        m = results[(y, "bl", "both")]
        ev = metrics.excess_value(m, chain)
        start = m["portfolio_value"].iloc[0]
        carry = ev.iloc[-1] - (m["portfolio_value"].iloc[-1] - m["interest"].iloc[-1])  # net carry
        path = m["portfolio_value"].iloc[-1] - m["interest"].iloc[-1] - start           # trading, marked to market
        settle = m.attrs["final_value"] - m["portfolio_value"].iloc[-1]                 # terminal settlement step
        tot = carry + path + settle
        print(f"  {y}: carry {carry:+6.2f}  trading path {path:+6.2f}  settlement {settle:+5.2f}  "
              f"= {tot:+6.2f} in excess of rf  | settlement is "
              f"{abs(settle)/(abs(carry)+abs(path)+abs(settle))*100:.0f}% and carry "
              f"{abs(carry)/(abs(carry)+abs(path)+abs(settle))*100:.0f}% of |P&L|")


if __name__ == "__main__":
    main()
