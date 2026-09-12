"""Does blending model prices toward market mids (the paper's "control variate",
lambda) help the risk-neutral density and the strategy, or hurt it?

Builds the BL density from a price curve that is a lambda-blend of the smooth
SABR-smile BSM price and the interpolated market mid:

    C_blend(K) = (1-lambda) * BSM_smile(K) + lambda * mid_interp(K)

lambda=0 is the production density (pure smile). We compare density quality and
full strategy metrics across lambda in {0, 0.2, 0.5} to isolate the blend's
effect. Run from repo root:  python analysis/control_variate_test.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from kalshi_arb import pipeline, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import smoothing, pricing, buckets as bkt
from kalshi_arb.transform.density import _isotonic_increasing, bucket_prob
from kalshi_arb.strategy import signals, backtest, metrics


def bl_density_blend(day_df, lam):
    """BL density from a lambda-blend of smile price and interpolated market mid.
    lam=0 reproduces the production (pure-smile) density."""
    sm = smoothing.fit_daily_smile(day_df)
    if sm is None:
        return None
    S = day_df["underlying"].iloc[0]; T = day_df["T"].iloc[0]; r = day_df["r"].iloc[0]
    grid = np.linspace(max(sm.k_min - config.GRID_PAD_LOW, 1.0),
                       sm.k_max + config.GRID_PAD_HIGH, config.GRID_POINTS)
    c_smile = pricing.bsm_call(S, grid, T, r, smoothing.eval_smile(sm, grid))
    if lam > 0:
        # market mid as a function of strike (OTM calls carry call mids; OTM puts
        # converted to call value via put-call parity C = P + S - K*DF)
        F = day_df["forward"].iloc[0]; DF = np.exp(-r * T)
        d = day_df.dropna(subset=["Mid"]).copy()
        d["cval"] = np.where(d["option_type"] == "c", d["Mid"], d["Mid"] + S - d["strike"] * DF)
        d = d[((d.option_type == "p") & (d.strike < F)) | ((d.option_type == "c") & (d.strike >= F))]
        d = d.sort_values("strike")
        mid_interp = np.interp(grid, d["strike"], d["cval"],
                               left=d["cval"].iloc[0], right=d["cval"].iloc[-1])
        calls = (1 - lam) * c_smile + lam * mid_interp
    else:
        calls = c_smile
    DF = np.exp(-r * T)
    fp = _isotonic_increasing(np.clip(np.gradient(calls, grid), -DF, 0.0))
    dens = np.clip(np.exp(r * T) * np.gradient(fp, grid), 0.0, None)
    w = config.DENSITY_SMOOTH_WINDOW
    dens = np.convolve(dens, np.ones(w) / w, mode="same")
    return grid[2:-2], dens[2:-2]


def density_quality(chain, lam, dates):
    rough, arb, modes = [], [], []
    for ds in dates:
        res = bl_density_blend(chain[chain.quote == pd.Timestamp(ds)], lam)
        if res is None:
            continue
        g, d = res
        # signed 2nd-derivative arbitrage measured separately
        peak = d.max() if d.max() > 0 else 1.0
        rough.append(np.sum(np.abs(np.diff(d))) / peak)
        modes.append(int(((np.diff(np.sign(np.diff(d))) < 0) & (d[1:-1] > 0.05 * peak)).sum()))
    return np.mean(rough), np.mean(modes)


def build_pmf_blend(chain, kalshi, lam):
    out = {}
    for year in kalshi:
        bks = list(kalshi[year]["price"].columns)
        tmpl = kalshi[year]["price"].copy(); tmpl[:] = np.nan
        by_day = {d: g for d, g in chain[chain.quote.dt.year == year].groupby("quote", observed=True)}
        for date in tmpl.index:
            day = by_day.get(pd.Timestamp(date))
            if day is None:
                continue
            res = bl_density_blend(day, lam)
            if res is None:
                continue
            g, d = res
            tmpl.loc[date] = pd.Series({b: bucket_prob(g, d, *bkt.bucket_bounds(b)) for b in bks})
        out[year] = tmpl
    return out


def main():
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()
    dates = ["2022-02-25", "2022-06-16", "2022-10-04", "2023-06-23", "2023-11-15", "2024-05-29"]

    print("Density quality vs blend lambda (lower = smoother/cleaner):")
    print(f"{'lambda':>7}{'roughness':>12}{'modes':>8}")
    for lam in [0.0, 0.2, 0.5]:
        rq, mq = density_quality(chain, lam, dates)
        print(f"{lam:>7.1f}{rq:>12.2f}{mq:>8.1f}")

    print("\nStrategy (BL, both-side) vs blend lambda:")
    print(f"{'lambda':>7}{'year':>6}{'sharpe':>9}{'total%':>9}{'maxDD%':>9}")
    for lam in [0.0, 0.2, 0.5]:
        pmf = build_pmf_blend(chain, k, lam)     # {year: date x bucket}
        for year in config.YEARS:
            tr = signals.generate(pmf, k, year, side="both")
            m = backtest.run(tr, pmf, k, year)
            s = metrics.summarize(m, chain, year, n_trades=m.attrs["n_executed"])
            print(f"{lam:>7.1f}{year:>6}{s['sharpe']:>9.2f}{s['total_return']*100:>9.2f}{s['max_dd']*100:>9.2f}")


if __name__ == "__main__":
    main()
