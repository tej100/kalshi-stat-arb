"""Execution-timing check: same-close fills versus a lookahead-free lag.

The headline backtest fills each signal at the Kalshi close of the day whose
option chain produced it. The option quotes are the day's LAST quotes (likely
the 4:15pm SPXW close) while the Kalshi daily candle closes at 4:00pm, so that
convention can use up to 15 minutes of information the trader would not have
had. Here the model is lagged by one and two priced days (density.lag_pmf_table)
and every signal is re-tested against the Kalshi book of the day it fills, so
no fill uses information from after it. The gap between the columns is an
upper bound on what the timing convention contributes.

Also reports the markout of each same-close fill: the move of the Kalshi mid in
the trade's favour 1, 5 and 20 days later, in cents per contract.

Run from repo root:  python analysis/execution_lag.py
Writes artifacts/execution_lag.csv (untracked).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from kalshi_arb import pipeline, run, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import kalshi_pmf
from kalshi_arb.strategy import metrics


def markouts(results, kalshi, horizons=(1, 5, 20)):
    """One row per fill: the Kalshi mid's move in the trade's favour h valid-quote
    days later (cents per contract), gross and net of that fill's own fee."""
    rows = []
    for (y, mth, side), m in results.items():
        if side != "both":
            continue
        mid = kalshi_pmf.market_pmf(kalshi, y)
        for _, t in m.attrs["executed_trades"].iterrows():
            later = mid.loc[mid.index > t["day"], t["bucket"]].dropna()
            fee_c = t["fee"] / abs(t["qty"]) * 100
            row = dict(year=y, model=mth.upper(), day=t["day"])
            for h in horizons:
                g = (np.sign(t["qty"]) * (later.iloc[h - 1] - t["price"]) * 100
                     if len(later) >= h else np.nan)
                row[f"h{h}"], row[f"net{h}"] = g, g - fee_c
            rows.append(row)
    return pd.DataFrame(rows)


def clustered_mean(x, groups):
    """Mean and its t-statistic with standard errors clustered by `groups`
    (fills on the same day share one density and one Kalshi snapshot)."""
    d = pd.DataFrame(dict(x=x, g=groups)).dropna()
    m, n = d.x.mean(), len(d)
    G = d.g.nunique()
    s = d.groupby("g").x.apply(lambda v: (v - m).sum())
    se = np.sqrt((s ** 2).sum()) / n * np.sqrt(G / (G - 1))
    return m, m / se, n, G


def main(reps=10000):
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()
    rows, pooled, marks = [], [], []
    for lag in (0, 1, 2):
        _, res, _ = run.full(chain=chain, kalshi=k, sides=("both",), verbose=False, lag=lag)
        for (y, mth, _), m in res.items():
            s = metrics.summarize(m, chain, y, n_trades=m.attrs["n_executed"], kalshi=k)
            ci = metrics.sharpe_bootstrap_ci(m, chain, y, reps=reps, kalshi=k)
            rows.append(dict(model=mth.upper(), year=y, lag=lag, sharpe=s["sharpe"], ci_lo=ci["lo"],
                             ci_hi=ci["hi"], ann_excess=s["ann_excess"], total_return=s["total_return"],
                             fills=s["n_trades"]))
        for mth in ("bl", "gbm"):
            p = metrics.pooled_sharpe_ci({y: res[(y, mth, "both")] for y in config.YEARS},
                                         chain, reps=reps, kalshi=k)
            pooled.append(dict(model=mth.upper(), lag=lag, sharpe=p["sharpe"], lo=p["lo"], hi=p["hi"],
                               p_gt0=p["p_gt0"], days=p["n"]))
        if lag <= 1:
            M = markouts(res, k)
            for mdl, g in M.groupby("model"):
                for h in (1, 5, 20):
                    mg, tg, n, G = clustered_mean(g[f"h{h}"], g["day"])
                    mn, tn, _, _ = clustered_mean(g[f"net{h}"], g["day"])
                    marks.append(dict(model=mdl, lag=lag, horizon=h, gross_c=mg, t_gross=tg,
                                      net_c=mn, t_net=tn, fills=n, days=G))
    R = pd.DataFrame(rows)
    print("=== Sharpe by execution convention (both-side book) ===")
    print("  lag 0 = same close (headline); lag 1/2 = model from 1/2 priced days earlier,")
    print("  signal re-tested at the fill day's Kalshi book.\n")
    print(R.pivot_table(index=["model", "year"], columns="lag", values="sharpe").round(2).to_string())
    print()
    show = R.assign(ann_excess=(R.ann_excess * 100).round(2), total_return=(R.total_return * 100).round(2))
    print(show.round(2).to_string(index=False))
    print("\n=== Pooled 2022-2024 Sharpe (daily excess returns stacked, block bootstrap) ===")
    print(pd.DataFrame(pooled).round(3).to_string(index=False))
    print("\n=== Markouts: Kalshi mid move in the trade's favour after h days (cents/contract) ===")
    print("  net = after the fill's own fee; t clustered by fill date")
    print(pd.DataFrame(marks).round(2).to_string(index=False))
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    R.to_csv(config.ARTIFACT_DIR / "execution_lag.csv", index=False)
    pd.DataFrame(pooled).to_csv(config.ARTIFACT_DIR / "pooled_sharpe.csv", index=False)
    pd.DataFrame(marks).to_csv(config.ARTIFACT_DIR / "markouts.csv", index=False)


if __name__ == "__main__":
    main()
