"""Risk profile of the backtest: collateral use, worst-case settlement stress, and
where the P&L came from position by position.

Run from repo root:  python analysis/risk_profile.py
Writes artifacts/risk_profile.csv and artifacts/position_episodes.csv (both untracked).
The write-up is in RISK_PROFILE.md.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from kalshi_arb import pipeline, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import density
from kalshi_arb.strategy import signals, backtest, diagnostics, metrics


def main():
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()
    risk_rows, ep_rows, freq_rows = [], [], []
    for method in ("bl", "gbm"):
        pmf = density.build_pmf_table(chain, k, method=method)
        for y in config.YEARS:
            tr = signals.generate(pmf, k, y, side="both")
            m = backtest.run(tr, pmf, k, y)
            risk_rows.append(dict(model=method.upper(), year=y, **diagnostics.risk_summary(m)))
            td = metrics.trading_days(chain, y, m.index, k)
            r1 = metrics.excess_value(m, chain).loc[td].pct_change().dropna()
            freq_rows.append(dict(model=method.upper(), year=y,
                                  sharpe_daily=metrics.summarize(m, chain, y, kalshi=k)["sharpe"],
                                  sharpe_5day=metrics.sharpe_at_frequency(m, chain, y, 5, k),
                                  sharpe_20day=metrics.sharpe_at_frequency(m, chain, y, 20, k),
                                  lag1_autocorr=r1.autocorr(1)))
            e = diagnostics.episodes(m, y)
            e.insert(0, "model", method.upper())
            ep_rows.append(e)
    R = pd.DataFrame(risk_rows)
    E = pd.concat(ep_rows, ignore_index=True)

    print("=== 1. Collateral use and worst-case settle-now stress (both-side book) ===")
    show = R.assign(peak_util=(R.peak_collateral_util * 100).round(1),
                    worst_pnl=R.worst_settle_pnl.round(2),
                    worst_pct=(R.worst_settle_pct_capital * 100).round(1))
    print(show[["model", "year", "peak_collateral", "peak_util", "days_over_100pct", "avg_open_buckets",
                "max_open_buckets", "peak_net_short_buckets", "worst_pnl", "worst_pct",
                "days_below_start"]].round(2).to_string(index=False))
    print("  peak_util = collateral posted / account value; > 100% would mean hidden leverage.")
    print("  worst_pnl = P&L vs starting capital if the year had settled on that date in the")
    print("  worst possible bucket for the open book (one bucket pays, the rest are worthless).")

    print("\n=== 2. P&L by how each position ended (net of fees) ===")
    print("  'closed' = exited by an opposite signal; 'settled' = held to year-end.")
    for method, g in E.groupby("model"):
        t = g.groupby("how").agg(n=("actual", "size"), total=("actual", "sum"), mean=("actual", "mean"),
                                 win_rate=("actual", lambda x: (x > 0).mean()))
        t["share_of_total"] = t.total / g.actual.sum()
        print(f"\n  {method}: net P&L over {len(g)} positions = ${g.actual.sum():+.2f}")
        print(t.round(3).to_string())

    print("\n=== 3. Would holding each closed position to settlement have been better? ===")
    C = E[E.how == "closed"]
    for method, g in C.groupby("model"):
        lose = g[g.actual < 0]
        print(f"  {method}: {len(g)} closed positions | realised {g.actual.sum():+.2f} vs "
              f"if held {g.hold.sum():+.2f} -> closing changed P&L by {g.actual.sum() - g.hold.sum():+.2f}")
        print(f"        {len(lose)} exited at a loss: realised {lose.actual.sum():+.2f} vs if held "
              f"{lose.hold.sum():+.2f} ({int((lose.hold > 0).sum())} of them would have ended profitable)")
    print("  NOTE: the settled positions' win rate is SELECTED by the exit rule -- they are the ones")
    print("  the model kept agreeing with -- so it is not a property of the strategy as a whole.")

    F = pd.DataFrame(freq_rows)
    print("\n=== 4. Does the Sharpe depend on the sampling frequency? ===")
    print("  The daily Sharpe scales by sqrt(252), assuming uncorrelated daily returns; they are not.")
    print("  5-/20-day figures use non-overlapping returns averaged over every starting phase.")
    print(F.round(2).to_string(index=False))
    same = ((np.sign(F.sharpe_daily) == np.sign(F.sharpe_5day)) & (np.sign(F.sharpe_daily) == np.sign(F.sharpe_20day)))
    print(f"  sign identical at all three frequencies in {int(same.sum())} of {len(F)} model-years.")

    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    R.to_csv(config.ARTIFACT_DIR / "risk_profile.csv", index=False)
    E.to_csv(config.ARTIFACT_DIR / "position_episodes.csv", index=False)
    F.to_csv(config.ARTIFACT_DIR / "sharpe_by_frequency.csv", index=False)


if __name__ == "__main__":
    main()
