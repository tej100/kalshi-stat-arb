"""Step 8 - performance metrics, consistently annualized (252) for strat + bench.

Reported: annualized return/vol, Sharpe, strategy max drawdown, daily alpha and
beta vs SPX, and the mark-to-model <-> mark-to-market correlation (model_rho).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config


def _daily_returns(level: pd.Series) -> pd.Series:
    r = level.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r


def spx_benchmark(chain, year, index_like) -> pd.Series:
    """Daily SPX (raw spot) level reindexed onto the backtest calendar."""
    s = (chain[chain["quote"].dt.year == year][["quote", "spot"]]
         .drop_duplicates("quote").set_index("quote")["spot"].sort_index())
    return s.reindex(index_like, method="ffill").bfill()


def summarize(m: pd.DataFrame, chain, year, n_trades=None) -> dict:
    """Compute the performance row for one backtest result `m`."""
    pv = m["portfolio_value"]
    rf_d = config.BENCH_RF / config.TRADING_DAYS
    rs = _daily_returns(pv)

    ann_ret = rs.mean() * config.TRADING_DAYS
    ann_vol = rs.std() * np.sqrt(config.TRADING_DAYS)
    sharpe = (ann_ret - config.BENCH_RF) / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((pv - pv.cummax()) / pv.cummax()).min()

    spx = spx_benchmark(chain, year, m.index)
    rm = _daily_returns(spx)
    idx = rs.index.intersection(rm.index)
    beta = alpha = np.nan
    if len(idx) > 2:
        a, b = rs.loc[idx] - rf_d, rm.loc[idx] - rf_d
        var = b.var()
        beta = a.cov(b) / var if var > 0 else np.nan
        alpha = a.mean() - beta * b.mean()            # daily alpha

    rmod = _daily_returns(m["model_value"])
    j = rs.index.intersection(rmod.index)
    model_rho = rs.loc[j].corr(rmod.loc[j]) if len(j) > 2 else np.nan

    # Total return INCLUDING the year-end settlement payoff (the MTM path above
    # is pre-settlement; ann_return/Sharpe are path metrics, total_return is the
    # realized end-to-end result once held buckets settle).
    start = pv.iloc[0]
    final_value = m.attrs.get("final_value", pv.iloc[-1])
    total_return = final_value / start - 1 if start else np.nan

    return dict(
        ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd,
        alpha_daily=alpha, beta=beta, model_rho=model_rho,
        settlement=m.attrs.get("settlement", np.nan),
        final_value=final_value, total_return=total_return,
        n_trades=n_trades,
    )


def format_table(records: list[dict]) -> pd.DataFrame:
    """records: list of dicts with year, model, side + summarize() output."""
    df = pd.DataFrame(records)
    pct = ["ann_return", "ann_vol", "max_dd", "alpha_daily", "total_return"]
    for c in pct:
        df[c] = (df[c] * 100).round(2)
    df["sharpe"] = df["sharpe"].round(2)
    df["beta"] = df["beta"].round(3)
    df["model_rho"] = df["model_rho"].round(2)
    df["final_value"] = df["final_value"].round(2)
    return df
