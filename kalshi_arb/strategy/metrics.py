"""Step 8 - performance metrics, consistently annualized (252) for strat + bench.

Reported: annualized return/vol, Sharpe, strategy max drawdown, daily alpha and
beta vs SPX, and the mark-to-model <-> mark-to-market correlation (model_rho).

RETURN SAMPLING -- all return-based statistics are computed on the TRADING-day
calendar, not on the backtest's own index. The backtest is indexed by Kalshi's
calendar (365/366 rows: Kalshi trades 24/7), so a naive pct_change there yields
a calendar-daily series, which the 252 annualization factor does not describe.
Weekend rows are not inert either -- 49-88 of them per year carry a genuine
non-zero mark. Two things therefore go wrong if the raw index is used:

  1. ann_return/ann_vol are scaled by 252 while the series has ~365 obs/yr.
  2. The SPX benchmark only exists on trading days, so reindexing it onto the
     Kalshi calendar forward-fills 124-142 artificial zero-return rows per year,
     which shrinks cov(strategy, SPX) and biases beta toward zero.

Sampling the portfolio on trading days fixes both: a Friday->Monday return
correctly absorbs the weekend move (the standard equity convention), and the
benchmark is then real on every row. The trading calendar is taken from the
option chain's own quote dates, so it is data-derived (no hard-coded holiday
list) and identical to the benchmark's calendar by construction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config


def _daily_returns(level: pd.Series) -> pd.Series:
    r = level.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r


def trading_days(chain, year, index_like, kalshi=None) -> pd.DatetimeIndex:
    """The real trading calendar for `year`: days BOTH markets were available.

    Using the chain's quote dates (rather than a weekday mask) excludes market
    holidays without hard-coding a holiday calendar, and guarantees the
    benchmark has a genuine observation on every sampled row.

    When `kalshi` is supplied the window is additionally clipped to the span in
    which the Kalshi market actually existed. This matters for 2022: the Kalshi
    SPX market did not open until 2022-07-07, so 127 of 240 otherwise-eligible
    observations (53%) sit before the first possible trade and are structurally
    flat. Those are not days the strategy declined to trade, they are days it
    could not exist, and including them drags the mean and (more so) the
    standard deviation toward zero -- 2022's Sharpe reads 1.53 with them and
    2.15 without. The clip is the same rule already applied at the other end of
    the sample, where 2024's path stops with the option feed on 11-19; applying
    it at both ends makes the evaluation window the intersection of the two
    data sources, symmetrically.
    """
    q = chain.loc[chain["quote"].dt.year == year, "quote"]
    days = sorted(set(q) & set(index_like))
    if kalshi is not None and year in kalshi:
        quoted = kalshi[year]["bid"].notna().any(axis=1)
        live = quoted[quoted].index
        if len(live):
            lo, hi = live.min(), live.max()
            days = [d for d in days if lo <= d <= hi]
    return pd.DatetimeIndex(days)


def spx_benchmark(chain, year, index_like) -> pd.Series:
    """Daily SPX (raw spot) level reindexed onto the backtest calendar."""
    s = (chain[chain["quote"].dt.year == year][["quote", "spot"]]
         .drop_duplicates("quote").set_index("quote")["spot"].sort_index())
    return s.reindex(index_like, method="ffill").bfill()


def summarize(m: pd.DataFrame, chain, year, n_trades=None, kalshi=None) -> dict:
    """Compute the performance row for one backtest result `m`."""
    pv = m["portfolio_value"]
    rf_d = config.BENCH_RF / config.TRADING_DAYS
    # Return statistics are sampled on trading days so that the x252 factor
    # describes the series (see module docstring); the drawdown below stays on
    # the FULL calendar path, since a trough reached over a weekend is a real
    # drawdown the position lived through and is not an annualized quantity.
    td = trading_days(chain, year, m.index, kalshi)
    rs = _daily_returns(pv.loc[td])

    ann_ret = rs.mean() * config.TRADING_DAYS
    ann_vol = rs.std() * np.sqrt(config.TRADING_DAYS)
    sharpe = (ann_ret - config.BENCH_RF) / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((pv - pv.cummax()) / pv.cummax()).min()

    spx = spx_benchmark(chain, year, td)
    rm = _daily_returns(spx)
    idx = rs.index.intersection(rm.index)
    beta = alpha = np.nan
    if len(idx) > 2:
        a, b = rs.loc[idx] - rf_d, rm.loc[idx] - rf_d
        var = b.var()
        beta = a.cov(b) / var if var > 0 else np.nan
        alpha = a.mean() - beta * b.mean()            # daily alpha

    rmod = _daily_returns(m["model_value"].loc[td])
    j = rs.index.intersection(rmod.index)
    model_rho = rs.loc[j].corr(rmod.loc[j]) if len(j) > 2 else np.nan

    # Return NET of the accrued Kalshi APY -- an ATTRIBUTION figure, splitting
    # the headline return into passive platform carry and actual trading P&L.
    #
    # There is deliberately no second "ex-APY Sharpe" here. Because BENCH_RF is
    # set to KALSHI_APY, the headline `sharpe` ALREADY measures excess over the
    # carry (holding cash on Kalshi is the risk-free alternative, and it earns
    # exactly that yield). Subtracting the risk-free rate again from a series
    # that has had the carry removed double-counts it and reads far too harsh --
    # 2024 BL both-side would show -0.11 rather than its correct 1.03. Use
    # `sharpe` for risk-adjusted performance and `ann_return_ex_apy` only to say
    # how much of the raw return was carry.
    ex = pv - m["interest"] if "interest" in m else pv
    rx = _daily_returns(ex.loc[td])
    ann_ret_ex = rx.mean() * config.TRADING_DAYS

    # Total return INCLUDING the year-end settlement payoff (the MTM path above
    # is pre-settlement; ann_return/Sharpe are path metrics, total_return is the
    # realized end-to-end result once held buckets settle).
    start = pv.iloc[0]
    final_value = m.attrs.get("final_value", pv.iloc[-1])
    total_return = final_value / start - 1 if start else np.nan

    return dict(
        ann_return=ann_ret, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd,
        ann_return_ex_apy=ann_ret_ex,
        alpha_daily=alpha, beta=beta, model_rho=model_rho,
        settlement=m.attrs.get("settlement", np.nan),
        final_value=final_value, total_return=total_return,
        n_trades=n_trades,
    )


def sharpe_bootstrap_ci(m: pd.DataFrame, chain, year, reps=10000,
                        mean_block=10, alpha=0.05, seed=0, kalshi=None) -> dict:
    """Stationary block-bootstrap confidence interval for the Sharpe ratio.

    The point estimate is fragile here and should never be quoted bare: the book
    is a ~$200 base turning over a few dozen lots a year, so daily returns are
    lumpy and heavy-tailed (excess kurtosis 6.6-28.4 by year). A plain i.i.d.
    bootstrap would understate the spread because the mark-to-market path is
    serially dependent, so blocks of geometrically-distributed length (mean
    `mean_block` days) are resampled instead, which preserves short-run
    dependence.

    Returns the point estimate, the interval, and the share of resamples above
    0 and above 1 -- the useful summary being that the SIGN is well determined
    while the MAGNITUDE is not.
    """
    rng = np.random.default_rng(seed)
    td = trading_days(chain, year, m.index, kalshi)
    r = _daily_returns(m["portfolio_value"].loc[td])
    v = r.values
    n = len(v)
    if n < 30:
        return dict(sharpe=np.nan, lo=np.nan, hi=np.nan, p_gt0=np.nan, p_gt1=np.nan, n=n)

    def _sharpe(x):
        ar, av = x.mean() * config.TRADING_DAYS, x.std() * np.sqrt(config.TRADING_DAYS)
        return (ar - config.BENCH_RF) / av if av > 0 else np.nan

    out = np.empty(reps)
    for i in range(reps):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(((s + np.arange(rng.geometric(1 / mean_block))) % n).tolist())
        out[i] = _sharpe(v[np.array(idx[:n])])
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(sharpe=_sharpe(v), lo=lo, hi=hi,
                p_gt0=float(np.nanmean(out > 0)), p_gt1=float(np.nanmean(out > 1)),
                n=n, kurtosis=float(r.kurt()))


def format_table(records: list[dict]) -> pd.DataFrame:
    """records: list of dicts with year, model, side + summarize() output."""
    df = pd.DataFrame(records)
    pct = ["ann_return", "ann_vol", "max_dd", "alpha_daily", "total_return",
           "ann_return_ex_apy"]
    for c in pct:
        df[c] = (df[c] * 100).round(2)
    df["sharpe"] = df["sharpe"].round(2)
    df["beta"] = df["beta"].round(3)
    df["model_rho"] = df["model_rho"].round(2)
    df["final_value"] = df["final_value"].round(2)
    return df
