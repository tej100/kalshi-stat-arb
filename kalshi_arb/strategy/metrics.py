"""Step 8 - performance metrics, consistently annualized (252) for strat + bench.

Reported: annualized return/vol, Sharpe, strategy max drawdown, daily alpha and
beta vs SPX, and the mark-to-model <-> mark-to-market correlation (model_rho).

RETURN SAMPLING -- all return-based statistics are computed on the TRADING-day
calendar, not on the backtest's own index. The backtest is indexed by Kalshi's
calendar (365/366 rows: Kalshi trades 24/7), so a naive pct_change there yields
a calendar-daily series, which the 252 annualization factor does not describe.
Weekend rows are not inert either -- 49-88 of them per year carry a genuine
non-zero mark. Two things therefore go wrong if the raw index is used:

  1. annualised return/vol are scaled by 252 while the series has ~365 obs/yr.
  2. The SPX benchmark only exists on trading days, so reindexing it onto the
     Kalshi calendar forward-fills 124-142 artificial zero-return rows per year,
     which shrinks cov(strategy, SPX) and biases beta toward zero.

Sampling the portfolio on trading days fixes both: a Friday->Monday return
correctly absorbs the weekend move (the standard equity convention), and the
benchmark is then real on every row. The trading calendar is taken from the
option chain's own quote dates, so it is data-derived (no hard-coded holiday
list) and identical to the benchmark's calendar by construction.

EXCESS RETURNS AND CARRY -- every risk-adjusted statistic is computed on the
return in EXCESS of the risk-free rate, built by `excess_value`. The account is
treated as two pieces: the collateral posted for open positions, which has to
sit at Kalshi, and the idle cash, which is assumed to earn the risk-free rate
elsewhere. Idle cash therefore earns exactly rf and contributes nothing to the
excess return; the collateral earns Kalshi's rate (zero before 2024-10-10) and
so costs (rf - Kalshi rate) per year while it is posted. The daily excess P&L is

    trading P&L  +  Kalshi interest on collateral  -  rf * collateral.

The risk-free rate is the option-implied rate of the day (config), so no rate
is subtracted again in the Sharpe ratio or the alpha regression.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config


def rf_series(chain, index_like) -> pd.Series:
    """Option-implied risk-free rate (annual, continuous) on each date of `index_like`,
    carried forward from the most recent quote date."""
    r = chain.drop_duplicates("quote").set_index("quote")["r"].sort_index()
    idx = pd.DatetimeIndex(index_like)
    return r.reindex(r.index.union(idx)).ffill().bfill().reindex(idx)


def excess_value(m: pd.DataFrame, chain) -> pd.Series:
    """Account value in excess of the risk-free rate (see module docstring).

    `portfolio_value` already holds trading P&L and Kalshi's interest on the
    posted collateral; this subtracts the cost of capital on that collateral,
    accrued daily at the previous day's option-implied rate. On a day with no
    open positions the excess P&L is exactly the trading P&L, which is zero."""
    rf = rf_series(chain, m.index)
    days = m.index.to_series().diff().dt.days.fillna(0)
    charge = (rf.shift().bfill() * m["collateral"].shift().fillna(0) * days / 365.0).cumsum()
    return m["portfolio_value"] - charge


def _daily_returns(level: pd.Series) -> pd.Series:
    r = level.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return r


def _newey_west_t(y, x):
    """OLS of y on [1, x] with Newey-West (Bartlett) standard errors.

    Returns (intercept, slope, t_intercept, t_slope). Daily strategy returns here
    are autocorrelated (thin books bounce and revert; lag-1 ranges from -0.18 to
    +0.04 across model-years), so plain OLS standard errors misstate significance.

    The truncation lag is the Newey-West (1994) rule of thumb floor(4*(n/100)^(2/9)),
    which is a function of the sample size rather than a chosen constant (4 for
    the ~240 daily observations in a year).
    """
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    L = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    u = X * e[:, None]
    S = u.T @ u
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    return beta[0], beta[1], beta[0] / se[0], beta[1] / se[1]


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
    standard deviation toward zero -- 2022's Sharpe reads 1.35 with them and
    1.97 without. The clip is the same rule already applied at the other end of
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
    """Compute the performance row for one backtest result `m`.

    `ann_excess` is the annualised excess return and `sharpe` = ann_excess /
    ann_vol. It splits into `ann_trading` (trading P&L alone) and `ann_carry`
    (Kalshi interest on collateral minus its cost of capital, negative whenever
    Kalshi pays less than rf); the three are computed on the same denominator,
    so ann_excess = ann_trading + ann_carry exactly."""
    ev = excess_value(m, chain)
    # Return statistics are sampled on trading days so that the x252 factor
    # describes the series (see module docstring); the drawdown below stays on
    # the FULL calendar path, since a trough reached over a weekend is a real
    # drawdown the position lived through and is not an annualized quantity.
    td = trading_days(chain, year, m.index, kalshi)
    evt = ev.loc[td]
    rs = _daily_returns(evt)

    ann_ex = rs.mean() * config.TRADING_DAYS
    ann_vol = rs.std() * np.sqrt(config.TRADING_DAYS)
    sharpe = ann_ex / ann_vol if ann_vol > 0 else np.nan
    max_dd = ((ev - ev.cummax()) / ev.cummax()).min()

    trade_pnl = (m["portfolio_value"] - m["interest"]).loc[td].diff()
    carry_pnl = evt.diff() - trade_pnl
    base = evt.shift()
    ann_trading = (trade_pnl / base).dropna().mean() * config.TRADING_DAYS
    ann_carry = (carry_pnl / base).dropna().mean() * config.TRADING_DAYS

    spx = spx_benchmark(chain, year, td)
    rf = rf_series(chain, td)
    gap = td.to_series().diff().dt.days
    rm = (_daily_returns(spx) - (rf.shift() * gap / 365.0)).dropna()
    idx = rs.index.intersection(rm.index)
    beta = alpha = alpha_t = beta_t = np.nan
    if len(idx) > 2:
        a, b = rs.loc[idx], rm.loc[idx]
        var = b.var()
        beta = a.cov(b) / var if var > 0 else np.nan
        alpha = a.mean() - beta * b.mean()            # daily alpha
        if len(idx) > 30 and var > 0:
            _, _, alpha_t, beta_t = _newey_west_t(a.values, b.values)

    rmod = _daily_returns(m["model_value"].loc[td])
    rpv = _daily_returns(m["portfolio_value"].loc[td])
    j = rpv.index.intersection(rmod.index)
    model_rho = rpv.loc[j].corr(rmod.loc[j]) if len(j) > 2 else np.nan

    # Total return INCLUDING the year-end settlement payoff (the MTM path above
    # is pre-settlement; ann_excess/Sharpe are path metrics, total_return is the
    # realized end-to-end result once held buckets settle). It is the nominal
    # P&L of the Kalshi account itself -- trades, fees, settlement and Kalshi
    # interest -- and excludes the rf earned on idle cash held elsewhere.
    pv = m["portfolio_value"]
    start = pv.iloc[0]
    final_value = m.attrs.get("final_value", pv.iloc[-1])
    total_return = final_value / start - 1 if start else np.nan

    return dict(
        ann_excess=ann_ex, ann_vol=ann_vol, sharpe=sharpe, max_dd=max_dd,
        ann_trading=ann_trading, ann_carry=ann_carry, rf_mean=float(rf.mean()),
        alpha_daily=alpha, alpha_ann=alpha * config.TRADING_DAYS, alpha_t=alpha_t,
        beta=beta, beta_t=beta_t, model_rho=model_rho,
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
    r = _daily_returns(excess_value(m, chain).loc[td])
    v = r.values
    n = len(v)
    if n < 30:
        return dict(sharpe=np.nan, lo=np.nan, hi=np.nan, p_gt0=np.nan, p_gt1=np.nan, n=n)

    def _sharpe(x):
        ar, av = x.mean() * config.TRADING_DAYS, x.std() * np.sqrt(config.TRADING_DAYS)
        return ar / av if av > 0 else np.nan

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


def sharpe_at_frequency(m: pd.DataFrame, chain, year, step: int, kalshi=None) -> float:
    """Sharpe from non-overlapping `step`-trading-day returns, annualised by 252/step.

    The daily Sharpe scales by sqrt(252), which assumes daily returns are serially
    uncorrelated. They are not (thin books bounce and revert), so a Sharpe read
    from weekly or four-weekly returns is a check that the conclusion does not
    hinge on that assumption. Which day the non-overlapping grid starts on would
    otherwise be an arbitrary choice, so the result is averaged over all `step`
    possible starting phases.
    """
    if step <= 1:
        raise ValueError("step must be > 1 (use `summarize` for the daily Sharpe)")
    td = trading_days(chain, year, m.index, kalshi)
    pv = excess_value(m, chain).loc[td]
    per = config.TRADING_DAYS / step
    out = []
    for off in range(step):
        r = _daily_returns(pv.iloc[off::step])
        if len(r) >= 5 and r.std() > 0:
            out.append(r.mean() * per / (r.std() * np.sqrt(per)))
    return float(np.mean(out)) if out else np.nan


def format_table(records: list[dict]) -> pd.DataFrame:
    """records: list of dicts with year, model, side + summarize() output."""
    df = pd.DataFrame(records)
    pct = ["ann_excess", "ann_vol", "max_dd", "total_return", "ann_trading",
           "ann_carry", "rf_mean", "alpha_ann"]
    for c in pct:
        df[c] = (df[c] * 100).round(2)
    # alpha_daily is a per-day PERCENT of ~0.01-0.06; two decimals would round most
    # of the information away (0.01%/day is 2.5%/yr), so keep four.
    df["alpha_daily"] = (df["alpha_daily"] * 100).round(4)
    df["alpha_t"] = df["alpha_t"].round(2)
    df["beta_t"] = df["beta_t"].round(2)
    df["sharpe"] = df["sharpe"].round(2)
    df["beta"] = df["beta"].round(3)
    df["model_rho"] = df["model_rho"].round(2)
    df["final_value"] = df["final_value"].round(2)
    return df
