"""Post-backtest diagnostics: where the P&L came from, and how much risk carried it.

None of this feeds back into the trading logic; it only reads the result of a
`backtest.run`. It exists so the reported Sharpe is never the only number about a
strategy that trades a few dozen binary lots on a $200 base:

  * `risk_summary`  -- collateral use and a settle-now stress along the whole path.
  * `episodes`      -- one row per position, with the P&L it actually realised and
                       the P&L it would have had if simply held to settlement.

Both rely on the one-lot-per-bucket cap (`MAX_LOTS_PER_BUCKET == 1`): with no
pyramiding and no flips, every executed trade on a bucket alternates open, close.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from ..transform import buckets
from .backtest import _apply_trade


def _snapshots(m: pd.DataFrame) -> dict:
    """{date: {bucket: {qty, cost}}} -- the open book at the end of each day."""
    ex = m.attrs["executed_trades"]
    by_day = {d: g for d, g in ex.groupby("day")} if len(ex) else {}
    pos, snaps = {}, {}
    for d in m.index:
        g = by_day.get(d)
        if g is not None:
            for _, t in g.iterrows():
                _apply_trade(pos, t["bucket"], t["qty"], t["price"])
        snaps[d] = {b: dict(p) for b, p in pos.items()}
    return snaps


def risk_summary(m: pd.DataFrame) -> dict:
    """Collateral use and a path-wise settle-now stress for one backtest result.

    COLLATERAL. Kalshi is fully collateralised, so a long costs p per contract
    and a short (sold YES) requires posting 1 - p. `collateral_util` is that total
    over account value; anything above 100% would mean the backtest is quietly
    using leverage the account cannot provide.

    STRESS. Because the buckets are mutually exclusive, exactly one of them
    (or none, if SPX settles outside all 13) pays out. On every date, suppose the
    year had settled right then in the worst possible bucket for the open book;
    `worst_settle_pnl` is the resulting P&L vs starting capital. The payoff of
    the worst bucket is the most negative single-bucket quantity, or 0 if the
    book has no shorts. This bounds the tail loss independently of the three
    settlement outcomes that happened to occur.
    """
    snaps = _snapshots(m)
    coll = pd.Series({d: sum(p["cost"] if p["qty"] > 0 else abs(p["qty"]) + p["cost"]
                             for p in s.values()) for d, s in snaps.items()})
    util = coll / m["portfolio_value"]
    worst_payoff = pd.Series({d: min([0.0] + [p["qty"] for p in s.values()])
                              for d, s in snaps.items()})
    adv = m["cash"] + m["interest"] + worst_payoff - config.START_CASH
    n_open = pd.Series({d: len(s) for d, s in snaps.items()})
    net_short = pd.Series({d: sum(1 for p in s.values() if p["qty"] < 0)
                           - sum(1 for p in s.values() if p["qty"] > 0)
                           for d, s in snaps.items()})
    return dict(
        peak_collateral=float(coll.max()),
        peak_collateral_util=float(util.max()),
        days_over_100pct=int((util > 1).sum()),
        avg_open_buckets=float(n_open.mean()),
        max_open_buckets=int(n_open.max()),
        peak_net_short_buckets=int(net_short.max()),
        worst_settle_pnl=float(adv.min()),
        worst_settle_date=adv.idxmin(),
        worst_settle_pct_capital=float(adv.min() / config.START_CASH),
        days_below_start=int((adv < 0).sum()),
    )


def episodes(m: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per position: the P&L it realised, and the P&L it would have made
    if held to settlement.

    `actual` is net of BOTH fees for a position the rule closed, and net of the
    entry fee alone for one held to settlement (settlement is fee-free).
    `hold` is the same position's P&L had it never been closed. For a position
    that was held to settlement the two are identical by construction.

    Read `hold` as a COUNTERFACTUAL, not a strategy: it is what closing each
    position would have cost or saved, position by position. Do not read the
    settled positions' win rate as a property of the strategy -- they are the
    ones the model kept agreeing with, so that rate is selected by the exit rule.
    """
    assert config.MAX_LOTS_PER_BUCKET == 1, "episode accounting assumes one lot per bucket"
    ex = m.attrs["executed_trades"]
    close = config.SPX_YEAR_END_CLOSE[year]

    def settle_value(b):
        lo, hi = buckets.bucket_bounds(b)
        return 1.0 if lo <= close <= hi else 0.0

    open_, out = {}, []
    for _, t in ex.iterrows():
        b = t["bucket"]
        if b not in open_:
            open_[b] = dict(side=int(np.sign(t["qty"])), n=abs(t["qty"]), px=t["price"],
                            fee=t["fee"], day=t["day"])
            continue
        o = open_.pop(b)
        out.append(dict(year=year, bucket=b, side=o["side"], how="closed", entry_day=o["day"],
                        entry_px=o["px"], held_days=(t["day"] - o["day"]).days,
                        actual=o["side"] * (t["price"] - o["px"]) * o["n"] - o["fee"] - t["fee"],
                        hold=o["side"] * (settle_value(b) - o["px"]) * o["n"] - o["fee"]))
    for b, o in open_.items():
        pnl = o["side"] * (settle_value(b) - o["px"]) * o["n"] - o["fee"]
        out.append(dict(year=year, bucket=b, side=o["side"], how="settled", entry_day=o["day"],
                        entry_px=o["px"], held_days=np.nan, actual=pnl, hold=pnl))
    return pd.DataFrame(out)
