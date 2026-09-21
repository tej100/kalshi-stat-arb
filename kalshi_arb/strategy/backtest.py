"""Step 7 - portfolio engine for the Kalshi bucket strategy.

Marks open positions to the Kalshi order book each day (long -> exit at bid,
short -> exit at ask), accrues the 3.75% APY monthly on total portfolio value,
and settles held buckets at the true year-end SPX close ($1 if in-bucket).

Produces both a mark-to-MARKET series (only fillable prices) and a
mark-to-MODEL series (positions valued at the model probability), so their
correlation can quantify how executable the ideal strategy is.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from ..transform import buckets
from ..transform.kalshi_pmf import is_valid_book


def _apply_trade(positions, bucket, qty, price):
    """Update signed position + cost basis; return realized PnL for this fill."""
    pos = positions.get(bucket, {"qty": 0.0, "cost": 0.0})
    cq, cc = pos["qty"], pos["cost"]
    trade_cost = qty * price
    realized = 0.0
    if cq == 0 or (cq > 0) == (qty > 0):            # open / add same direction
        new_qty, new_cost = cq + qty, cc + trade_cost
    else:                                            # reduce / close / flip
        avg = cc / cq
        close = min(abs(qty), abs(cq))
        realized = (price - avg) * close * np.sign(cq)
        new_qty = cq + qty
        new_cost = cc - avg * close if abs(qty) <= abs(cq) else new_qty * price
    if abs(new_qty) < 1e-9:
        positions.pop(bucket, None)
    else:
        positions[bucket] = {"qty": new_qty, "cost": new_cost}
    return realized


def run(trades, pmf_table, kalshi, year, start_cash=None):
    """Run the backtest for one year. Returns a DataFrame indexed by date with
    portfolio_value (mark-to-market, incl. APY) and model_value (mark-to-model),
    plus realized/unrealized/cash columns; the final row includes settlement.

    `trades` is the raw SIGNAL log from signals.generate() -- it re-fires every
    day a mispricing persists, not just on the day a new order would actually
    be placed. Most of those rows are suppressed by the no-pyramiding cap below
    (already-at-capacity same-direction signals are skipped); the ones that
    actually change a position are recorded in `.attrs["executed_trades"]` on
    the returned frame, which is the correct trade count to report."""
    start_cash = config.START_CASH if start_cash is None else start_cash
    bid, ask = kalshi[year]["bid"], kalshi[year]["ask"]
    pmf = pmf_table[year]
    dates = pmf.index
    trades_by_day = {d: g for d, g in trades.groupby("day")} if len(trades) else {}

    cash = start_cash
    positions = {}
    last_mark = {}
    realized_cum = 0.0
    rows = []
    executed = []

    for d in dates:
        # execute trades (no pyramiding: skip a same-direction fill once the
        # per-bucket exposure cap is reached; opposite-direction fills close)
        if d in trades_by_day:
            for _, t in trades_by_day[d].iterrows():
                cur = positions.get(t["bucket"], {}).get("qty", 0.0)
                cap = config.MAX_LOTS_PER_BUCKET * abs(t["qty"])
                if cur != 0 and np.sign(cur) == np.sign(t["qty"]) and abs(cur) >= cap:
                    continue
                realized_cum += _apply_trade(positions, t["bucket"], t["qty"], t["price"])
                cash -= t["qty"] * t["price"] + t["fee"]
                executed.append(t)

        # Mark to the MID of a valid two-sided book. Degenerate/empty books
        # (bid 0 / ask 100) and missing quotes are not tradable prices, so we
        # fall back to the last valid mid, then to the model probability.
        mtm_pos = 0.0
        model_pos = 0.0
        unreal = 0.0
        for b, pos in positions.items():
            q = pos["qty"]
            kb = bid.loc[d, b] / 100.0 if (d in bid.index and not pd.isna(bid.loc[d, b])) else np.nan
            ka = ask.loc[d, b] / 100.0 if (d in ask.index and not pd.isna(ask.loc[d, b])) else np.nan
            mp = pmf.loc[d, b] if (d in pmf.index and not pd.isna(pmf.loc[d, b])) else np.nan
            if is_valid_book(kb, ka):
                m = 0.5 * (kb + ka)
                last_mark[b] = m
            elif b in last_mark:
                m = last_mark[b]
            else:
                m = mp if not np.isnan(mp) else 0.0
            mtm_pos += q * m
            unreal += q * m - pos["cost"]
            model_pos += q * (mp if not np.isnan(mp) else m)

        rows.append(dict(day=d, cash=cash, mtm_positions=mtm_pos,
                         model_positions=model_pos, unrealized=unreal,
                         realized_cum=realized_cum))

    m = pd.DataFrame(rows).set_index("day")
    m["portfolio_value"] = m["cash"] + m["mtm_positions"]
    m["model_value"] = m["cash"] + m["model_positions"]

    # monthly APY accrual on total portfolio value, added to both series
    apy_m = config.KALSHI_APY / 12
    interest = pd.Series(0.0, index=m.index)
    for i in range(1, len(m)):
        if m.index[i].month != m.index[i - 1].month:
            interest.iloc[i] = m["portfolio_value"].iloc[i - 1] * apy_m
    interest = interest.cumsum()
    m["portfolio_value"] += interest
    m["model_value"] += interest

    # settlement at true year-end close: held buckets pay $1 if in range
    close = config.SPX_YEAR_END_CLOSE[year]
    settle = 0.0
    for b, pos in positions.items():
        lo, hi = buckets.bucket_bounds(b)
        settle += pos["qty"] * (1.0 if lo <= close <= hi else 0.0)
    m.attrs["settlement"] = settle
    m.attrs["final_value"] = m["portfolio_value"].iloc[-1] - m["mtm_positions"].iloc[-1] + settle
    m.attrs["year_end_close"] = close
    m.attrs["n_signals"] = len(trades)                 # raw signal-log rows (re-fires daily)
    m.attrs["executed_trades"] = pd.DataFrame(executed) if executed else pd.DataFrame(
        columns=["day", "bucket", "qty", "price", "fee", "model_p"])
    m.attrs["n_executed"] = len(executed)               # actual fills after the pyramiding cap
    return m
