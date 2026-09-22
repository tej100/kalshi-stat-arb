"""Step 7 - portfolio engine for the Kalshi bucket strategy.

Marks open positions to the MID of a valid two-sided Kalshi book each day (see
the marking block in `run` for the fallback chain), accrues Kalshi's interest
on the collateral posted for open positions (only from the date Kalshi started
paying it), and settles held buckets at the true year-end SPX close ($1 if
in-bucket).

Produces both a mark-to-MARKET series (only fillable prices) and a
mark-to-MODEL series (positions valued at the model probability), so their
correlation can quantify how executable the ideal strategy is.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from ..transform import buckets
from ..transform.kalshi_pmf import is_valid_book, feed_outage_days


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


def collateral(positions) -> float:
    """Cash Kalshi holds against the open book: p per long, 1 - p per short."""
    return sum(p["cost"] if p["qty"] > 0 else abs(p["qty"]) + p["cost"]
               for p in positions.values())


def run(trades, pmf_table, kalshi, year, start_cash=None, stale_mark="book"):
    """Run the backtest for one year. Returns a DataFrame indexed by date with
    portfolio_value (mark-to-market, incl. Kalshi interest) and model_value (mark-to-model),
    plus realized/unrealized/cash columns; the final row includes settlement.

    `trades` is the raw SIGNAL log from signals.generate() -- it re-fires every
    day a mispricing persists, not just on the day a new order would actually
    be placed. Most of those rows are suppressed by the no-pyramiding cap below
    (already-at-capacity same-direction signals are skipped); the ones that
    actually change a position are recorded in `.attrs["executed_trades"]` on
    the returned frame, which is the correct trade count to report.

    `stale_mark` decides how a position is marked on a day its bucket has no
    valid book: "book" (default) carries the last valid book forward; "model"
    marks it to that day's model probability where one exists. A carried mark
    does not move, which can understate volatility; the "model" setting is the
    check on how much (see RISK_PROFILE.md). `.attrs["stale_share"]` reports the
    fraction of position-days marked from a carried book."""
    start_cash = config.START_CASH if start_cash is None else start_cash
    bid, ask = kalshi[year]["bid"], kalshi[year]["ask"]
    pmf = pmf_table[year]
    dates = pmf.index
    trades_by_day = {d: g for d, g in trades.groupby("day")} if len(trades) else {}

    # Dates whose whole quote set is internally incoherent (see
    # kalshi_pmf.feed_outage_days). Their books are treated exactly like
    # degenerate ones: fall back to the last valid mid, then the model.
    outage = set(feed_outage_days(kalshi, year))

    cash = start_cash
    positions = {}
    last_mark = {}
    n_marks = n_stale = 0
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

        # Mark each position to the side it would actually LIQUIDATE against: a
        # long can only be sold at the bid, a short can only be covered at the
        # ask. Marking to the mid would carry every open position at half a
        # spread better than it could be realised. Degenerate/empty books (bid 0
        # / ask 100) and outage days are not tradable prices, so we fall back to
        # the last valid book, then to the model probability.
        mtm_pos = 0.0
        model_pos = 0.0
        unreal = 0.0
        for b, pos in positions.items():
            q = pos["qty"]
            kb = bid.loc[d, b] / 100.0 if (d in bid.index and not pd.isna(bid.loc[d, b])) else np.nan
            ka = ask.loc[d, b] / 100.0 if (d in ask.index and not pd.isna(ask.loc[d, b])) else np.nan
            mp = pmf.loc[d, b] if (d in pmf.index and not pd.isna(pmf.loc[d, b])) else np.nan
            fresh = d not in outage and is_valid_book(kb, ka)
            if fresh:
                last_mark[b] = (kb, ka)
            n_marks += 1
            if not fresh and b in last_mark:
                n_stale += 1
            if not fresh and stale_mark == "model" and not np.isnan(mp):
                m = mp
            elif b in last_mark:
                lb, la = last_mark[b]
                m = lb if q > 0 else la
            else:
                m = mp if not np.isnan(mp) else 0.0
            mtm_pos += q * m
            unreal += q * m - pos["cost"]
            model_pos += q * (mp if not np.isnan(mp) else m)

        rows.append(dict(day=d, cash=cash, mtm_positions=mtm_pos,
                         model_positions=model_pos, unrealized=unreal,
                         realized_cum=realized_cum, collateral=collateral(positions)))

    m = pd.DataFrame(rows).set_index("day")
    m["portfolio_value"] = m["cash"] + m["mtm_positions"]
    m["model_value"] = m["cash"] + m["model_positions"]

    # Kalshi interest on posted collateral, accrued daily on the previous day's
    # collateral and only from the date Kalshi began paying it. Idle cash is not
    # credited here: it is assumed to earn the risk-free rate outside the account,
    # which metrics.excess_value accounts for by charging the collateral, not
    # the cash, for the cost of capital.
    days = m.index.to_series().diff().dt.days.fillna(0).values
    rate = np.where(m.index >= pd.Timestamp(config.KALSHI_INTEREST_START),
                    config.KALSHI_INTEREST_RATE, 0.0)
    interest = pd.Series(np.cumsum(m["collateral"].shift().fillna(0).values * rate * days / 365.0),
                         index=m.index)
    m["interest"] = interest
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
    m.attrs["stale_share"] = n_stale / n_marks if n_marks else np.nan
    m.attrs["n_executed"] = len(executed)               # actual fills after the pyramiding cap
    return m
