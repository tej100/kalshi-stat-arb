"""Step 6 - trade signals from model-vs-market bucket mispricing.

A bucket is bought when the model probability exceeds the Kalshi ask by more
than the per-contract fee hurdle, and sold when it is below the Kalshi bid by
the same margin. Fees follow Kalshi's schedule; see config. `side` filters
buy-only / sell-only sub-strategies.

UNITS -- `kalshi_fee` returns the fee for the WHOLE LOT in dollars (that is how
`backtest.run` spends it), while `model_p`, `bid` and `ask` are PER-CONTRACT
prices in [0, 1]. Comparing the two directly demands LOT_SIZE times the edge a
trade actually needs, so the hurdle must be divided by the lot -- see
`entry_hurdle`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from ..transform.kalshi_pmf import feed_outage_days, is_valid_book


def kalshi_fee(price, contracts):
    """Kalshi fee in DOLLARS FOR THE WHOLE LOT: ceil(rate*C*100*p(1-p))/100."""
    contracts = abs(contracts)
    p = np.clip(price, 0.0, 1.0)
    return np.ceil(config.KALSHI_FEE_RATE * contracts * 100 * p * (1 - p)) / 100


def entry_hurdle(price, contracts):
    """Per-contract edge (in probability units) a signal must clear.

    Buying C contracts at `price` costs C*price + fee(price, C) and returns
    C*model_p, so the trade is worth taking when

        C*(model_p - price) > fee   <=>   model_p - price > fee / C

    i.e. the hurdle is the fee PER CONTRACT, which is what `model_p` and
    `price` are denominated in. Comparing against the undivided lot fee would
    demand C times too much edge.

    Kalshi charges the taker fee on both fills of a market-order round trip, so
    the hurdle is `config.FEE_ROUND_TRIP_FILLS` such fees. The exit fee is
    approximated at the entry price, since the exit price is unknown when the
    signal fires; it is a hurdle for deciding whether to trade, while the
    backtest charges each fill its own actual fee.
    """
    return (config.FEE_ROUND_TRIP_FILLS
            * kalshi_fee(price, contracts) / abs(contracts))


def generate(pmf_table, kalshi, year, side="both", lot=None):
    """Return a trades DataFrame for one year.

    Columns: day, bucket, qty (signed contracts), price (execution, 0-1),
             fee ($ per lot), model_p.
    side in {'both','buy','sell'}.
    """
    lot = config.LOT_SIZE if lot is None else lot
    pmf = pmf_table[year]
    bid = kalshi[year]["bid"]
    ask = kalshi[year]["ask"]
    # A day whose whole quote set is internally incoherent is a broken feed, not
    # a tradable market: no order would be placed against those prices.
    outage = set(feed_outage_days(kalshi, year))
    rows = []
    for day in pmf.index:
        if day not in bid.index or day not in ask.index or day in outage:
            continue
        for bucket in pmf.columns:
            mp = pmf.loc[day, bucket]
            kb, ka = bid.loc[day, bucket], ask.loc[day, bucket]
            if pd.isna(mp) or pd.isna(kb) or pd.isna(ka):
                continue
            kb, ka = kb / 100.0, ka / 100.0
            if not is_valid_book(kb, ka):      # same validity rule as marking and hedging
                continue
            if side in ("both", "buy") and mp > ka + entry_hurdle(ka, lot):
                rows.append(dict(day=day, bucket=bucket, qty=lot, price=ka,
                                 fee=kalshi_fee(ka, lot), model_p=mp))
            elif side in ("both", "sell") and mp < kb - entry_hurdle(kb, lot):
                rows.append(dict(day=day, bucket=bucket, qty=-lot, price=kb,
                                 fee=kalshi_fee(kb, lot), model_p=mp))
    return pd.DataFrame(rows)
