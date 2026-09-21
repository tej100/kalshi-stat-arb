"""Step 6 - trade signals from model-vs-market bucket mispricing.

A bucket is bought when the model probability exceeds the Kalshi ask (plus fee)
and sold when it is below the Kalshi bid (minus fee). Fees follow Kalshi's
schedule; see config. `side` filters buy-only / sell-only sub-strategies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from ..transform.kalshi_pmf import feed_outage_days


def kalshi_fee(price, contracts):
    """Kalshi fee in dollars for a lot: ceil(rate * C * 100 * p(1-p)) / 100."""
    contracts = abs(contracts)
    p = np.clip(price, 0.0, 1.0)
    return np.ceil(config.KALSHI_FEE_RATE * contracts * 100 * p * (1 - p)) / 100


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
            if side in ("both", "buy") and mp > ka + kalshi_fee(ka, lot):
                rows.append(dict(day=day, bucket=bucket, qty=lot, price=ka,
                                 fee=kalshi_fee(ka, lot), model_p=mp))
            elif side in ("both", "sell") and mp < kb - kalshi_fee(kb, lot):
                rows.append(dict(day=day, bucket=bucket, qty=-lot, price=kb,
                                 fee=kalshi_fee(kb, lot), model_p=mp))
    return pd.DataFrame(rows)
