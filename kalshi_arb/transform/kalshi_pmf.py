"""Transform - Kalshi order book -> market-implied bucket PMF.

A Kalshi 'yes' price is already a probability, so the transform is light: take
the mid of a valid two-sided book per bucket per day. This puts the market on
the same date x bucket PMF footing as the options-implied model densities, so
the strategy stage compares two homogeneous objects. Execution still uses the
raw bid/ask (in `kalshi[year]`), which this PMF does not replace.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config


def _valid(kb, ka):
    if np.isnan(kb) or np.isnan(ka):
        return False
    if kb <= config.MARK_MIN_BID and ka >= config.MARK_MAX_ASK:
        return False
    return True


def market_pmf(kalshi: dict, year: int) -> pd.DataFrame:
    """date x bucket table of Kalshi mid-implied probabilities for one year.

    Degenerate/empty books (bid 0 / ask 100) and missing quotes are left NaN
    rather than trusted as a probability.
    """
    bid = kalshi[year]["bid"] / 100.0
    ask = kalshi[year]["ask"] / 100.0
    mid = 0.5 * (bid + ask)
    valid = (~bid.isna()) & (~ask.isna()) & ~((bid <= config.MARK_MIN_BID) &
                                              (ask >= config.MARK_MAX_ASK))
    return mid.where(valid)
