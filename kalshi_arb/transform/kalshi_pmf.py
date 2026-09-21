"""Transform - Kalshi order book -> market-implied bucket PMF.

A Kalshi 'yes' price is already a probability, so the transform is light: take
the mid of a valid two-sided book per bucket per day. This puts the market on
the same date x bucket PMF footing as the options-implied model densities, so
the two can be compared as homogeneous objects.

SCOPE -- this module is ANALYSIS-FACING, not part of the traded path. The
strategy never consumes `market_pmf`: `signals.generate` must compare the model
probability to the price it would actually PAY (the ask) or RECEIVE (the bid),
not to the mid, so it reads the raw book in `kalshi[year]` directly. The mid PMF
exists so the two venues can be studied on equal footing (see
analysis/cointegration.py, which measures the Kalshi-vs-model spread and which
venue error-corrects). `is_valid_book` below IS on the traded path -- it is the
single definition of book validity, shared with backtest marking.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config


def is_valid_book(bid, ask):
    """Is (bid, ask) a real two-sided market? Elementwise; PROBABILITY units (0-1).

    Accepts scalars or pandas/numpy containers and returns a matching boolean.
    An empty book quotes bid 0 / ask 100c -- e.g. a bucket SPX has left entirely,
    which no one will offer at any price. That ask is a phantom quote, not a
    liquidation value, so it must not mark a position or imply a probability.

    This is the ONE definition of book validity: `market_pmf` here and
    `backtest.run`'s marking both call it, so the traded path and the analysis
    path cannot drift apart.

    NOTE (known gap, deliberate): this is a CORNER test (bid pinned at the floor
    AND ask pinned at the ceiling), not a spread test. A merely terrible book
    such as bid 2c / ask 97c is a genuine quote by someone and passes, even
    though its 49.5c mid carries little information. Measured incidence is small
    (2 marks in 2024, 0 in 2022-23; ~1% of marks have a >20c spread). Widening
    this into a spread-quality filter would require a threshold that no hard
    rule pins down, so it is documented rather than guessed at -- see
    PROJECT_CONTEXT.md section 12.
    """
    b = np.asarray(bid, dtype=float)
    a = np.asarray(ask, dtype=float)
    quoted = ~(np.isnan(b) | np.isnan(a))
    degenerate = (b <= config.MARK_MIN_BID) & (a >= config.MARK_MAX_ASK)
    return quoted & ~degenerate


def market_pmf(kalshi: dict, year: int) -> pd.DataFrame:
    """date x bucket table of Kalshi mid-implied probabilities for one year.

    Degenerate/empty books (bid 0 / ask 100) and missing quotes are left NaN
    rather than trusted as a probability.

    The rows do NOT sum to 1, for two reasons: the open-ended tail buckets are
    excluded from the dataset (so genuine mass sits outside the 13 buckets), and
    a mid sits above the bid, so summing mids overstates the tradable mass. Sums
    above 1 are therefore expected and are not an arbitrage: measured over
    all-13-quoted days, sum(bid) never exceeds 1 (max 0.99) while sum(ask)
    averages ~1.0-1.2.
    """
    bid = kalshi[year]["bid"] / 100.0
    ask = kalshi[year]["ask"] / 100.0
    mid = 0.5 * (bid + ask)
    return mid.where(is_valid_book(bid, ask))
