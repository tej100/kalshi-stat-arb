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

    The columns really are yes_bid / yes_ask, NOT a yes/no pair: if `ask` were
    the NO bid, then bid + ask would sit at ~100 by construction (yes_ask =
    100 - no_bid). Measured, bid + ask has a median of 13-16c and lands in
    99..101 for only 0.0-2.3% of quotes, so the labels are correct.

    NOTE (known gap, deliberate): this is a CORNER test (bid pinned at the floor
    AND ask pinned at the ceiling), not a spread test, so it catches an ask of
    98c but not 97c. That matters because near-ceiling asks are a BOOK-OUTAGE
    artifact, not real pricing: in 2024 the 4800-4999.99 bucket quotes an ask of
    2-3c through 11/12, then 100, 99, 100, 98, 97 on 11/16-11/21, then snaps back
    to 2c on 11/22, with no trade in between -- one contiguous outage whose tail
    happens to dip below the 98c threshold. On 11/16 every one of the 13 buckets
    reads 0/100 while the last-trade column still shows sane 1-13c values, which
    is the cleanest demonstration that the quote side, not the trade side, is
    what degrades. Incidence is small (2 marks in 2024, 0 in 2022-23), so this
    is documented rather than patched with a fresh cutoff: the principled repair
    is a marking policy that cannot be fooled by a one-sided book (mark to the
    side a position would actually liquidate against), which is a Phase 8
    decision. See PROJECT_CONTEXT.md section 12.
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
