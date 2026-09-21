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

    A book is rejected if any of three things holds:
      1. a side is missing;
      2. it is EMPTY: bid pinned at the floor and ask at the ceiling (a corner test);
      3. its spread is at least `config.MAX_BOOK_SPREAD` (half the probability
         range): one side is then a stale or unfilled offer, not a price.

    (1)-(2) alone let stray quotes through. Near-ceiling asks are a quote-feed
    outage artifact, not pricing: in 2024 the 4800-4999.99 bucket quotes an ask of
    2-3c through 11/12, then 100, 99, 100, 98, 97 on 11/16-11/21, then 2c again on
    11/22 with no trade between (the whole-day version is handled by
    `feed_outage_days`). Rule (3) covers the isolated version -- one dead bucket
    quoting an ask of 85c against a bid of 0 on 2024-11-23, or bid 13c / ask 99c
    on a live bucket on 2024-06-14/15 -- which marking a short at the ask turned
    into phantom losses. The 0.50 cutoff is not fitted; see `config.MAX_BOOK_SPREAD`.
    """
    b = np.asarray(bid, dtype=float)
    a = np.asarray(ask, dtype=float)
    quoted = ~(np.isnan(b) | np.isnan(a))
    degenerate = (b <= config.MARK_MIN_BID) & (a >= config.MARK_MAX_ASK)
    with np.errstate(invalid="ignore"):
        too_wide = (a - b) >= config.MAX_BOOK_SPREAD
    return quoted & ~degenerate & ~too_wide


def feed_outage_days(kalshi: dict, year: int) -> pd.DatetimeIndex:
    """Dates whose Kalshi quotes are internally incoherent and must not be used.

    Kalshi's API declares these events `mutually_exclusive: true`, so at most ONE
    bucket can be highly probable. Two or more buckets quoting an ask at or above
    `config.OUTAGE_ASK_LEVEL` therefore cannot both be genuine offers -- it would
    imply two outcomes whose probabilities sum well past 100% -- and marks an
    empty offer side being reported at the ceiling.

    This catches an outage the per-cell `is_valid_book` corner test cannot. On
    2024-11-16 all 13 buckets read bid 0 / ask 100 while the last-trade column
    still showed sane 1-13c values, and the outage decays through 11/21 with
    asks of 97-99c that slip under the 98c per-cell threshold. Those marks are
    not cosmetic: two 8-lot shorts marked at 49.5c instead of ~1c on 2024-11-21
    produce the whole of the reported 2024 maximum drawdown (-4.35% with them,
    -1.58% without).

    The detector is a LEVEL test rather than a sum-of-asks test on purpose: a
    summed-ask bound conflates a merely WIDE market with a broken one. On
    2022-07-07 the asks sum to 2.81 -- above any reasonable coherence bound --
    yet every individual ask is <= 25c against 0-1c bids, which is a real if
    lazy market-maker ladder on the first day the market existed, not a feed
    failure. The level test leaves such days alone.

    Flagged days are treated exactly like a degenerate book -- positions fall
    back to the last valid mid, then to the model -- rather than being dropped,
    so the mark-to-market path stays continuous.
    """
    ask = kalshi[year]["ask"] / 100.0
    n_high = (ask >= config.OUTAGE_ASK_LEVEL).sum(axis=1)
    return pd.DatetimeIndex(n_high.index[n_high >= config.OUTAGE_MIN_BUCKETS])


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
    out = mid.where(is_valid_book(bid, ask))
    out.loc[out.index.intersection(feed_outage_days(kalshi, year))] = np.nan
    return out
