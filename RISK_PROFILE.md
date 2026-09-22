# Risk Profile

The Sharpe ratio alone says little about a book that trades a few dozen binary lots on
a $200 base, so this note reports three risk facts a reader should have: whether the
strategy uses hidden leverage, how bad the tail could be, and where the profit came
from. Everything is reproduced by [`analysis/risk_profile.py`](analysis/risk_profile.py),
built on [`kalshi_arb/strategy/diagnostics.py`](kalshi_arb/strategy/diagnostics.py).
Figures are for the both-side book, one lot (8 contracts) per bucket.

## 1. Collateral: no hidden leverage

Kalshi is fully collateralised: a long costs `p` per contract, and a short (sold YES)
requires posting `1 − p`. The backtest credits short proceeds to cash without reserving
that collateral, so the question is whether the book would have been fundable.

| model | year | peak collateral | peak use of account | days above 100% | avg / max open buckets |
|---|---|---|---|---|---|
| BL | 2022 | $70.16 | 33.4% | 0 | 5.4 / 13 |
| BL | 2023 | $88.32 | 42.7% | 0 | 12.2 / 13 |
| BL | 2024 | $89.28 | 44.4% | 0 | 10.8 / 12 |
| GBM | 2022 | $58.32 | 27.7% | 0 | 5.7 / 13 |
| GBM | 2023 | $90.72 | 45.1% | 0 | 11.9 / 13 |
| GBM | 2024 | $87.76 | 45.5% | 0 | 12.3 / 13 |

Peak use never exceeds 46% of the account, so the unmodelled collateral does not
overstate returns; the reported P&L was achievable without leverage. The collateral
is not free, though: it is the capital that earns Kalshi's rate (nothing before
10 October 2024) instead of the risk-free rate, and the metrics charge it that
difference every day it is posted (`metrics.excess_value`).

## 2. Tail risk: worst-case settlement along the whole path

Only three year-end outcomes occurred, which says nothing about how a different
outcome would have hurt. The buckets are mutually exclusive, so exactly one pays (or
none, if SPX settles outside all thirteen). On every date this stress supposes the year
had settled that day in the worst bucket for the open book:

| model | year | worst settle-now P&L | as % of capital | date | days below start capital |
|---|---|---|---|---|---|
| BL | 2022 | −$6.38 | −3.2% | 2022-07-22 | 94 |
| BL | 2023 | −$6.14 | −3.1% | 2023-04-18 | 287 |
| BL | 2024 | −$11.02 | −5.5% | 2024-01-03 | 302 |
| GBM | 2022 | −$7.83 | −3.9% | 2022-08-16 | 81 |
| GBM | 2023 | −$12.73 | −6.4% | 2023-06-02 | 363 |
| GBM | 2024 | −$14.20 | −7.1% | 2024-10-21 | 365 |

The loss is bounded because one lot per bucket caps the worst single-bucket payout at
$8, so the stress cannot exceed roughly 4% of capital beyond the fees and losses
already booked, however SPX settles. (An earlier version of this table credited a
3.75% APY that Kalshi did not pay before October 2024, which made every row look
$3-7 better.) This is a stylised bound (it ignores the path to settlement), not a VaR.

Worth knowing: in 2024 the book was at times short 12 of 13 mutually exclusive buckets,
i.e. a bet that SPX would not land in any of them. It is bounded, but it is a
concentrated bet on the shape of the whole distribution.

## 3. Where the profit came from

Each row of the table is one position, net of fees. "Closed" positions were exited by
an opposite signal; "settled" ones were held to year-end.

| model | how it ended | positions | net P&L | mean per position | win rate |
|---|---|---|---|---|---|
| BL | closed | 94 | +$17.25 | $0.18 | 64% |
| BL | settled | 32 | +$27.61 | $0.86 | 97% |
| GBM | closed | 120 | +$6.02 | $0.05 | 49% |
| GBM | settled | 36 | +$24.41 | $0.68 | 89% |

Settled positions produce most of the profit, but **their win rate is selected by the
exit rule, not a property of the strategy**: they are the positions the model kept
agreeing with, while the ones that went wrong were closed. The counterfactual makes
this concrete. Had each closed position instead been held to settlement:

| model | closed positions | realised | if held to settlement | effect of closing |
|---|---|---|---|---|
| BL | 94 | +$17.25 | −$30.63 | **+$47.88** |
| GBM | 120 | +$6.02 | +$5.76 | +$0.26 |

For BL, 34 positions were exited at a loss (−$6.89 realised); held, they would have
lost $20.94. So the exit rule works as a model-based stop, and removing it (never
closing anything) turns 2022 BL total return from +9.8% into −7.4%. For GBM it is roughly neutral.
Note also that 16 of the 34 losing BL exits would have ended profitable if held, so the
rule is not perfect, only net-positive.

## 4. How much does the Sharpe depend on how it is measured?

The daily Sharpe scales by the square root of 252, which assumes daily returns are
uncorrelated. Here they are not (lag-1 autocorrelation runs from −0.18 to +0.04:
thin order books bounce and revert), so the same P&L gives different Sharpe ratios at
different sampling frequencies. The 5- and 20-day figures below use non-overlapping
returns, averaged over every possible starting phase so no arbitrary start day is
chosen.

| model | year | daily | 5-day | 20-day | lag-1 autocorr |
|---|---|---|---|---|---|
| BL | 2022 | 1.97 | 2.80 | 3.57 | −0.16 |
| BL | 2023 | 1.90 | 1.72 | 1.16 | +0.04 |
| BL | 2024 | 0.68 | 0.70 | 0.90 | +0.03 |
| GBM | 2022 | 2.38 | 3.45 | 3.14 | −0.10 |
| GBM | 2023 | 1.24 | 0.63 | 0.04 | −0.09 |
| GBM | 2024 | −1.98 | −2.63 | −2.93 | −0.18 |

**The sign is identical at all three frequencies in 6 of 6 model-years; the magnitude
is not.** BL is positive at every frequency and GBM 2024 is negative at every
frequency. The 20-day figures rest on only 6-12 observations each, so they are noisy;
the point is that the conclusion does not hinge on the daily scaling, not that any one
number is right. This is one more reason to quote a Sharpe only with its bootstrap
interval.

Alpha against SPX is only marginally distinguishable from zero: one model-year
reaches t of about 2 in the right direction (GBM 2022: 14.7% annualised, Newey-West
t = 2.05), GBM 2024 is significantly negative (t = −2.39), and BL is between 0.55
and 1.80 in every year. Beta is at most 0.03 in absolute value throughout.

## 5. How much depends on the execution timing?

The headline fills each signal at the Kalshi close of the same day whose option
chain produced it. The option quotes are the day's last quotes (probably 4:15pm) and
the Kalshi candle closes at 4:00pm, so that convention can see up to 15 minutes of
option-market news before the trade. [`analysis/execution_lag.py`](analysis/execution_lag.py)
removes it: the model is taken from the previous priced day and each signal is
re-tested against the Kalshi book it actually fills against.

| model | year | same close | model 1 day old | model 2 days old |
|---|---|---|---|---|
| BL | 2022 | 1.97 | 0.86 | −0.05 |
| BL | 2023 | 1.90 | −0.09 | −0.26 |
| BL | 2024 | 0.68 | 0.50 | 0.38 |
| GBM | 2022 | 2.38 | 0.79 | 0.59 |
| GBM | 2023 | 1.24 | −1.75 | −1.63 |
| GBM | 2024 | −1.98 | −2.32 | −2.74 |

**Most of the headline Sharpe is front-loaded into the first day.** With a one-day-old
model, BL stays positive in 2022 and 2024 and is flat in 2023, with intervals that all
include zero; GBM loses money in two of three years. Consistent with this, the Kalshi
mid moves in a fill's favour by 0.65¢ per contract the next day and about 1.2¢ after
five days, then stops. Part of the same-close result may therefore be the timing
convention rather than the model, and the lagged column is the conservative reading.

## Limits

Three years and 40–120 positions per model: these are descriptions of what happened,
not estimates with tight uncertainty. Collateral is measured at entry prices, and the
stress assumes a single-bucket loss on the day it is evaluated.
