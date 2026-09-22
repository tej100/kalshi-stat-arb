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
| BL | 2022 | $69.92 | 33.3% | 0 | 5.4 / 13 |
| BL | 2023 | $88.32 | 42.7% | 0 | 12.2 / 13 |
| BL | 2024 | $89.60 | 44.5% | 0 | 10.9 / 12 |
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
| BL | 2023 | −$6.14 | −3.1% | 2023-04-18 | 286 |
| BL | 2024 | −$11.02 | −5.5% | 2024-01-03 | 365 |
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
| BL | closed | 93 | +$15.76 | $0.17 | 61% |
| BL | settled | 33 | +$28.46 | $0.86 | 97% |
| GBM | closed | 119 | +$4.94 | $0.04 | 49% |
| GBM | settled | 36 | +$23.76 | $0.66 | 89% |

Settled positions produce most of the profit, but **their win rate is selected by the
exit rule, not a property of the strategy**: they are the positions the model kept
agreeing with, while the ones that went wrong were closed. The counterfactual makes
this concrete. Had each closed position instead been held to settlement:

| model | closed positions | realised | if held to settlement | effect of closing |
|---|---|---|---|---|
| BL | 93 | +$15.76 | −$33.56 | **+$49.32** |
| GBM | 119 | +$4.94 | −$0.27 | +$5.21 |

For BL, 36 positions were exited at a loss (−$7.38 realised); held, they would have
lost $22.51. So the exit rule works as a model-based stop, and removing it (never
closing anything) turns 2022 BL total return from +9.8% into −7.6%. For GBM it matters
little. Note also that 16 of the 36 losing BL exits would have ended profitable if held,
so the rule is not perfect, only net-positive. A convergence exit (close as soon as the
bid or ask reaches the model) gives the same pooled Sharpe, 1.46 against 1.44, with 26%
more fills, so the simpler rule is kept.

## 4. How much does the Sharpe depend on how it is measured?

The daily Sharpe scales by the square root of 252, which assumes daily returns are
uncorrelated. Here they are not (lag-1 autocorrelation runs from −0.18 to +0.05:
thin order books bounce and revert), so the same P&L gives different Sharpe ratios at
different sampling frequencies. The 5- and 20-day figures below use non-overlapping
returns, averaged over every possible starting phase so no arbitrary start day is
chosen.

| model | year | daily | 5-day | 20-day | lag-1 autocorr |
|---|---|---|---|---|---|
| BL | 2022 | 1.98 | 2.81 | 3.57 | −0.16 |
| BL | 2023 | 1.90 | 1.73 | 1.17 | +0.04 |
| BL | 2024 | 0.55 | 0.57 | 0.58 | +0.05 |
| GBM | 2022 | 2.31 | 3.42 | 3.14 | −0.10 |
| GBM | 2023 | 1.08 | 0.56 | 0.05 | −0.13 |
| GBM | 2024 | −1.98 | −2.63 | −2.93 | −0.18 |

**The sign is identical at all three frequencies in 6 of 6 model-years; the magnitude
is not.** BL is positive at every frequency and GBM 2024 is negative at every
frequency. The 20-day figures rest on only 6-12 observations each, so they are noisy;
the point is that the conclusion does not hinge on the daily scaling, not that any one
number is right. This is one more reason to quote a Sharpe only with its bootstrap
interval.

Alpha against SPX is only marginally distinguishable from zero year by year: one
model-year reaches t of about 2 in the right direction (GBM 2022: 14.2% annualised,
Newey-West t = 2.00), GBM 2024 is significantly negative (t = −2.39), and BL is between
0.40 and 1.80 in every year. Beta is at most 0.03 in absolute value throughout. The
test with enough power is the pooled one below.

## 5. How much depends on the execution timing?

The headline fills each signal at the Kalshi close of the same day whose option
chain produced it. The option quotes are the day's last quotes (probably 4:15pm) and
the Kalshi candle closes at 4:00pm, so that convention can see up to 15 minutes of
option-market news before the trade. [`analysis/execution_lag.py`](analysis/execution_lag.py)
removes it: the model is taken from the previous priced day and each signal is
re-tested against the Kalshi book it actually fills against.

| model | year | same close | model 1 day old | model 2 days old |
|---|---|---|---|---|
| BL | 2022 | 1.98 | 0.67 | 0.00 |
| BL | 2023 | 1.90 | −0.13 | −0.22 |
| BL | 2024 | 0.55 | 0.52 | 0.37 |
| GBM | 2022 | 2.31 | 0.79 | 0.50 |
| GBM | 2023 | 1.08 | −1.70 | −1.62 |
| GBM | 2024 | −1.98 | −2.32 | −2.74 |
| **BL pooled [95% CI]** | 2022–24 | **1.44 [0.36, 2.49]** | 0.32 [−0.70, 1.34] | 0.03 [−0.97, 1.05] |
| GBM pooled | 2022–24 | 0.70 [−0.58, 1.88] | −1.02 [−2.09, 0.11] | −1.12 [−2.22, 0.01] |

Pooling the three years' daily excess returns is the test with enough power: a single
year is too short for an interval to exclude zero. **At the same close the BL edge is
significant (99.5% of bootstrap resamples positive); with a one-day delay it is not.**
The markouts show why. After a same-close fill the Kalshi mid moves in the trade's
favour by 0.66¢ per contract the next day (t = 2.3, clustered by fill date) and 1.21¢
after five days (t = 3.1), which is 0.83¢ net of the fill's fee (t = 2.1), then stops.
A fill placed a day late moves *against* the trade by 0.62¢ the next day (t = −3.4):
most of a mispricing is corrected within a day, so a late trade enters after the
correction has begun. The edge is real and belongs to a trader who acts on the day it
appears; the one-day lag is a conservative bound, since the actual lookahead is 15
minutes. Hourly Kalshi candles would allow trading after the option close on the same
day and measure it directly.

## 6. Stale marks

Late in each year the buckets the index has left stop being quoted, so an open
position there has no valid book. The headline marks it to the last valid book, which
does not move. The share of position-days marked this way is 18% / 36% / 66% for BL
(20% / 37% / 68% for GBM). A mark that does not move understates daily volatility,
so the same run was repeated marking those days to the model probability instead:

| model | year | stale share | daily Sharpe, book | daily Sharpe, model | 20-day Sharpe, book | 20-day Sharpe, model |
|---|---|---|---|---|---|---|
| BL | 2022 | 18% | 1.98 | 2.15 | 3.57 | 3.77 |
| BL | 2023 | 36% | 1.90 | 0.99 | 1.17 | 1.09 |
| BL | 2024 | 66% | 0.55 | 0.44 | 0.58 | 0.44 |
| GBM | 2022 | 20% | 2.31 | 2.40 | 3.14 | 3.20 |
| GBM | 2023 | 37% | 1.08 | 0.72 | 0.05 | 0.11 |
| GBM | 2024 | 68% | −1.98 | −0.40 | −2.93 | −2.23 |

Neither column is the truth. The carried book understates volatility; the model mark
overstates it, because a position then jumps between a liquidation-side book price and a
model probability as the book comes and goes. Total returns are identical under both,
since the endpoints do not change. **The daily Sharpe is sensitive to this choice in
2023 and 2024; the 20-day Sharpe, which averages over the switching, is not** (BL 1.17 vs
1.09 in 2023, 0.58 vs 0.44 in 2024). Pooled, marking to the model gives 1.04 [0.19, 1.90],
still significant. BL 2023's daily 1.90 should be read with that in
mind: its 20-day figure is closer to 1.1. GBM 2024's daily −1.98 is exaggerated by the
same effect (steady carry losses on a book that barely moves); its sign is not in doubt.

## Limits

Three years and 40–120 positions per model: these are descriptions of what happened,
not estimates with tight uncertainty. Collateral is measured at entry prices, and the
stress assumes a single-bucket loss on the day it is evaluated.
