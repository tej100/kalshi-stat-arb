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
| BL | 2022 | $70.16 | 32.3% | 0 | 5.4 / 13 |
| BL | 2023 | $88.32 | 41.3% | 0 | 12.2 / 13 |
| BL | 2024 | $89.28 | 43.3% | 0 | 10.8 / 12 |
| GBM | 2022 | $58.32 | 26.8% | 0 | 5.7 / 13 |
| GBM | 2023 | $90.72 | 43.6% | 0 | 11.9 / 13 |
| GBM | 2024 | $87.76 | 44.2% | 0 | 12.3 / 13 |

Peak use never exceeds 44% of the account, so the unmodelled collateral does not
overstate returns; the reported P&L was achievable without leverage.

## 2. Tail risk: worst-case settlement along the whole path

Only three year-end outcomes occurred, which says nothing about how a different
outcome would have hurt. The buckets are mutually exclusive, so exactly one pays (or
none, if SPX settles outside all thirteen). On every date this stress supposes the year
had settled that day in the worst bucket for the open book:

| model | year | worst settle-now P&L | as % of capital | date | days below start capital |
|---|---|---|---|---|---|
| BL | 2022 | −$2.63 | −1.3% | 2022-07-22 | 27 |
| BL | 2023 | −$4.36 | −2.2% | 2023-03-29 | 166 |
| BL | 2024 | −$11.02 | −5.5% | 2024-01-03 | 189 |
| GBM | 2022 | −$3.46 | −1.7% | 2022-08-16 | 52 |
| GBM | 2023 | −$10.59 | −5.3% | 2023-04-27 | 339 |
| GBM | 2024 | −$8.89 | −4.4% | 2024-01-03 | 365 |

The loss is bounded because one lot per bucket caps the worst single-bucket payout at
$8, so the tail cannot exceed roughly 4% of capital plus accrued costs, however SPX
settles. This is a stylised bound (it ignores the path to settlement), not a VaR.

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
closing anything) drops 2022 BL total return from 13.3% to 0.25%. For GBM it is roughly neutral.
Note also that 16 of the 34 losing BL exits would have ended profitable if held, so the
rule is not perfect, only net-positive.

## 4. How much does the Sharpe depend on how it is measured?

The daily Sharpe scales by the square root of 252, which assumes daily returns are
uncorrelated. Here they are not (lag-1 autocorrelation runs from −0.17 to +0.07:
thin order books bounce and revert), so the same P&L gives different Sharpe ratios at
different sampling frequencies. The 5- and 20-day figures below use non-overlapping
returns, averaged over every possible starting phase so no arbitrary start day is
chosen.

| model | year | daily | 5-day | 20-day | lag-1 autocorr |
|---|---|---|---|---|---|
| BL | 2022 | 1.99 | 2.84 | 3.73 | −0.17 |
| BL | 2023 | 2.25 | 2.11 | 1.67 | +0.07 |
| BL | 2024 | 1.08 | 1.22 | 1.92 | −0.06 |
| GBM | 2022 | 2.35 | 3.49 | 3.22 | −0.12 |
| GBM | 2023 | 1.63 | 1.13 | 0.78 | −0.12 |
| GBM | 2024 | −1.26 | −1.54 | −1.92 | −0.13 |

**The sign is identical at all three frequencies in 6 of 6 model-years; the magnitude
is not.** BL is positive at every frequency and GBM 2024 is negative at every
frequency. The 20-day figures rest on only 6-12 observations each, so they are noisy;
the point is that the conclusion does not hinge on the daily scaling, not that any one
number is right. This is one more reason to quote a Sharpe only with its bootstrap
interval.

Alpha against SPX is only marginally distinguishable from zero: two of six
model-years reach |t| of about 2 (BL 2023: 7.5% annualised, Newey-West t = 2.09;
GBM 2022: 14.5%, t = 2.04) and the rest are below 1.9. Beta is at most 0.04 in absolute value throughout.

## Limits

Three years and 40–120 positions per model: these are descriptions of what happened,
not estimates with tight uncertainty. Collateral is measured at entry prices, and the
stress assumes a single-bucket loss on the day it is evaluated.
