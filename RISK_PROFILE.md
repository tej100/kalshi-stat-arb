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
| BL | 2023 | $87.04 | 40.8% | 0 | 12.4 / 13 |
| BL | 2024 | $89.44 | 43.1% | 0 | 10.8 / 12 |
| GBM | 2022 | $58.32 | 26.8% | 0 | 5.7 / 13 |
| GBM | 2023 | $87.36 | 42.8% | 0 | 11.8 / 13 |
| GBM | 2024 | $87.76 | 45.4% | 0 | 12.3 / 13 |

Peak use never exceeds 45% of the account, so the unmodelled collateral does not
overstate returns; the reported P&L was achievable without leverage.

## 2. Tail risk: worst-case settlement along the whole path

Only three year-end outcomes occurred, which says nothing about how a different
outcome would have hurt. The buckets are mutually exclusive, so exactly one pays (or
none, if SPX settles outside all thirteen). On every date this stress supposes the year
had settled that day in the worst bucket for the open book:

| model | year | worst settle-now P&L | as % of capital | date | days below start capital |
|---|---|---|---|---|---|
| BL | 2022 | −$2.63 | −1.3% | 2022-07-22 | 27 |
| BL | 2023 | −$4.37 | −2.2% | 2023-04-18 | 166 |
| BL | 2024 | −$11.77 | −5.9% | 2024-01-03 | 189 |
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
| BL | closed | 95 | +$18.82 | $0.20 | 64% |
| BL | settled | 31 | +$27.78 | $0.90 | 97% |
| GBM | closed | 122 | +$6.84 | $0.06 | 49% |
| GBM | settled | 35 | +$20.89 | $0.60 | 89% |

Settled positions produce most of the profit, but **their win rate is selected by the
exit rule, not a property of the strategy**: they are the positions the model kept
agreeing with, while the ones that went wrong were closed. The counterfactual makes
this concrete. Had each closed position instead been held to settlement:

| model | closed positions | realised | if held to settlement | effect of closing |
|---|---|---|---|---|
| BL | 95 | +$18.82 | −$31.06 | **+$49.88** |
| GBM | 122 | +$6.84 | +$6.84 | $0.00 |

For BL, 34 positions were exited at a loss (−$5.62 realised); held, they would have
lost $21.36. So the exit rule works as a model-based stop, and removing it (never
closing anything) drops 2022 BL total return from 13.2% to 0.25%. For GBM it is neutral.
Note also that 17 of the 34 losing BL exits would have ended profitable if held, so the
rule is not perfect, only net-positive.

## 4. How much does the Sharpe depend on how it is measured?

The daily Sharpe scales by the square root of 252, which assumes daily returns are
uncorrelated. Here they are not (lag-1 autocorrelation runs from −0.39 to +0.07:
thin order books bounce and revert), so the same P&L gives different Sharpe ratios at
different sampling frequencies. The 5- and 20-day figures below use non-overlapping
returns, averaged over every possible starting phase so no arbitrary start day is
chosen.

| model | year | daily | 5-day | 20-day | lag-1 autocorr |
|---|---|---|---|---|---|
| BL | 2022 | 1.93 | 2.85 | 3.72 | −0.18 |
| BL | 2023 | 2.85 | 2.32 | 1.71 | +0.07 |
| BL | 2024 | 1.03 | 1.16 | 1.87 | −0.04 |
| GBM | 2022 | 2.35 | 3.49 | 3.22 | −0.12 |
| GBM | 2023 | 1.70 | 1.15 | 0.80 | −0.11 |
| GBM | 2024 | −0.59 | −1.30 | −1.85 | −0.39 |

**The sign is identical at all three frequencies in 6 of 6 model-years; the magnitude
is not.** BL is positive at every frequency and GBM 2024 is negative at every
frequency. The 20-day figures rest on only 6-12 observations each, so they are noisy;
the point is that the conclusion does not hinge on the daily scaling, not that any one
number is right. This is one more reason to quote a Sharpe only with its bootstrap
interval.

Alpha against SPX is statistically distinguishable from zero in one of six
model-years (BL 2023: annualised 10.2%, Newey-West t = 2.59); the other five have
|t| below 2.1. Beta is at most 0.04 in absolute value throughout.

## Limits

Three years and 40–120 positions per model: these are descriptions of what happened,
not estimates with tight uncertainty. Collateral is measured at entry prices, and the
stress assumes a single-bucket loss on the day it is evaluated.
