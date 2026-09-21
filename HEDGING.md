# Cross-Market Check: Kalshi Buckets vs SPX-Option Replication

A Kalshi bucket and a strip of SPX options can be built to pay off on the same event,
so their prices should agree. This note measures how closely they do, and how much of
any gap could actually be captured. Everything is reproduced by
[`analysis/hedging_edge.py`](analysis/hedging_edge.py), built on
[`kalshi_arb/strategy/hedging.py`](kalshi_arb/strategy/hedging.py).

## Method

A bucket `[L, U]` pays `1[S>L] − 1[S>U]`. Each indicator is a tight spread across the two
listed strikes that bracket the edge, scaled by `1/width` so it climbs from 0 to 1. By
put-call parity a call spread and `1 −` a put spread are the same payoff, so each edge is
built from the **out-of-the-money** side (puts below the forward, calls above); when the two
edges use different types a zero-coupon bond leg (worth the discount factor) makes up the
difference. Only strikes listed as both a call and a put are used.

Comparison rules:
- **Same day only.** A bucket-day is used only if the option chain for that exact date exists.
- **The Kalshi book, not the last trade.** Mid for the like-for-like gap; bid and ask for
  execution.
- **No discounting adjustment.** Both prices are present values of the same terminal claim.
- **Executable edge** crosses every spread: sell Kalshi at its bid and buy the replication at
  the option asks (and bids for the short legs), or the reverse; Kalshi's taker fee is
  deducted. Option commissions are not modelled.
- **Replication error** is the density-weighted expected payoff gap `E_Q|replication − binary|`
  under the day's own Breeden-Litzenberger density, not an average over an arbitrary grid.

## Verification

On 30 random dates (361 replications, all three option-type combinations including the
bond-leg case): every replication pays exactly 0 far above and far below its strikes
(361/361), and the flat top of the replication covers the bucket centre in 95%. The
remainder are buckets too coarse for the strike grid at that point. 29 further buckets
cannot be replicated at all (an edge outside the listed strikes, or both edges inside one
strike gap) and are excluded.

## Results

4,257 bucket-days with a valid Kalshi book, same-day chain and two-sided quotes on every leg:

| year | bucket-days | mean \|Kalshi − replication\| | option round-trip spread | Kalshi spread | edge > 0 after spreads | after Kalshi fee | also above replication error | replication error |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1,196 | 2.81¢ | 42¢ | 3¢ | 0.3% | 0.2% | 0.2% | 0.36¢ |
| 2023 | 2,043 | 3.33¢ | 23¢ | 1¢ | 6.6% | 5.7% | 2.1% | 1.11¢ |
| 2024 | 1,018 | 2.31¢ | 26¢ | 2¢ | 0.3% | 0.1% | 0.0% | 0.78¢ |
| all | 4,257 | 2.94¢ | 26¢ | 2¢ | 3.3% | 2.8% | 1.1% | 0.82¢ |

(Spreads are medians, in cents of a $1 bucket. Replication error is the mean `E_Q|gap|`.)

**Reading it.**
- **The two markets agree closely.** The average gap is about 3¢ and there is no persistent
  premium: the mean signed gap is +1.0¢, of which about 0.8¢ is the replication's own
  expected bias (its signed mean payoff gap is −0.81¢), leaving +0.2¢.
- **Almost none of the gap is capturable.** Replicating a bucket with options costs about
  26¢ round trip against Kalshi's roughly 2¢ spread, so only 3.3% of bucket-days show any
  positive edge after crossing spreads (median 0.9¢ when positive), and 1.1% beat the
  replication's own error. 2023, with the tightest option markets, is the only year where
  this is not close to zero.
- **So this is a consistency check, not evidence of a tradable arbitrage.** The strategy's
  profits come from statistical convergence (`DUAL_TRADING_ANALYSIS.md`), not from a
  locked-in hedge.

## What changed from the earlier version

The earlier module produced the figures in the paper draft (about 24% of quotes more than
5¢ apart; mean gap about 4-6¢; mean "basis" 2-3¢). Those did not survive scrutiny:

| earlier behaviour | effect measured |
|---|---|
| compared the Kalshi **last trade** to an option mid | using the Kalshi mid changes the headline by under 1 point |
| used the most recent chain **at or before** the date, up to 42 days stale (only 67% same-day) | restricting to same-day drops a third of the sample, headline unchanged (23.8%) |
| replicated with **in-the-money calls** | the main driver: their spreads are wide, and each leg carries weight `1/width`, so one index point of quote error moves the bucket price by 10¢ at a 10-point strike spacing (4¢ at 25). Round-trip option spread 97¢ median (26¢ with out-of-the-money options); mean gap 4.4¢ → 2.95¢; share above 5¢ 24% → 19% (2024: 25% → 9.5%) |
| never crossed a spread, yet claimed the gap exceeded the replication cost | with ITM calls 98% of the cases above 5¢ had no executable edge; with OTM options 97% |
| averaged replication error over a fixed uniform grid `linspace(L−400, U+400)` | dilutes a payoff gap that exists only in two narrow ramps; the density-weighted figure is 0.82¢ mean (0.41¢ median) rather than a grid average |
| returned an **empty** leg list when both edges fell in one strike gap, priced at zero | 80 of 8,730 bucket-days (0.9%) were compared against a zero "replication"; now not replicable |
| compared to a 5¢ threshold | an arbitrary constant; replaced by comparison with actual costs and the replication's own error |

The large "dislocations" in the earlier version were concentrated exactly where quote noise
is amplified most (narrow ramps, wide option spreads), so much of what looked like
mispricing was measurement error.

## Limits

- Static, same-day comparison; not a daily-marked hedged backtest (future work).
- Option commissions and margin are not modelled; the Kalshi fee is.
- Only buckets with a valid Kalshi book and two-sided quotes on every option leg are
  included, which excludes the thinnest wings.
- The replication error uses the model density, so it inherits smile-fit error.
- Three years, one index, one Kalshi product.
