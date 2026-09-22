# Dual-Trading vs. Single-Leg: Cointegration & Lead-Lag Analysis

Paper-facing writeup of whether this strategy should trade **both legs**
(long the cheap venue, short the same claim on the other — a true hedged
arbitrage) or **only Kalshi** (a directional bet informed by the options
market). All numbers are reproducible via `analysis/cointegration.py`
(3-year backtest, 2022–2024, Breeden–Litzenberger model, active near-the-money
buckets). Spread is defined throughout as **Kalshi mid − options-implied
probability** for the same bucket-day. Position-level attribution and tail risk are
in [`RISK_PROFILE.md`](RISK_PROFILE.md).

---

## 1. Framing: is this a pairs / cointegration trade?

Two venues price the *same underlying event* (SPX close on 31 Dec, in $200
buckets). This resembles pairs trading, but with one structural difference that
makes it **stronger** than a statistical pairs trade:

- **Classic pairs trade** relies on a *statistical* assumption — the spread
  between two price series has been historically stationary, so it should
  revert. There is no guarantee; you are betting the past relationship holds.
- **Here** there is a **hard settlement anchor**: at expiry the Kalshi bucket
  and its options replication pay on the *identical* event, so the two claims
  are contractually equal at T. Convergence is not merely assumed — it is
  enforced by settlement.

So the relevant question is empirical: *do* the two series behave as a
cointegrated pair with a tradeable convergence, and if so, **which venue leads
and which lags?** That determines whether a hedge (dual-trading) helps or hurts.

## 2. Two distinct profit mechanisms (they are not the same bet)

- **(a) Lead-lag convergence** — Kalshi is slow to incorporate information and
  re-prices toward the (faster) options market. Profit is realized in the
  **mark-to-market path** as the spread closes. This is the pairs-trade
  mechanism.
- **(b) Settlement forecast accuracy** — if the options-implied probability were a
  better forecast of the realized outcome than Kalshi's price, you would collect at
  **settlement** whether or not Kalshi ever visibly converges first. (Measured
  Brier scores over all bucket-days are within noise of Kalshi's, so no forecast
  advantage is evident; see section 3.3.)

The backtest captures both (it marks to market daily *and* settles at year-end),
so we must decompose which one actually drives returns.

## 3. Empirical findings

### 3.1 The spread is mean-reverting and reverts to ~zero
- **58%** of 31 active buckets reject a unit root (ADF, 5%) — moderate but real
  evidence of stationarity given ~240 daily obs/year (ADF has low power in short
  samples, so this understates it).
- **Mean spread ≈ +0.003 (0.3¢), median |mean| ≈ 0.8¢** — the spread reverts to
  essentially **zero**, not to a persistent biased premium.
- **Half-life ≈ 5 trading days** (median 5.0, IQR 3.7–8.1) once bid-ask bounce is
  removed. The plain AR(1) fit gives 2.0 days (IQR 1.2–3.0), but that estimate is
  biased: bounce in the Kalshi mid is noise that appears in both the spread level
  and its next-day change, which mechanically looks like reversion. Instrumenting
  the level with the previous day's spread removes it (`half_life_iv`).
  Convergence takes about a week, not two days.

### 3.2 Kalshi is the laggard — it does the correcting (the key result)
Error-correction test on 2,344 bucket-days (change in each venue regressed on
the current spread):

| Change | Coef vs spread | t-stat | Interpretation |
|---|---|---|---|
| Kalshi, t → t+1 | −0.186 | −21.8 | inflated: shares the Kalshi mid at t with the spread |
| **Kalshi, t+1 → t+2** | **−0.043** | **−4.6** | no shared endpoint, so bounce cannot produce it (n = 2,188) |
| Options (model), t → t+1 | −0.001 | −0.1 | options do not move toward Kalshi at all |

**The spread correction is done by the Kalshi leg**, and the options market does
not move toward Kalshi. That direction is the finding and it survives the bounce
correction. The size does not: the next-day regression puts the Kalshi mid at time
t on both sides, so any bid-ask bounce in it reverts mechanically. Measured one day
later, where no endpoint is shared, the coefficient is about a quarter as large,
which means Kalshi closes roughly 4% of a gap per day rather than 19%. The
t-statistics are plain OLS and should be read as an upper bound on significance
(see section 6).

### 3.3 What the P&L is made of
The daily mark-to-market path is **not** all "Kalshi converging to the options
market". Three separate things are in it, and the earlier version of this note
mislabelled the first as convergence. For the BL both-side backtest:

| Year | Net carry on collateral | Trading path (marked to market) | Settlement step | Total, in excess of rf |
|---|---|---|---|---|
| 2022 | −0.88 | +18.88 | +0.72 | +18.72 |
| 2023 | −3.84 | +17.35 | +1.12 | +14.63 |
| 2024 | −2.93 | +6.07 | +0.72 | +3.86 |

- **Carry is a cost, not a source of return.** Kalshi paid no interest before
  10 October 2024 (4.05% after), so the collateral posted for open positions earned
  nothing while it could have earned the risk-free rate (3.8% / 5.5% / 5.3% over the
  three windows). Idle cash is assumed to earn rf elsewhere and so nets to zero.
  An earlier version of this note credited a 3.75% APY in every year and showed carry
  as 25–52% of profit; that APY did not exist over most of the sample.
- **Settlement (3–5%)** is small because most positions have already priced toward
  0 or 1 before expiry, not because the edge is unrelated to the outcome.
- **The trading path** is where the strategy's edge lives, but it combines two
  things this decomposition cannot separate: Kalshi re-pricing toward the options
  market (the lead-lag effect of section 3.2), and a contract's price drifting
  toward its realised 0/1 outcome as expiry approaches, which the path books before
  the settlement step. A short sold at 5¢ that decays to 0 earns along the path
  without any convergence to the options market. Section 3.2 tests the lead-lag
  effect directly; this table does not.

Position-level attribution ([`RISK_PROFILE.md`](RISK_PROFILE.md)) adds that positions
held to settlement earn most of the profit, but that is selected by the exit rule
(they are the positions the model kept agreeing with), and that closing early on an
opposite signal is what removes the losers: for BL it was worth about $50 versus
holding everything to expiry. Measured forecast accuracy shows no edge either way:
Brier scores of the BL and GBM PMFs against Kalshi's mid are within noise, so the
profit comes from mispricing dynamics, not from a better forecast of the settlement.

## 4. Implication for dual-trading — the counterintuitive part

Because the edge comes from **Kalshi-specific repricing** rather than a better
settlement forecast, adding the options short leg (full dual-trading) **does not
help this strategy and would most likely reduce net performance**:

1. **The hedge protects the wrong risk.** A long-Kalshi / short-options hedge
   neutralizes the *shared* (SPX-directional) exposure. But the both-side bucket
   portfolio is already near market-neutral (β ≈ 0 in every year), so there is
   little directional risk to remove. The risk that actually matters —
   Kalshi *failing* to converge, or diverging further before exit — is
   **idiosyncratic to Kalshi and cannot be hedged by the options leg**.
2. **The hedge would cancel the source of profit.** The P&L comes from the
   Kalshi leg moving (essentially all of the correction). The options leg barely moves
   (it leads and is already "correct"), so shorting it adds little offsetting
   P&L — it mainly layers on the option bid/ask spread (measured: the replication's
   round-trip spread is about 26¢ per $1 bucket at the median, against Kalshi's own
   ~2¢) and about 0.8¢ of expected replication error, plus a second venue's
   execution, margin, and management burden. See [`HEDGING.md`](HEDGING.md).
3. **The hedge's theoretical benefit is nearly irrelevant here.** Dual-trading's
   real advantage is converting a forecast bet into a *settlement-guaranteed*
   arbitrage. But settlement is only 4–7% of P&L and the spread half-life is about
   a week, so the guarantee buys little while costing spread + basis. (A slower
   convergence than first estimated makes this point weaker, not void: a week of
   exposure to Kalshi-specific risk is still not what an options hedge removes.)

### When dual-trading *would* be the right call (for a different configuration)
- If the edge were **settlement-forecast-driven** (mechanism b dominant) rather
  than convergence-driven — then holding to expiry with an options hedge to strip
  directional risk would be correct.
- If running a **concentrated, one-directional book** (not the both-side
  portfolio) with meaningful residual β, where the options hedge removes real
  directional risk.
- If the goal is a **provable arbitrage bound** (capped, near-riskless return)
  rather than maximizing a statistical convergence edge — accepting a lower
  yield in exchange for certainty, and if options execution costs are low enough
  not to erase the ~few-cent gap.

## 5. Recommendation

**Publish the single-leg Kalshi strategy as the primary result**, now with the
evidence above to justify it: the Kalshi/options relationship is a genuine
cointegrated lead-lag pair in which Kalshi is the error-correcting laggard, and
the strategy efficiently harvests that convergence by trading Kalshi directly.
Present **dual-trading as an analyzed alternative and future work** — not merely
as an unexplored idea, but with the finding that, *for this convergence-driven
edge*, hedging with options would neutralize already-small directional risk
while cancelling the profit source and adding basis/execution cost. The
model-free option replication ([`HEDGING.md`](HEDGING.md)) is a consistency check,
not evidence of a tradable arbitrage: it shows Kalshi and the options market agree
to about 3¢ on average with no persistent premium, and that after crossing spreads
only about 3% of bucket-days show any positive edge (about 1% beyond the
replication's own error). The strategy's profits come from statistical convergence,
not from a locked-in hedge.

## 6. Caveats (state these in the paper)
- **3-year sample**; ADF and error-correction estimates have wide confidence
  intervals. Treat magnitudes as indicative and the direction as the finding.
- **The reported t-statistics are plain OLS and overstate significance.** The
  2,344 bucket-days are serially dependent within a bucket and cross-sectionally
  dependent within a day (all buckets derive from one density). Bid-ask bounce is
  now handled (the t+1 → t+2 regression and the instrumented half-life), but
  clustered standard errors are not yet done.
- **Timestamps are not perfectly aligned.** The option quotes are the day's last
  quotes (probably the 4:15pm SPXW close) and the Kalshi candle closes at 4:00pm,
  so a same-day spread can contain up to 15 minutes of option-market news that
  Kalshi has not yet seen. That would show up as Kalshi "correcting" the next day.
  The one-day-lagged backtest (`analysis/execution_lag.py`) bounds how much of the
  trading result depends on it.
- The direction test uses **next-day** changes; it establishes Kalshi-leads-
  correction at daily frequency, not intraday.
- Kalshi *following* options does not by itself prove options are a *correct*
  forecast of settlement — only that Kalshi tracks options. The near-zero
  settlement P&L is consistent with, but not proof of, either.
- Dual-trading costs are measured statically (option round-trip spread, expected
  replication error, Kalshi fee; see `HEDGING.md`) but **not** simulated as a full
  hedged backtest with daily marking and margin — that remains the concrete
  future-work item. Option commissions are not modelled.
