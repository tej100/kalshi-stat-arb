# Model Assumptions, Thresholds & Parameters

Complete registry of every assumption, threshold, and free parameter in the
`kalshi_arb` pipeline. Each entry lists the value, where it lives in code, why
it was chosen, and any mismatch with the paper text that must be reconciled.
This is the authoritative reference for the paper's Methodology/Appendix.

All numeric parameters live in one place: [`kalshi_arb/config.py`](kalshi_arb/config.py).
Nothing downstream hard-codes an economic constant.

---

## 1. Data cleaning (`transform/clean.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Drop rule | drop quote iff Bid, Ask **and** IV all missing | `clean_chain` | keep any quote with usable pricing info |
| One-sided mid | missing side assumed 0 → mid = present/2 | `_mid_price` | paper's stated rule (old code did NOT implement this) |
| `ONE_SIDED_BSM_TOL` | **$0.50** | config | keep a one-sided quote only if vendor-IV BSM price is within this of the theoretical mid |
| `MAX_SPREAD_FRAC` | **0.30** | config | drop illiquid two-sided quotes with (ask−bid)/mid > 30% |
| `MONEYNESS_SIGMA` | **2.0** | config | drop deep ITM/OTM: |moneyness − mean| > 2σ (global) |
| `NO_ARB_TOL` | **$0.01** | config | drop quotes violating no-arbitrage price bounds: `mid < intrinsic` (negative time value) or `mid > upper`. European bounds on div-adj spot S, DF=e^{−rT}: call ∈ [max(S−K·DF,0), S], put ∈ [max(K·DF−S,0), K·DF]. Catches stale deep-ITM quotes (~1%, all ITM, never enter the OTM density) |
| Moneyness def. | calls K/F, puts 2 − K/F | `_transform` | centers distribution at 1 |
| `IV_MIN, IV_MAX` | **0.01, 5.0** | config | bisection search bounds (decimal vol) for IV backfill |

**Unit conventions (fixed, not tunable):** vendor IV percent→decimal (÷100);
T days→years (÷365); implied rate `r = −ln(DF)/T`; dividend-adjusted spot
`S = spot − div·DF`; forward `F = S·e^{rT}`.

**Validated:** 344,720 → 310,525 rows; IV-missing 3,372 → 89 after backfill
(paper reported →87). Pricing MAE $4.52 (paper $4.50).

> ⚠️ **Paper reconciliation:** paper says IV-missing 7,474→87 from ~81k rows; our
> starting missing count differs because upstream `eoy_chains.csv` is already
> partly filtered. Restate the exact figures in the paper from this pipeline.

---

## 2. Volatility smoothing (`transform/smoothing.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Smoother | LSQ univariate spline | `fit_iv_spline` | paper's chosen method (LOWESS/cubic/poly/PCHIP/SABR de-scoped) |
| `DEFAULT_KNOTS` | **10** interior | `smoothing.py` | balances smoothness vs local flexibility (pricing); density uses `DENSITY_KNOTS`=6 |
| Knot placement | interior **quantiles** of strikes | `fit_iv_spline` | guarantees Schoenberg–Whitney (uniform knots failed on clustered strikes) |
| Min points | ≥ 4 unique strikes | `fit_iv_spline` | below this, no spline fit |
| Extrapolation | **flat** outside observed strike range | `eval_on_grid` | avoids spurious wings / negative vols when widening the density grid |

> ✅ RESOLVED: pricing MAE $4.38 / RMSE $7.54 (paper $4.50 / $7.72). The earlier
> $9.36 RMSE gap was caused by ~1% stale deep-ITM quotes with mid below intrinsic
> value; the no-arb filter (§1) removes them, reconciling RMSE with the paper.
> Not knot-count related.

---

## 3. Pricing (`transform/pricing.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Model | spot-based BSM on dividend-adjusted S with rate r | `bsm_call/put` | European SPX; no separate dividend-yield term needed |
| `CONTROL_VARIATE_LAMBDA` | **0.20** | config | Price_adj = BSM + λ(Mid − BSM); partial market anchoring w/o overfitting |

---

## 4. Density formation (`transform/density.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| **Primary method** | Breeden–Litzenberger | `bl_density` | model-free RND from option convexity |
| BL discount factor | f_Q(K) = e^{rT}·∂²C/∂K² (= 1/DF) | `bl_density` | recovers a **true** (undiscounted) probability density |
| No forced rescaling | bucket probs NOT scaled to sum to 1 | `build_pmf_table` | residual = P(SPX breaches all buckets) is kept as signal |
| No-arbitrage enforcement | clip dC/dK to [−DF, 0] + **isotonic** (non-decreasing) | `bl_density`, `_isotonic_increasing` | guarantees C convex ⇒ density ≥ 0 and removes spurious modes without ad-hoc surgery; makes the density a proper law integrating to ≈1 on the wide grid |
| `DENSITY_KNOTS` | **6** (vs 10 for pricing) | config | fewer knots ⇒ smile not over-fit; the 2nd derivative no longer amplifies noise into extra modes |
| `DENSITY_SMOOTH_WINDOW` | **25** grid pts (~$30) | config | light mass-preserving smoothing; removes the flat-extrapolation boundary kink |
| IV curve for BL | **OTM-combined** (puts below F, calls above) | `_iv_curve(otm=True)` | market standard; less noisy than deep-ITM |
| `GRID_PAD_LOW / HIGH` | **2000 / 1000** | config | widen strike grid beyond observed strikes (flat-vol) so density has near-full support |
| `GRID_POINTS` | **5000** | config | finite-difference resolution for ∂²C/∂K² |
| Edge trim | drop 2 grid points each end | `bl_density` | np.gradient boundary error |
| GBM law | lognormal, S₀ = **spot** (adj), drift (r−½σ²)T | `gbm_pmf` | correct reading of paper's d₂ (which carries r-drift ⇒ spot, not forward) |
| GBM σ | smoothed ATM IV (strike nearest F) | `_atm_vol` | single diffusion parameter |
| `DISCOUNT_PROBABILITY` | **False** | config | compare undiscounted P(event) to Kalshi price; APY captures time value |
| **Calendar matching** | **strict same-day, hard rule (no parameter)**: a NEW position requires a genuine same-calendar-day quote from BOTH the option chain and Kalshi | `build_pmf_table` | **design decision, not a data-gap workaround** — see below |
| Same-Kalshi-year expiry | option rows filtered to `exp.year == quote.year` | `transform/clean.py` | drops the late-Dec expiry-rollover window where the raw feed only has next year's ~365-day contract |

> **Why strict same-day, and why there is no staleness-tolerance parameter at
> all:** the strategy's thesis is that the options market is the live,
> informed reference and Kalshi sometimes lags it. That only holds while
> options are actually trading. On a day they aren't (weekends, holidays, or
> — before the clean-stage expiry filter above — the late-December window
> where only next year's contract was quoted), the "model" side is frozen
> while Kalshi keeps moving. A divergence on such a day no longer tells you
> Kalshi is wrong; it could equally mean the model is stale and Kalshi is
> right. Trading it would rest on an unstated, different mechanism (a bet
> that Kalshi's own after-hours move reverts), not the options-information
> edge the paper claims. Crucially, **this problem is categorical, not a
> matter of degree**: it is already fully present after a single day of
> staleness, so there is no "safe" bound to calibrate — a 4-day tolerance
> (matching the longest US holiday weekend) is not a smaller version of the
> problem, it is the same problem for four days instead of one. For that
> reason `build_pmf_table` takes no staleness parameter at all (removed after
> initially being added as a bounded compromise — see CLEANUP_LOG.md for the
> before/after). New signals only fire on days both markets are genuinely
> live; existing positions still mark-to-market and settle normally every day
> regardless. Also evaluated and rejected: unbounded stale-fill through the
> December gap specifically — empirically the worst option, since the frozen
> `T` never sharpens toward expiry.
>
> ⚠️ **Paper reconciliation:** (a) paper says BL uses "call options" — we use
> OTM-combined; (b) paper says GBM S₀ = forward — we use spot (formula-consistent);
> (c) paper says PMFs "scaled so each day sums to 100%" — we do **not** (breach mass
> is signal); (d) paper says quotes are "back-filled" for non-trading days — the
> code actually does the reverse (forward-fill: reuse the last KNOWN/past chain,
> never a future one, to avoid look-ahead bias), and as of this decision, does
> not fill non-trading days for new signals at all. All four must be corrected
> in the paper text.

---

## 5. Signals & strategy (`strategy/signals.py`, `strategy/backtest.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Entry rule | buy if model_p > ask+fee; sell if model_p < bid−fee | `signals.generate` | trade only past-fee mispricings |
| `LOT_SIZE` | **8** contracts | config | paper's tuned lot (fee-efficiency vs liquidity) |
| **`MAX_LOTS_PER_BUCKET`** | **1** (no pyramiding) | config | **risk-managed design**: cap per-bucket exposure so PV stays > 0 and metrics are well-defined |
| `START_CASH` | **$200** | config | buffer for consecutive losses on the small base |
| `KALSHI_FEE_RATE` | **0.035** | config | fee = ⌈0.035·C·p·(1−p)·100⌉/100 $ per lot (limit orders would be fee-free — future work) |
| `KALSHI_APY` | **3.75%**, monthly | config, `backtest.run` | yield on cash **and** open positions |
| Settlement | $1 if year-end close ∈ [L,U] | `backtest.run` | `SPX_YEAR_END_CLOSE` = {2022:3839.50, 2023:4769.83, 2024:5881.63} (actual S&P 500 cash closes) |
| Marking | **mid of a valid book**; empty book (bid ≤ `MARK_MIN_BID`=2c & ask ≥ `MARK_MAX_ASK`=98c) or missing → last valid mid → model | `_valid_book`, `backtest.run` | an empty book (bid 0 / ask 100 when SPX has left a bucket) is a phantom price, NOT a liquidation value; marking shorts there caused the 2024 −33% artifact |
| Sub-strategies | both / buy-only / sell-only | `signals.generate(side=...)` | isolates directional performance |
| **Exit rule / holding period** | **none fixed** — hold until the OPPOSITE entry test fires on that bucket, or year-end settlement | `signals.generate` (re-evaluated daily), `backtest.run` | there is no take-profit, stop-loss, or time-based exit; observed holding periods in the actual backtest range from 1 day to 6+ months (same bucket, 2023) |
| **`n_trades` reporting** | = **executed fills** (`backtest.run().attrs["n_executed"]`), NOT the raw signal-log length | `run.full`, `backtest.run` | `signals.generate()` re-fires a signal every day a mispricing persists even while already at the 1-lot cap; those redundant rows are correctly suppressed by execution but were being mis-reported as trade count (found while tracing a real position: 137 raw signals vs 8 actual fills for one bucket). Raw count still available as `n_signals`. |

> **Design decision (locked):** the risk-managed (no-pyramiding) variant is the
> published strategy. The unbounded-accumulation variant produces the old
> dramatic-but-ill-defined numbers (PV crosses 0) and is NOT used.
>
> **On the exit rule:** because there is no convergence-target exit, a position
> is not guaranteed to close once the spread reaches zero — it can ride past
> zero into an overshoot (locking a gain), or reverse against the position
> before ever converging (a real loss; observed in the actual backtest — see
> CLEANUP_LOG.md round 6). This means the "97–99% convergence P&L" finding in
> `DUAL_TRADING_ANALYSIS.md` is a mix of realized gains/losses from these
> reversal-triggered exits plus unrealized mark-to-market on positions still
> open on any given date — both correctly flow into `portfolio_value`, but they
> are not the same thing and should be described precisely, not conflated, in
> the paper.

---

## 6. Performance metrics (`strategy/metrics.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| `TRADING_DAYS` | **252** | config | annualization factor — **same for strategy and benchmark** (old code was inconsistent) |
| `BENCH_RF` | **3%** annual | config | risk-free for Sharpe/alpha; rf_daily = 3%/252 |
| Benchmark | SPX raw spot, ffill to backtest calendar | `spx_benchmark` | index buy-and-hold |
| Returns | simple daily pct_change, drop ±inf | `_daily_returns` | |
| Sharpe | (annReturn − rf)/annVol | `summarize` | |
| Max drawdown | on **strategy** PV (not benchmark) | `summarize` | old code computed DD on SPX by mistake |
| `total_return` | (final incl. settlement)/start − 1 | `summarize` | end-to-end realized result INCLUDING year-end payoff (ann_return/Sharpe are pre-settlement path metrics) |
| Alpha/Beta | OLS of daily excess strat vs excess SPX; alpha reported **daily** | `summarize` | |
| Model ρ | corr of mark-to-market vs mark-to-model daily returns | `summarize` | executability of the ideal strategy |

---

## 7. Future work (logged, not implemented)

- **Daily mark-to-market HEDGED backtest**: pair each Kalshi bucket trade with its
  offsetting SPX iron-condor and track both legs' MTM daily (options are the EOY
  expiry, so daily marking is feasible). Current hedging module (`hedging.py`) does
  static replication + settlement/basis-risk analysis only.
- Limit-order (fee-free) execution modeling.
- Bid/ask **size**-aware position sizing.
