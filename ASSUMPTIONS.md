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
| `MONEYNESS_SIGMA` | **2.0** | config | drop deep ITM/OTM: |moneyness − mean| > 2σ, with the mean and σ computed **within each quote date**. An earlier version pooled them over the whole 2022-2024 sample, so 2022 cleaning used statistics from later years; that leak was measured and found immaterial (BL both-side Sharpe pooled 1.93/2.85/1.03, per-year 1.92/2.85/1.03, per-quote-date 1.93/2.86/1.04; dropping the filter entirely 2.16/2.73/0.91) but was removed on principle. A date with a single quote is kept (it cannot be an outlier relative to itself) |
| No-arbitrage bound filter | **exact rule, no parameter** | `_drop_arbitrage_violations` | keep iff `intrinsic ≤ mid ≤ upper`. European bounds on div-adj spot S, DF=e^{−rT}: call ∈ [max(S−K·DF,0), S], put ∈ [max(K·DF−S,0), K·DF]. Drops ~1% stale deep-ITM quotes (all ITM, never enter the OTM density). No tolerance: the bound is a hard law and a strict test vs a 1e-9 epsilon drops identical rows (no float-noise risk) |
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

**ONE smile per quote date, used everywhere** (pricing `LSQ_Vol`, GBM ATM vol,
and the BL density all call `fit_daily_smile` + `eval_smile`). No second/per-type
smoother — see the unification note below.

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| **`SMILE_METHOD`** | **`sabr`** (swappable) | config, `smiles.REGISTRY` | chosen by head-to-head comparison (SMOOTHING_COMPARISON.md): smoothest + ~arbitrage-free, 705/705 days calibrate. Options: `sabr\|svi\|lsq\|cubic\|poly\|pchip\|lowess` — set to study smoothing→strategy sensitivity; nothing deprecated |
| Smile support | **OTM-combined** (puts K<F, calls K≥F) | `fit_daily_smile` | each strike taken from its less-noisy OTM side; by put-call parity this one IV(K) prices either type |
| SABR params | β=0.5 fixed; α, ρ, ν calibrated (least-squares) | `smiles.SABR` | Hagan 2002 lognormal; β=0.5 standard for equity index |
| `SMILE_KNOTS` | **6** interior | config | only used when `SMILE_METHOD="lsq"`; modest because the BL density (2nd derivative) amplifies over-fitting |
| Min points | ≥ 4 unique OTM strikes | `fit_daily_smile` | below this / on calibration failure → no smile (day unpriceable) |
| Strike extrapolation | **flat** outside observed [k_min,k_max] | `eval_smile` | avoids spurious wings (applied to every method) |
| IV-envelope clamp | IV clipped to observed **[iv_min, iv_max]** | `eval_smile` | data-derived (no constant): stops any fit overshooting to absurd/negative vols in sparse far-OTM strike gaps (~12% of days have a >$500 gap; unclamped LSQ hit 913% vol) |

> ✅ Pricing (HONEST model accuracy, no control-variate blend): MAE **$1.89** /
> RMSE **$3.74** (paper's original $4.50 / $7.72). Progression: per-type LSQ
> $4.38/$7.54 → unified LSQ → unified SABR → SABR w/o CV $1.93/$4.46 → per-quote-date moneyness filter $1.89/$3.74.

> **Unification note (why there is one smoother, not two):** earlier the density
> refit its own OTM 6-knot smile while pricing/GBM used a per-type 10-knot smile
> — an inconsistency that also contradicted the paper ("LSQ_Vol is the basis for
> pricing AND density estimation"). Collapsed to a single `fit_daily_smile`. This
> also fixed the call/put ATM disagreement (there is now one ATM vol) and exposed
> + fixed the spline-overshoot bug (the IV-envelope clamp above), which the old
> density had been silently masking via its isotonic repair.

---

## 3. Pricing (`transform/pricing.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Model | spot-based BSM on dividend-adjusted S with rate r | `bsm_call/put` | European SPX; no separate dividend-yield term needed |
| Control variate | **removed** (no blend) | `price_chain` | `BSM` = pure model price. The old λ=0.2 blend toward market mid only flattered the pricing diagnostic and, if used to build the density, corrupted it (modes 2→54); it never touched the strategy. Verified in `analysis/control_variate_test.py`. Verified: parity `C−P=S−K·DF` exact, call BSM ↓ / put BSM ↑ in strike 100%, 0 negative, 0 NaN |

---

## 4. Density formation (`transform/density.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| **Methods** | BL (primary) + GBM (benchmark) | `_METHODS` | the paper's 3rd, an option-price-spread heuristic, was **removed** — not a valid bucket PMF ([C(L)−C(U)]/(U−L) ≈ −dC/dK = tail prob P(S_T>mid), not the bucket mass; sums to ~4) |
| BL discount factor | f_Q(K) = e^{rT}·∂²C/∂K² (= 1/DF) | `bl_density` | recovers a **true** (undiscounted) probability density |
| **Coverage guard** | bucket priced only if it overlaps observed [k_min,k_max]; else NaN (untradeable) | `bl_pmf` | don't assign a probability to a bucket with no option support (pure flat-vol extrapolation). Zero result impact (deep-tail buckets don't trade) but honest |
| No forced rescaling | bucket probs NOT scaled to sum to 1 | `build_pmf_table` | residual = P(SPX breaches all buckets) is kept as signal |
| Smile input | the **shared** daily smile (`fit_daily_smile`, SABR) | `bl_density` | same surface as pricing/GBM — §2; no separate density smoother |
| No-arbitrage enforcement | clip dC/dK to [−DF, 0] + **isotonic** (non-decreasing) | `bl_density`, `_isotonic_increasing` | guarantees C convex ⇒ density ≥ 0; a safety net that is now largely a no-op under SABR (arb_neg 0.002) but kept so any `SMILE_METHOD` yields a valid density |
| `DENSITY_SMOOTH_WINDOW` | **25** grid pts (~$30) | config | light mass-preserving smoothing; removes any flat-extrapolation boundary kink |
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
| Entry rule | buy if model_p > ask + hurdle; sell if model_p < bid − hurdle | `signals.generate` | trade only past-fee mispricings |
| Fee hurdle | `2 · fee(price, lot) / lot` (PER CONTRACT) | `signals.entry_hurdle` | a lot of C contracts breaks even when C·(model_p−price) > fee, i.e. at fee/C per contract; comparing the undivided lot fee to a per-contract price demanded 8× too much edge (fixed Phase 7) |
| `FEE_ROUND_TRIP_FILLS` | **2** | config | not tunable: Kalshi charges the taker fee on BOTH fills of a market-order round trip. Positions held to settlement pay no exit fee, so the hurdle is mildly conservative |
| `LOT_SIZE` | **8** contracts | config | paper's tuned lot (fee-efficiency vs liquidity) |
| **`MAX_LOTS_PER_BUCKET`** | **1** (no pyramiding) | config | **risk-managed design**: cap per-bucket exposure so PV stays > 0 and metrics are well-defined |
| `START_CASH` | **$200** | config | buffer for consecutive losses on the small base |
| `KALSHI_FEE_RATE` | **0.035** | config | fee = ⌈0.035·C·p·(1−p)·100⌉/100 $ per lot. This is the taker rate **in force over the 2022–2024 sample**, which is what the backtest must charge. Kalshi's *current* schedule is 0.07 taker / 0.0175 maker, so limit orders are no longer fee-free; results are robust to the change (BL both-side Sharpe 1.46/2.41/0.76 at 0.07, versus 1.93/2.86/1.04 at 0.035) |
| `KALSHI_APY` | **3.75%**, monthly | config, `backtest.run` | yield on cash **and** open positions |
| Settlement | $1 if year-end close ∈ [L,U] | `backtest.run` | `SPX_YEAR_END_CLOSE` = {2022:3839.50, 2023:4769.83, 2024:5881.63} (actual S&P 500 cash closes) |
| Marking | **liquidation-side**: a long is marked at the bid, a short at the ask, of the last valid book; a book that is missing, empty (bid ≤ `MARK_MIN_BID`=2c & ask ≥ `MARK_MAX_ASK`=98c) or on a feed-outage day falls back to the last valid book, then the model | `kalshi_pmf.is_valid_book`, `feed_outage_days`, `backtest.run` | a position can only be exited at the side it would trade against, so mid overstated open positions by half a spread; an empty or outage book quotes phantom prices (2024-11-16..21 asks near 100c on mutually exclusive buckets) that once produced the whole reported 2024 drawdown |
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
| `TRADING_DAYS` | **252** | config | annualization factor, same for strategy and benchmark |
| Return calendar | **trading days**: chain quote dates, clipped to the span Kalshi actually quoted | `trading_days` | the backtest ledger is calendar-daily (Kalshi trades 24/7) but the x252 factor describes trading days; sampling here also gives the SPX benchmark a real observation on every row instead of forward-filled zeros, and drops the 2022 days before the Kalshi market existed |
| `BENCH_RF` | **= `KALSHI_APY`** (3.75%) | config | idle capital in a Kalshi account earns the APY, so that is the risk-free hurdle; rf_daily = APY/252 |
| Benchmark | SPX raw spot (price index) on the trading calendar | `spx_benchmark` | buy-and-hold; excludes dividends, which moves alpha by beta x ~1.5%/yr (under 0.1 pt here) |
| Returns | simple daily pct_change, drop ±inf | `_daily_returns` | annualised return is arithmetic (mean x 252); the geometric figure differs by about sigma^2/2, under 0.1 pt here |
| Sharpe | (annReturn − rf)/annVol | `summarize` | point estimate only; see the two rows below |
| Sharpe interval | stationary block bootstrap, 10,000 resamples, mean block 10 days | `sharpe_bootstrap_ci` | geometric block length preserves short-run dependence; intervals are 3-5 Sharpe units wide, so never quote the point estimate alone |
| Sharpe sampling check | non-overlapping 5- and 20-day returns, averaged over all starting phases | `sharpe_at_frequency` | the sqrt(252) scaling assumes uncorrelated daily returns (lag-1 ranges -0.39 to +0.07). Magnitude moves with frequency (e.g. BL 2023: 2.86 daily, 2.29 at 5 days, 1.66 at 20) but the sign is identical at all three in 6 of 6 model-years |
| Max drawdown | on **strategy** PV, full calendar path | `summarize` | a weekend trough is a real drawdown; not an annualised quantity |
| `total_return` | (final incl. settlement)/start − 1 | `summarize` | end-to-end result INCLUDING the year-end payoff; for 2022 it covers only the ~6 months Kalshi existed, so it is not comparable to a full-year figure |
| `ann_return_ex_apy` | annualised return of PV minus accrued interest | `summarize` | attribution only: how much of the raw return is passive platform carry (about 3%/yr). There is deliberately no separate ex-APY Sharpe, since with rf = APY the headline Sharpe already excludes the carry |
| Alpha / beta | OLS of daily excess strategy return on excess SPX return | `summarize` | `alpha_ann` = daily alpha x 252; **t-statistics are Newey-West** (`_newey_west_t`), because daily returns are autocorrelated |
| Newey-West lag | floor(4·(n/100)^(2/9)), i.e. 4 for a year of data | `_newey_west_t` | the standard data-derived rule, not a chosen constant; verified against statsmodels to 1e-15 |
| Model ρ | corr of mark-to-market vs mark-to-model daily returns | `summarize` | executability of the ideal strategy |

Alpha is statistically distinguishable from zero in only one of six model-years (BL 2023, t = 2.56); beta is small everywhere (|β| ≤ 0.04), and although BL 2023's is nominally significant (t = 3.0) it is economically negligible.

---

## 7. Future work (logged, not implemented)

- **Daily mark-to-market HEDGED backtest**: pair each Kalshi bucket trade with its
  offsetting SPX iron-condor and track both legs' MTM daily (options are the EOY
  expiry, so daily marking is feasible). Current hedging module (`hedging.py`) does
  a static same-day comparison of each Kalshi bucket with its out-of-the-money option
  replication (mid gap, executable edge after spreads and fee, density-weighted
  replication error); see `HEDGING.md`.
- Limit-order execution modeling (note: no longer fee-free under Kalshi's current 0.0175 maker rate).
- Bid/ask **size**-aware position sizing.
