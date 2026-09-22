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
| One-sided quotes | **dropped** | `_mid_price` | a quote with one side has no mid. The earlier rule (missing side = 0, keep if BSM within $0.50) was meaningless for bid-only quotes and its tolerance had no effect. 1,892 quotes, 1,881 ask-only; pricing error improves and BL Sharpe moves by at most 0.08 |
| Bid-ask spread filter | **none** (`MAX_SPREAD_FRAC = None`) | config, `clean_chain` | an inherited 30%-of-mid cutoff was removed: not derived, and the smile uses out-of-the-money quotes only. With it BL Sharpe is 1.89 / 1.90 / 0.68 and pooled 1.45; without, 1.98 / 1.90 / 0.55 and pooled 1.44. The knob stays only for the appendix sensitivity row |
| `MONEYNESS_SIGMA` | **2.0** | config | drop deep ITM/OTM: |moneyness − mean| > 2σ, with the mean and σ computed **within each quote date**. An earlier version pooled them over the whole 2022-2024 sample, so 2022 cleaning used statistics from later years; that leak was measured at the time and found immaterial (Sharpe within 0.01 between pooled, per-year and per-quote-date versions; dropping the filter entirely moved it by up to 0.23) but was removed on principle. A date with a single quote is kept (it cannot be an outlier relative to itself) |
| No-arbitrage bound filter | **exact rule, no parameter** | `_drop_arbitrage_violations` | keep iff `intrinsic ≤ mid ≤ upper`. European bounds on div-adj spot S, DF=e^{−rT}: call ∈ [max(S−K·DF,0), S], put ∈ [max(K·DF−S,0), K·DF]. Drops ~1% stale deep-ITM quotes (all ITM, never enter the OTM density). No tolerance: the bound is a hard law and a strict test vs a 1e-9 epsilon drops identical rows (no float-noise risk) |
| Moneyness def. | calls K/F, puts 2 − K/F | `_transform` | centers distribution at 1 |
| `IV_MIN, IV_MAX` | **0.01, 5.0** | config | bisection search bounds (decimal vol) for IV backfill |

**Unit conventions (fixed, not tunable):** vendor IV percent→decimal (÷100);
T days→years (÷365); implied rate `r = −ln(DF)/T`; dividend-adjusted spot
`S = spot − div·DF`; forward `F = S·e^{rT}`.

**Validated:** 344,720 → 314,626 rows (3,282 on expiry-roll dates, 1,892 one-sided, 3,035
outside the no-arbitrage bounds, the rest by moneyness); 4,216 missing IVs recovered, none
left. Pricing MAE $1.99, RMSE $3.50.

---

## 2. Volatility smoothing (`transform/smoothing.py`)

**ONE smile per quote date, used everywhere** (pricing `LSQ_Vol`, GBM ATM vol,
and the BL density all call `fit_daily_smile` + `eval_smile`). No second/per-type
smoother — see the unification note below.

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| **`SMILE_METHOD`** | **`sabr`** (swappable) | config, `smiles.REGISTRY` | chosen by head-to-head comparison (SMOOTHING_COMPARISON.md): smoothest + ~arbitrage-free, 705/705 days calibrate. Options: `sabr\|svi\|lsq\|cubic\|poly\|pchip\|lowess` — set to study smoothing→strategy sensitivity; nothing deprecated |
| Smile support | **OTM-combined** (puts K<F, calls K≥F) | `fit_daily_smile` | each strike taken from its less-noisy OTM side; by put-call parity this one IV(K) prices either type |
| `SABR_BETA` | **0.5** fixed; α, ρ, ν calibrated (least-squares) | config, `smiles.SABR` | Hagan 2002 lognormal. For one expiry β and ρ both act on the smile's slope and are nearly unidentified jointly, so β is fixed. Pooled BL Sharpe 1.46 at β=0, 1.44 at 0.5, 1.23 at 1 (`analysis/sensitivity.py`) |
| `SMILE_KNOTS` | **6** interior | config | only used when `SMILE_METHOD="lsq"`; modest because the BL density (2nd derivative) amplifies over-fitting |
| Min points | ≥ 4 unique OTM strikes | `fit_daily_smile` | below this / on calibration failure → no smile (day unpriceable) |
| `SMILE_WINGS` | **flat** outside observed [k_min,k_max] (`"model"` extends SABR/SVI) | config, `eval_smile` | conservative. It has no effect on any result, because the quoted strikes extend beyond every listed bucket on most days; it will matter for the open-ended tail contracts. Non-parametric smoothers are always flat |
| IV envelope / floor | **non-parametric** smoothers (LSQ, cubic, poly, PCHIP, LOWESS): IV clipped to the observed **[iv_min, iv_max]**. **Parametric** smoothers (SABR, SVI): only a positivity floor (`IV_MIN`) | `eval_smile`, `Smoother.parametric` | data-derived (no constant). The envelope stops a spline overshooting into a sparse far-OTM strike gap (an unclamped LSQ spline hit 913% vol), but for a smooth SABR/SVI fit it was harmful: a least-squares fit legitimately dips below the lowest observed IV between data points, and clamping there bound inside the strike range on 66% of days, creating kinks that Breeden-Litzenberger turns into density cliffs and spikes. Removing it for the parametric forms raised unimodal days from 26.7% to 43.5% and cut days with 3+ density peaks from 54.3% to 9.9% |

> ✅ Pricing (model accuracy, no control-variate blend): MAE **$1.99** /
> RMSE **$3.50** (paper's original $4.50 / $7.72). Progression: per-type LSQ
> $4.38/$7.54 → unified LSQ → unified SABR → SABR w/o CV $1.93/$4.46 → per-quote-date moneyness filter $1.89/$3.74 → no IV clamp on SABR/SVI $1.93/$3.77 → one-sided quotes dropped $1.85/$3.33 → spread filter removed $1.99/$3.50 (wide quotes stay in).

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
| No-arbitrage enforcement | clip dC/dK to [−DF, 0] + **isotonic** (non-decreasing, pool-adjacent-violators) | `_density_from_smile`, `_isotonic_increasing` | guarantees C convex ⇒ density ≥ 0. Not a no-op: SABR is arbitrage-free inside the quoted strikes on all 705 days, but the flat-wing kink at the last strike leaves a negative lobe of a median 1.8% of mass (max 9.1%). On most days that lies beyond every bucket; on 220 days it reaches an outer bucket and the repair moves it by up to 3.7¢ |
| Density smoothing | **removed** | — | a 25-point moving average was inert under SABR (identical results with and without it) |
| `GRID_PAD_LOW / HIGH` | **2000 / 1000** | config | widen strike grid beyond observed strikes (flat-vol) so density has near-full support |
| `GRID_POINTS` | **5000** | config | finite-difference resolution for ∂²C/∂K² |
| Edge trim | drop 2 grid points each end | `bl_density` | np.gradient boundary error |
| GBM law | lognormal, S₀ = **spot** (adj), drift (r−½σ²)T | `gbm_pmf` | correct reading of paper's d₂ (which carries r-drift ⇒ spot, not forward) |
| GBM σ | smoothed ATM IV (strike nearest F) | `_atm_vol` | single diffusion parameter |
| Undiscounted P(event) | compared directly to the Kalshi price | `density` | a $1 year-end claim is strictly worth DF·P when collateral earns nothing; the gap averages 0.15¢ per bucket-day, inside the entry hurdle, and moves Sharpe by about −0.1 when applied. The cost of posting collateral is instead charged explicitly in `metrics.excess_value` |
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
| `LOT_SIZE` | **8** contracts | config | inherited, not derived (no depth data to size from). Because each fee is rounded up to the cent per order, small lots pay more per contract: pooled BL Sharpe 1.04 at 1 contract, 1.31 at 4, 1.44 at 8, 1.47 at 16 (capital scaled with the lot) |
| Order type | **market (taker) orders only** | `signals.generate` | daily candles carry no fill information, so resting limit orders cannot be simulated honestly |
| **`MAX_LOTS_PER_BUCKET`** | **1** (no pyramiding) | config | **risk-managed design**: cap per-bucket exposure so PV stays > 0 and metrics are well-defined |
| `START_CASH` | **$200** | config | buffer for consecutive losses on the small base |
| `KALSHI_FEE_RATE` | **0.035** | config | fee = ⌈0.035·C·p·(1−p)·100⌉/100 $ per lot. This is the taker rate **in force over the 2022–2024 sample**, which is what the backtest must charge. Kalshi's *current* schedule is 0.07 taker / 0.0175 maker, so limit orders are no longer fee-free; BL stays positive in every year at the new rate (both-side Sharpe 1.51/1.66/0.40 and pooled 1.16 at 0.07, versus 1.98/1.90/0.55 and pooled 1.44 at 0.035) |
| Kalshi interest | **4.05% from 2024-10-10, nothing before** (`KALSHI_INTEREST_START`, `KALSHI_INTEREST_RATE`), accrued daily on posted collateral | config, `backtest.run` | Kalshi announced interest on cash and open positions on 10 Oct 2024. Only posted collateral is credited because idle cash is assumed to earn rf elsewhere (see §6). An earlier version credited 3.75% on the whole account in every year, which was wrong for 2022-2023 |
| Execution timing | **same close**: a signal fills at the Kalshi close of the day whose option chain produced it | `signals.generate`, `backtest.run` | option quotes are the day's last quotes (probably the 4:15pm SPXW close) against Kalshi's 4:00pm candle, so this can carry up to 15 minutes of lookahead. `analysis/execution_lag.py` reruns with the model one and two priced days old (`density.lag_pmf_table`, `run.full(lag=)`), re-testing every signal at the fill-day book: BL 0.67 / −0.13 / 0.52 (pooled 0.32, not significant) at one day |
| Settlement | $1 if year-end close ∈ [L,U] | `backtest.run` | `SPX_YEAR_END_CLOSE` = {2022:3839.50, 2023:4769.83, 2024:5881.63} (actual S&P 500 cash closes) |
| Marking | **liquidation-side**: a long is marked at the bid, a short at the ask, of the last valid book; a book that is missing, empty (bid ≤ `MARK_MIN_BID`=2c & ask ≥ `MARK_MAX_ASK`=98c), wider than half the probability range (spread ≥ `MAX_BOOK_SPREAD`=0.50; every legitimate spread in 2022-2024 is ≤ 44c, so any cutoff from 45c to 84c gives identical results) or on a feed-outage day falls back to the last valid book, then the model | `kalshi_pmf.is_valid_book`, `feed_outage_days`, `backtest.run` | a position can only be exited at the side it would trade against, so mid overstated open positions by half a spread; an empty or outage book quotes phantom prices (2024-11-16..21 asks near 100c on mutually exclusive buckets) that once produced the whole reported 2024 drawdown |
| Sub-strategies | both / buy-only / sell-only | `signals.generate(side=...)` | isolates directional performance |
| **Exit rule / holding period** | **none fixed** — hold until the OPPOSITE entry test fires on that bucket, or year-end settlement | `signals.generate` (re-evaluated daily), `backtest.run` | there is no take-profit, stop-loss, or time-based exit; observed holding periods range from 1 day to 6+ months (median 32 days). A convergence exit (`exit_rule="convergence"`: also close once the bid/ask reaches the model) was tested and changed nothing: pooled 1.46 vs 1.44, with 26% more fills |
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
| Risk-free rate | **option-implied**, r = −ln(DF)/T of each quote date (3.8% / 5.5% / 5.3% on average) | `rf_series` | a T-bill-like rate to the same year-end expiry the contracts settle on; no constant is hard-coded |
| Excess return | trading P&L + Kalshi interest on collateral − rf × collateral | `excess_value` | the account is split into posted collateral (must sit at Kalshi, earns Kalshi's rate) and idle cash (assumed to earn rf elsewhere, so zero excess). All Sharpe, alpha, drawdown and bootstrap figures use this series, with no second rf subtraction |
| Benchmark | SPX raw spot (price index) on the trading calendar | `spx_benchmark` | buy-and-hold; excludes dividends, which moves alpha by beta x ~1.5%/yr (under 0.1 pt here) |
| Returns | simple daily pct_change, drop ±inf | `_daily_returns` | annualised return is arithmetic (mean x 252); the geometric figure differs by about sigma^2/2, under 0.1 pt here |
| Sharpe | annExcess / annVol | `summarize` | point estimate only; see the two rows below |
| Sharpe interval | stationary block bootstrap, 10,000 resamples, mean block 10 days | `sharpe_bootstrap_ci` | geometric block length preserves short-run dependence; single-year intervals are 3-5 Sharpe units wide, so never quote the point estimate alone |
| **Pooled Sharpe (headline)** | the three years' daily excess returns stacked, same bootstrap | `pooled_sharpe_ci` | one year (114-240 days) is too short for an interval to exclude zero even at a Sharpe near 2; pooled BL 1.44 [0.36, 2.49], P(>0) = 0.995 |
| Sharpe sampling check | non-overlapping 5- and 20-day returns, averaged over all starting phases | `sharpe_at_frequency` | the sqrt(252) scaling assumes uncorrelated daily returns (lag-1 ranges −0.18 to +0.04). Magnitude moves with frequency (e.g. BL 2023: 1.90 daily, 1.73 at 5 days, 1.17 at 20) but the sign is identical at all three in 6 of 6 model-years |
| Max drawdown | on the **excess-value** series, full calendar path | `summarize` | a weekend trough is a real drawdown; not an annualised quantity |
| `total_return` | (final incl. settlement)/start − 1 | `summarize` | end-to-end result INCLUDING the year-end payoff; for 2022 it covers only the ~6 months Kalshi existed, so it is not comparable to a full-year figure |
| `ann_trading` / `ann_carry` | the two parts of `ann_excess`, on the same denominator so they add exactly | `summarize` | carry = Kalshi interest on collateral − rf × collateral, a cost of 0.6-1.9%/yr here; `total_return` is the nominal P&L of the Kalshi account itself and excludes rf on idle cash |
| Alpha / beta | OLS of daily excess strategy return on excess SPX return | `summarize` | `alpha_ann` = daily alpha x 252; **t-statistics are Newey-West** (`_newey_west_t`), because daily returns are autocorrelated |
| Newey-West lag | floor(4·(n/100)^(2/9)), i.e. 4 for a year of data | `_newey_west_t` | the standard data-derived rule, not a chosen constant; verified against statsmodels to 1e-15 |
| Model ρ | corr of mark-to-market vs mark-to-model daily returns | `summarize` | executability of the ideal strategy |

Alpha is only marginally distinguishable from zero year by year: GBM 2022 reaches t = 2.00, GBM 2024 is significantly negative (t = −2.39), and BL lies between t = 0.40 and 1.80. Beta is small everywhere (|β| ≤ 0.03); BL 2023's is nominally significant (t = 2.37) but economically negligible.

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
