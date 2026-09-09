# Model Assumptions, Thresholds & Parameters

Complete registry of every assumption, threshold, and free parameter in the
`kalshi_arb` pipeline. Each entry lists the value, where it lives in code, why
it was chosen, and any mismatch with the paper text that must be reconciled.
This is the authoritative reference for the paper's Methodology/Appendix.

All numeric parameters live in one place: [`kalshi_arb/config.py`](kalshi_arb/config.py).
Nothing downstream hard-codes an economic constant.

---

## 1. Data cleaning (`cleaning.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Drop rule | drop quote iff Bid, Ask **and** IV all missing | `clean_chain` | keep any quote with usable pricing info |
| One-sided mid | missing side assumed 0 → mid = present/2 | `_mid_price` | paper's stated rule (old code did NOT implement this) |
| `ONE_SIDED_BSM_TOL` | **$0.50** | config | keep a one-sided quote only if vendor-IV BSM price is within this of the theoretical mid |
| `MAX_SPREAD_FRAC` | **0.30** | config | drop illiquid two-sided quotes with (ask−bid)/mid > 30% |
| `MONEYNESS_SIGMA` | **2.0** | config | drop deep ITM/OTM: |moneyness − mean| > 2σ (global) |
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

## 2. Volatility smoothing (`vol.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Smoother | LSQ univariate spline | `fit_iv_spline` | paper's chosen method (LOWESS/cubic/poly/PCHIP/SABR de-scoped) |
| `DEFAULT_KNOTS` | **10** interior | `vol.py` | balances smoothness vs local flexibility |
| Knot placement | interior **quantiles** of strikes | `fit_iv_spline` | guarantees Schoenberg–Whitney (uniform knots failed on clustered strikes) |
| Min points | ≥ 4 unique strikes | `fit_iv_spline` | below this, no spline fit |
| Extrapolation | **flat** outside observed strike range | `eval_on_grid` | avoids spurious wings / negative vols when widening the density grid |

> ⚠️ RMSE $9.36 vs paper $7.72 — likely knot-count/tail differences. Revisit knot
> count if matching the paper's RMSE matters; not material to the strategy.

---

## 3. Pricing (`pricing.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Model | spot-based BSM on dividend-adjusted S with rate r | `bsm_call/put` | European SPX; no separate dividend-yield term needed |
| `CONTROL_VARIATE_LAMBDA` | **0.20** | config | Price_adj = BSM + λ(Mid − BSM); partial market anchoring w/o overfitting |

---

## 4. Density formation (`density.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| **Primary method** | Breeden–Litzenberger | `bl_density` | model-free RND from option convexity |
| BL discount factor | f_Q(K) = e^{rT}·∂²C/∂K² (= 1/DF) | `bl_density` | recovers a **true** (undiscounted) probability density |
| **Renormalization** | **NONE** | `bl_density` | density integrates to ≈1 naturally on the wide grid; residual = P(SPX breaches all buckets) |
| IV curve for BL | **OTM-combined** (puts below F, calls above) | `_iv_curve(otm=True)` | market standard; less noisy than deep-ITM |
| `GRID_PAD_LOW / HIGH` | **2000 / 1000** | config | widen strike grid beyond observed strikes (flat-vol) so density has near-full support |
| `GRID_POINTS` | **5000** | config | finite-difference resolution for ∂²C/∂K² |
| Edge trim | drop 2 grid points each end | `bl_density` | np.gradient boundary error |
| Negative clip | f = max(f, 0) | `bl_density` | remove differentiation-noise negatives |
| GBM law | lognormal, S₀ = **spot** (adj), drift (r−½σ²)T | `gbm_pmf` | correct reading of paper's d₂ (which carries r-drift ⇒ spot, not forward) |
| GBM σ | smoothed ATM IV (strike nearest F) | `_atm_vol` | single diffusion parameter |
| `DISCOUNT_PROBABILITY` | **False** | config | compare undiscounted P(event) to Kalshi price; APY captures time value |
| PMF calendar fill | ffill→bfill onto Kalshi dates | `build_pmf_table` | option data back-filled to non-trading days |

> ⚠️ **Paper reconciliation:** (a) paper says BL uses "call options" — we use
> OTM-combined; (b) paper says GBM S₀ = forward — we use spot (formula-consistent);
> (c) paper says PMFs "scaled so each day sums to 100%" — we do **not** (breach mass
> is signal). All three must be corrected in the paper text.

---

## 5. Signals & strategy (`signals.py`, `backtest.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| Entry rule | buy if model_p > ask+fee; sell if model_p < bid−fee | `signals.generate` | trade only past-fee mispricings |
| `LOT_SIZE` | **8** contracts | config | paper's tuned lot (fee-efficiency vs liquidity) |
| **`MAX_LOTS_PER_BUCKET`** | **1** (no pyramiding) | config | **risk-managed design**: cap per-bucket exposure so PV stays > 0 and metrics are well-defined |
| `START_CASH` | **$200** | config | buffer for consecutive losses on the small base |
| `KALSHI_FEE_RATE` | **0.035** | config | fee = ⌈0.035·C·p·(1−p)·100⌉/100 $ per lot (limit orders would be fee-free — future work) |
| `KALSHI_APY` | **3.75%**, monthly | config, `backtest.run` | yield on cash **and** open positions |
| Settlement | $1 if year-end close ∈ [L,U] | `backtest.run` | `SPX_YEAR_END_CLOSE` = {2022:3839.50, 2023:4769.83, 2024:5881.63} (actual S&P 500 cash closes) |
| Marking | long→bid, short→ask; carry last if missing | `_mark` | conservative, executable exit prices |
| Sub-strategies | both / buy-only / sell-only | `signals.generate(side=...)` | isolates directional performance |

> **Design decision (locked):** the risk-managed (no-pyramiding) variant is the
> published strategy. The unbounded-accumulation variant produces the old
> dramatic-but-ill-defined numbers (PV crosses 0) and is NOT used.

---

## 6. Performance metrics (`metrics.py`)

| Parameter | Value | Where | Rationale |
|---|---|---|---|
| `TRADING_DAYS` | **252** | config | annualization factor — **same for strategy and benchmark** (old code was inconsistent) |
| `BENCH_RF` | **3%** annual | config | risk-free for Sharpe/alpha; rf_daily = 3%/252 |
| Benchmark | SPX raw spot, ffill to backtest calendar | `spx_benchmark` | index buy-and-hold |
| Returns | simple daily pct_change, drop ±inf | `_daily_returns` | |
| Sharpe | (annReturn − rf)/annVol | `summarize` | |
| Max drawdown | on **strategy** PV (not benchmark) | `summarize` | old code computed DD on SPX by mistake |
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
- Knot-count tuning to reconcile pricing RMSE with the paper.
