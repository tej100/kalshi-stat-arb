# IV Smoothing Method Comparison

Reusable head-to-head of implied-volatility smoothing methods for risk-neutral
density (RND) extraction. Framework: [`analysis/smoothers.py`](analysis/smoothers.py)
(self-contained smoother classes + one shared RND pipeline) and
[`analysis/compare_smoothers.py`](analysis/compare_smoothers.py) (driver → metrics
+ `analysis/figures/smoother_comparison.png`). Standalone enough to lift into a
separate volatility-surface methods paper.

## Methods (all those in the original paper + SVI)
LSQ spline (current production), cubic smoothing spline, polynomial (deg 4),
PCHIP (monotone), LOWESS, SABR (Hagan 2002, β=0.5), SVI (Gatheral raw). SABR and
SVI are implemented directly (no `pysabr` dependency).

## Metric definitions
Each smoother is fit on the same OTM-combined smile; only the smile differs,
the density pipeline is identical.

- **fit_rmse** — RMS(fitted IV − market IV) at data strikes (fit tightness).
- **overshoot** — max fitted IV / max observed IV; ~1 ideal, >1.05 = inventing high vol.
- **arb_neg** — fraction of the grid where the *raw* density is negative
  (static arbitrage the smile introduces, before any repair). Lower = better.
- **roughness** — total-variation / peak of the raw density; smooth unimodal ≈ 2.
- **modes** — count of local maxima of the raw density above 5% of peak; 1 = clean.

## Results (mean over 6 sample dates, 2022–2024; all evaluated through the
## production `eval_smile`: strike-clamp + IV-envelope clamp)

| method | fit_rmse | overshoot | arb_neg | roughness | modes | verdict |
|---|---|---|---|---|---|---|
| **SABR** (Hagan, β=0.5) — **CHOSEN** | 0.004 | 0.99 | **0.002** | **2.30** | 2.2 | smoothest, ~arbitrage-free, 705/705 days calibrate |
| **SVI** (Gatheral raw) — kept | 0.004 | 0.99 | **0.002** | 2.69 | 2.3 | ~arbitrage-free, flexible wings |
| cubic smoothing spline | 0.005 | 0.98 | 0.023 | 2.53 | 2.2 | smooth but oversmooths wings |
| polynomial (deg 4) | 0.004 | 1.00 | 0.019 | 3.01 | 3.0 | ok body, Runge in the wings |
| LSQ spline (former default) | 0.002 | 1.00 | 0.021 | 2.72 | 2.5 | tight fit, spiky raw density, needs repair |
| PCHIP (monotone) | 0.000 | 1.00 | 0.199 | 6.38 | 14 | **interpolates noise → 20% arbitrage** |
| LOWESS | 0.003 | 0.99 | 0.012 | 29.4 | 93 | **density is pure noise** |

## Findings
1. **PCHIP and LOWESS are disqualified.** PCHIP interpolates every noisy point
   (zero fit error but 20% of the density negative, 14 modes); LOWESS gives a
   density with ~93 spurious modes. Both crush the plot axis — dropped from the
   density panels.
2. **LSQ (production) is the noisiest of the *reasonable* methods.** Its raw
   density has boundary spikes (flat-extrapolation kinks) and a bumpy left
   shoulder — it only looks clean in production because the isotonic + smoothing
   repair fixes it afterward. It leans on that repair the most.
3. **SABR and SVI are the smoothest and most arbitrage-free**, by construction —
   they would produce a clean density *without* the isotonic/smoothing band-aids,
   which could then be removed. SABR is nearly arbitrage-free (arb_neg 0.002);
   SVI matches its smoothness with more flexible wings (5 params vs SABR's 3+β).
4. Cubic/polynomial are middling: smooth in the body, but polynomial Runge-
   oscillates in sparse wings and cubic oversmooths the skew.

## Decision (implemented)
**Global smoother = SABR** (`config.SMILE_METHOD = "sabr"`). It is the smoothest,
most arbitrage-free form, calibrates on all 705 days, and tightened pricing to
MAE $1.55 / RMSE $3.57 (from LSQ's $1.86 / $4.26). BL results improved on the
cleaner density (e.g. 2023 BL Sharpe 0.58 → 0.81).

**Every method is retained and swappable.** All seven live in
`kalshi_arb.transform.smiles.REGISTRY`; set `config.SMILE_METHOD` to any of
`sabr | svi | lsq | cubic | poly | pchip | lowess` to study how the smoothing
model propagates to strategy results. SVI is kept specifically for that A/B
comparison. Nothing is deprecated; the production pipeline and this comparison
share the one implementation.

> NOTE: the raw density is shown pre-repair to expose each method's *native*
> quality. In production every method passes through the isotonic + smoothing
> step, so even LSQ yields a usable density — but SVI/SABR need no repair at all.
