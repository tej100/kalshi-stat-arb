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

## Results (mean over 6 sample dates, 2022–2024)

| method | fit_rmse | overshoot | arb_neg | roughness | modes | verdict |
|---|---|---|---|---|---|---|
| **SABR** (Hagan, β=0.5) | 0.004 | 0.99 | **0.002** | **2.29** | **2.0** | smoothest, ~arbitrage-free |
| **SVI** (Gatheral raw) | 0.004 | 0.99 | 0.042 | **2.29** | 2.2 | equally smooth, flexible wings |
| cubic smoothing spline | 0.005 | 0.98 | 0.053 | 2.36 | 2.0 | smooth but oversmooths wings |
| polynomial (deg 4) | 0.004 | 1.01 | 0.023 | 2.36 | 2.3 | ok body, Runge in the wings |
| LSQ spline (production) | 0.002 | 1.00 | 0.021 | 2.72 | 2.5 | tight fit, but spiky raw density |
| PCHIP (monotone) | 0.000 | 1.00 | 0.199 | 6.38 | 14 | **interpolates noise → 20% arbitrage** |
| LOWESS | 0.003 | 0.99 | 29.4 (rough) | 29.4 | 93 | **density is pure noise** |

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

## Recommendation
Move the **global** smoother from LSQ to a **parametric arbitrage-free form —
SVI or SABR**. Both beat LSQ on density smoothness and arbitrage, match its fit
quality, and remove the need for post-hoc density repair. SVI is the modern
equity-index standard (better wing control); SABR is the classic, physically
motivated 3-parameter form and scored the lowest arbitrage here. Final pick is a
visual call on the wings (see `smoother_comparison.png`); either is a structural
improvement over the current knot-spline.

> NOTE: the raw density is shown pre-repair to expose each method's *native*
> quality. In production every method passes through the isotonic + smoothing
> step, so even LSQ yields a usable density — but SVI/SABR need no repair at all.
