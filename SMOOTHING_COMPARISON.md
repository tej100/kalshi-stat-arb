# IV Smoothing Method Comparison

Reusable head-to-head of implied-volatility smoothing methods for risk-neutral
density (RND) extraction. The smoothers live in
[`kalshi_arb/transform/smiles.py`](kalshi_arb/transform/smiles.py) (one registry,
shared with the production pipeline) and
[`analysis/compare_smoothers.py`](analysis/compare_smoothers.py) drives the
comparison and writes its plot to `analysis/figures/` (generated, not tracked).
Standalone enough to lift into a separate volatility-surface methods paper.

## Methods (all those in the original paper + SVI)
LSQ spline (former default), cubic smoothing spline, polynomial (deg 4),
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
## production `eval_smile`: strike clamp for every method, IV envelope for the
## spline smoothers only, positivity floor for SABR/SVI)

| method | fit_rmse | overshoot | arb_neg | roughness | modes | verdict |
|---|---|---|---|---|---|---|
| **SABR** (Hagan, β=0.5) — **CHOSEN** | 0.004 | 0.99 | **0.002** | **2.16** | 1.7 | smoothest, ~arbitrage-free, 705/705 days calibrate |
| **SVI** (Gatheral raw) — kept | 0.004 | 1.00 | **0.002** | 2.28 | 1.7 | ~arbitrage-free, flexible wings |
| cubic smoothing spline | 0.005 | 0.99 | 0.027 | 2.38 | 2.0 | smooth but oversmooths wings |
| polynomial (deg 4) | 0.004 | 1.00 | 0.024 | 2.63 | 2.2 | ok body, Runge in the wings |
| LSQ spline (former default) | 0.002 | 1.00 | 0.031 | 2.75 | 2.7 | tight fit, spiky raw density, needs repair |
| PCHIP (monotone) | 0.000 | 1.00 | 0.216 | 6.37 | 14 | **interpolates noise → 20% arbitrage** |
| LOWESS | 0.003 | 0.98 | 0.013 | 27.2 | 97 | **density is pure noise** |

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

## Strategy-level robustness: does the smile model change the result?

The table above judges smoothers on density quality. The question that matters for
the strategy is whether the choice changes what is traded and earned.
[`analysis/smile_robustness.py`](analysis/smile_robustness.py) reruns the full
backtest (both-side, liquidation-side marking, risk-free = Kalshi APY) with four
treatments of the smile, none of which adds a tuned parameter: **SABR** (default),
**SVI**, the **average** of the two bucket PMFs, and an **agreement** filter that
keeps a SABR signal only when SVI independently signals the same direction.

| variant | 2022 Sharpe [95% CI] | 2023 Sharpe [95% CI] | 2024 Sharpe [95% CI] | fills 22 / 23 / 24 |
|---|---|---|---|---|
| SABR | 1.99 [0.26, 3.78] | 2.25 [0.27, 3.98] | 1.08 [−0.29, 2.37] | 100 / 81 / 39 |
| SVI | 1.94 [0.11, 3.83] | 2.74 [0.41, 4.73] | 0.67 [−0.54, 1.87] | 89 / 93 / 41 |
| average | 1.87 [0.12, 3.72] | 2.16 [0.28, 3.77] | 0.89 [−0.51, 2.24] | 91 / 89 / 45 |
| agreement | 2.08 [0.22, 4.00] | 2.19 [0.32, 3.83] | 0.69 [−0.51, 1.85] | 83 / 74 / 37 |

**The four are statistically indistinguishable, though not identical.** The largest
Sharpe gap between any two variants is 0.21 / 0.58 / 0.41 by year, against confidence
intervals about 3.3 Sharpe units wide. The one visible spread is BL 2023, where the
point estimate runs from 2.16 (average) to 2.74 (SVI): the smile treatment moves that
single number by roughly ±0.3 around SABR's 2.25, so it should be quoted with its
interval and not read as a property of the model. The densities do differ (mean
absolute SABR−SVI bucket probability 0.41¢), so this is not a case of the toggle doing
nothing. The agreement filter removes trades without improving results, so it is not
worth its complexity.

**Density quality in production** (all 704 days, SABR): after removing the IV-envelope
clamp for the parametric smiles, unimodal densities rose from 26.7% to 43.5% of days,
days with three or more peaks fell from 54.3% to 9.9%, mean peaks from 2.32 to 1.66
and roughness (total variation / peak) from 2.50 to 2.22, with mass unchanged at
0.9996. Some interior extra peaks remain (about 0.45 a day), which need not be errors:
a left-tail shoulder is normal in an equity skew.

Consequences: the choice of SABR over SVI rests on smoothness and calibration
(above), not on strategy performance, and the paper can report the result as
insensitive to the smile model. Sharpe figures in the decision section above
predate later audit fixes; use this table for current numbers.
