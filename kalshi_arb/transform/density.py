"""Step 5 - risk-neutral terminal distributions of SPX, discretized to Kalshi buckets.

Three methods (paper Density Formation):
  * Breeden-Litzenberger (PRIMARY): f_Q(K) = (1/DF) d^2C/dK^2 on a smoothed,
    WIDE strike grid (flat-vol extrapolation). NOT renormalized to 1 - the
    residual mass is the probability SPX breaches every Kalshi bucket.
  * GBM (benchmark): lognormal terminal law from ATM vol.
  * Option-price spread (heuristic, secondary).

A bucket probability is the raw integral of the density over [L, U]; the
undiscounted P(event) is compared to Kalshi prices (APY captures time value).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .. import config
from . import smoothing, pricing
from . import buckets as bkt


# --------------------------------------------------------------------------- #
# Breeden-Litzenberger
# --------------------------------------------------------------------------- #
def _isotonic_increasing(y):
    """Least-squares projection onto non-decreasing sequences (pool-adjacent-
    violators). Used to enforce that dC/dK is monotone (i.e. C is convex, so the
    recovered density is non-negative) without ad-hoc tail surgery."""
    y = np.asarray(y, float)
    vals, cnts = [], []
    for yi in y:
        vals.append(yi); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            nc = cnts[-1] + cnts[-2]
            nv = (vals[-1] * cnts[-1] + vals[-2] * cnts[-2]) / nc
            vals.pop(); cnts.pop(); vals[-1] = nv; cnts[-1] = nc
    out = np.empty(len(y)); i = 0
    for v, c in zip(vals, cnts):
        out[i:i + c] = v; i += c
    return out


def bl_density(day_df):
    """Return (grid, density) for the Breeden-Litzenberger RND on a wide grid,
    built from the SAME daily OTM-combined smile used for pricing and GBM
    (smoothing.fit_daily_smile)."""
    sm = smoothing.fit_daily_smile(day_df)
    if sm is None:
        return None
    S = day_df["underlying"].iloc[0]      # dividend-adjusted spot
    T = day_df["T"].iloc[0]
    r = day_df["r"].iloc[0]

    lo = max(sm.k_min - config.GRID_PAD_LOW, 1.0)
    hi = sm.k_max + config.GRID_PAD_HIGH
    grid = np.linspace(lo, hi, config.GRID_POINTS)

    sigma = smoothing.eval_smile(sm, grid)   # flat wings + IV envelope clamp
    calls = pricing.bsm_call(S, grid, T, r, sigma)
    # Enforce no-arbitrage before differentiating: dC/dK must lie in [-DF, 0]
    # and be non-decreasing (C convex), which makes the density non-negative and
    # removes spurious multi-modality from smile noise -- no ad-hoc surgery.
    DF = np.exp(-r * T)
    fp = np.clip(np.gradient(calls, grid), -DF, 0.0)
    fp = _isotonic_increasing(fp)
    fpp = np.gradient(fp, grid)
    density = np.clip(np.exp(r * T) * fpp, 0.0, None)
    # light, mass-preserving smoothing to remove the flat-extrapolation kink at
    # the observed-strike boundary (a thin spike in the raw 2nd derivative)
    w = config.DENSITY_SMOOTH_WINDOW
    if w and w > 1:
        density = np.convolve(density, np.ones(w) / w, mode="same")
    grid, density = grid[2:-2], density[2:-2]   # trim gradient boundary error
    return grid, density


def bucket_prob(grid, density, L, U):
    """Integrate a density over [L, U] (trapezoidal); 0 if no grid points."""
    mask = (grid >= L) & (grid <= U)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(density[mask], grid[mask]))


def bl_pmf(day_df, buckets):
    """Bucket -> probability for one day via Breeden-Litzenberger."""
    res = bl_density(day_df)
    if res is None:
        return None
    grid, density = res
    return {b: bucket_prob(grid, density, *bkt.bucket_bounds(b)) for b in buckets}


# --------------------------------------------------------------------------- #
# GBM (lognormal) benchmark
# --------------------------------------------------------------------------- #
def _atm_vol(day_df):
    """Smoothed IV at the strike closest to the forward (ATM)."""
    F = day_df["forward"].iloc[0]
    d = day_df.dropna(subset=["LSQ_Vol"])
    if d.empty:
        return None
    return d.loc[(d["strike"] - F).abs().idxmin(), "LSQ_Vol"]


def gbm_pmf(day_df, buckets):
    """Bucket -> probability under a lognormal terminal law.

    log S_T ~ N(log S0 + (r - sigma^2/2)T, sigma^2 T), with S0 the
    dividend-adjusted spot and sigma the ATM vol. (This is the correct reading
    of the paper's d2 formula, which carries an r-drift and hence uses spot,
    not forward.)
    """
    sigma = _atm_vol(day_df)
    if sigma is None or sigma <= 0:
        return None
    S0 = day_df["underlying"].iloc[0]
    T = day_df["T"].iloc[0]
    r = day_df["r"].iloc[0]

    def cdf(K):  # P(S_T <= K)
        return norm.cdf((np.log(K / S0) - (r - 0.5 * sigma ** 2) * T) /
                        (sigma * np.sqrt(T)))

    out = {}
    for b in buckets:
        L, U = bkt.bucket_bounds(b)
        out[b] = max(cdf(U) - cdf(L), 0.0)
    return out


# --------------------------------------------------------------------------- #
# Option-price-spread heuristic (secondary; paper ultimately discarded it)
# --------------------------------------------------------------------------- #
def spread_pmf(day_df, buckets):
    """Bucket -> probability from call/put price spreads, distance-weighted."""
    F = day_df["forward"].iloc[0]
    calls = day_df[day_df["option_type"] == "c"].set_index("strike")["BSM"]
    puts = day_df[day_df["option_type"] == "p"].set_index("strike")["BSM"]
    if calls.empty or puts.empty:
        return None

    def interp(series, K):
        return float(np.interp(K, series.index, series.values,
                               left=series.iloc[0], right=series.iloc[-1]))

    mids = np.array([np.mean(bkt.bucket_bounds(b)) for b in buckets])
    d_max = max(abs(mids.min() - F), abs(mids.max() - F)) or 1.0
    out = {}
    for b in buckets:
        L, U = bkt.bucket_bounds(b)
        p_call = (interp(calls, L) - interp(calls, U)) / (U - L)
        p_put = (interp(puts, U) - interp(puts, L)) / (U - L)
        M = 0.5 * (L + U)
        w_put = np.clip(0.5 * (1 - (M - F) / d_max), 0, 1)
        out[b] = max((1 - w_put) * p_call + w_put * p_put, 0.0)
    return out


# --------------------------------------------------------------------------- #
# Table builder
# --------------------------------------------------------------------------- #
_METHODS = {"bl": bl_pmf, "gbm": gbm_pmf, "spread": spread_pmf}


def build_pmf_table(chain, kalshi, method="bl"):
    """{year: DataFrame[date x bucket]} of model probabilities.

    A date is priced only if a same-year-expiry SPXW option chain exists for
    that EXACT calendar date -- no forward-fill, back-fill, or staleness
    tolerance of any kind. This is a deliberate design choice, not a
    simplification: the strategy's edge comes from the options market being
    the live, informed reference, which only holds while it is actually
    trading. The moment it isn't (a weekend/holiday, or the late-December
    expiry-rollover gap handled in transform/clean.py), a divergence between
    a frozen model view and a moving Kalshi price no longer shows Kalshi is
    mispriced -- it could equally mean the model is stale and Kalshi is
    right. There is no "safe" amount of staleness to tolerate here: the
    problem is categorical (is the reference live or not), not a matter of
    degree, so do not reintroduce a staleness-bound parameter without
    re-deriving this reasoning (see ASSUMPTIONS.md, Density formation).
    Downstream, signals.generate() skips any date with a NaN PMF, so no new
    position is opened that day; existing positions still mark-to-market and
    settle normally regardless. NO renormalization to 1 (see module
    docstring).
    """
    fn = _METHODS[method]
    out = {}
    for year in kalshi:
        buckets = list(kalshi[year]["price"].columns)
        template = kalshi[year]["price"].copy()
        template[:] = np.nan
        yr_chain = chain[chain["quote"].dt.year == year]
        by_day = {d: g for d, g in yr_chain.groupby("quote", observed=True)}
        for date in template.index:
            day_df = by_day.get(pd.Timestamp(date))
            if day_df is None:
                continue
            probs = fn(day_df, buckets)
            if probs is not None:
                template.loc[date] = pd.Series(probs)
        out[year] = template
    return out
