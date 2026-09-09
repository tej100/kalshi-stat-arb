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
from . import config, vol, pricing, io


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


def _iv_curve(day_df, otm=True, n_knots=None):
    """Fit a smoothed IV(strike) spline for one quote date.

    otm=True builds the market-standard OTM curve (puts below forward, calls
    above), which is less noisy than deep-ITM quotes; otm=False uses calls only
    (the paper's literal text).
    """
    F = day_df["forward"].iloc[0]
    if otm:
        sel = ((day_df["option_type"] == "p") & (day_df["strike"] < F)) | \
              ((day_df["option_type"] == "c") & (day_df["strike"] >= F))
        d = day_df[sel]
    else:
        d = day_df[day_df["option_type"] == "c"]
    d = d.dropna(subset=["IV"]).sort_values("strike")
    if d["strike"].nunique() < 4:
        return None
    n_knots = config.DENSITY_KNOTS if n_knots is None else n_knots
    spline = vol.fit_iv_spline(d["strike"], d["IV"], n_knots=n_knots)
    if spline is None:
        return None
    return spline, d["strike"].min(), d["strike"].max()


def bl_density(day_df, otm=True):
    """Return (grid, density) for the Breeden-Litzenberger RND on a wide grid."""
    curve = _iv_curve(day_df, otm=otm)
    if curve is None:
        return None
    spline, k_min, k_max = curve
    S = day_df["underlying"].iloc[0]      # dividend-adjusted spot
    T = day_df["T"].iloc[0]
    r = day_df["r"].iloc[0]

    lo = max(k_min - config.GRID_PAD_LOW, 1.0)
    hi = k_max + config.GRID_PAD_HIGH
    grid = np.linspace(lo, hi, config.GRID_POINTS)

    sigma = vol.eval_on_grid(spline, grid, k_min, k_max)   # flat wings
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


def bl_pmf(day_df, buckets, otm=True):
    """Bucket -> probability for one day via Breeden-Litzenberger."""
    res = bl_density(day_df, otm=otm)
    if res is None:
        return None
    grid, density = res
    return {b: bucket_prob(grid, density, *io.bucket_bounds(b)) for b in buckets}


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
        L, U = io.bucket_bounds(b)
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
    ks = np.sort(day_df["strike"].unique())

    def interp(series, K):
        return float(np.interp(K, series.index, series.values,
                               left=series.iloc[0], right=series.iloc[-1]))

    mids = np.array([np.mean(io.bucket_bounds(b)) for b in buckets])
    d_max = max(abs(mids.min() - F), abs(mids.max() - F)) or 1.0
    out = {}
    for b in buckets:
        L, U = io.bucket_bounds(b)
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


def build_pmf_table(chain, kalshi, method="bl", ffill=True, **kw):
    """{year: DataFrame[date x bucket]} of model probabilities.

    Dates follow the Kalshi calendar; option data is forward/back-filled to
    non-trading days. NO renormalization to 1 (see module docstring).
    """
    fn = _METHODS[method]
    out = {}
    for year in kalshi:
        buckets = list(kalshi[year]["price"].columns)
        template = kalshi[year]["price"].copy()
        template[:] = np.nan
        yr_chain = chain[chain["quote"].dt.year == year]
        by_day = {d: g for d, g in yr_chain.groupby("quote", observed=True)}
        avail = pd.Series(sorted(by_day)).values
        for date in template.index:
            # nearest available option quote on/before this calendar date
            prior = avail[avail <= np.datetime64(date)]
            if len(prior) == 0:
                continue
            day_df = by_day[pd.Timestamp(prior[-1])]
            probs = fn(day_df, buckets, **kw) if method == "bl" else fn(day_df, buckets)
            if probs is not None:
                template.loc[date] = pd.Series(probs)
        if ffill:
            template = template.ffill().bfill()
        out[year] = template
    return out
