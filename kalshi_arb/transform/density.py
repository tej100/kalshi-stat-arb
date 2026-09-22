"""Step 5 - risk-neutral terminal distributions of SPX, discretized to Kalshi buckets.

Two methods (the paper's third, an option-price-spread heuristic, was tested and
removed -- it is not a valid bucket PMF; see the note near the table builder):
  * Breeden-Litzenberger (PRIMARY): f_Q(K) = (1/DF) d^2C/dK^2 on a smoothed,
    WIDE strike grid (flat-vol extrapolation). NOT renormalized to 1 - the
    residual mass is the probability SPX breaches every Kalshi bucket.
  * GBM (benchmark): lognormal terminal law from ATM vol -- the risk-neutral
    P(L < S_T < U) = N(-d2(L)) - N(-d2(U)), i.e. the same lognormal ITM
    probability (N(d2)) that appears in Black-Scholes, but with a single ATM
    vol so it ignores the skew the BL density captures.

A bucket probability is the raw integral of the density over [L, U]; the
undiscounted P(event) is compared to Kalshi prices. Strictly, a $1 claim paid at
year-end is worth DF * P today when the collateral earns nothing; the difference
averages 0.15c per bucket-day, well inside the two-fee entry hurdle, and moves
Sharpe by about -0.1 when applied. The cost of tying up collateral is booked
explicitly in the P&L instead (metrics.excess_value).
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


def _density_from_smile(sm, S, T, r):
    """(grid, density) for the BL RND on a wide grid, given a fitted smile."""
    grid = np.linspace(max(sm.k_min - config.GRID_PAD_LOW, 1.0),
                       sm.k_max + config.GRID_PAD_HIGH, config.GRID_POINTS)
    calls = pricing.bsm_call(S, grid, T, r, smoothing.eval_smile(sm, grid))
    # Enforce no-arbitrage before differentiating: dC/dK must lie in [-DF, 0]
    # and be non-decreasing (C convex), which makes the density non-negative and
    # removes spurious multi-modality from smile noise -- no ad-hoc surgery.
    DF = np.exp(-r * T)
    fp = _isotonic_increasing(np.clip(np.gradient(calls, grid), -DF, 0.0))
    density = np.clip(np.exp(r * T) * np.gradient(fp, grid), 0.0, None)
    # light, mass-preserving smoothing to remove the flat-extrapolation kink at
    # the observed-strike boundary (a thin spike in the raw 2nd derivative)
    w = config.DENSITY_SMOOTH_WINDOW
    if w and w > 1:
        density = np.convolve(density, np.ones(w) / w, mode="same")
    return grid[2:-2], density[2:-2]   # trim gradient boundary error


def bl_density(day_df):
    """Return (grid, density) for the Breeden-Litzenberger RND, built from the
    SAME daily OTM-combined smile used for pricing and GBM."""
    sm = smoothing.fit_daily_smile(day_df)
    if sm is None:
        return None
    return _density_from_smile(sm, day_df["underlying"].iloc[0],
                               day_df["T"].iloc[0], day_df["r"].iloc[0])


def bucket_prob(grid, density, L, U):
    """Integrate a density over [L, U] (trapezoidal); 0 if no grid points."""
    mask = (grid >= L) & (grid <= U)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(density[mask], grid[mask]))


def bl_pmf(day_df, buckets):
    """Bucket -> probability for one day via Breeden-Litzenberger.

    A bucket is priced only if it overlaps the observed strike range
    [k_min, k_max]; a bucket entirely outside that range would be priced purely
    off flat-vol extrapolation with no option support, so it is left NaN
    (untradeable) rather than assigned a spurious probability.
    """
    sm = smoothing.fit_daily_smile(day_df)
    if sm is None:
        return None
    grid, density = _density_from_smile(sm, day_df["underlying"].iloc[0],
                                        day_df["T"].iloc[0], day_df["r"].iloc[0])
    out = {}
    for b in buckets:
        L, U = bkt.bucket_bounds(b)
        out[b] = (bucket_prob(grid, density, L, U)
                  if (L < sm.k_max and U > sm.k_min) else np.nan)
    return out


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


# NOTE: the paper's third method -- estimating bucket probabilities from option
# PRICE SPREADS, p ~ [C(L) - C(U)] / (U - L) -- was implemented, tested, and
# removed. It is not a valid bucket PMF: that first difference approximates
# -dC/dK = DF * P(S_T > midpoint), a TAIL probability, not the bucket mass (the
# SECOND difference d^2C/dK^2). It overstates every bucket ~2-4x and sums to ~4,
# not 1. The paper already discarded it; it is not reconstructed here.


# --------------------------------------------------------------------------- #
# Table builder
# --------------------------------------------------------------------------- #
_METHODS = {"bl": bl_pmf, "gbm": gbm_pmf}


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


def lag_pmf_table(pmf_table, lag=1):
    """Shift each year's model PMF back by `lag` PRICED days (execution check).

    Same-close execution assumes a trade fills at the Kalshi close of the day
    whose option chain produced the signal. The option quotes are the day's last
    quotes (likely the 4:15pm SPXW close) while the Kalshi candle closes at
    4:00pm, so that assumption can carry up to 15 minutes of lookahead. With
    lag=1, each priced day instead uses the model computed from the PREVIOUS
    priced day's chain, and signals.generate re-tests it against that day's
    Kalshi book, so every fill uses only information that existed before it.
    Days the options did not trade stay unpriced, as in build_pmf_table."""
    out = {}
    for year, t in pmf_table.items():
        priced = t.dropna(how="all")
        out[year] = priced.shift(lag).reindex(t.index)
    return out
