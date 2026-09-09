"""Step 3 - implied-volatility smoothing via least-squares splines (LSQ_Vol).

The paper evaluated several smoothers (LOWESS, cubic, poly, PCHIP, SABR) and
selected the LSQ univariate spline. That bake-off is out of scope for the core
arbitrage paper; only the chosen LSQ smoother lives here. Flat extrapolation is
provided so the density module can widen the strike grid without spurious wings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline

DEFAULT_KNOTS = 10


def fit_iv_spline(strikes, iv, n_knots=DEFAULT_KNOTS):
    """LSQ spline of IV vs strike with uniformly spaced interior knots.

    Knot count is reduced when data are sparse (needs >= n_knots+2 points).
    Returns a fitted LSQUnivariateSpline, or None if < 3 usable points.
    """
    strikes = np.asarray(strikes, float)
    iv = np.asarray(iv, float)
    order = np.argsort(strikes)
    strikes, iv = strikes[order], iv[order]
    # collapse duplicate strikes (spline requires strictly increasing x)
    strikes, idx = np.unique(strikes, return_index=True)
    iv = iv[idx]
    if len(strikes) < 4:
        return None
    # Interior knots at interior QUANTILES of the strikes so every knot interval
    # contains data (satisfies the Schoenberg-Whitney condition). Retry with
    # fewer knots if the fit is still rejected for sparse/clustered strikes.
    for k in range(min(n_knots, len(strikes) - 2), 0, -1):
        interior = np.quantile(strikes, np.linspace(0, 1, k + 2)[1:-1])
        interior = np.unique(interior)
        interior = interior[(interior > strikes.min()) & (interior < strikes.max())]
        if len(interior) == 0:
            continue
        try:
            return LSQUnivariateSpline(strikes, iv, interior)
        except ValueError:
            continue
    return None


def eval_on_grid(spline, grid, k_min, k_max):
    """Evaluate a fitted IV spline on an arbitrary grid with FLAT extrapolation
    outside the observed [k_min, k_max] strike range."""
    grid = np.asarray(grid, float)
    clamped = np.clip(grid, k_min, k_max)
    return spline(clamped)


def smooth_chain(df: pd.DataFrame, n_knots=DEFAULT_KNOTS) -> pd.DataFrame:
    """Add an 'LSQ_Vol' column: smoothed IV at each observed strike, fit per
    (quote date, option type)."""
    df = df.copy()
    df["LSQ_Vol"] = np.nan
    for (date, opt), grp in df.groupby(["quote", "option_type"], observed=True):
        valid = grp.dropna(subset=["IV"])
        spline = fit_iv_spline(valid["strike"], valid["IV"], n_knots)
        if spline is None:
            continue
        df.loc[grp.index, "LSQ_Vol"] = spline(grp["strike"].values)
    return df
