"""Step 3 - implied-volatility smoothing (LSQ spline over OTM options).

ONE smile is fitted per quote date, on out-of-the-money options (puts below the
forward, calls above -- the less-noisy side of each strike), and is the single
smoothed surface used everywhere downstream: option pricing (`LSQ_Vol` column),
the GBM at-the-money vol, and the Breeden-Litzenberger density all consume it.
There is deliberately no second/per-type smoother.

The paper evaluated several smoothers (LOWESS, cubic, poly, PCHIP, SABR) and
selected the LSQ univariate spline; that bake-off is out of scope for the core
arbitrage paper, so only the chosen LSQ smoother lives here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline
from .. import config


def fit_iv_spline(strikes, iv, n_knots=None):
    """LSQ spline of IV vs strike with interior knots at strike quantiles.

    Quantile knots guarantee the Schoenberg-Whitney condition on clustered
    strikes; the knot count is reduced on sparse data and the fit retried.
    Returns a fitted LSQUnivariateSpline, or None if < 4 usable points.
    """
    n_knots = config.SMILE_KNOTS if n_knots is None else n_knots
    strikes = np.asarray(strikes, float)
    iv = np.asarray(iv, float)
    order = np.argsort(strikes)
    strikes, iv = strikes[order], iv[order]
    strikes, idx = np.unique(strikes, return_index=True)   # strictly increasing x
    iv = iv[idx]
    if len(strikes) < 4:
        return None
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


def fit_daily_smile(day_df, n_knots=None):
    """Fit the single OTM-combined IV(strike) smile for one quote date.

    OTM-combined = puts where strike < forward, calls where strike >= forward
    (each strike taken from its less-noisy, out-of-the-money side; by put-call
    parity this one IV(K) is correct for either option type at that strike).

    Returns a `curve` tuple (spline, k_min, k_max, iv_min, iv_max) for eval_smile,
    or None if too few OTM strikes. k_min/k_max bound the observed strikes;
    iv_min/iv_max bound the observed IVs.
    """
    F = day_df["forward"].iloc[0]
    sel = ((day_df["option_type"] == "p") & (day_df["strike"] < F)) | \
          ((day_df["option_type"] == "c") & (day_df["strike"] >= F))
    d = day_df[sel].dropna(subset=["IV"]).sort_values("strike")
    if d["strike"].nunique() < 4:
        return None
    spline = fit_iv_spline(d["strike"], d["IV"], n_knots=n_knots)
    if spline is None:
        return None
    return (spline, float(d["strike"].min()), float(d["strike"].max()),
            float(d["IV"].min()), float(d["IV"].max()))


def eval_smile(curve, strikes):
    """Evaluate a fitted daily smile at arbitrary strikes.

    Two bounds are enforced, both data-derived (no tunable constant):
      * strikes are clamped to [k_min, k_max]  -> FLAT extrapolation in the wings;
      * the resulting IV is clamped to [iv_min, iv_max] -> the smoothed IV can
        never leave the envelope of observed market IVs, which prevents the LSQ
        spline from overshooting to absurd (or negative) vols inside sparse
        far-OTM strike gaps (~12% of days have a >$500 gap between OTM strikes).
    """
    spline, k_min, k_max, iv_min, iv_max = curve
    strikes = np.asarray(strikes, float)
    return np.clip(spline(np.clip(strikes, k_min, k_max)), iv_min, iv_max)


def smooth_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'LSQ_Vol' column: the daily OTM-combined smile evaluated at every
    option's strike (both calls and puts). This is the same surface, evaluated
    the same way (eval_smile), that the BL density is built from."""
    df = df.copy()
    df["LSQ_Vol"] = np.nan
    for date, grp in df.groupby("quote", observed=True):
        curve = fit_daily_smile(grp)
        if curve is None:
            continue
        df.loc[grp.index, "LSQ_Vol"] = eval_smile(curve, grp["strike"].values)
    return df
