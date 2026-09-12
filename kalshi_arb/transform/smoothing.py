"""Step 3 - implied-volatility smoothing (pipeline layer).

ONE smile is fitted per quote date, on out-of-the-money options (puts below the
forward, calls above -- the less-noisy side of each strike), with the functional
form chosen by `config.SMILE_METHOD` (SABR by default; LSQ/SVI/cubic/poly/
PCHIP/LOWESS also available in `smiles.REGISTRY`). That single surface is used
everywhere downstream: option pricing (`LSQ_Vol` column), the GBM at-the-money
vol, and the Breeden-Litzenberger density.

The functional forms themselves live in `smiles.py`; this module owns the OTM
selection and the evaluation clamps.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from . import smiles


def fit_daily_smile(day_df, method=None):
    """Fit the configured smoother on one quote date's OTM-combined smile.

    OTM-combined = puts where strike < forward, calls where strike >= forward
    (each strike from its less-noisy side; by put-call parity this IV(K) is
    correct for either option type). Returns the fitted smoother object (which
    carries k_min/k_max/iv_min/iv_max and evaluates at any strike), or None if
    too few strikes or the fit/calibration fails.
    """
    F = day_df["forward"].iloc[0]
    sel = ((day_df["option_type"] == "p") & (day_df["strike"] < F)) | \
          ((day_df["option_type"] == "c") & (day_df["strike"] >= F))
    d = day_df[sel].dropna(subset=["IV"]).sort_values("strike")
    if d["strike"].nunique() < 4:
        return None
    return smiles.fit_smile(d["strike"].values, d["IV"].values, F,
                            float(day_df["T"].iloc[0]), method=method)


def eval_smile(smoother, strikes):
    """Evaluate a fitted daily smile at arbitrary strikes, with two data-derived
    bounds (no tunable constant):
      * strikes clamped to [k_min, k_max]  -> flat extrapolation in the wings;
      * IV clamped to the observed [iv_min, iv_max] envelope -> a smoothed IV can
        never leave the range of real market IVs (guards against any fit
        overshooting into a sparse far-OTM strike gap).
    """
    iv = smoother(np.asarray(strikes, float), clamp_wings=True)
    return np.clip(iv, smoother.iv_min, smoother.iv_max)


def smooth_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 'LSQ_Vol' column: the daily OTM-combined smile evaluated at every
    option's strike. Same surface, same evaluation, the BL density uses.
    (Column name kept as LSQ_Vol for continuity; it is whatever SMILE_METHOD is.)"""
    df = df.copy()
    df["LSQ_Vol"] = np.nan
    for date, grp in df.groupby("quote", observed=True):
        sm = fit_daily_smile(grp)
        if sm is None:
            continue
        df.loc[grp.index, "LSQ_Vol"] = eval_smile(sm, grp["strike"].values)
    return df
