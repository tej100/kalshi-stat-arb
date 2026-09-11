"""Step 2 - option-chain cleaning, transforms, and IV backfill.

Mirrors the paper's Data Preprocessing section. Deviations from the old
notebook are documented inline (marked NOTE) and in CLEANUP_LOG.md.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from . import pricing


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Scale units and derive rate, dividend-adjusted spot, forward, moneyness."""
    df = df.copy()
    df["IV"] = df["Implied Volatility"] / 100.0        # percent -> decimal
    df["T"] = df["T"] / 365.0                          # days -> years
    df["r"] = -np.log(df["DF"]) / df["T"]              # implied continuous rate
    df["spot"] = df["underlying"]                      # keep raw spot
    df["underlying"] = df["spot"] - df["div"] * df["DF"]  # strip PV of dividends
    df["forward"] = df["underlying"] * np.exp(df["r"] * df["T"])
    # moneyness centered at 1: K/F for calls, 2 - K/F for puts (paper eq.)
    kf = df["strike"] / df["forward"]
    df["moneyness"] = np.where(df["option_type"] == "c", kf, 2.0 - kf)
    return df


def _mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """Mid = (bid+ask)/2 when both present; one-sided quotes assume the missing
    side = 0 (mid = present/2) and are kept only if a vendor-IV BSM price is
    within ONE_SIDED_BSM_TOL of that theoretical mid."""
    df = df.copy()
    both = df["Bid"].notna() & df["Ask"].notna()
    one = (df["Bid"].notna() ^ df["Ask"].notna())

    df["Mid"] = np.nan
    df.loc[both, "Mid"] = df.loc[both, ["Bid", "Ask"]].mean(axis=1)
    # one-sided: present side / 2
    present = df[["Bid", "Ask"]].max(axis=1)
    df.loc[one, "Mid"] = present[one] / 2.0

    # validate one-sided quotes against BSM at vendor IV (drop if no IV or too far)
    val = df[one & df["IV"].notna()].copy()
    if not val.empty:
        model = pricing.bsm_price(val["option_type"].values, val["underlying"],
                                  val["strike"], val["T"], val["r"], val["IV"])
        keep = np.abs(model - val["Mid"]) <= config.ONE_SIDED_BSM_TOL
        drop_idx = val.index[~keep.values]
    else:
        drop_idx = pd.Index([])
    # one-sided with no vendor IV cannot be validated -> drop
    drop_idx = drop_idx.union(df.index[one & df["IV"].isna()])
    return df.drop(index=drop_idx)


def _backfill_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Invert BSM (bisection) for rows missing IV but having a valid mid."""
    df = df.copy()
    need = df["IV"].isna() & df["Mid"].notna()
    if need.any():
        df.loc[need, "IV"] = df[need].apply(
            lambda row: pricing.implied_vol(
                row["underlying"], row["strike"], row["T"], row["r"],
                row["Mid"], row["option_type"]),
            axis=1)
    return df


def clean_chain(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline. Returns the cleaned, priced-input chain."""
    n0 = len(df)
    # drop rows with no usable pricing info at all
    df = df.dropna(subset=["Bid", "Ask", "Implied Volatility"], how="all")
    df = _transform(df)
    df = _mid_price(df)

    # liquidity: drop wide two-sided spreads
    both = df["Bid"].notna() & df["Ask"].notna()
    spread_frac = (df["Ask"] - df["Bid"]) / df["Mid"]
    df = df.drop(index=df.index[both & (spread_frac > config.MAX_SPREAD_FRAC)])

    # moneyness outlier filter (global)
    mu, sd = df["moneyness"].mean(), df["moneyness"].std()
    df = df[np.abs(df["moneyness"] - mu) <= config.MONEYNESS_SIGMA * sd]

    n_missing_before = int(df["IV"].isna().sum())
    df = _backfill_iv(df)
    n_missing_after = int(df["IV"].isna().sum())

    df = df.drop(columns=["Implied Volatility"]).reset_index(drop=True)
    if verbose:
        print(f"[clean] {n0} -> {len(df)} rows | IV missing "
              f"{n_missing_before} -> {n_missing_after}")
    return df
