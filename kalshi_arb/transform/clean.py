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


def _drop_arbitrage_violations(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Drop quotes whose mid violates the European no-arbitrage price bounds.

    Every option price = intrinsic value + time value, and time value can never
    be negative (optionality only helps the holder). For a European option on
    the dividend-adjusted spot S with discount factor DF = e^{-rT}:

        call:  max(S - K*DF, 0)  <=  mid  <=  S
        put:   max(K*DF - S, 0)  <=  mid  <=  K*DF

    A mid below the intrinsic floor (negative implied time value) is impossible
    for a live quote -- it is a STALE deep-ITM quote whose bid/ask did not
    refresh after the underlying moved. Such rows corrupt the pricing-error
    statistics (large-magnitude outliers) and cannot be inverted for IV; they
    are all deep-ITM and never enter the OTM-combined density smile, so removing
    them does not affect the risk-neutral density or any strategy result.
    """
    S, K = df["underlying"], df["strike"]
    DF = np.exp(-df["r"] * df["T"])
    is_call = df["option_type"] == "c"
    intrinsic = np.where(is_call, np.maximum(S - K * DF, 0.0),
                         np.maximum(K * DF - S, 0.0))
    upper = np.where(is_call, S, K * DF)
    # Exact no-arbitrage box: keep iff intrinsic <= mid <= upper. No tolerance --
    # the bound is a hard law, violations here span $0-$28 (median $0.69), and a
    # strict test vs a 1e-9 epsilon drops the identical rows (no float-noise risk).
    bad = df["Mid"].notna() & ((df["Mid"] < intrinsic) | (df["Mid"] > upper))
    if verbose and bad.any():
        print(f"[clean] dropped {int(bad.sum())} rows violating no-arbitrage price "
              f"bounds (stale deep-ITM quotes: mid outside [intrinsic, upper])")
    return df[~bad]


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


def _restrict_to_current_year_expiry(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Keep only quotes whose expiry falls in the same calendar year as the
    quote date, matching the paper's design ("only SPXW options whose
    expiration dates aligned with Kalshi's event horizons were retained").

    The raw feed rolls to next year's contract in the final ~2 weeks of
    December (the expiring contract disappears from the feed before it
    settles), so without this filter those late-December quote dates would be
    priced off a ~365-day-to-maturity chain instead of a days-to-maturity one
    -- exactly the window where the density should be sharpening the most.
    Dropped dates are left with no same-year chain at all; build_pmf_table
    requires an exact same-calendar-day match, so those dates are correctly
    treated as unpriceable rather than silently reusing a stale prior chain.
    """
    same_year = df["exp"].dt.year == df["quote"].dt.year
    dropped = len(df) - int(same_year.sum())
    if verbose and dropped:
        bad_dates = df.loc[~same_year, "quote"]
        print(f"[clean] dropped {dropped} rows on {bad_dates.nunique()} quote date(s) "
              f"whose only available expiry rolled to next year "
              f"(e.g. {bad_dates.min().date()}..{bad_dates.max().date()})")
    return df[same_year]


def clean_chain(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Full cleaning pipeline. Returns the cleaned, priced-input chain."""
    n0 = len(df)
    # drop rows with no usable pricing info at all
    df = df.dropna(subset=["Bid", "Ask", "Implied Volatility"], how="all")
    df = _restrict_to_current_year_expiry(df, verbose)
    df = _transform(df)
    df = _mid_price(df)

    # liquidity: drop wide two-sided spreads
    both = df["Bid"].notna() & df["Ask"].notna()
    spread_frac = (df["Ask"] - df["Bid"]) / df["Mid"]
    df = df.drop(index=df.index[both & (spread_frac > config.MAX_SPREAD_FRAC)])

    # moneyness outlier filter (global)
    mu, sd = df["moneyness"].mean(), df["moneyness"].std()
    df = df[np.abs(df["moneyness"] - mu) <= config.MONEYNESS_SIGMA * sd]

    # no-arbitrage price-bound filter (removes stale deep-ITM quotes)
    df = _drop_arbitrage_violations(df, verbose)

    n_missing_before = int(df["IV"].isna().sum())
    df = _backfill_iv(df)
    n_missing_after = int(df["IV"].isna().sum())

    df = df.drop(columns=["Implied Volatility"]).reset_index(drop=True)
    if verbose:
        print(f"[clean] {n0} -> {len(df)} rows | IV missing "
              f"{n_missing_before} -> {n_missing_after}")
    return df
