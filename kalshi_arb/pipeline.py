"""End-to-end orchestration across the three stages.

    build_chain() : extract.refinitiv -> transform.clean -> smoothing -> pricing
                    (the cleaned, priced option chain, single source of truth)
"""
from __future__ import annotations
from . import config
from .extract import refinitiv
from .transform import clean, smoothing, pricing


def build_chain(path=None, save=False, verbose=True):
    """Load the raw chain, clean, smooth vols, and price. Single source of truth
    for the cleaned+priced option chain used downstream."""
    df = refinitiv.load_raw_chain(path)
    df = clean.clean_chain(df, verbose=verbose)
    df = smoothing.smooth_chain(df)
    df = pricing.price_chain(df)
    if verbose:
        pe = (df["BSM"] - df["Mid"]).dropna()
        print(f"[price] MAE={pe.abs().mean():.2f}  RMSE={(pe**2).mean()**0.5:.2f}  "
              f"n={len(pe)}")
    if save:
        config.ARTIFACT_DIR.mkdir(exist_ok=True)
        out = config.ARTIFACT_DIR / "clean_chain.parquet"
        df.to_parquet(out)
        if verbose:
            print(f"[save] {out}")
    return df
