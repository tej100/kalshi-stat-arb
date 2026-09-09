"""End-to-end orchestration of the reproducible pipeline.

    build_chain() : io.load_raw_chain -> cleaning -> vol -> pricing
"""
from __future__ import annotations
import pandas as pd
from . import io, cleaning, vol, pricing, config


def build_chain(path=None, save=False, verbose=True) -> pd.DataFrame:
    """Load raw chain, clean, smooth vols, and price. Single source of truth
    for the cleaned+priced option chain used downstream."""
    df = io.load_raw_chain(path)
    df = cleaning.clean_chain(df, verbose=verbose)
    df = vol.smooth_chain(df)
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
