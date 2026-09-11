"""Extract - raw SPXW option chain (Refinitiv Elektron via Hanlon Labs).

The chain is delivered as a CSV; IV is in percent and T in calendar days at this
stage (the transform stage scales them). Column contract:
    Ask, Bid, Implied Volatility, quote, exp, T, option_type, strike,
    underlying, DF, div, fwd
"""
from __future__ import annotations
import pandas as pd
from .. import config

_RAW_NUMERIC = ["Ask", "Bid", "Implied Volatility", "T", "strike",
                "underlying", "DF", "div", "fwd"]


def load_raw_chain(path=None) -> pd.DataFrame:
    """Load the raw SPXW end-of-year option chain, typed and sorted."""
    path = path or config.CHAIN_CSV
    df = pd.read_csv(path)
    for col in _RAW_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["quote"] = pd.to_datetime(df["quote"], format="%Y-%m-%d")
    df["exp"] = pd.to_datetime(df["exp"], format="%Y-%m-%d")
    df["option_type"] = df["option_type"].astype("category")
    return df.sort_values(["quote", "strike"]).reset_index(drop=True)
