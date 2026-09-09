"""Data loading: raw SPXW option chain and Kalshi bucket order-book data."""
from __future__ import annotations
import pickle
import pandas as pd
from . import config

# Columns expected in the raw chain CSV (IV in percent, T in calendar days).
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


def load_kalshi(path=None) -> dict:
    """Load the Kalshi data pickle: {year: {tickers, bid, ask, price}}.

    Each of bid/ask/price is a DataFrame indexed by calendar date with one
    column per $200 bucket (e.g. '5000.0-5199.99'); values are cents (0-100).
    """
    path = path or config.KALSHI_PKL
    with open(path, "rb") as f:
        data = pickle.load(f)
    for year in data:
        for key in ("bid", "ask", "price"):
            data[year][key].index = pd.to_datetime(data[year][key].index)
    return data


def bucket_bounds(bucket: str) -> tuple[float, float]:
    """'5000.0-5199.99' -> (5000.0, 5199.99)."""
    lo, hi = bucket.split("-")
    return float(lo), float(hi)
