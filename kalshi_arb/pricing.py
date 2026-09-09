"""Black-Scholes-Merton pricing primitives, implied-vol inversion, control variate.

Convention: spot-based BSM on the dividend-adjusted underlying S with continuous
rate r (so no separate dividend yield term). T is in YEARS, sigma in decimal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from . import config


def _d1_d2(S, K, T, r, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return d1, d1 - sigma * sqrtT


def bsm_call(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_put(S, K, T, r, sigma):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_price(option_type, S, K, T, r, sigma):
    """Vectorized BSM price; option_type is 'c'/'p' array-like or scalar."""
    call = bsm_call(S, K, T, r, sigma)
    put = bsm_put(S, K, T, r, sigma)
    return np.where(np.asarray(option_type) == "c", call, put)


def implied_vol(S, K, T, r, market_price, option_type,
                tol=1e-5, max_iter=100):
    """Invert BSM for implied vol via bisection (decimal). NaN if no root."""
    lo, hi = config.IV_MIN, config.IV_MAX
    price_fn = bsm_call if option_type == "c" else bsm_put
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = price_fn(S, K, T, r, mid)
        if np.abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid
    return np.nan


def apply_control_variate(model_price, mid, lam=None):
    """Price_adj = model + lam*(mid - model). Anchors model toward market."""
    lam = config.CONTROL_VARIATE_LAMBDA if lam is None else lam
    return model_price + lam * (mid - model_price)


def price_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Add a control-variate-adjusted BSM price column using LSQ_Vol."""
    df = df.copy()
    raw = bsm_price(df["option_type"].values, df["underlying"], df["strike"],
                    df["T"], df["r"], df["LSQ_Vol"])
    df["BSM"] = apply_control_variate(raw, df["Mid"].values)
    return df
