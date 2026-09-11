"""Stage 2 - TRANSFORM: raw data -> discrete bucket probability mass functions.

Both sources converge to the same representation (a date x bucket PMF on the
shared grid in `buckets`), so the strategy stage can treat them symmetrically:

    buckets     : the shared discrete bucket grid (standardization)
    clean       : option-chain cleaning + unit/rate/forward transforms
    smoothing   : implied-volatility smoothing (LSQ spline)
    pricing     : Black-Scholes-Merton + control variate
    density     : options  -> risk-neutral density (BL) / GBM -> model bucket PMF
    kalshi_pmf  : Kalshi order book -> market-implied bucket PMF
"""
