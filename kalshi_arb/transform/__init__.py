"""Stage 2 - TRANSFORM: raw data -> discrete bucket probability mass functions.

Both sources converge to the same representation (a date x bucket PMF on the
shared grid in `buckets`), so the strategy stage can treat them symmetrically:

    buckets     : the shared discrete bucket grid (standardization)
    clean       : option-chain cleaning + unit/rate/forward transforms
    smiles      : the swappable IV-smile functional forms (SABR default; REGISTRY)
    smoothing   : fits ONE daily smile per quote date and evaluates it
    pricing     : Black-Scholes-Merton model price (no blend toward market)
    density     : options  -> risk-neutral density (BL) / GBM -> model bucket PMF
    kalshi_pmf  : Kalshi order book -> market-implied bucket PMF (analysis-facing)
                  + is_valid_book, the shared book-validity rule used for marking
"""
