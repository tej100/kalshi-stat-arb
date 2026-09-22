"""Transform - the shared discrete bucket grid.

Kalshi defines the discrete outcome grid (13 non-overlapping $200 buckets per
year, named e.g. '5000.0-5199.99'). Every source's probability estimate is
standardized onto this grid so the strategy stage can compare them directly.
"""
from __future__ import annotations
import numpy as np


def bucket_bounds(bucket: str) -> tuple[float, float]:
    """'5000.0-5199.99' -> (5000.0, 5199.99)."""
    lo, hi = bucket.split("-")
    return float(lo), float(hi)


def edges(buckets) -> np.ndarray:
    """Sorted array of unique bucket boundaries across a set of buckets."""
    e = set()
    for b in buckets:
        lo, hi = bucket_bounds(b)
        e.update((lo, hi))
    return np.array(sorted(e))
