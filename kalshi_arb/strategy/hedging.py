"""SPX iron-condor replication of Kalshi buckets, and basis-risk analysis.

A Kalshi bucket [L, U] pays $1 if L <= S_T <= U. It is replicated with SPX
options as the difference of two tight call spreads:

    long-range[L,U]  =  callspread(A1->A2 around L)  -  callspread(B1->B2 around U)

where A1<A2 bracket L and B1<B2 bracket U (nearest available strikes). Each
spread is scaled by 1/width so it climbs from 0 to 1, giving a payoff that is
~1 on [A2, B1] ~ [L, U] and ramps at the edges. The ramp (finite strike
spacing) is the irreducible BASIS RISK versus the exact binary.

Buying the Kalshi bucket and selling this replicating condor (or vice-versa)
locks the cross-market mispricing up to that basis risk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config
from ..transform import buckets


def _bracket(strikes, target):
    """Nearest available strikes (below_or_eq, above) bracketing `target`."""
    below = strikes[strikes <= target]
    above = strikes[strikes > target]
    if len(below) == 0 or len(above) == 0:
        return None
    return float(below.max()), float(above.min())


def replicate_bucket(day_df, L, U):
    """Return replicating call legs [(strike, qty), ...] for bucket [L, U].

    qty > 0 long, < 0 short. None if strikes don't bracket both edges.
    """
    calls = day_df[day_df["option_type"] == "c"]
    ks = np.sort(calls["strike"].unique())
    lo = _bracket(ks, L)
    hi = _bracket(ks, U)
    if lo is None or hi is None:
        return None
    a1, a2 = lo
    b1, b2 = hi
    if a2 == a1 or b2 == b1:
        return None
    wl, wu = a2 - a1, b2 - b1
    # low spread climbs 0->1 across [a1,a2]; high spread (subtracted) across [b1,b2]
    legs = [(a1, +1 / wl), (a2, -1 / wl), (b1, -1 / wu), (b2, +1 / wu)]
    # merge duplicate strikes (e.g. a2 == b1)
    merged = {}
    for k, q in legs:
        merged[k] = merged.get(k, 0.0) + q
    return [(k, q) for k, q in merged.items() if abs(q) > 1e-12]


def condor_price(day_df, legs, side="mid"):
    """Cost to enter the replicating condor. side: 'mid' | 'buy' (pay ask on
    longs / receive bid on shorts) | 'sell' (the reverse)."""
    calls = day_df[day_df["option_type"] == "c"].set_index("strike")
    cost = 0.0
    for k, q in legs:
        if k not in calls.index:
            return np.nan
        row = calls.loc[k]
        if side == "mid":
            px = row["Mid"]
        elif side == "buy":
            px = row["Ask"] if q > 0 else row["Bid"]
        else:
            px = row["Bid"] if q > 0 else row["Ask"]
        if pd.isna(px):
            return np.nan
        cost += q * px
    return float(cost)


def condor_payoff(legs, S_T):
    """Terminal payoff of the replicating condor at underlying level S_T."""
    return float(sum(q * max(S_T - k, 0.0) for k, q in legs))


def binary_payoff(L, U, S_T):
    return 1.0 if L <= S_T <= U else 0.0


def basis_risk(legs, L, U, grid=None, reduce="mean"):
    """Payoff gap between condor and binary over a price grid.

    reduce='mean' returns the average |gap| (expected replication error, small -
    concentrated in the two ramp regions); reduce='max' returns the worst-case
    gap (~1 at the binary's knife-edge, irreducible)."""
    if grid is None:
        grid = np.linspace(L - 400, U + 400, 400)
    diff = np.abs([condor_payoff(legs, s) - binary_payoff(L, U, s) for s in grid])
    return float(np.max(diff) if reduce == "max" else np.mean(diff))


def analyze_year(chain, kalshi, year, verbose=True):
    """Cross-market comparison for every tradable bucket-day: Kalshi mid vs
    replicating-condor mid, plus settlement basis risk at the true year-end close.
    Returns a per-observation DataFrame."""
    price = kalshi[year]["price"]
    by_day = {d: g for d, g in chain[chain["quote"].dt.year == year].groupby("quote")}
    avail = np.array(sorted(by_day))
    close = None
    close = config.SPX_YEAR_END_CLOSE[year]
    rows = []
    for date in price.index:
        prior = avail[avail <= np.datetime64(date)]
        if len(prior) == 0:
            continue
        day_df = by_day[pd.Timestamp(prior[-1])]
        for bucket in price.columns:
            kp = price.loc[date, bucket]
            if pd.isna(kp):
                continue
            L, U = buckets.bucket_bounds(bucket)
            legs = replicate_bucket(day_df, L, U)
            if legs is None:
                continue
            cmid = condor_price(day_df, legs, "mid")
            if np.isnan(cmid):
                continue
            rows.append(dict(
                day=date, bucket=bucket, kalshi=kp / 100.0, condor_mid=cmid,
                edge=kp / 100.0 - cmid,          # >0: Kalshi rich vs options
                basis=basis_risk(legs, L, U),
                settle_binary=binary_payoff(L, U, close),
                settle_condor=condor_payoff(legs, close)))
    out = pd.DataFrame(rows)
    if verbose and len(out):
        print(f"[hedge {year}] {len(out)} obs | mean |edge|={out['edge'].abs().mean():.3f} "
              f"| mean basis={out['basis'].mean():.3f} "
              f"| corr(kalshi,condor)={out['kalshi'].corr(out['condor_mid']):.2f}")
    return out
