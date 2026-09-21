"""SPX-option replication of Kalshi buckets, and the cross-market edge it implies.

A Kalshi bucket [L, U] pays $1 if L <= S_T <= U, i.e. 1[S>L] - 1[S>U]. Each
indicator is approximated by a TIGHT SPREAD across the two listed strikes that
bracket the edge, scaled by 1/width so it climbs from 0 to 1 across the gap:

    1[S>K]  ~  call spread (long C(k1), short C(k2)) / (k2-k1)       payoff 0 -> 1
            =  1 - put spread (long P(k2), short P(k1)) / (k2-k1)    (put-call parity)

Each edge is built from the OUT-OF-THE-MONEY spread (puts below the forward,
calls above it). The two constructions have the same payoff and, by parity, the
same fair value, but in-the-money options are illiquid and quoted with wide
spreads, and because each leg carries weight 1/width, one index point of error in a
leg's quote moves the bucket price by 1/width of a unit (10 cents at a 10-point strike
spacing, 4 cents at 25). An ITM-call replication therefore turns quote noise into
apparent "mispricing". When the
two edges use different option types the identity 1[S>K] = 1 - put spread adds a
zero-coupon bond leg worth DF.

WHAT IS AND IS NOT MEASURED
  * The comparison is Kalshi vs replication on the SAME DAY only (no stale chain),
    using the Kalshi book (mid, bid, ask), never the last trade.
  * Both prices are present values of the same terminal claim, so they are
    compared directly with no discounting adjustment.
  * `edge` is the mid-to-mid gap. The EXECUTABLE edge crosses every spread:
    sell Kalshi at its bid and buy the replication at the option asks, or the
    reverse. Kalshi's taker fee is deducted; option commissions are not modelled.
  * The replication is imperfect by construction (the ramps). Its risk is the
    density-weighted expected payoff gap E_Q|condor - binary| under the day's
    own Breeden-Litzenberger density, not an average over an arbitrary grid.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .. import config
from ..transform import buckets, density
from ..transform.kalshi_pmf import is_valid_book, feed_outage_days
from .signals import kalshi_fee


@dataclass
class Replication:
    """legs: (option_type 'c'|'p', strike, qty); bond: units of a $1 zero-coupon
    bond at expiry; edges: the bracketing strikes (a1, a2) at L and (b1, b2) at U."""
    legs: list
    bond: float
    edges: tuple


def _bracket(strikes, target):
    """Nearest available strikes (below_or_eq, above) bracketing `target`."""
    below = strikes[strikes <= target]
    above = strikes[strikes > target]
    if len(below) == 0 or len(above) == 0:
        return None
    return float(below.max()), float(above.min())


def replicate_bucket(day_df, L, U):
    """Replication of bucket [L, U] from listed options, or None if it cannot be built.

    None when a bucket edge lies outside the listed strikes, or when both edges
    fall inside the same strike gap (the two ramps overlap and the legs would
    cancel, leaving a zero-priced "replication" of a bucket that is not zero).
    Only strikes listed as BOTH a call and a put are used, so either option type
    is available at every edge.
    """
    F = float(day_df["forward"].iloc[0])
    kc = np.sort(day_df.loc[day_df["option_type"] == "c", "strike"].unique())
    kp = np.sort(day_df.loc[day_df["option_type"] == "p", "strike"].unique())
    ks = np.intersect1d(kc, kp)
    lo, hi = _bracket(ks, L), _bracket(ks, U)
    if lo is None or hi is None:
        return None
    (a1, a2), (b1, b2) = lo, hi
    if a2 > b1:
        return None
    wl, wu = a2 - a1, b2 - b1
    legs, bond = [], 0.0
    if L < F:                                   # 1[S>L] = 1 - put spread
        bond += 1.0
        legs += [("p", a1, +1 / wl), ("p", a2, -1 / wl)]
    else:                                       # 1[S>L] = call spread
        legs += [("c", a1, +1 / wl), ("c", a2, -1 / wl)]
    if U < F:                                   # -1[S>U] = put spread - 1
        bond -= 1.0
        legs += [("p", b2, +1 / wu), ("p", b1, -1 / wu)]
    else:                                       # -1[S>U] = - call spread
        legs += [("c", b1, -1 / wu), ("c", b2, +1 / wu)]
    return Replication(legs=legs, bond=bond, edges=(a1, a2, b1, b2))


def condor_price(day_df, rep, side="mid"):
    """Present value of the replication. side: 'mid' | 'buy' (pay the ask on long
    legs, receive the bid on short legs) | 'sell' (the reverse: proceeds from
    selling the whole replication). NaN if any leg is missing or has no quote."""
    DF = float(day_df["DF"].iloc[0])
    tabs = {t: day_df[day_df["option_type"] == t].drop_duplicates("strike").set_index("strike")
            for t in ("c", "p")}
    total = rep.bond * DF
    for t, K, q in rep.legs:
        if K not in tabs[t].index:
            return np.nan
        row = tabs[t].loc[K]
        if side == "mid":
            px = row["Mid"]
        elif side == "buy":
            px = row["Ask"] if q > 0 else row["Bid"]
        else:
            px = row["Bid"] if q > 0 else row["Ask"]
        if pd.isna(px):
            return np.nan
        total += q * px
    return float(total)


def condor_payoff(rep, S_T):
    """Terminal payoff of the replication at index level(s) S_T (scalar or array)."""
    S = np.asarray(S_T, float)
    out = np.full(S.shape, rep.bond, dtype=float)
    for t, K, q in rep.legs:
        out = out + q * (np.maximum(S - K, 0.0) if t == "c" else np.maximum(K - S, 0.0))
    return float(out) if out.ndim == 0 else out


def binary_payoff(L, U, S_T):
    S = np.asarray(S_T, float)
    out = ((S >= L) & (S <= U)).astype(float)
    return float(out) if out.ndim == 0 else out


def basis_risk(rep, L, U, grid, dens):
    """(E|gap|, E[gap]) of replication payoff minus binary payoff under the density.

    E|gap| is the expected size of the replication error (the risk left in a
    hedged position); E[gap] is its signed mean (a systematic pricing bias of the
    replication). Both are in probability units and use the same day's risk-neutral
    density, so the weighting reflects where the index can actually settle.
    """
    gap = condor_payoff(rep, grid) - binary_payoff(L, U, grid)
    return float(np.trapz(np.abs(gap) * dens, grid)), float(np.trapz(gap * dens, grid))


def analyze_year(chain, kalshi, year, verbose=True):
    """Cross-market comparison for every bucket-day where BOTH markets are live and
    the replication is fully quoted. Returns a per-observation DataFrame.

    A bucket-day is included only if: the option chain for that exact date exists;
    the Kalshi book is valid (not empty, date not a feed outage); the bucket can be
    replicated; and every leg has a two-sided option quote (so mid, buy and sell
    prices are all defined and comparable).
    """
    bid, ask = kalshi[year]["bid"] / 100.0, kalshi[year]["ask"] / 100.0
    outage = set(feed_outage_days(kalshi, year))
    close = config.SPX_YEAR_END_CLOSE[year]
    lot = config.LOT_SIZE
    rows = []
    for date, day_df in chain[chain["quote"].dt.year == year].groupby("quote"):
        if date not in bid.index or date in outage:
            continue
        dens = None
        for bucket in bid.columns:
            kb, ka = bid.loc[date, bucket], ask.loc[date, bucket]
            if not bool(is_valid_book(kb, ka)):
                continue
            L, U = buckets.bucket_bounds(bucket)
            rep = replicate_bucket(day_df, L, U)
            if rep is None:
                continue
            cm, cbuy, csell = (condor_price(day_df, rep, s) for s in ("mid", "buy", "sell"))
            if np.isnan(cm) or np.isnan(cbuy) or np.isnan(csell):
                continue
            if dens is None:
                dens = density.bl_density(day_df)
                if dens is None:
                    break
            e_abs, e_sig = basis_risk(rep, L, U, *dens)
            a1, a2, b1, b2 = rep.edges
            e_sell_k = kb - cbuy               # sell Kalshi at the bid, buy the replication
            e_buy_k = csell - ka               # buy Kalshi at the ask, sell the replication
            direction = "sell_kalshi" if e_sell_k >= e_buy_k else "buy_kalshi"
            best = max(e_sell_k, e_buy_k)
            fee = kalshi_fee(kb if direction == "sell_kalshi" else ka, lot) / lot
            rows.append(dict(
                day=date, bucket=bucket, kalshi=0.5 * (kb + ka), condor_mid=cm,
                edge=0.5 * (kb + ka) - cm,                 # >0: Kalshi rich vs options
                exec_edge=best, direction=direction, fee=fee, net_edge=best - fee,
                option_spread=cbuy - csell, kalshi_spread=ka - kb,
                basis=e_abs, basis_signed=e_sig, ramp_width=(a2 - a1) + (b2 - b1),
                plateau_ok=bool(a2 <= 0.5 * (L + U) <= b1),
                settle_binary=binary_payoff(L, U, close), settle_condor=condor_payoff(rep, close)))
    out = pd.DataFrame(rows)
    if verbose and len(out):
        s = summarize_edges(out)
        print(f"[hedge {year}] {s['n']} obs | mean |edge| {s['mean_abs_edge']*100:.2f}c | "
              f"executable>0 {s['share_exec_positive']*100:.1f}% | net of fee>0 {s['share_net_positive']*100:.1f}% | "
              f"net>basis {s['share_net_beats_basis']*100:.1f}% | option spread median {s['median_option_spread']*100:.0f}c")
    return out


def summarize_edges(df):
    """Headline cross-market statistics for a frame from `analyze_year`."""
    return dict(
        n=len(df),
        mean_abs_edge=float(df["edge"].abs().mean()),
        median_option_spread=float(df["option_spread"].median()),
        median_kalshi_spread=float(df["kalshi_spread"].median()),
        share_exec_positive=float((df["exec_edge"] > 0).mean()),
        share_net_positive=float((df["net_edge"] > 0).mean()),
        share_net_beats_basis=float((df["net_edge"] > df["basis"]).mean()),
        median_net_when_positive=float(df.loc[df["net_edge"] > 0, "net_edge"].median())
        if (df["net_edge"] > 0).any() else float("nan"),
        mean_basis=float(df["basis"].mean()),
        mean_basis_signed=float(df["basis_signed"].mean()),
    )
