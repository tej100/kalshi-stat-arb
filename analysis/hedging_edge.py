"""Cross-market check: Kalshi buckets versus SPX-option replication.

Verifies the replication construction, then reports how large the Kalshi-vs-options
gap is and how much of it survives crossing the bid-ask spreads.

Run from repo root:  python analysis/hedging_edge.py
Writes artifacts/hedging_edges.csv (untracked); write-up in HEDGING.md.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from kalshi_arb import pipeline, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import buckets
from kalshi_arb.strategy import hedging


def verify_construction(chain, k, n_days=30, seed=0):
    """Payoff checks on real chains, broken out by which option types build the edges.

    Every replication must pay 0 far below and far above its strikes (a construction
    property that must hold exactly). Separately, report how often the bucket centre
    sits on the flat top of the replication (payoff 1), which fails only when the strike
    grid is too coarse for the bucket -- a replication-quality issue, not a maths error.
    """
    days = chain["quote"].drop_duplicates().sample(n_days, random_state=seed)
    stats = {}
    n_none = 0
    for d in days:
        day_df = chain[chain["quote"] == d]
        F = float(day_df["forward"].iloc[0])
        for b in k[d.year]["price"].columns:
            L, U = buckets.bucket_bounds(b)
            rep = hedging.replicate_bucket(day_df, L, U)
            if rep is None:
                n_none += 1
                continue
            combo = f"{'put' if L < F else 'call'} @L / {'put' if U < F else 'call'} @U"
            ks = [K for _, K, _ in rep.legs]
            ends_ok = (abs(hedging.condor_payoff(rep, min(ks) - 500)) < 1e-9
                       and abs(hedging.condor_payoff(rep, max(ks) + 500)) < 1e-9)
            top_ok = abs(hedging.condor_payoff(rep, 0.5 * (L + U)) - 1.0) < 1e-9
            s = stats.setdefault(combo, dict(n=0, ends=0, top=0))
            s["n"] += 1; s["ends"] += ends_ok; s["top"] += top_ok
    return stats, n_none


def main():
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()

    stats, n_none = verify_construction(chain, k)
    print("=== 1. Construction check (real chains, 30 random dates) ===")
    for combo, s in sorted(stats.items()):
        print(f"  {combo:>22}: {s['n']:4d} replications | pay 0 far below/above: {s['ends']}/{s['n']} "
              f"| flat top covers bucket centre: {s['top']}/{s['n']}")
    print(f"  not replicable (edge outside the listed strikes, or both edges in one strike gap): {n_none}")

    frames = []
    print("\n=== 2. Kalshi vs replication, same-day, both markets live ===")
    for y in config.YEARS:
        f = hedging.analyze_year(chain, k, y)
        f.insert(0, "year", y)
        frames.append(f)
    A = pd.concat(frames, ignore_index=True)

    print("\n=== 3. Summary ===")
    rows = []
    for y, g in A.groupby("year"):
        rows.append(dict(year=y, **hedging.summarize_edges(g)))
    rows.append(dict(year="all", **hedging.summarize_edges(A)))
    S = pd.DataFrame(rows)
    show = S.assign(mean_abs_edge_c=S.mean_abs_edge * 100, opt_spread_c=S.median_option_spread * 100,
                    kalshi_spread_c=S.median_kalshi_spread * 100, exec_pos_pct=S.share_exec_positive * 100,
                    net_pos_pct=S.share_net_positive * 100, net_beats_basis_pct=S.share_net_beats_basis * 100,
                    basis_c=S.mean_basis * 100, basis_signed_c=S.mean_basis_signed * 100)
    print(show[["year", "n", "mean_abs_edge_c", "opt_spread_c", "kalshi_spread_c", "exec_pos_pct", "net_pos_pct",
                "net_beats_basis_pct", "basis_c", "basis_signed_c"]].round(2).to_string(index=False))
    print("  mean_abs_edge_c    : mean |Kalshi mid - replication mid|, in cents")
    print("  opt_spread_c       : median round-trip bid-ask cost of the option legs, in cents of bucket price")
    print("  exec_pos / net_pos : share of bucket-days with a positive edge after crossing spreads / after Kalshi's fee")
    print("  net_beats_basis    : share where that net edge also exceeds the replication's expected error")
    print("  basis_c            : density-weighted E|replication payoff - binary payoff|")
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    A.to_csv(config.ARTIFACT_DIR / "hedging_edges.csv", index=False)


if __name__ == "__main__":
    main()
