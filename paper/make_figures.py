"""Regenerate every figure used in the paper from the kalshi_arb pipeline.

Run from repo root:  python paper/make_figures.py
Outputs -> paper/figures/*.png  (referenced by paper/main.tex)
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kalshi_arb import pipeline, io, vol, density, run, metrics, hedging, config

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)
NAVY, GREEN, GREY, RED, ORANGE = "#1f4e79", "#27ae60", "#7f8c8d", "#c0392b", "#e67e22"


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=140, bbox_inches="tight"); plt.close()
    print("  ", name)


def main():
    chain = pipeline.build_chain(save=False)
    kalshi = io.load_kalshi()

    # 1. moneyness distribution
    plt.figure(figsize=(5, 3.4))
    plt.hist(chain["moneyness"], bins=80, color=NAVY, alpha=0.85)
    plt.axvline(1.0, color=RED, ls="--"); plt.xlabel("moneyness"); plt.ylabel("count")
    plt.title("Moneyness distribution"); save("moneyness_distribution.png")

    # 2. smoothing example (raw IV vs LSQ spline, calls, one date)
    date = pd.Timestamp("2023-06-23")
    d = chain[(chain["quote"] == date) & (chain["option_type"] == "c")].sort_values("strike")
    plt.figure(figsize=(5.5, 3.6))
    plt.scatter(d["strike"], d["IV"], s=10, color=GREY, label="market IV")
    plt.plot(d["strike"], d["LSQ_Vol"], color=NAVY, lw=1.8, label="LSQ spline")
    plt.axvline(d["forward"].iloc[0], color=RED, ls="--", label="forward")
    plt.xlabel("strike"); plt.ylabel("implied vol"); plt.legend(frameon=False)
    plt.title(f"IV smoothing — calls, {date.date()}"); save("smoothing_example.png")

    # 3. model vs market prices (calls, one date)
    plt.figure(figsize=(5.5, 3.6))
    plt.plot(d["strike"], d["Mid"], color=GREY, lw=1.6, label="market mid")
    plt.plot(d["strike"], d["BSM"], color=NAVY, lw=1.4, ls="--", label="model (adj. BSM)")
    plt.xlabel("strike"); plt.ylabel("call price"); plt.legend(frameon=False)
    plt.title(f"Model vs market — {date.date()}"); save("model_vs_market.png")

    # 4. density: BL RND + bucket comparison
    day = chain[chain["quote"] == date]
    g, dens = density.bl_density(day); F = day["forward"].iloc[0]
    buckets = list(kalshi[2023]["price"].columns)
    bl = density.bl_pmf(day, buckets); gb = density.gbm_pmf(day, buckets)
    kp = kalshi[2023]["price"].reindex([date]).iloc[0] / 100
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(g, dens, color=NAVY); ax[0].fill_between(g, dens, alpha=0.15, color=NAVY)
    ax[0].axvline(F, color=RED, ls="--", label=f"forward {F:.0f}")
    ax[0].set_xlim(3000, 5800); ax[0].set_xlabel("SPX level"); ax[0].set_ylabel("density")
    ax[0].legend(frameon=False); ax[0].set_title(f"Breeden–Litzenberger RND — {date.date()}")
    x = np.arange(len(buckets)); w = 0.27
    ax[1].bar(x - w, kp.values, w, label="Kalshi", color=GREY)
    ax[1].bar(x, [bl[b] for b in buckets], w, label="BL", color=NAVY)
    ax[1].bar(x + w, [gb[b] for b in buckets], w, label="GBM", color=ORANGE)
    mids = [int(np.mean(io.bucket_bounds(b))) for b in buckets]
    ax[1].set_xticks(x); ax[1].set_xticklabels(mids, rotation=60, fontsize=7)
    ax[1].set_xlabel("bucket midpoint"); ax[1].set_ylabel("probability")
    ax[1].legend(frameon=False); ax[1].set_title("Model vs Kalshi bucket probabilities")
    save("density_example.png")

    # 5+6. strategy vs benchmark and vs model (BL both, all years)
    table, results, _ = run.full(chain=chain, kalshi=kalshi, verbose=False)
    table.to_csv(os.path.join(FIG, "..", "results_table.csv"), index=False)
    for tag, other, olabel, ocolor, fname in [
        ("benchmark", "spx", "SPX (norm.)", GREY, "strategy_vs_benchmark.png"),
        ("model", "model_value", "mark-to-model", GREEN, "strategy_vs_model.png")]:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for i, y in enumerate(config.YEARS):
            m = results[(y, "bl", "both")]
            ax[i].plot(m.index, m["portfolio_value"], color=NAVY, label="strategy (MtM)")
            if other == "spx":
                s = metrics.spx_benchmark(chain, y, m.index); s = config.START_CASH * s / s.iloc[0]
                ax[i].plot(s.index, s, color=ocolor, alpha=0.8, label=olabel)
            else:
                ax[i].plot(m.index, m[other], color=ocolor, ls="--", alpha=0.85, label=olabel)
            ax[i].axhline(config.START_CASH, color=GREY, lw=0.4)
            ax[i].set_title(f"{y}"); ax[i].tick_params(axis="x", rotation=45, labelsize=7)
            if i == 0: ax[i].legend(frameon=False, fontsize=8); ax[i].set_ylabel("portfolio ($)")
        save(fname)

    # 7. hedging cross-market edge distribution
    allh = pd.concat([hedging.analyze_year(chain, kalshi, y, verbose=False).assign(year=y)
                      for y in config.YEARS])
    plt.figure(figsize=(5.5, 3.4))
    plt.hist(allh["edge"], bins=60, color=NAVY, alpha=0.85)
    plt.axvline(0, color=RED, ls="--")
    plt.axvline(0.05, color=GREY, ls=":"); plt.axvline(-0.05, color=GREY, ls=":")
    plt.xlabel("Kalshi price − options replication ($)"); plt.ylabel("count")
    plt.title("Cross-market dislocation (hedging)"); save("hedging_edge.png")

    print("done.")


if __name__ == "__main__":
    main()
