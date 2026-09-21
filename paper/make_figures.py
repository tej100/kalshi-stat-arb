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
from kalshi_arb import pipeline, run, config
from kalshi_arb.extract import kalshi as kalshi_src
from kalshi_arb.transform import density, smoothing, kalshi_pmf, buckets as bkt
from kalshi_arb.strategy import metrics, hedging

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)
NAVY, GREEN, GREY, RED, ORANGE = "#1f4e79", "#27ae60", "#7f8c8d", "#c0392b", "#e67e22"


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=140, bbox_inches="tight"); plt.close()
    print("  ", name)


def main():
    chain = pipeline.build_chain(save=False)
    kalshi = kalshi_src.load_kalshi()

    # 1. moneyness distribution
    plt.figure(figsize=(5, 3.4))
    plt.hist(chain["moneyness"], bins=80, color=NAVY, alpha=0.85)
    plt.axvline(1.0, color=RED, ls="--"); plt.xlabel("moneyness"); plt.ylabel("count")
    plt.title("Moneyness distribution"); save("moneyness_distribution.png")

    date = pd.Timestamp("2023-06-23")
    day = chain[chain["quote"] == date]
    F = day["forward"].iloc[0]

    # 2. the smile the pipeline actually fits (SABR on OUT-OF-THE-MONEY options)
    sm = smoothing.fit_daily_smile(day)
    otm = ((day["option_type"] == "p") & (day["strike"] < F)) | ((day["option_type"] == "c") & (day["strike"] >= F))
    used, excl = day[otm].dropna(subset=["IV"]), day[~otm].dropna(subset=["IV"])
    grid = np.linspace(sm.k_min, sm.k_max, 400)
    plt.figure(figsize=(5.8, 3.8))
    plt.scatter(excl["strike"], excl["IV"], s=9, facecolors="none", edgecolors=RED, alpha=0.6,
                label="in-the-money quotes (not used)")
    plt.scatter(used["strike"], used["IV"], s=10, color=GREY, label="out-of-the-money quotes (fitted)")
    plt.plot(grid, smoothing.eval_smile(sm, grid), color=NAVY, lw=1.8, label=f"{config.SMILE_METHOD.upper()} smile")
    plt.axvline(F, color=RED, ls="--", lw=0.8)
    plt.xlabel("strike"); plt.ylabel("implied vol"); plt.legend(frameon=False, fontsize=8)
    plt.title(f"IV smile fit — {date.date()}"); save("smoothing_example.png")

    # 3. model vs market prices, with the residual (overlaid price curves hide any error)
    dd = day.dropna(subset=["BSM", "Mid"]).sort_values("strike")
    res = dd["BSM"] - dd["Mid"]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.5))
    c = dd[dd["option_type"] == "c"]
    ax[0].plot(c["strike"], c["Mid"], color=GREY, lw=1.6, label="market mid")
    ax[0].plot(c["strike"], c["BSM"], color=NAVY, lw=1.4, ls="--", label="model (BSM on fitted smile)")
    ax[0].set_xlabel("strike"); ax[0].set_ylabel("call price"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(f"Model vs market — {date.date()}")
    for t, col, lab in (("c", NAVY, "calls"), ("p", ORANGE, "puts")):
        r = res[dd["option_type"] == t]
        ax[1].scatter(dd.loc[r.index, "strike"], r, s=7, color=col, alpha=0.7, label=lab)
    ax[1].axhline(0, color=GREY, lw=0.6)
    ax[1].set_xlabel("strike"); ax[1].set_ylabel("model − market ($)"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title(f"Pricing error, this day: MAE \${res.abs().mean():.2f}, RMSE \${np.sqrt((res**2).mean()):.2f}")
    save("model_vs_market.png")

    # 4. density: BL RND + bucket comparison (Kalshi = mid of a valid book)
    g, dens = density.bl_density(day)
    buckets = list(kalshi[2023]["price"].columns)
    bl = density.bl_pmf(day, buckets); gb = density.gbm_pmf(day, buckets)
    kp = kalshi_pmf.market_pmf(kalshi, 2023).loc[date]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(g, dens, color=NAVY); ax[0].fill_between(g, dens, alpha=0.15, color=NAVY)
    ax[0].axvline(F, color=RED, ls="--", label=f"forward {F:.0f}")
    ax[0].set_xlim(3000, 5800); ax[0].set_xlabel("SPX level"); ax[0].set_ylabel("density")
    ax[0].legend(frameon=False); ax[0].set_title(f"Breeden–Litzenberger RND — {date.date()}")
    x = np.arange(len(buckets)); w = 0.27
    ax[1].bar(x - w, kp.values, w, label="Kalshi (mid; blank = no valid quote)", color=GREY)
    ax[1].bar(x, [bl[b] for b in buckets], w, label="BL", color=NAVY)
    ax[1].bar(x + w, [gb[b] for b in buckets], w, label="GBM", color=ORANGE)
    mids = [int(np.mean(bkt.bucket_bounds(b))) for b in buckets]
    ax[1].set_xticks(x); ax[1].set_xticklabels(mids, rotation=60, fontsize=7)
    ax[1].set_xlabel("bucket midpoint"); ax[1].set_ylabel("probability")
    ax[1].legend(frameon=False, fontsize=8); ax[1].set_title("Model vs Kalshi bucket probabilities")
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
            td = metrics.trading_days(chain, y, m.index)              # days the option chain exists
            ax[i].plot(m.index, m["portfolio_value"], color=NAVY, label="strategy (MtM)")
            if other == "spx":
                s = (chain[chain["quote"].dt.year == y].drop_duplicates("quote")
                     .set_index("quote")["spot"].sort_index())
                s = config.START_CASH * s / s.iloc[0]                 # option-chain days only: no flat fill
                ax[i].plot(s.index, s, color=ocolor, alpha=0.8, label=olabel)
            else:
                # the model exists only on option-chain days; plotting it on those days avoids
                # a weekday/weekend sawtooth from mixing two valuations, and ends where the feed ends
                ax[i].plot(td, m.loc[td, other], color=ocolor, ls="--", alpha=0.9, label=olabel)
            ax[i].axhline(config.START_CASH, color=GREY, lw=0.4)
            live = kalshi[y]["bid"].notna().any(axis=1)
            first = live[live].index.min()
            if first > pd.Timestamp(f"{y}-02-01"):
                ax[i].axvline(first, color=RED, ls=":", lw=0.9)
                ax[i].text(first, ax[i].get_ylim()[1], " Kalshi market opens", color=RED, fontsize=7, va="top")
            if td.max() < pd.Timestamp(f"{y}-12-01"):
                ax[i].axvline(td.max(), color=GREY, ls=":", lw=0.9)
                ax[i].text(td.max(), ax[i].get_ylim()[0], " option data ends", color=GREY, fontsize=7, va="bottom")
            ax[i].set_title(f"{y}"); ax[i].tick_params(axis="x", rotation=45, labelsize=7)
            if i == 0: ax[i].legend(frameon=False, fontsize=8, loc="upper left"); ax[i].set_ylabel("portfolio ($)")
        save(fname)

    # 7. hedging cross-market edge distribution
    allh = pd.concat([hedging.analyze_year(chain, kalshi, y, verbose=False).assign(year=y)
                      for y in config.YEARS])
    plt.figure(figsize=(5.8, 3.5))
    plt.hist(allh["edge"] * 100, bins=60, range=(-15, 15), color=NAVY, alpha=0.75,
             label="mid-to-mid gap")
    plt.hist(allh["net_edge"] * 100, bins=60, range=(-15, 15), color=RED, alpha=0.6,
             label="capturable after spreads and fee")
    plt.axvline(0, color=GREY, ls="--")
    plt.xlabel("Kalshi vs SPX-option replication (cents; x-axis clipped to +/-15)")
    plt.ylabel("count"); plt.legend(fontsize=8)
    plt.title("Cross-market gap vs what execution allows"); save("hedging_edge.png")

    print("done.")


if __name__ == "__main__":
    main()
