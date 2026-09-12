"""Head-to-head comparison of IV smoothing methods for RND extraction.

Fits every smoother in `smoothers.py` on the same OTM-combined smile for a set
of sample dates, scores them on discriminating metrics, and plots the smiles and
their resulting risk-neutral densities so the choice can be made by eye + number.

Run from repo root:  python analysis/compare_smoothers.py
Outputs: analysis/figures/smoother_comparison.png  and a printed metrics table.

Reusable for a separate volatility-surface methods paper: `smoothers.py` is
self-contained; only this driver reaches into kalshi_arb for the cleaned chain.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kalshi_arb import pipeline
import smoothers as S

# sample dates spanning regimes: sparse far-OTM gap, noisy spike, calm, steep
# skew, high-vol, later year. Missing dates are skipped.
SAMPLE_DATES = ["2022-02-25", "2022-06-16", "2022-10-04",
                "2023-06-23", "2023-11-15", "2024-05-29"]
COLORS = ["#1f4e79", "#e67e22", "#27ae60", "#c0392b",
          "#8e44ad", "#16a085", "#7f8c8d"]


def otm_smile(day):
    F = day["forward"].iloc[0]
    sel = ((day.option_type == "p") & (day.strike < F)) | \
          ((day.option_type == "c") & (day.strike >= F))
    d = day[sel].dropna(subset=["IV"]).sort_values("strike")
    K, idx = np.unique(d.strike.values, return_index=True)
    return (K, d.IV.values[idx], F, float(day.underlying.iloc[0]),
            float(day["T"].iloc[0]), float(day.r.iloc[0]))


def score(sm, K, iv, grid, S_, T, r):
    """Metrics for one fitted smoother on one day."""
    if not sm.ok:
        return dict(ok=False, fit_rmse=np.nan, overshoot=np.nan,
                    arb_neg=np.nan, rough=np.nan, modes=np.nan)
    iv_hat_data = sm(K)
    iv_hat_grid = sm(grid)
    raw, rep, signed, g = S.rnd_from_iv(grid, iv_hat_grid, S_, T, r)
    peak = raw.max() if raw.max() > 0 else 1.0
    modes = int(((np.diff(np.sign(np.diff(raw))) < 0) & (raw[1:-1] > 0.05 * peak)).sum())
    return dict(
        ok=True,
        fit_rmse=float(np.sqrt(np.mean((iv_hat_data - iv) ** 2))),
        overshoot=float(iv_hat_grid.max() / iv.max()),        # >1 = above observed
        arb_neg=float((signed < 0).mean()),                   # frac negative density (pre-clip)
        rough=float(np.sum(np.abs(np.diff(raw))) / peak),
        modes=modes,
    )


def main():
    chain = pipeline.build_chain(verbose=False)
    dates = [d for d in SAMPLE_DATES
             if pd.Timestamp(d) in set(chain.quote.unique())]
    smoothers = [cls() for cls in S.ALL_SMOOTHERS]

    fig, ax = plt.subplots(len(dates), 2, figsize=(14, 3.5 * len(dates)))
    if len(dates) == 1:
        ax = ax.reshape(1, 2)
    records = []

    for row, ds in enumerate(dates):
        day = chain[chain.quote == pd.Timestamp(ds)]
        K, iv, F, S_, T, r = otm_smile(day)
        grid = np.linspace(max(K.min() - 500, 1), K.max() + 500, 4000)

        # PCHIP + LOWESS produce such noisy densities they crush the y-axis;
        # they stay in the smile panel + metrics but are dropped from density view.
        DENSITY_SKIP = {"PCHIP (monotone)", "LOWESS"}
        ax[row, 0].scatter(K, iv, s=12, c="#95a5a6", alpha=0.6,
                           label="raw OTM IV", zorder=1)
        dens_peak = 0.0
        for sm, col in zip(smoothers, COLORS):
            sm.fit(K, iv, F, T)
            m = score(sm, K, iv, grid, S_, T, r)
            records.append(dict(date=ds, method=sm.name, **m))
            if not sm.ok:
                continue
            ax[row, 0].plot(grid, sm(grid), c=col, lw=1.4, label=sm.name)
            if sm.name in DENSITY_SKIP:
                continue
            raw, rep, signed, g = S.rnd_from_iv(grid, sm(grid), S_, T, r)
            ax[row, 1].plot(g, raw, c=col, lw=1.4, label=sm.name)
            dens_peak = max(dens_peak, np.quantile(raw, 0.999))
        for a in ax[row]:
            a.axvline(F, c="k", ls="--", lw=0.7, alpha=0.6)
        ax[row, 0].set_ylim(max(0, iv.min() - 0.05), iv.max() + 0.08)
        if dens_peak > 0:
            ax[row, 1].set_ylim(-0.02 * dens_peak, 1.35 * dens_peak)
        ax[row, 0].set_title(f"{ds}  —  OTM IV smile", fontsize=10)
        ax[row, 1].set_title(f"{ds}  —  raw density (smooth methods)", fontsize=10)
        ax[row, 0].set_ylabel("IV"); ax[row, 1].set_ylabel("density")
        if row == 0:
            ax[row, 0].legend(fontsize=7, frameon=False, ncol=2)
            ax[row, 1].legend(fontsize=7, frameon=False)
    ax[-1, 0].set_xlabel("strike"); ax[-1, 1].set_xlabel("SPX level")
    plt.tight_layout()
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "smoother_comparison.png"), dpi=125,
                bbox_inches="tight")
    print("saved analysis/figures/smoother_comparison.png")

    # aggregate metrics table (mean over dates)
    df = pd.DataFrame(records)
    agg = df.groupby("method").agg(
        fits=("ok", "sum"),
        fit_rmse=("fit_rmse", "mean"),
        overshoot=("overshoot", "mean"),
        arb_neg=("arb_neg", "mean"),
        roughness=("rough", "mean"),
        modes=("modes", "mean"),
    ).reindex([c().name for c in S.ALL_SMOOTHERS])
    pd.set_option("display.width", 200)
    print(f"\nMetrics averaged over {len(dates)} dates "
          f"(lower is better except 'fits'; overshoot~1 ideal):")
    print(agg.round(3).to_string())
    print("\n  fit_rmse  = RMS(fitted IV - market IV) at data strikes")
    print("  overshoot = max fitted IV / max observed IV  (>1.05 => invents high vol)")
    print("  arb_neg   = fraction of grid with NEGATIVE raw density (arbitrage before repair)")
    print("  roughness = total-variation / peak of raw density  (smooth unimodal ~2)")
    print("  modes     = # local maxima of raw density above 5% of peak  (1 = clean)")


if __name__ == "__main__":
    main()
