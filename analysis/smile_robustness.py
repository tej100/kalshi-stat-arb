"""Is the strategy's result sensitive to the choice of smile model?

Runs the full backtest with the BL density built from four smile treatments and
compares them on the same footing:
  sabr       production default
  svi        Gatheral raw SVI
  average    mean of the SABR and SVI bucket probabilities
  agreement  SABR signals kept only where SVI independently signals the same way

None adds a tuned parameter. Run from repo root:  python analysis/smile_robustness.py
Writes artifacts/smile_robustness.csv (untracked); write-up in SMOOTHING_COMPARISON.md.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from kalshi_arb import pipeline, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import density
from kalshi_arb.strategy import signals, backtest, metrics


def _key(df):
    return set(zip(df["day"], df["bucket"], np.sign(df["qty"]))) if len(df) else set()


def main(reps=2000):
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()
    default = config.SMILE_METHOD
    try:
        config.SMILE_METHOD = "sabr"
        sabr = density.build_pmf_table(chain, k, method="bl")
        config.SMILE_METHOD = "svi"
        svi = density.build_pmf_table(chain, k, method="bl")
    finally:
        config.SMILE_METHOD = default
    avg = {y: (sabr[y] + svi[y]) / 2 for y in config.YEARS}
    gap = np.nanmean([np.nanmean(np.abs(sabr[y].values - svi[y].values)) for y in config.YEARS])
    print(f"mean |SABR - SVI| bucket probability: {gap * 100:.2f}c (the two densities genuinely differ)\n")

    rows = []
    for name in ("sabr", "svi", "average", "agreement"):
        for y in config.YEARS:
            if name == "sabr":
                tr, pm = signals.generate(sabr, k, y, "both"), sabr
            elif name == "svi":
                tr, pm = signals.generate(svi, k, y, "both"), svi
            elif name == "average":
                tr, pm = signals.generate(avg, k, y, "both"), avg
            else:
                a = signals.generate(sabr, k, y, "both")
                b = _key(signals.generate(svi, k, y, "both"))
                keep = np.array([(d, bk, np.sign(q)) in b for d, bk, q in
                                 zip(a["day"], a["bucket"], a["qty"])], dtype=bool) if len(a) else []
                tr, pm = (a[keep] if len(a) else a), sabr
            m = backtest.run(tr, pm, k, y)
            s = metrics.summarize(m, chain, y, n_trades=m.attrs["n_executed"], kalshi=k)
            ci = metrics.sharpe_bootstrap_ci(m, chain, y, reps=reps, kalshi=k)
            rows.append(dict(variant=name, year=y, sharpe=s["sharpe"], ci_lo=ci["lo"], ci_hi=ci["hi"],
                             ann_return=s["ann_return"], ann_return_ex_apy=s["ann_return_ex_apy"],
                             max_dd=s["max_dd"], signals=len(tr), fills=m.attrs["n_executed"]))
    R = pd.DataFrame(rows)
    show = R.assign(ann_return=(R.ann_return * 100).round(2), ann_return_ex_apy=(R.ann_return_ex_apy * 100).round(2),
                    max_dd=(R.max_dd * 100).round(2)).round(2)
    print(show.to_string(index=False))
    spread = R.pivot(index="year", columns="variant", values="sharpe")
    width = (R.ci_hi - R.ci_lo).mean()
    print(f"\nlargest Sharpe gap between any two variants, by year: "
          f"{(spread.max(axis=1) - spread.min(axis=1)).round(2).to_dict()}")
    print(f"typical 95% CI width: {width:.1f} Sharpe units -> variants are statistically indistinguishable "
          f"if the gaps above are well inside it.")
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    R.to_csv(config.ARTIFACT_DIR / "smile_robustness.csv", index=False)


if __name__ == "__main__":
    main()
