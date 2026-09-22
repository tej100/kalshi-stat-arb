"""Appendix robustness table: every modelling choice varied one at a time.

Each row changes ONE assumption against the headline configuration and reports
the Breeden-Litzenberger both-side Sharpe per year, the pooled three-year Sharpe
with its block-bootstrap interval, and (where the option chain changes) the
pricing error. Nothing here was used to choose the headline configuration; the
point is to show which choices the result does and does not depend on.

Run from repo root:  python analysis/sensitivity.py
Writes artifacts/sensitivity.csv (untracked).
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

K = ksrc.load_kalshi()
DEFAULTS = {n: getattr(config, n) for n in
            ("MAX_SPREAD_FRAC", "MONEYNESS_SIGMA", "SABR_BETA", "SMILE_WINGS", "SMILE_METHOD",
             "LOT_SIZE", "START_CASH", "KALSHI_FEE_RATE")}


def run_bl(chain, lag=0, exit_rule="signal", stale="book"):
    pmf = density.build_pmf_table(chain, K, method="bl")
    if lag:
        pmf = density.lag_pmf_table(pmf, lag)
    out = {}
    for y in config.YEARS:
        tr = signals.generate(pmf, K, y, "both", exit_rule=exit_rule)
        out[y] = backtest.run(tr, pmf, K, y, stale_mark=stale)
    return out


def row(name, group, chain, res):
    per = [metrics.summarize(res[y], chain, y, kalshi=K)["sharpe"] for y in config.YEARS]
    p = metrics.pooled_sharpe_ci(res, chain, reps=10000, kalshi=K)
    mae = float((chain.BSM - chain.Mid).abs().mean())
    fills = sum(res[y].attrs["n_executed"] for y in config.YEARS)
    return dict(group=group, variant=name, s2022=per[0], s2023=per[1], s2024=per[2],
                pooled=p["sharpe"], lo=p["lo"], hi=p["hi"], mae=mae, fills=fills)


def main():
    rows = []
    base_chain = pipeline.build_chain(verbose=False)
    rows.append(row("as in the paper", "baseline", base_chain, run_bl(base_chain)))

    chain_variants = [
        ("cleaning", "spread filter 30% of mid", dict(MAX_SPREAD_FRAC=0.30)),
        ("cleaning", "moneyness cut 3 sigma", dict(MONEYNESS_SIGMA=3.0)),
        ("cleaning", "no moneyness cut", dict(MONEYNESS_SIGMA=1e9)),
        ("smile", "SABR beta = 0", dict(SABR_BETA=0.0)),
        ("smile", "SABR beta = 1", dict(SABR_BETA=1.0)),
        ("smile", "SABR wings (no flat extrapolation)", dict(SMILE_WINGS="model")),
        ("smile", "SVI instead of SABR", dict(SMILE_METHOD="svi")),
    ]
    for group, name, over in chain_variants:
        for n, v in DEFAULTS.items(): setattr(config, n, v)
        for n, v in over.items(): setattr(config, n, v)
        ch = pipeline.build_chain(verbose=False)
        rows.append(row(name, group, ch, run_bl(ch)))
        print(f"  done: {name}", flush=True)
    for n, v in DEFAULTS.items(): setattr(config, n, v)

    for lot in (1, 4, 16):
        config.LOT_SIZE, config.START_CASH = lot, DEFAULTS["START_CASH"] * lot / DEFAULTS["LOT_SIZE"]
        rows.append(row(f"lot {lot} contracts (capital ${config.START_CASH:.0f})", "sizing",
                        base_chain, run_bl(base_chain)))
    for n, v in DEFAULTS.items(): setattr(config, n, v)

    config.KALSHI_FEE_RATE = 0.07
    rows.append(row("current Kalshi fee (0.07)", "costs", base_chain, run_bl(base_chain)))
    config.KALSHI_FEE_RATE = DEFAULTS["KALSHI_FEE_RATE"]

    rows.append(row("convergence exit", "strategy", base_chain, run_bl(base_chain, exit_rule="convergence")))
    rows.append(row("model 1 day old", "timing", base_chain, run_bl(base_chain, lag=1)))
    rows.append(row("model 2 days old", "timing", base_chain, run_bl(base_chain, lag=2)))
    rows.append(row("stale marks to model", "marking", base_chain, run_bl(base_chain, stale="model")))

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.round(2).to_string(index=False))
    config.ARTIFACT_DIR.mkdir(exist_ok=True)
    R.to_csv(config.ARTIFACT_DIR / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
