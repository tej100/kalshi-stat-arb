"""Convenience runner: extract -> transform (PMFs) -> strategy sweep.

    from kalshi_arb import run
    table, results, pmf = run.full()   # results[(year, model, side)] -> backtest df
"""
from __future__ import annotations
from . import config, pipeline
from .extract import kalshi as kalshi_src
from .transform import density
from .strategy import signals, backtest, metrics


def full(chain=None, kalshi=None, methods=("bl", "gbm"),
         sides=("both", "buy", "sell"), verbose=True):
    if chain is None:
        chain = pipeline.build_chain(verbose=verbose)
    if kalshi is None:
        kalshi = kalshi_src.load_kalshi()

    # transform: each density method -> {year: date x bucket model PMF}
    pmf = {mth: density.build_pmf_table(chain, kalshi, method=mth) for mth in methods}

    # strategy: signals -> backtest -> metrics, per model/year/side
    records, results = [], {}
    for mth in methods:
        for year in config.YEARS:
            for side in sides:
                tr = signals.generate(pmf[mth], kalshi, year, side=side)
                m = backtest.run(tr, pmf[mth], kalshi, year)
                results[(year, mth, side)] = m
                # n_trades = actual executed fills (post pyramiding-cap), NOT the
                # raw signal-log length in `tr`, which re-fires daily while a
                # mispricing persists -- see backtest.run's docstring.
                rec = dict(year=year, model=mth.upper(), side=side,
                           **metrics.summarize(m, chain, year, n_trades=m.attrs["n_executed"],
                                               kalshi=kalshi))
                rec["n_signals"] = m.attrs["n_signals"]
                records.append(rec)
    table = metrics.format_table(records)
    return table, results, pmf
