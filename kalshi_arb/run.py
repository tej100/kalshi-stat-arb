"""Convenience runner: build chain -> PMFs -> signals -> backtest -> metrics.

    from kalshi_arb import run
    table, results = run.full()   # results[(year, model, side)] -> backtest df
"""
from __future__ import annotations
import pandas as pd
from . import io, pipeline, density, signals, backtest, metrics, config


def full(chain=None, kalshi=None, methods=("bl", "gbm"),
         sides=("both", "buy", "sell"), verbose=True):
    if chain is None:
        chain = pipeline.build_chain(verbose=verbose)
    if kalshi is None:
        kalshi = io.load_kalshi()

    pmf = {mth: density.build_pmf_table(chain, kalshi, method=mth) for mth in methods}

    records, results = [], {}
    for mth in methods:
        for year in config.YEARS:
            for side in sides:
                tr = signals.generate(pmf[mth], kalshi, year, side=side)
                m = backtest.run(tr, pmf[mth], kalshi, year)
                results[(year, mth, side)] = m
                rec = dict(year=year, model=mth.upper(), side=side,
                           **metrics.summarize(m, chain, year, n_trades=len(tr)))
                records.append(rec)
    table = metrics.format_table(records)
    return table, results, pmf
