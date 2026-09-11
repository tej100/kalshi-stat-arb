"""kalshi_arb - options-implied probabilities vs Kalshi event-contract mispricing.

Three-stage ETL pipeline (see the sub-packages):

    extract/    pull raw data from each source (refinitiv option chain, kalshi book)
    transform/  raw -> discrete bucket PMFs on the shared grid (both sources)
    strategy/   PMFs -> signals -> backtest -> metrics (+ hedging)

Orchestration:
    pipeline.build_chain()  extract + transform the option chain
    run.full()              end-to-end sweep (BL/GBM x years x sides)
"""
from . import config, pipeline, run
from . import extract, transform, strategy

__all__ = ["config", "pipeline", "run", "extract", "transform", "strategy"]
