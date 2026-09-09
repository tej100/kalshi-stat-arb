"""kalshi_arb: options-implied probabilities vs Kalshi event-contract mispricing.

Pipeline (see paper "Event Contract Mispricing via Options-Implied Probabilities"):
    io -> cleaning -> vol -> pricing -> density -> signals -> backtest -> metrics
    hedging  (SPX iron-condor replication of Kalshi buckets)
"""
from . import config

__all__ = ["config"]
