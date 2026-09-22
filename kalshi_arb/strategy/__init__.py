"""Stage 3 - STRATEGY: consume bucket PMFs, trade, evaluate.

    signals  : model vs market bucket mispricing -> trades
    backtest : portfolio engine (fees, marking, Kalshi interest, year-end settlement)
    metrics  : performance (return/vol/Sharpe/DD, alpha/beta, model rho)
    diagnostics : post-backtest risk (collateral, settle-now stress) and per-position P&L
    hedging  : SPX-option replication of Kalshi buckets, executable edge, replication error
"""
