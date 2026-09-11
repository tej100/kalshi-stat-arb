"""Central configuration: paths and economic constants.

All tunable assumptions live here so the pipeline has a single source of truth
(the old notebooks scattered these as inline literals that drifted apart).
"""
from __future__ import annotations
from pathlib import Path

# ---- paths -----------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent                      # project root (kalshi_stat_arb/)
DATA_DIR = ROOT / "spx_w"
CHAIN_CSV = DATA_DIR / "eoy_chains.csv"     # raw SPXW EOY chain (IV in %, T in days)
KALSHI_PKL = ROOT / "kalshi_data.pkl"       # {year: {tickers, bid, ask, price}}
ARTIFACT_DIR = ROOT / "artifacts"           # generated outputs (created on demand)

# ---- cleaning thresholds ---------------------------------------------------
MAX_SPREAD_FRAC = 0.30       # drop quote if (ask-bid)/mid exceeds this
MONEYNESS_SIGMA = 2.0        # drop quotes > this many std from mean moneyness
ONE_SIDED_BSM_TOL = 0.50     # $ tolerance to keep a one-sided (assumed-0) mid-price
IV_MIN, IV_MAX = 0.01, 5.0   # bisection search bounds for implied vol (decimal)

# ---- pricing ---------------------------------------------------------------
CONTROL_VARIATE_LAMBDA = 0.20   # Price_adj = BSM + lambda*(Mid - BSM)

# ---- density ---------------------------------------------------------------
# Strike grid is widened beyond observed strikes (flat-vol extrapolation) so the
# raw Breeden-Litzenberger density has near-full support and is NOT renormalized
# to 1 (residual mass = probability SPX breaches all Kalshi buckets).
GRID_PAD_LOW = 2000.0        # extend grid this far below min observed strike
GRID_PAD_HIGH = 1000.0       # extend grid this far above max observed strike
GRID_POINTS = 5000
DENSITY_KNOTS = 6             # spline knots for the BL smile (fewer than pricing's
                             # 10 -> avoids over-fitting noise that the 2nd
                             # derivative would amplify into spurious modes)
DENSITY_SMOOTH_WINDOW = 25    # grid points (~$30) for light mass-preserving
                              # smoothing of the density; removes boundary kinks
DISCOUNT_PROBABILITY = False  # compare undiscounted P(event) to price (APY covers TV)
# Maximum calendar-day gap for which a prior day's option-derived PMF may be
# reused for a NEW trade signal. Default 0 = strict same-day matching: a new
# position is only opened on a date where the SPXW option chain and the
# Kalshi order book both have a genuine live quote that day.
#
# Design rationale: the strategy's thesis is that the options market is the
# informed, current reference and Kalshi sometimes lags it. That only holds
# while both markets are live. On a day options don't trade (weekends,
# holidays) or don't have the correct-maturity contract (the late-December
# expiry-rollover gap in transform/clean.py), the "model" side is frozen while
# Kalshi keeps moving -- so a divergence no longer tells you Kalshi is wrong,
# it could equally mean the model is stale and Kalshi is right. Trading it
# would rest on a different, unstated mechanism (a bet that Kalshi's own
# after-hours move reverts), not the options-information edge the paper
# claims. Existing positions still mark-to-market and settle normally on
# these days -- only NEW signal generation is restricted.
PMF_MAX_STALE_DAYS = 0

# ---- strategy / backtest ---------------------------------------------------
LOT_SIZE = 8                 # contracts per trade (paper's tuned value)
MAX_LOTS_PER_BUCKET = 1      # no pyramiding: cap open exposure per bucket to N lots
                             # (bounds risk on the small capital base; keeps PV > 0)
# Marking: a book with bid <= MARK_MIN_BID and ask >= MARK_MAX_ASK is an empty
# (degenerate) market whose quotes are phantom and must not mark a position.
MARK_MIN_BID = 0.02
MARK_MAX_ASK = 0.98
START_CASH = 200.0           # initial cash ($)
KALSHI_FEE_RATE = 0.035      # fee = ceil(rate * contracts * p * (1-p)) cents
KALSHI_APY = 0.0375          # yield on cash + open positions, accrued monthly
BENCH_RF = 0.03              # annual risk-free rate for Sharpe/alpha benchmarking
TRADING_DAYS = 252           # annualization factor (consistent across strat + bench)

YEARS = (2022, 2023, 2024)

# Actual S&P 500 cash-index year-end closes, used to settle held Kalshi buckets
# at expiry. NOTE: 2024 option data ends 2024-11-19, so the 2024 mark-to-market
# path stops early, but positions still settle at the true year-end close.
SPX_YEAR_END_CLOSE = {2022: 3839.50, 2023: 4769.83, 2024: 5881.63}
