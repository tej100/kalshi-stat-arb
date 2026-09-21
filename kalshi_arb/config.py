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
MONEYNESS_SIGMA = 2.0        # drop quotes > this many std from that QUOTE DATE's mean moneyness
ONE_SIDED_BSM_TOL = 0.50     # $ tolerance to keep a one-sided (assumed-0) mid-price
IV_MIN, IV_MAX = 0.01, 5.0   # bisection search bounds for implied vol (decimal)
# No-arbitrage price-bound filter (transform/clean._drop_arbitrage_violations) is
# an EXACT rule (intrinsic <= mid <= upper) with no tunable threshold, so there is
# deliberately no parameter for it here.

# ---- volatility smile ------------------------------------------------------
# ONE smoothed IV smile is fitted per quote date, on OTM-combined options, and
# used identically for pricing, the GBM ATM vol, and the Breeden-Litzenberger
# density -- so every model consumes the same surface (no second smoother).
# The functional form is swappable (see transform/smiles.REGISTRY) so its effect
# on strategy results can be studied; SABR is the chosen default (smoothest,
# ~arbitrage-free -- see SMOOTHING_COMPARISON.md). Options: sabr | svi | lsq |
# cubic | poly | pchip | lowess.
SMILE_METHOD = "sabr"
SMILE_KNOTS = 6             # interior knots when SMILE_METHOD="lsq" (modest: the
                            # BL density is the 2nd derivative and amplifies
                            # over-fitting).

# ---- pricing ---------------------------------------------------------------
# (no control-variate blend: the smile is already fit to market IVs, so the BSM
# column is the pure model price. See analysis/control_variate_test.py.)

# ---- density ---------------------------------------------------------------
# Strike grid is widened beyond observed strikes (flat-vol extrapolation) so the
# raw Breeden-Litzenberger density has near-full support and is NOT renormalized
# to 1 (residual mass = probability SPX breaches all Kalshi buckets).
GRID_PAD_LOW = 2000.0        # extend grid this far below min observed strike
GRID_PAD_HIGH = 1000.0       # extend grid this far above max observed strike
GRID_POINTS = 5000
DENSITY_SMOOTH_WINDOW = 25    # grid points (~$30) for light mass-preserving
                              # smoothing of the density; removes boundary kinks
DISCOUNT_PROBABILITY = False  # compare undiscounted P(event) to price (APY covers TV)
# NOTE: there is deliberately NO staleness-tolerance parameter for the model
# PMF (transform/density.build_pmf_table requires an exact same-calendar-day
# option chain; no forward/back-fill of any kind). This was a considered and
# rejected design: the strategy's thesis is that the options market is the
# live, informed reference and Kalshi sometimes lags it, which only holds
# while options are actually trading. On a day they aren't (weekends,
# holidays, or the late-December expiry-rollover gap in transform/clean.py),
# a divergence between a frozen model view and a moving Kalshi price no
# longer shows Kalshi is wrong -- it could equally mean the model is stale
# and Kalshi is right. The problem is categorical, not a matter of degree, so
# there is no "safe" amount of staleness to bound it by. Existing positions
# still mark-to-market and settle normally on these days regardless -- only
# new-signal generation requires a live match. See ASSUMPTIONS.md (Density
# formation) before reintroducing any fill/staleness mechanism here.

# ---- strategy / backtest ---------------------------------------------------
LOT_SIZE = 8                 # contracts per trade (paper's tuned value)
MAX_LOTS_PER_BUCKET = 1      # no pyramiding: cap open exposure per bucket to N lots
                             # (bounds risk on the small capital base; keeps PV > 0)
# Marking: a book with bid <= MARK_MIN_BID and ask >= MARK_MAX_ASK is an empty
# (degenerate) market whose quotes are phantom and must not mark a position.
MARK_MIN_BID = 0.02
MARK_MAX_ASK = 0.98
# Day-level quote-feed outage detector. Kalshi's own API declares these events
# `mutually_exclusive: true`, so at most ONE bucket can be highly probable: two
# outcomes cannot both have probability >= 0.90 (that sums to 180%). Even
# allowing a generous 10c spread, two genuine offers at >= 0.90 would imply two
# true probabilities >= 0.80, summing to 160% -- impossible. So two or more
# buckets quoting an ask this high on the same day is not a market, it is an
# empty offer side being reported at the ceiling.
#
# NOT fitted: the flagged set is IDENTICAL for any level from 0.70 to 0.97
# (exactly the 2024-11-16..21 outage, nothing else in three years), so 0.90 sits
# mid-plateau. Going below ~0.60 starts catching genuine near-year-end races
# where two adjacent buckets are both live (e.g. 2022-12-23, quoted 22/66 and
# 34/70); going to 0.98 misses 2024-11-21, whose 97c asks produce the whole of
# the reported 2024 drawdown. Deliberately a LEVEL test, not a sum-of-asks test:
# summed asks conflate a merely WIDE market (2022-07-07 sums to 2.81 with every
# ask <= 25c -- lazy market-maker offers, but real) with a BROKEN one.
# See transform/kalshi_pmf.feed_outage_days.
OUTAGE_ASK_LEVEL = 0.90
OUTAGE_MIN_BUCKETS = 2
START_CASH = 200.0           # initial cash ($)
KALSHI_FEE_RATE = 0.035      # fee = ceil(rate * contracts * p * (1-p)) cents
# NOTE (verified 2026-09): this is the taker rate that applied over the
# 2022-2024 sample. Kalshi's CURRENT published schedule is 0.07 (with a separate
# 0.0175 maker rate), i.e. double, so a present-day replication would face
# higher costs -- see PAPER_CHANGES.md. Kept at 0.035 because the backtest must
# price the fees the strategy would actually have paid in-sample. With the
# per-contract hurdle below, results are not fragile to this: BL both-side
# Sharpe is 1.93/2.86/1.04 at 0.035 versus 1.46/2.41/0.76 at 0.07.
# Round trips per position, used to size the entry hurdle in signals.generate.
# NOT a tunable: Kalshi charges the taker fee on BOTH fills of a market-order
# round trip (open and close), so a signal must clear two fees to be worth
# taking. The backtest charges each fill's fee separately, so this only decides
# which signals fire, not what they cost. Positions that instead run to year-end
# settlement pay no exit fee, which makes this hurdle mildly conservative.
FEE_ROUND_TRIP_FILLS = 2
KALSHI_APY = 0.0375          # yield on cash + open positions, accrued monthly
# Risk-free hurdle for Sharpe/alpha. Set to the KALSHI APY, not an external
# T-bill rate: the capital backing this strategy sits in a Kalshi account, where
# doing nothing at all earns KALSHI_APY. That is the genuine opportunity cost, so
# it is the rate the strategy must beat. Using a lower external rate credits the
# strategy with (KALSHI_APY - rate) per year of excess it did not generate --
# at 3% that was 0.75%/yr of spurious alpha. Metrics additionally report the
# return NET of the accrued APY (`*_ex_apy`), which isolates the mispricing edge
# from the platform carry; the gap is large (2024 BL Sharpe 1.34 vs 0.18).
BENCH_RF = KALSHI_APY        # annual risk-free rate for Sharpe/alpha benchmarking
TRADING_DAYS = 252           # annualization factor (consistent across strat + bench)

YEARS = (2022, 2023, 2024)

# Actual S&P 500 cash-index year-end closes, used to settle held Kalshi buckets
# at expiry. NOTE: 2024 option data ends 2024-11-19, so the 2024 mark-to-market
# path stops early, but positions still settle at the true year-end close.
SPX_YEAR_END_CLOSE = {2022: 3839.50, 2023: 4769.83, 2024: 5881.63}
