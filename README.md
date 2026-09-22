# Event Contract Mispricing via Options-Implied Probabilities

Kalshi lists binary contracts on where the S&P 500 will close at year-end, in thirteen
$200 buckets. The SPX options market prices the same event far more liquidly. This
project extracts the risk-neutral distribution of the year-end SPX close from SPXW
options each day, prices every Kalshi bucket from it, and trades the gaps on Kalshi,
2022–2024. The paper is [`paper/main.tex`](paper/main.tex).

## Findings

- **The two markets agree.** Replicating each bucket from out-of-the-money option
  spreads, the Kalshi mid and the replication mid differ by 2.9 cents on average with no
  persistent premium on either side. The replication itself costs about 26 cents round
  trip in option spreads, so almost none of the gap is capturable as an arbitrage
  ([`HEDGING.md`](HEDGING.md)).
- **A skew-aware density trades the gaps profitably, but the edge is small and front-loaded.**
  Both-side book, one lot per bucket, $200 capital, returns in excess of the
  option-implied risk-free rate:

  | | 2022 (from 7 Jul) | 2023 | 2024 (to 19 Nov) |
  |---|---|---|---|
  | Breeden–Litzenberger, Sharpe [95% CI] | 1.97 [0.19, 3.89] | 1.90 [−0.31, 3.72] | 0.68 [−0.89, 2.13] |
  | same, model lagged one day (lookahead-free) | 0.86 | −0.09 | 0.50 |
  | GBM at the ATM vol (skew-free benchmark) | 2.38 | 1.24 | −1.98 |

  The same-close figures are an upper bound: most of the edge is realized within a day
  of the signal ([`RISK_PROFILE.md`](RISK_PROFILE.md) §5). Beta to SPX is within ±0.03.
- **Kalshi is the laggard.** It corrects toward the options market, at about 4% of a gap
  a day (half-life about a week, after removing bid-ask bounce); the options market does
  not move toward Kalshi ([`DUAL_TRADING_ANALYSIS.md`](DUAL_TRADING_ANALYSIS.md)).
- **Skew matters.** The lognormal benchmark systematically overprices buckets 2–10%
  below spot by about 4–5 cents. In 2024, when SPX rallied through the board, that bias
  produced its whole loss ([`GBM_BIAS.md`](GBM_BIAS.md)).

## Method

```
extract    SPXW end-of-year chain (Refinitiv)   Kalshi daily order books
transform  clean -> SABR smile on OTM quotes -> BSM pricing -> Breeden-Litzenberger density
           -> bucket probabilities                          Kalshi mid / valid-book rules
strategy   signal: model p vs ask/bid after two fees -> backtest (liquidation-side marks,
           collateral carry, year-end settlement) -> metrics (trading-day sampling,
           block-bootstrap Sharpe CI, Newey-West alpha/beta)
```

Every parameter and the reason for its value is in [`ASSUMPTIONS.md`](ASSUMPTIONS.md);
the stage-by-stage flow is in [`PIPELINE.md`](PIPELINE.md); the smile-model comparison
is in [`SMOOTHING_COMPARISON.md`](SMOOTHING_COMPARISON.md).

## Running it

```bash
pip install -r requirements.txt
python -c "from kalshi_arb import run; t, res, pmf = run.full(); print(t)"
python analysis/execution_lag.py     # same-close vs lagged execution
python analysis/risk_profile.py      # collateral, tail stress, exit rule, stale marks
python analysis/cointegration.py     # lead-lag and P&L attribution
python analysis/hedging_edge.py      # cross-market replication
python paper/make_figures.py         # all paper figures
```

`notebooks/01_pipeline.ipynb` walks through the pipeline with plots.

## Data

The code expects two files that are **not in this repository**:

- `spx_w/eoy_chains.csv`: SPXW option chains, obtained from Refinitiv through
  Hanlon Labs at Stevens Institute of Technology. It is licensed data and cannot be
  redistributed.
- `kalshi_data.pkl`: Kalshi daily candles for the 2022–2024 year-end SPX markets,
  pulled with `kalshi_arb.extract.kalshi.build_kalshi_dataset` (needs a Kalshi API key in
  a local `api_key.py`). Kalshi no longer serves these settled markets, so the file
  cannot be re-created from the API.

Without them the results cannot be reproduced end to end, but every figure and table
in the documents above states the script that produced it.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The license does not cover the data.
