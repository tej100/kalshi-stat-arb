# Event Contract Mispricing via Options-Implied Probabilities

Kalshi lists binary contracts on where the S&P 500 will close at year-end, in thirteen
$200 buckets. The SPX options market prices the same event far more liquidly. This
project asks whether the two markets price it consistently, or whether Kalshi can be
exploited using the information in option prices. It extracts the risk-neutral
distribution of the year-end SPX close from SPXW options each day (Breeden–Litzenberger
on a SABR smile), prices every Kalshi bucket from it, and tests three levels of
efficiency, 2022–2024. The paper is [`paper/main.tex`](paper/main.tex).

## Findings

1. **No riskless arbitrage.** Replicating each bucket from out-of-the-money option
   spreads gives a no-arbitrage band; Kalshi stays inside it on 97.5% of bucket-days.
   Replication costs about 26¢ round trip against Kalshi's 2¢ ([`HEDGING.md`](HEDGING.md)).
2. **Kalshi lags the options market.** When the two disagree, Kalshi corrects, at
   about 4% of a gap per day (t = −2.7 clustered by date; half-life about a week, after
   removing bid-ask bounce); the options market does not move toward Kalshi
   ([`DUAL_TRADING_ANALYSIS.md`](DUAL_TRADING_ANALYSIS.md)).
3. **The lag is exploitable at the close.** Trading Kalshi against the density, after
   fees and the cost of posted collateral, one lot per bucket on $200, returns in excess
   of the option-implied risk-free rate:

   | Sharpe ratio | 2022 (from 7 Jul) | 2023 | 2024 (to 19 Nov) | **pooled [95% CI]** |
   |---|---|---|---|---|
   | Breeden–Litzenberger, same close | 1.98 | 1.90 | 0.55 | **1.44 [0.36, 2.49]** |
   | same, model one day old | 0.67 | −0.13 | 0.52 | 0.32 [−0.70, 1.34] |
   | skew-free GBM baseline, same close | 2.31 | 1.08 | −1.98 | 0.70 [−0.58, 1.88] |

   Fills move 0.8¢ per contract in the trade's favour net of fees within five days, and
   most of a mispricing is corrected within a day, so the edge belongs to a trader who
   acts the same day ([`RISK_PROFILE.md`](RISK_PROFILE.md) §5). It survives Kalshi's
   current doubled fee (pooled 1.16) and every modelling variant except execution delay
   (`analysis/sensitivity.py`).
4. **The skew is what matters.** The lognormal baseline overprices buckets 2–10% below
   spot by about 4–5¢; in 2024, when SPX rallied through the board, that bias produced
   its whole loss ([`GBM_BIAS.md`](GBM_BIAS.md)).

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
python analysis/execution_lag.py     # same-close vs lagged execution, pooled Sharpe, markouts
python analysis/sensitivity.py       # appendix robustness table (then: python paper/make_tables.py)
python analysis/gbm_bias.py          # why the lognormal baseline loses in 2024
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
