"""Cointegration / lead-lag analysis of the Kalshi vs options-implied bucket
probabilities, and a P&L attribution (convergence vs settlement).

Answers: is the strategy a pairs/convergence trade, and which venue leads?
Run from repo root:  python analysis/cointegration.py
Findings are written up in DUAL_TRADING_ANALYSIS.md.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from kalshi_arb import pipeline, run, config
from kalshi_arb.extract import kalshi as ksrc
from kalshi_arb.transform import density, kalshi_pmf
from kalshi_arb.strategy import metrics

ACTIVE_PROB = 0.03      # a bucket is "active" on a day if either side prices it > 3%
MIN_ACTIVE_DAYS = 20    # only analyze buckets that are near-the-money enough to trade


def half_life(s):
    """Mean-reversion half-life (days) from an AR(1) fit; inf if not reverting."""
    s = s.dropna()
    if len(s) < 15:
        return np.nan
    ds = s.diff().dropna()
    lag = s.shift(1).dropna().loc[ds.index]
    beta = np.polyfit(lag.values, ds.values, 1)[0]      # ds = a + beta*level
    return np.inf if beta >= 0 else -np.log(2) / np.log(1 + beta)


def adf_p(s):
    s = s.dropna()
    if len(s) < 15 or s.std() < 1e-6:
        return np.nan
    try:
        return adfuller(s.values, autolag="AIC")[1]
    except Exception:
        return np.nan


def main():
    chain = pipeline.build_chain(verbose=False)
    k = ksrc.load_kalshi()
    model = density.build_pmf_table(chain, k, method="bl")          # options-implied PMF
    market = {y: kalshi_pmf.market_pmf(k, y) for y in k}            # Kalshi mid PMF

    # ---- 1. spread stationarity + half-life, per active bucket ----
    rows, dk, dm, sp = [], [], [], []
    for y in k:
        mdl, mkt = model[y], market[y]
        common = mdl.index.intersection(mkt.index)
        for b in mdl.columns:
            mm, dd = mkt.loc[common, b], mdl.loc[common, b]
            spread = (mm - dd)                                       # Kalshi - model
            active = (mm.fillna(0) > ACTIVE_PROB) | (dd.fillna(0) > ACTIVE_PROB)
            if active.sum() < MIN_ACTIVE_DAYS or spread.dropna().shape[0] < 20:
                continue
            rows.append(dict(year=y, bucket=b, n=int(spread.dropna().shape[0]),
                             mean_spread=spread.mean(), adf_p=adf_p(spread),
                             half_life=half_life(spread)))
            # pooled next-day changes vs current spread, for the direction test
            v = spread.notna() & mm.diff().shift(-1).notna() & dd.diff().shift(-1).notna() & active
            sp += spread[v].tolist()
            dk += mm.diff().shift(-1)[v].tolist()
            dm += dd.diff().shift(-1)[v].tolist()

    df = pd.DataFrame(rows)
    df["stationary"] = df["adf_p"] < 0.05
    fin = df[np.isfinite(df["half_life"])]
    print("=== 1. Spread stationarity & mean reversion (active buckets) ===")
    print(f"buckets analyzed:            {len(df)}")
    print(f"spread stationary (ADF<5%):  {df['stationary'].mean()*100:.0f}%")
    print(f"mean spread (Kalshi-model):  {df['mean_spread'].mean():+.4f}  "
          f"(|median| {df['mean_spread'].abs().median():.4f}) -> reverts to ~0")
    print(f"half-life (days):            median {fin['half_life'].median():.1f}  "
          f"[{fin['half_life'].quantile(.25):.1f}-{fin['half_life'].quantile(.75):.1f} IQR]")

    # ---- 2. direction of adjustment (which venue error-corrects) ----
    sp, dk, dm = np.array(sp), np.array(dk), np.array(dm)
    rk = sm.OLS(dk, sm.add_constant(sp)).fit()
    rm = sm.OLS(dm, sm.add_constant(sp)).fit()
    bk, bm = rk.params[1], rm.params[1]
    print("\n=== 2. Direction of adjustment (error correction), spread = Kalshi - model ===")
    print(f"n bucket-days:               {len(sp)}")
    print(f"d(Kalshi)_next ~ spread:     coef {bk:+.3f}  t {rk.tvalues[1]:+.1f}  "
          f"(negative => Kalshi corrects toward model)")
    print(f"d(model)_next  ~ spread:     coef {bm:+.3f}  t {rm.tvalues[1]:+.1f}  "
          f"(positive => options drift toward Kalshi)")
    print("  (plain OLS t-stats: bucket-days are serially and cross-sectionally dependent and the")
    print("   Kalshi mid carries bid-ask bounce, so treat them as an upper bound on significance.)")
    print(f"share of correction by KALSHI leg: {abs(bk)/(abs(bk)+abs(bm))*100:.0f}%")

    # ---- 3. P&L attribution: carry vs trading path vs settlement ----
    # The daily mark-to-market path is NOT all "convergence": it also contains the
    # carry on posted collateral (Kalshi interest, paid only from 2024-10-10, minus
    # the cost of capital at the option-implied rate), and it books a contract's
    # drift toward its realised 0/1 outcome before the settlement step arrives. So
    # three pieces are separated here, and the lead-lag question is answered by the
    # regression in section 2, not by this split.
    print("\n=== 3. P&L attribution (BL, both-side): carry / trading path / settlement ===")
    _, results, _ = run.full(chain=chain, kalshi=k, verbose=False)
    for y in config.YEARS:
        m = results[(y, "bl", "both")]
        ev = metrics.excess_value(m, chain)
        start = m["portfolio_value"].iloc[0]
        carry = ev.iloc[-1] - (m["portfolio_value"].iloc[-1] - m["interest"].iloc[-1])  # net carry
        path = m["portfolio_value"].iloc[-1] - m["interest"].iloc[-1] - start           # trading, marked to market
        settle = m.attrs["final_value"] - m["portfolio_value"].iloc[-1]                 # terminal settlement step
        tot = carry + path + settle
        print(f"  {y}: carry {carry:+6.2f}  trading path {path:+6.2f}  settlement {settle:+5.2f}  "
              f"= {tot:+6.2f} in excess of rf  | settlement is "
              f"{abs(settle)/(abs(carry)+abs(path)+abs(settle))*100:.0f}% and carry "
              f"{abs(carry)/(abs(carry)+abs(path)+abs(settle))*100:.0f}% of |P&L|")


if __name__ == "__main__":
    main()
