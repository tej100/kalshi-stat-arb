"""Extract - Kalshi event-contract order-book data.

`load_kalshi` reads the cached pickle used by the pipeline; the fetch helpers
(`build_kalshi_dataset` and friends) pull fresh data from the Kalshi API and
rebuild that pickle. Fetching is never triggered on import - call it explicitly
or run this module as a script. The API key is loaded lazily from the gitignored
`api_key.py` so importing this module never requires credentials.

!! `kalshi_data.pkl` CANNOT BE REGENERATED - TREAT IT AS PRIMARY SOURCE DATA !!
Verified 2026-09-21: Kalshi no longer serves the settled 2022-2024 markets.
`GET /events/{ticker}` returns the event with an EMPTY markets list, and every
one of the 39 stored market tickers 404s on `GET /markets/{ticker}` (0/13 in
each year), so the candlesticks behind this pickle are unreachable. The API,
the credentials and the candlestick endpoint are all fine - live markets under
the current series return data normally - so this is retention, not breakage.
The pickle (written 2025-04-22) is the only surviving copy of this dataset;
back it up, and do not overwrite it with a partial re-fetch.

Consequence: the 2024-11-16..21 quote-feed outage documented in
`transform/kalshi_pmf.feed_outage_days` can never be repaired at source, which
is why it is handled in code.

Note also that the product has since been renamed and restructured (the current
series uses tickers like `KXINXY-26DEC31H1600-T4000`, threshold-style, versus
the `INXY-22DEC30-B3300` bucket style stored here), so EVENT_TICKERS and
`_rename_buckets` below describe the historical schema and would need rework to
target current markets.
"""
from __future__ import annotations
import base64
import datetime
import pickle
import pandas as pd
# NOTE: `requests` and `cryptography` are needed only for FETCHING (build_kalshi_
# dataset). They are imported lazily inside those functions so that load_kalshi -
# and importing this module at all - works without the fetch dependencies.

from .. import config

BASE_URL = "https://api.elections.kalshi.com"
SERIES_TICKER = "KXINXY"
EVENT_TICKERS = {2022: "INXY-22DEC30", 2023: "INXY-23DEC29", 2024: "INXD-24DEC31"}
PERIOD_INTERVAL = 1440
PRIVATE_KEY_FILE = "kalshiprivatekey.key"


# --------------------------------------------------------------------------- #
# Load (cached)
# --------------------------------------------------------------------------- #
def load_kalshi(path=None) -> dict:
    """Load the Kalshi data pickle: {year: {tickers, bid, ask, price}}.

    Each of bid/ask/price is a DataFrame indexed by calendar date with one
    column per $200 bucket (e.g. '5000.0-5199.99'); values are cents (0-100).
    """
    path = path or config.KALSHI_PKL
    with open(path, "rb") as f:
        data = pickle.load(f)
    for year in data:
        for key in ("bid", "ask", "price"):
            data[year][key].index = pd.to_datetime(data[year][key].index)
    return data


# --------------------------------------------------------------------------- #
# Fetch (Kalshi API) - refactored so nothing runs on import
# --------------------------------------------------------------------------- #
def _load_private_key(file_path=PRIVATE_KEY_FILE):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    with open(file_path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None,
                                                  backend=default_backend())


def _sign_pss_text(private_key, text: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.exceptions import InvalidSignature
    try:
        sig = private_key.sign(
            text.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return base64.b64encode(sig).decode("utf-8")
    except InvalidSignature as e:
        raise ValueError("RSA sign PSS failed") from e


def _auth_headers(path):
    """Build Kalshi auth headers. Key is imported lazily (gitignored api_key.py)."""
    from api_key import key as API_KEY
    ts = str(int(datetime.datetime.now().timestamp() * 1000))
    sig = _sign_pss_text(_load_private_key(), ts + "GET" + path)
    return {"accept": "application/json", "KALSHI-ACCESS-KEY": API_KEY,
            "KALSHI-ACCESS-SIGNATURE": sig, "KALSHI-ACCESS-TIMESTAMP": ts}


def get_markets_from_event(event_ticker: str) -> list[str]:
    import requests
    path = "/trade-api/v2/events/" + event_ticker
    r = requests.get(BASE_URL + path, headers=_auth_headers(path))
    if r.status_code != 200:
        print(f"Error: {r.text}")
        return []
    return [m["ticker"] for m in r.json()["markets"]]


def get_market_candlesticks(*, market_ticker, series_ticker, start_ts, end_ts,
                            period_interval):
    import requests
    path = f"/trade-api/v2/series/{series_ticker}/markets/{market_ticker}/candlesticks"
    r = requests.get(BASE_URL + path, headers=_auth_headers(path),
                     params={"start_ts": start_ts, "end_ts": end_ts,
                             "period_interval": period_interval})
    if r.status_code != 200:
        print(f"Error: {r.text}")
        return []
    return r.json()


def _rename_buckets(cols):
    """Market ticker -> '<lo>-<hi>' $200 bucket label."""
    return [f"{float(c.split('-')[2][1:]) - 100}-{float(c.split('-')[2][1:]) + 99.99}"
            for c in cols]


def build_kalshi_dataset(series_ticker=SERIES_TICKER, event_tickers=None,
                         period_interval=PERIOD_INTERVAL, out_path=None) -> dict:
    """Pull bid/ask/price candlesticks for each year's buckets and cache to pickle.

    Drops the open-ended lower/upper tail markets (first and last), matching the
    defined-bucket dataset used downstream.
    """
    event_tickers = event_tickers or EVENT_TICKERS
    out_path = out_path or config.KALSHI_PKL
    master = {}
    for year, event in event_tickers.items():
        tickers = get_markets_from_event(event)[1:-1]   # drop tail markets
        start = datetime.datetime(year, 1, 1, 16, 0, 0)
        end = datetime.datetime(year, 12, 31, 16, 0, 0)
        idx = pd.date_range(start=start.date(), end=end.date(), freq="D")
        bid = pd.DataFrame(columns=tickers, index=idx)
        ask = pd.DataFrame(columns=tickers, index=idx)
        price = pd.DataFrame(columns=tickers, index=idx)
        for t in tickers:
            data = get_market_candlesticks(
                market_ticker=t, series_ticker=series_ticker,
                start_ts=int(start.timestamp()), end_ts=int(end.timestamp()),
                period_interval=period_interval)
            data = pd.DataFrame(data["candlesticks"]) if data else pd.DataFrame()
            if data.empty:
                continue
            data.index = pd.to_datetime(data["end_period_ts"], unit="s").normalize()
            p = data[["yes_bid", "yes_ask", "price"]].apply(
                lambda x: x.apply(lambda y: y["close"]))
            bid[t], ask[t], price[t] = p["yes_bid"], p["yes_ask"], p["price"]
        for d in (bid, ask, price):
            d.columns = _rename_buckets(d.columns.tolist())
        master[year] = {"tickers": tickers, "bid": bid, "ask": ask, "price": price}
    with open(out_path, "wb") as f:
        pickle.dump(master, f)
    print(f"Data saved to {out_path}")
    return master


if __name__ == "__main__":
    build_kalshi_dataset()
