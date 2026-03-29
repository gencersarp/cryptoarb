"""
Historical data downloader for Binance, Bybit, OKX.

Downloads:
  - OHLCV (8h bars) for spot and perpetual futures
  - Funding rate history
  - Saves raw CSVs to data/raw/{venue}/{asset}_{market}.csv

All times stored in UTC. No timezone assumptions.
"""
import logging
import time
from pathlib import Path
from typing import List
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---- Binance -----------------------------------------------------------

def _binance_ohlcv(symbol: str, interval: str,
                   start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch OHLCV from Binance via REST.
    Paginates automatically across the full requested range.
    """
    url    = "https://fapi.binance.com/fapi/v1/klines"
    rows   = []
    cur_ms = start_ms
    while cur_ms < end_ms:
        params = dict(symbol=symbol, interval=interval,
                      startTime=cur_ms, endTime=end_ms, limit=1500)
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Binance OHLCV fetch error {symbol}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        rows.extend(data)
        cur_ms = int(data[-1][0]) + 1
        time.sleep(0.1)  # rate limit courtesy
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])
    df["ts"]    = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"]= df["volume"].astype(float)
    return df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts")


def _binance_spot_ohlcv(symbol: str, interval: str,
                        start_ms: int, end_ms: int) -> pd.DataFrame:
    url  = "https://api.binance.com/api/v3/klines"
    rows = []
    cur_ms = start_ms
    while cur_ms < end_ms:
        params = dict(symbol=symbol, interval=interval,
                      startTime=cur_ms, endTime=end_ms, limit=1000)
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Binance spot OHLCV error {symbol}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        rows.extend(data)
        cur_ms = int(data[-1][0]) + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])
    df["ts"]    = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts")


def _binance_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """
    Fetch historical funding rates from Binance.
    Binance funding interval: 8h (00:00, 08:00, 16:00 UTC)
    """
    url  = "https://fapi.binance.com/fapi/v1/fundingRate"
    rows = []
    cur  = start_ms
    while cur < end_ms:
        params = dict(symbol=symbol, startTime=cur, endTime=end_ms, limit=1000)
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Binance funding error {symbol}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        rows.extend(data)
        cur = int(data[-1]["fundingTime"]) + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"]           = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].drop_duplicates("ts")


# ---- Bybit -------------------------------------------------------------

def _bybit_ohlcv(symbol: str, interval: str,
                 start_ms: int, end_ms: int) -> pd.DataFrame:
    url  = "https://api.bybit.com/v5/market/kline"
    rows = []
    cur  = start_ms
    while cur < end_ms:
        params = dict(category="linear", symbol=symbol,
                      interval=interval, start=cur, end=end_ms, limit=1000)
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("result", {}).get("list", [])
        except Exception as e:
            logger.warning(f"Bybit OHLCV error {symbol}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        for r in data:
            rows.append({"ts": int(r[0]), "open": float(r[1]),
                         "high": float(r[2]), "low": float(r[3]),
                         "close": float(r[4]), "volume": float(r[5])})
        cur = int(data[-1][0]) + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts")


def _bybit_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url  = "https://api.bybit.com/v5/market/funding/history"
    rows = []
    cur  = start_ms
    while cur < end_ms:
        params = dict(category="linear", symbol=symbol,
                      startTime=cur, endTime=end_ms, limit=200)
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("result", {}).get("list", [])
        except Exception as e:
            logger.warning(f"Bybit funding error {symbol}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        for r in data:
            rows.append({"ts": int(r["fundingRateTimestamp"]),
                         "funding_rate": float(r["fundingRate"])})
        cur = int(data[-1]["fundingRateTimestamp"]) + 1
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts", "funding_rate"]].drop_duplicates("ts")


# ---- OKX ---------------------------------------------------------------

def _okx_ohlcv(inst_id: str, bar: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url  = "https://www.okx.com/api/v5/market/history-candles"
    rows = []
    after = None
    while True:
        params = dict(instId=inst_id, bar=bar, limit=300)
        if after:
            params["after"] = after
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"OKX OHLCV error {inst_id}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        for r in data:
            ts_ms = int(r[0])
            if ts_ms < start_ms:
                data = []
                break
            rows.append({"ts": ts_ms, "open": float(r[1]),
                         "high": float(r[2]), "low": float(r[3]),
                         "close": float(r[4]), "volume": float(r[5])})
        if not data:
            break
        after = data[-1][0]
        if int(after) < start_ms:
            break
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts").sort_values("ts")


def _okx_funding(inst_id: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    url  = "https://www.okx.com/api/v5/public/funding-rate-history"
    rows = []
    after = None
    while True:
        params = dict(instId=inst_id, limit=100)
        if after:
            params["after"] = after
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"OKX funding error {inst_id}: {e}")
            time.sleep(5)
            continue
        if not data:
            break
        for r in data:
            ts_ms = int(r["fundingTime"])
            if ts_ms < start_ms:
                data = []
                break
            rows.append({"ts": ts_ms, "funding_rate": float(r["fundingRate"])})
        if not data:
            break
        after = data[-1]["fundingTime"]
        if int(after) < start_ms:
            break
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts")


# ---- Orchestrator ------------------------------------------------------

ASSET_MAP = {
    "binance": {
        "BTC": {"spot": "BTCUSDT",  "perp": "BTCUSDT"},
        "ETH": {"spot": "ETHUSDT",  "perp": "ETHUSDT"},
        "SOL": {"spot": "SOLUSDT",  "perp": "SOLUSDT"},
        "BNB": {"spot": "BNBUSDT",  "perp": "BNBUSDT"},
    },
    "bybit": {
        "BTC": {"perp": "BTCUSDT"},
        "ETH": {"perp": "ETHUSDT"},
        "SOL": {"perp": "SOLUSDT"},
    },
    "okx": {
        "BTC": {"spot": "BTC-USDT", "perp": "BTC-USDT-SWAP"},
        "ETH": {"spot": "ETH-USDT", "perp": "ETH-USDT-SWAP"},
    },
}

INTERVAL_MAP = {
    "binance": {"spot": "8h", "perp": "8h"},
    "bybit":   {"perp": "480"},   # Bybit uses minutes: 480 = 8h
    "okx":     {"spot": "8H",  "perp": "8H"},
}


def download_all(assets: List[str], venues: List[str],
                 start: str, end: str, raw_dir: str = "data/raw"):
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    for venue in venues:
        venue_dir = Path(raw_dir) / venue
        venue_dir.mkdir(exist_ok=True)
        for asset in assets:
            _download_venue_asset(venue, asset, start_ms, end_ms, venue_dir)


def _download_venue_asset(venue: str, asset: str, start_ms: int,
                           end_ms: int, out_dir: Path):
    symbols = ASSET_MAP.get(venue, {}).get(asset, {})
    if not symbols:
        logger.warning(f"No symbol mapping for {venue}/{asset}")
        return

    logger.info(f"Downloading {venue}/{asset}...")

    # Spot OHLCV
    if "spot" in symbols:
        spot_sym = symbols["spot"]
        interval = INTERVAL_MAP[venue].get("spot", "8h")
        if venue == "binance":
            df = _binance_spot_ohlcv(spot_sym, interval, start_ms, end_ms)
        elif venue == "okx":
            df = _okx_ohlcv(spot_sym, interval, start_ms, end_ms)
        else:
            df = pd.DataFrame()
        if not df.empty:
            path = out_dir / f"{asset}_spot.csv"
            df.to_csv(path, index=False)
            logger.info(f"  Saved spot: {path} ({len(df)} bars)")

    # Perp OHLCV
    if "perp" in symbols:
        perp_sym = symbols["perp"]
        interval = INTERVAL_MAP[venue].get("perp", "8h")
        if venue == "binance":
            df = _binance_ohlcv(perp_sym, interval, start_ms, end_ms)
        elif venue == "bybit":
            df = _bybit_ohlcv(perp_sym, interval, start_ms, end_ms)
        elif venue == "okx":
            df = _okx_ohlcv(perp_sym, interval, start_ms, end_ms)
        else:
            df = pd.DataFrame()
        if not df.empty:
            path = out_dir / f"{asset}_perp.csv"
            df.to_csv(path, index=False)
            logger.info(f"  Saved perp: {path} ({len(df)} bars)")

    # Funding rates
    if "perp" in symbols:
        perp_sym = symbols["perp"]
        if venue == "binance":
            df = _binance_funding(perp_sym, start_ms, end_ms)
        elif venue == "bybit":
            df = _bybit_funding(perp_sym, start_ms, end_ms)
        elif venue == "okx":
            df = _okx_funding(perp_sym, start_ms, end_ms)
        else:
            df = pd.DataFrame()
        if not df.empty:
            path = out_dir / f"{asset}_funding.csv"
            df.to_csv(path, index=False)
            logger.info(f"  Saved funding: {path} ({len(df)} rates)")
