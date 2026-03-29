"""
Real historical data downloader.
Sources: Binance, Bybit, OKX public APIs — no API keys required for historical data.

Downloads:
  - Spot OHLCV (1h klines)
  - Perp OHLCV (1h klines)
  - Funding rate history (8h)

Saves raw CSVs to data/raw/<venue>/<asset>/

All pagination is handled automatically.
Files are skipped if they already exist (idempotent).
"""
import os
import time
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

BINANCE_BASE        = "https://api.binance.com"
BINANCE_SPOT_KLINES = f"{BINANCE_BASE}/api/v3/klines"
BINANCE_PERP_KLINES = f"{BINANCE_BASE}/fapi/v1/klines"
BINANCE_FUNDING     = f"{BINANCE_BASE}/fapi/v1/fundingRate"

BYBIT_BASE        = "https://api.bybit.com"
BYBIT_KLINES      = f"{BYBIT_BASE}/v5/market/kline"
BYBIT_FUNDING     = f"{BYBIT_BASE}/v5/market/funding/history"

OKX_BASE    = "https://www.okx.com"
OKX_KLINES  = f"{OKX_BASE}/api/v5/market/history-candles"
OKX_FUNDING = f"{OKX_BASE}/api/v5/public/funding-rate-history"


def _mkdir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} rows → {path}")


# ─── Binance ─────────────────────────────────────────────────────────────────────────────

def _binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                    endpoint: str) -> pd.DataFrame:
    """Paginate Binance klines, returning OHLCV + timestamp."""
    all_rows = []
    current  = start_ms
    limit    = 1000
    while current < end_ms:
        params = dict(symbol=symbol, interval=interval,
                      startTime=current, endTime=end_ms, limit=limit)
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        current = data[-1][6] + 1  # closeTime + 1ms
        if len(data) < limit:
            break
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df[["open_time", "open", "high", "low", "close",
             "volume", "quote_volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop(columns=["open_time"])


def _binance_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    current  = start_ms
    limit    = 1000
    while current < end_ms:
        params = dict(symbol=symbol, startTime=current, endTime=end_ms, limit=limit)
        r = requests.get(BINANCE_FUNDING, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        all_rows.extend(data)
        current = data[-1]["fundingTime"] + 1
        if len(data) < limit:
            break
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp"]    = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["timestamp", "funding_rate"]]


# ─── Bybit ─────────────────────────────────────────────────────────────────────────────

def _bybit_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                  category: str = "spot") -> pd.DataFrame:
    all_rows    = []
    current_end = end_ms
    limit       = 1000
    while True:
        params = dict(category=category, symbol=symbol, interval=interval,
                      start=start_ms, end=current_end, limit=limit)
        r = requests.get(BYBIT_KLINES, params=params, timeout=30)
        r.raise_for_status()
        result = r.json().get("result", {})
        data   = result.get("list", [])
        if not data:
            break
        all_rows.extend(data)
        oldest_ts = int(data[-1][0])
        if oldest_ts <= start_ms or len(data) < limit:
            break
        current_end = oldest_ts - 1
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["open_time", "open", "high", "low",
                                          "close", "volume", "turnover"])
    df["timestamp"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.rename(columns={"turnover": "quote_volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]


def _bybit_funding(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows    = []
    current_end = end_ms
    limit       = 200
    while True:
        params = dict(category="linear", symbol=symbol,
                      startTime=start_ms, endTime=current_end, limit=limit)
        r = requests.get(BYBIT_FUNDING, params=params, timeout=30)
        r.raise_for_status()
        result = r.json().get("result", {})
        data   = result.get("list", [])
        if not data:
            break
        all_rows.extend(data)
        oldest_ts = int(data[-1]["fundingRateTimestamp"])
        if oldest_ts <= start_ms or len(data) < limit:
            break
        current_end = oldest_ts - 1
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp"]    = pd.to_datetime(
        df["fundingRateTimestamp"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df[["timestamp", "funding_rate"]]


# ─── OKX ──────────────────────────────────────────────────────────────────────────────

def _okx_klines(inst_id: str, bar: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    after    = str(end_ms)
    limit    = 300
    while True:
        params = dict(instId=inst_id, bar=bar,
                      before=str(start_ms), after=after, limit=limit)
        r = requests.get(OKX_KLINES, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        all_rows.extend(data)
        oldest_ts = int(data[-1][0])
        if oldest_ts <= start_ms or len(data) < limit:
            break
        after = str(oldest_ts - 1)
        time.sleep(0.3)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows,
                       columns=["ts", "open", "high", "low", "close",
                                 "vol", "volCcy", "volCcyQuote", "confirm"])
    df["timestamp"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "vol", "volCcyQuote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.rename(columns={"vol": "volume", "volCcyQuote": "quote_volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume"]]


def _okx_funding(inst_id: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows = []
    after    = str(end_ms)
    limit    = 100
    while True:
        params = dict(instId=inst_id, before=str(start_ms), after=after, limit=limit)
        r = requests.get(OKX_FUNDING, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        all_rows.extend(data)
        oldest_ts = int(data[-1]["fundingTime"])
        if oldest_ts <= start_ms or len(data) < limit:
            break
        after = str(oldest_ts - 1)
        time.sleep(0.3)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["timestamp"]    = pd.to_datetime(
        df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df[["timestamp", "funding_rate"]]


# ─── Main API ───────────────────────────────────────────────────────────────────────

def download_venue_asset(venue: str, asset: str, start: str, end: str,
                          raw_dir: str = "data/raw") -> None:
    """
    Download spot klines, perp klines, and funding history for one venue/asset.
    Files are skipped if they already exist (idempotent).
    Saves to: data/raw/<venue>/<asset>/{spot_1h, perp_1h, funding_8h}.csv
    """
    out_dir  = _mkdir(f"{raw_dir}/{venue}/{asset}")
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms   = int(pd.Timestamp(end,   tz="UTC").timestamp() * 1000)
    logger.info(f"Downloading {venue}/{asset}  {start} → {end}")

    if venue == "binance":
        sym = f"{asset}USDT"
        paths = [
            (f"{out_dir}/spot_1h.csv",    lambda: _binance_klines(sym, "1h", start_ms, end_ms, BINANCE_SPOT_KLINES)),
            (f"{out_dir}/perp_1h.csv",    lambda: _binance_klines(sym, "1h", start_ms, end_ms, BINANCE_PERP_KLINES)),
            (f"{out_dir}/funding_8h.csv", lambda: _binance_funding(sym, start_ms, end_ms)),
        ]
    elif venue == "bybit":
        sym = f"{asset}USDT"
        paths = [
            (f"{out_dir}/spot_1h.csv",    lambda: _bybit_klines(sym, "60", start_ms, end_ms, "spot")),
            (f"{out_dir}/perp_1h.csv",    lambda: _bybit_klines(sym, "60", start_ms, end_ms, "linear")),
            (f"{out_dir}/funding_8h.csv", lambda: _bybit_funding(sym, start_ms, end_ms)),
        ]
    elif venue == "okx":
        spot_sym = f"{asset}-USDT"
        perp_sym = f"{asset}-USDT-SWAP"
        paths = [
            (f"{out_dir}/spot_1h.csv",    lambda: _okx_klines(spot_sym, "1H", start_ms, end_ms)),
            (f"{out_dir}/perp_1h.csv",    lambda: _okx_klines(perp_sym, "1H", start_ms, end_ms)),
            (f"{out_dir}/funding_8h.csv", lambda: _okx_funding(perp_sym, start_ms, end_ms)),
        ]
    else:
        raise ValueError(f"Unknown venue: {venue}")

    for fpath, fetch_fn in paths:
        if not os.path.exists(fpath):
            df = fetch_fn()
            if not df.empty:
                _save_csv(df, fpath)
            else:
                logger.warning(f"Empty result for {fpath}")
        else:
            logger.info(f"Skipping (already exists): {fpath}")


def download_all(assets: list, venues: list, start: str, end: str,
                 raw_dir: str = "data/raw") -> None:
    """Download all venue × asset combinations."""
    combos = [(v, a) for v in venues for a in assets]
    for venue, asset in tqdm(combos, desc="Downloading"):
        try:
            download_venue_asset(venue, asset, start, end, raw_dir)
        except Exception as e:
            logger.error(f"Failed {venue}/{asset}: {e}")
