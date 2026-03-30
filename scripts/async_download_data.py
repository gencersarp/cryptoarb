#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import aiohttp
import pandas as pd

from data.downloader import ASSET_MAP


async def _fetch_json(session: aiohttp.ClientSession, url: str, params: dict, retries: int = 5):
    backoff = 1.0
    for _ in range(retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status in (418, 429):
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 10)
                    continue
                resp.raise_for_status()
                return await resp.json()
        except Exception:
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)
    return None


async def _download_binance_funding(session, symbol: str, start_ms: int, end_ms: int):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    data = await _fetch_json(session, url, {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["ts", "funding_rate"]].drop_duplicates("ts")


async def main_async(args):
    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(args.end, tz="UTC").timestamp() * 1000)
    out = Path(args.raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.max_concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for asset in args.assets:
            symbol = ASSET_MAP.get("binance", {}).get(asset, {}).get("perp")
            if not symbol:
                continue
            async def wrapped(a=asset, s=symbol):
                async with sem:
                    df = await _download_binance_funding(session, s, start_ms, end_ms)
                    if not df.empty:
                        venue_dir = out / "binance"
                        venue_dir.mkdir(exist_ok=True)
                        df.to_csv(venue_dir / f"{a}_funding.csv", index=False)
            tasks.append(asyncio.create_task(wrapped()))
        await asyncio.gather(*tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--max-concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

