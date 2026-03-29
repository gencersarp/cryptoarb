"""
Data normalization layer.

Takes raw CSVs from data/raw/ and produces a clean, aligned, multi-indexed
panel DataFrame at the configured base frequency (default: 8h).

Output columns per (venue, asset):
  spot_close, spot_volume, spot_adv, perp_close, perp_volume,
  funding_rate, funding_ann, basis, basis_ann, realized_vol, adv
"""
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANNUALIZE_FACTOR = {"binance": 1095, "bybit": 1095, "okx": 1095}  # 8h*3*365


def _load_ohlcv(path: str) -> Optional[pd.DataFrame]:
    if not Path(path).exists():
        logger.warning(f"File not found: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp"))
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _load_funding(path: str) -> Optional[pd.DataFrame]:
    if not Path(path).exists():
        logger.warning(f"Funding file not found: {path}")
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp"))
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    return df


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {k: v for k, v in {
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum", "quote_volume": "sum"
    }.items() if k in df.columns}
    return df.resample(freq).agg(agg).dropna(subset=["close"])


def build_panel(assets: list, venues: list, raw_dir: str = "data/raw",
                freq: str = "8h", vol_lookback: int = 20,
                adv_lookback: int = 20) -> pd.DataFrame:
    """
    Load all raw data and produce a clean multi-indexed panel.
    Index: (ts, venue, asset). Columns: spot_close, perp_close,
           funding_rate, funding_ann, basis, realized_vol, adv, ...
    """
    records = []
    for venue in venues:
        for asset in assets:
            base     = f"{raw_dir}/{venue}/{asset}"
            spot_df  = _load_ohlcv(f"{base}/spot_1h.csv")
            perp_df  = _load_ohlcv(f"{base}/perp_1h.csv")
            fund_df  = _load_funding(f"{base}/funding_8h.csv")

            if spot_df is None or perp_df is None:
                logger.warning(f"Missing data for {venue}/{asset} — skipping")
                continue

            spot_r = _resample_ohlcv(spot_df, freq)
            perp_r = _resample_ohlcv(perp_df, freq)
            idx    = spot_r.index.intersection(perp_r.index)
            if len(idx) == 0:
                logger.warning(f"No overlapping timestamps for {venue}/{asset}")
                continue

            panel = pd.DataFrame(index=idx)
            panel["spot_close"]     = spot_r.loc[idx, "close"]
            panel["spot_volume"]    = spot_r.loc[idx].get("volume",  np.nan)
            panel["spot_quote_vol"] = spot_r.loc[idx].get("quote_volume", np.nan)
            panel["perp_close"]     = perp_r.loc[idx, "close"]
            panel["perp_volume"]    = perp_r.loc[idx].get("volume", np.nan)

            if fund_df is not None:
                fund_r = fund_df["funding_rate"].resample(freq).sum().fillna(0)
                panel["funding_rate"] = fund_r.reindex(idx, fill_value=0)
            else:
                panel["funding_rate"] = 0.0

            ann = ANNUALIZE_FACTOR.get(venue, 1095)
            panel["funding_ann"] = panel["funding_rate"] * ann
            panel["basis"]       = ((panel["perp_close"] - panel["spot_close"])
                                    / panel["spot_close"])
            panel["basis_ann"]   = panel["basis"] * ann

            log_ret = np.log(
                panel["spot_close"] / panel["spot_close"].shift(1))
            panel["realized_vol"] = log_ret.rolling(
                vol_lookback, min_periods=5).std()
            panel["adv"] = panel["spot_quote_vol"].rolling(
                adv_lookback, min_periods=5).mean()

            panel["venue"] = venue
            panel["asset"] = asset
            panel = panel.reset_index().rename(columns={"timestamp": "ts"})
            records.append(panel)

    if not records:
        raise ValueError(
            "No data loaded. Run: python scripts/download_data.py first.")

    combined = (pd.concat(records, ignore_index=True)
                  .set_index(["ts", "venue", "asset"])
                  .sort_index()
                  .dropna(subset=["spot_close", "perp_close"]))

    logger.info(
        f"Panel built: {len(combined)} rows | "
        f"{combined.index.get_level_values('ts').nunique()} timestamps | "
        f"{combined.index.get_level_values('venue').nunique()} venues | "
        f"{combined.index.get_level_values('asset').nunique()} assets")
    return combined


def get_single_series(panel: pd.DataFrame, venue: str, asset: str) -> pd.DataFrame:
    """Extract a single (venue, asset) time series from the multi-indexed panel."""
    try:
        return panel.xs((venue, asset), level=("venue", "asset")).copy()
    except KeyError:
        logger.error(f"No data for venue={venue} asset={asset}")
        return pd.DataFrame()


def compute_cross_venue_funding_diff(panel: pd.DataFrame,
                                      asset: str,
                                      venue_a: str,
                                      venue_b: str) -> pd.Series:
    """Funding differential (venue_a − venue_b) aligned on common timestamps."""
    try:
        fa = panel.xs((venue_a, asset), level=("venue", "asset"))["funding_rate"]
        fb = panel.xs((venue_b, asset), level=("venue", "asset"))["funding_rate"]
    except KeyError as e:
        logger.error(f"Missing cross-venue data: {e}")
        return pd.Series(dtype=float)
    aligned = fa.align(fb, join="inner")
    return aligned[0] - aligned[1]
