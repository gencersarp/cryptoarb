"""
Data normalizer: merges raw CSVs into a unified panel.

Output panel columns per (venue, asset):
  ts               UTC timestamp (index)
  spot_open/high/low/close  spot OHLCV
  perp_open/high/low/close  perp OHLCV
  spot_volume      spot volume in base asset
  perp_volume      perp volume
  funding_rate     8h funding rate (exchange-native convention)
  basis            (perp_close - spot_close) / spot_close
  realized_vol     rolling 24-bar (8-day) realized vol of perp returns
  adv              20-bar rolling average daily volume (USD notional)
  venue            string label
  asset            string label

All NaN-handling is explicit: forward-fill funding rates up to 1 interval,
interpolate small gaps in price, drop bars with critical missing data.
"""
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_raw(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Missing raw file: {path}")
        return None
    df = pd.read_csv(p, parse_dates=["ts"])
    if "ts" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").drop_duplicates("ts").set_index("ts")


def normalize_ohlcv(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Rename OHLCV columns with prefix, ensure numeric types,
    interpolate small gaps (max 2 bars), forward-fill volume.
    """
    cols = {"open": f"{prefix}_open", "high": f"{prefix}_high",
            "low":  f"{prefix}_low",  "close": f"{prefix}_close",
            "volume": f"{prefix}_volume"}
    df = df.rename(columns=cols)
    for col in cols.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Linear interpolation for price gaps (max 2 missing bars)
    price_cols = [c for c in cols.values() if "volume" not in c]
    df[price_cols] = df[price_cols].interpolate(method="linear", limit=2)
    # Forward-fill volume (no interpolation — volume not well-defined for missing bars)
    if f"{prefix}_volume" in df.columns:
        df[f"{prefix}_volume"] = df[f"{prefix}_volume"].fillna(0.0)
    return df


def compute_realized_vol(close: pd.Series, window: int = 24) -> pd.Series:
    """
    Annualized realized vol from log returns, rolling window.
    window=24 bars @ 8h each = 8 days.
    Annualization factor: sqrt(3 * 365) = sqrt(1095).
    """
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(1095)
    return rv.fillna(log_ret.std() * np.sqrt(1095))


def compute_adv(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Average Daily USD Volume proxy over rolling window.
    """
    notional = close * volume
    return notional.rolling(window).mean().fillna(notional.mean())


def build_single_panel(venue: str, asset: str, raw_dir: str) -> Optional[pd.DataFrame]:
    """
    Build a normalized, feature-enriched panel for one venue/asset pair.
    Returns None if critical data is missing.
    """
    base = Path(raw_dir) / venue
    spot_path    = base / f"{asset}_spot.csv"
    perp_path    = base / f"{asset}_perp.csv"
    funding_path = base / f"{asset}_funding.csv"

    spot_df    = load_raw(str(spot_path))
    perp_df    = load_raw(str(perp_path))
    funding_df = load_raw(str(funding_path))

    if perp_df is None:
        logger.warning(f"No perp data for {venue}/{asset}. Skipping.")
        return None

    # Build base index from perp bars
    panel = normalize_ohlcv(perp_df, "perp")

    # Merge spot if available
    if spot_df is not None:
        spot_norm = normalize_ohlcv(spot_df, "spot")
        panel = panel.join(spot_norm, how="left")
    else:
        # Use perp as proxy for spot (less ideal but functional)
        logger.warning(f"No spot data for {venue}/{asset}. Using perp as spot proxy.")
        for col in ["open", "high", "low", "close", "volume"]:
            panel[f"spot_{col}"] = panel[f"perp_{col}"]

    # Merge funding rates
    if funding_df is not None:
        funding_df = funding_df.rename(columns={"funding_rate": "funding_rate"})
        # Reindex to perp bars, forward-fill up to 3 bars (one funding interval)
        funding_reindexed = funding_df["funding_rate"].reindex(
            panel.index, method="ffill", limit=3
        )
        panel["funding_rate"] = funding_reindexed.fillna(0.0)
    else:
        logger.warning(f"No funding data for {venue}/{asset}. Setting to zero.")
        panel["funding_rate"] = 0.0

    # Derived features
    panel["basis"] = (
        (panel["perp_close"] - panel["spot_close"]) / panel["spot_close"]
    ).fillna(0.0)

    panel["realized_vol"] = compute_realized_vol(panel["perp_close"])
    panel["adv"]          = compute_adv(panel["perp_close"], panel["perp_volume"])

    panel["venue"] = venue
    panel["asset"] = asset

    # Drop bars where critical columns are NaN
    critical = ["perp_close", "spot_close", "realized_vol"]
    before = len(panel)
    panel = panel.dropna(subset=critical)
    after  = len(panel)
    if before - after > 0:
        logger.info(f"{venue}/{asset}: dropped {before-after} bars with missing critical data")

    if len(panel) < 100:
        logger.warning(f"{venue}/{asset}: only {len(panel)} bars after cleaning. Skipping.")
        return None

    logger.info(f"{venue}/{asset}: {len(panel)} clean bars "
                f"from {panel.index[0]} to {panel.index[-1]}")
    return panel


def build_panel(assets: List[str], venues: List[str],
                raw_dir: str, freq: str = "8h") -> pd.DataFrame:
    """
    Build multi-index panel for all (venue, asset) pairs.
    Returns flat DataFrame with venue/asset as columns for single-series strategies.
    Falls back to first available series if multi-index not needed.
    """
    panels = {}
    for venue in venues:
        for asset in assets:
            p = build_single_panel(venue, asset, raw_dir)
            if p is not None:
                panels[(venue, asset)] = p

    if not panels:
        raise ValueError("No valid panels built. Check raw data in data/raw/.")

    logger.info(f"Built {len(panels)} panels: {list(panels.keys())}")
    # Return dict; individual strategies pick the series they need
    # For backward compat: if only one pair, return flat DataFrame
    if len(panels) == 1:
        return list(panels.values())[0]
    return panels  # type: ignore


def get_single_series(panels, venue: str, asset: str) -> pd.DataFrame:
    if isinstance(panels, dict):
        return panels.get((venue, asset), pd.DataFrame())
    return panels  # already flat
