"""
Base interfaces for all strategies.

Design notes:
- Strategies are stateless between bars: all state is in Position objects.
- generate_signals() receives the panel up to (and including) bar_i.
  The engine enforces a 1-bar execution delay so signals from bar i are
  filled at bar i+1 prices.
- compute_position_size() must never exceed risk limits; engine also checks.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

import pandas as pd


class Side(str, Enum):
    LONG  = "long"
    SHORT = "short"
    FLAT  = "flat"


class Market(str, Enum):
    SPOT = "spot"
    PERP = "perp"


@dataclass
class Signal:
    signal_id:  str
    strategy:   str
    venue:      str
    asset:      str
    market:     Market
    side:       Side
    confidence: float = 1.0         # 0-1, used for position sizing
    metadata:   dict  = field(default_factory=dict)


@dataclass
class Position:
    signal_id:   str
    venue:       str
    asset:       str
    market:      Market
    side:        Side
    entry_price: float
    entry_ts:    pd.Timestamp
    size:        float
    notional:    float
    pnl:         float = 0.0        # accrued P&L (funding, borrow, unrealised)
    bars_held:   int   = 0
    metadata:    dict  = field(default_factory=dict)


class BaseStrategy(abc.ABC):
    """
    All strategies inherit from this. Subclasses implement:
      - generate_signals()
      - compute_position_size()
      - should_exit()
    """

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg  = cfg

    @abc.abstractmethod
    def generate_signals(
        self,
        panel:      pd.DataFrame,
        bar_i:      int,
        open_positions: List[Position],
    ) -> List[Signal]:
        """
        Return list of Signal objects.
        bar_i is the CURRENT bar (signal bar); fills happen at bar_i+1.
        """
        ...

    def compute_position_size(
        self,
        signal:  Signal,
        capital: float,
        price:   float,
        risk:    Dict[str, Any],
    ) -> float:
        """
        Default: flat pct-of-capital sizing. Override for volatility targeting.
        """
        pct = self.cfg.get("position_size_pct", 0.3)
        notional = capital * pct
        size = notional / max(price, 1e-9)
        return max(size, 0.0)

    def should_exit(
        self,
        position: Position,
        panel:    pd.DataFrame,
        bar_i:    int,
    ) -> bool:
        """Default: exit after max_hold_bars."""
        return position.bars_held >= self.cfg.get("max_hold_bars", 21)

    # ── Helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _zscore(series: pd.Series, lookback: int) -> float:
        """Rolling z-score of last value."""
        if len(series) < lookback:
            return 0.0
        window = series.iloc[-lookback:]
        mu, sigma = window.mean(), window.std()
        if sigma == 0 or pd.isna(sigma):
            return 0.0
        return float((series.iloc[-1] - mu) / sigma)

    @staticmethod
    def _col(row: pd.Series, *candidates: str, default: float = float("nan")) -> float:
        for c in candidates:
            if c in row.index and pd.notna(row[c]):
                return float(row[c])
        return default
