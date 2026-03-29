"""
Perp-Perp Funding Differential Strategy.

Logic:
  - Compare funding rates for the same asset across two venues.
  - Long perp on the low-funding venue, short perp on the high-funding venue.
  - Net P&L = rate differential minus round-trip fees.
  - Use z-score of the spread to filter noise and avoid thin-edge trades.

Risks modelled:
  - Venue A / venue B may not fill simultaneously → leg-in risk.
  - Funding timestamps may not align exactly (Binance vs Bybit differ
    by seconds; we snap to 8h bars so this is handled at normalisation).
  - Collateral is split across venues → capital drag.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, Position, Side, Market


class PerpPerpDiffStrategy(BaseStrategy):

    def __init__(self, cfg: dict):
        super().__init__("PerpPerpDiff", cfg)
        self.min_spread = cfg.get("min_funding_spread",  0.0002)
        self.lb         = cfg.get("zscore_lookback",     30)
        self.entry_z    = cfg.get("entry_z",             2.0)
        self.exit_z     = cfg.get("exit_z",              0.5)
        self.max_open   = cfg.get("max_open_positions",  2)
        self.venues     = cfg.get("venues",              ["binance", "bybit"])
        self.assets     = cfg.get("assets",              ["BTC", "ETH"])

    def generate_signals(self, panel: pd.DataFrame, bar_i: int,
                         open_positions: List[Position]) -> List[Signal]:
        if bar_i < self.lb or len(self.venues) < 2:
            return []

        row     = panel.iloc[bar_i]
        signals: List[Signal] = []
        n_open  = len([p for p in open_positions
                        if p.metadata.get("strategy") == self.name])

        v_hi, v_lo = self.venues[0], self.venues[1]

        for asset in self.assets:
            sid = f"{self.name}_{v_hi}_{v_lo}_{asset}"
            existing = next((p for p in open_positions if p.signal_id == sid), None)

            fr_hi = self._col(row, f"{v_hi}_{asset}_funding")
            fr_lo = self._col(row, f"{v_lo}_{asset}_funding")
            if np.isnan(fr_hi) or np.isnan(fr_lo):
                continue

            spread = fr_hi - fr_lo

            # Build spread series for z-score
            hi_col = f"{v_hi}_{asset}_funding"
            lo_col = f"{v_lo}_{asset}_funding"
            if hi_col not in panel.columns or lo_col not in panel.columns:
                continue
            spread_series = panel[hi_col].iloc[:bar_i+1] - panel[lo_col].iloc[:bar_i+1]
            z = self._zscore(spread_series, self.lb)

            # ── Exit ──────────────────────────────────────────────────────
            if existing is not None:
                if abs(z) < self.exit_z:
                    signals.append(Signal(
                        signal_id=sid, strategy=self.name,
                        venue=v_hi, asset=asset, market=Market.PERP,
                        side=Side.FLAT, metadata={"reason": "z_revert"}
                    ))
                continue

            if n_open >= self.max_open:
                continue
            if abs(spread) < self.min_spread:
                continue
            if abs(z) < self.entry_z:
                continue

            # Short the high-funding venue, long the low-funding venue
            side = Side.SHORT if spread > 0 else Side.LONG
            n_open += 1
            signals.append(Signal(
                signal_id=sid, strategy=self.name,
                venue=v_hi, asset=asset, market=Market.PERP, side=side,
                confidence=min(abs(z) / self.entry_z, 2.0) / 2.0,
                metadata={"strategy": self.name, "spread": spread, "z": z,
                           "venue_hi": v_hi, "venue_lo": v_lo}
            ))

        return signals
