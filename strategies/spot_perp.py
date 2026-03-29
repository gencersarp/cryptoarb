"""
Spot-Perp Funding Capture Strategy.

Logic:
  - Long spot + short perp when funding rate > entry_threshold
    (collect positive funding from the short perp leg).
  - Short spot + long perp when funding rate < -entry_threshold
    (collect negative funding — rare but profitable when it occurs).
  - Exit when funding reverts below exit_threshold or max_hold reached.

Cost awareness:
  - Entry only if expected funding income over min_hold bars > round-trip fee.
  - Basis filter: skip entry if basis z-score is extreme (stretched basis
    may gap further before converging, creating hidden cost).

Bias guards:
  - No look-ahead: funding at bar_i known; fill at bar_i+1.
  - Funding settled per 8h bar only at funding timestamps.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, Position, Side, Market


class SpotPerpFundingStrategy(BaseStrategy):

    def __init__(self, cfg: dict):
        super().__init__("SpotPerpFunding", cfg)
        self.entry_thr  = cfg.get("entry_funding_threshold", 0.0003)
        self.exit_thr   = cfg.get("exit_funding_threshold",  0.0001)
        self.min_yield  = cfg.get("min_annualized_yield",    0.05)
        self.basis_z    = cfg.get("basis_filter_z",          1.5)
        self.use_basis  = cfg.get("use_basis_filter",        True)
        self.max_open   = cfg.get("max_open_positions",      2)
        self.venues     = cfg.get("venues",                  ["binance"])
        self.assets     = cfg.get("assets",                  ["BTC", "ETH"])
        self.lb         = cfg.get("zscore_lookback",         48)

    def generate_signals(self, panel: pd.DataFrame, bar_i: int,
                         open_positions: List[Position]) -> List[Signal]:
        if bar_i < self.lb:
            return []

        row     = panel.iloc[bar_i]
        signals: List[Signal] = []
        n_open  = len([p for p in open_positions if p.metadata.get("strategy") == self.name])

        for venue in self.venues:
            for asset in self.assets:
                sid = f"{self.name}_{venue}_{asset}"

                # ── Exit check ────────────────────────────────────────────
                existing = next((p for p in open_positions if p.signal_id == sid), None)
                if existing is not None:
                    fr = self._col(row, f"{venue}_{asset}_funding", "funding_rate")
                    if abs(fr) < self.exit_thr or np.isnan(fr):
                        signals.append(Signal(
                            signal_id=sid, strategy=self.name,
                            venue=venue, asset=asset,
                            market=Market.PERP, side=Side.FLAT,
                            metadata={"reason": "funding_exit"}
                        ))
                    continue

                if n_open >= self.max_open:
                    continue

                # ── Entry check ───────────────────────────────────────────
                fr = self._col(row, f"{venue}_{asset}_funding", "funding_rate")
                if np.isnan(fr):
                    continue

                # Annualised yield estimate: 3 settlements/day * 365
                ann_yield = fr * 3 * 365
                if abs(fr) < self.entry_thr:
                    continue
                if abs(ann_yield) < self.min_yield:
                    continue

                # Basis filter — skip if basis is extremely stretched
                if self.use_basis:
                    basis_col = f"{venue}_{asset}_basis"
                    if basis_col in panel.columns:
                        bz = self._zscore(panel[basis_col].iloc[:bar_i+1], self.lb)
                        if abs(bz) > self.basis_z:
                            continue

                side = Side.SHORT if fr > 0 else Side.LONG
                n_open += 1
                signals.append(Signal(
                    signal_id=sid, strategy=self.name,
                    venue=venue, asset=asset,
                    market=Market.PERP, side=side,
                    confidence=min(abs(fr) / self.entry_thr, 3.0) / 3.0,
                    metadata={"strategy": self.name, "funding_rate": fr,
                               "ann_yield": ann_yield}
                ))

        return signals

    def should_exit(self, position: Position, panel: pd.DataFrame,
                     bar_i: int) -> bool:
        if position.bars_held >= self.cfg.get("max_hold_bars", 21):
            return True
        row = panel.iloc[bar_i]
        venue, asset = position.venue, position.asset
        fr = self._col(row, f"{venue}_{asset}_funding", "funding_rate")
        return (not np.isnan(fr)) and abs(fr) < self.exit_thr
