"""
Basis Mean-Reversion Strategy.

Basis = (perp_price - spot_price) / spot_price

Logic:
  - When basis is extremely positive (perp >> spot), short perp / long spot.
  - When basis is extremely negative, long perp / short spot.
  - Exit when basis z-score reverts toward zero or stop-loss hit.

Key design choices:
  - Z-score computed on rolling window (zscore_lookback bars).
  - Hard stop if z-score extends further to stop_z (position is wrong).
  - Convergence is natural near funding settlement (perp → index);
    do not assume continuous convergence.
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, Position, Side, Market


class BasisMeanRevertStrategy(BaseStrategy):

    def __init__(self, cfg: dict):
        super().__init__("BasisMeanRevert", cfg)
        self.lb       = cfg.get("zscore_lookback", 48)
        self.entry_z  = cfg.get("entry_z",         2.0)
        self.exit_z   = cfg.get("exit_z",          0.5)
        self.stop_z   = cfg.get("stop_z",          3.5)
        self.max_open = cfg.get("max_open_positions", 2)
        self.venues   = cfg.get("venues", ["binance"])
        self.assets   = cfg.get("assets", ["BTC", "ETH"])

    def generate_signals(self, panel: pd.DataFrame, bar_i: int,
                         open_positions: List[Position]) -> List[Signal]:
        if bar_i < self.lb:
            return []

        row     = panel.iloc[bar_i]
        signals: List[Signal] = []
        n_open  = len([p for p in open_positions
                        if p.metadata.get("strategy") == self.name])

        for venue in self.venues:
            for asset in self.assets:
                sid = f"{self.name}_{venue}_{asset}"
                existing = next((p for p in open_positions
                                  if p.signal_id == sid), None)

                basis_col = f"{venue}_{asset}_basis"
                if basis_col not in panel.columns:
                    # Fallback: compute from perp/spot close
                    spot_col = f"{venue}_{asset}_spot_close"
                    perp_col = f"{venue}_{asset}_perp_close"
                    if spot_col not in panel.columns or perp_col not in panel.columns:
                        continue
                    basis_series = (panel[perp_col] - panel[spot_col]) / panel[spot_col]
                else:
                    basis_series = panel[basis_col]

                z = self._zscore(basis_series.iloc[:bar_i+1], self.lb)

                # ── Exit / Stop ───────────────────────────────────────────
                if existing is not None:
                    if abs(z) > self.stop_z:
                        signals.append(Signal(
                            signal_id=sid, strategy=self.name,
                            venue=venue, asset=asset, market=Market.PERP,
                            side=Side.FLAT, metadata={"reason": "stop_loss_z"}
                        ))
                    elif abs(z) < self.exit_z:
                        signals.append(Signal(
                            signal_id=sid, strategy=self.name,
                            venue=venue, asset=asset, market=Market.PERP,
                            side=Side.FLAT, metadata={"reason": "z_revert"}
                        ))
                    continue

                if n_open >= self.max_open:
                    continue
                if abs(z) < self.entry_z:
                    continue

                # Positive z → basis too high → short perp, long spot
                side = Side.SHORT if z > 0 else Side.LONG
                n_open += 1
                signals.append(Signal(
                    signal_id=sid, strategy=self.name,
                    venue=venue, asset=asset, market=Market.PERP, side=side,
                    confidence=min(abs(z) / self.entry_z, 2.0) / 2.0,
                    metadata={"strategy": self.name, "basis_z": z}
                ))

        return signals

    def should_exit(self, position: Position, panel: pd.DataFrame,
                     bar_i: int) -> bool:
        if position.bars_held >= self.cfg.get("max_hold_bars", 36):
            return True
        venue, asset = position.venue, position.asset
        basis_col = f"{venue}_{asset}_basis"
        if basis_col not in panel.columns:
            return False
        z = self._zscore(panel[basis_col].iloc[:bar_i+1], self.lb)
        return abs(z) > self.stop_z or abs(z) < self.exit_z
