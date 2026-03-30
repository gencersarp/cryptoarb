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

from typing import List, Dict, Any
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
        self.hedge_with_spot = cfg.get("hedge_with_spot", True)
        self.min_funding_persistence = cfg.get("min_funding_persistence_bars", 2)
        self.min_expected_edge_bps = cfg.get("min_expected_edge_bps", 1.0)
        self.inventory_risk_bps = cfg.get("inventory_risk_bps", 0.5)
        self.expected_hold_bars = cfg.get("expected_hold_bars", 8)
        self.fee_bps_per_side = cfg.get("fee_bps_per_side", 4.0)
        self.slippage_bps_per_side = cfg.get("slippage_bps_per_side", 1.0)

    def generate_signals(self, panel: pd.DataFrame, bar_i: int,
                         open_positions: List[Position]) -> List[Signal]:
        if bar_i < self.lb:
            return []

        row     = panel.iloc[bar_i]
        signals: List[Signal] = []
        n_open  = len([p for p in open_positions if p.metadata.get("strategy") == self.name and p.market == Market.PERP])

        for venue in self.venues:
            for asset in self.assets:
                pair_id = f"{self.name}_{venue}_{asset}"
                perp_sid = f"{pair_id}_PERP" if self.hedge_with_spot else pair_id
                spot_sid = f"{pair_id}_SPOT"

                # ── Exit check ────────────────────────────────────────────
                existing_perp = next((p for p in open_positions if p.signal_id == perp_sid), None)
                existing_spot = next((p for p in open_positions if p.signal_id == spot_sid), None)
                if existing_perp is not None or existing_spot is not None:
                    fr = self._col(row, f"{venue}_{asset}_funding", "funding_rate")
                    if abs(fr) < self.exit_thr or np.isnan(fr):
                        if existing_perp is not None:
                            signals.append(Signal(
                                signal_id=perp_sid, strategy=self.name,
                                venue=venue, asset=asset,
                                market=Market.PERP, side=Side.FLAT,
                                metadata={"reason": "funding_exit", "strategy": self.name, "pair_id": pair_id}
                            ))
                        if existing_spot is not None:
                            signals.append(Signal(
                                signal_id=spot_sid, strategy=self.name,
                                venue=venue, asset=asset,
                                market=Market.SPOT, side=Side.FLAT,
                                metadata={"reason": "funding_exit", "strategy": self.name, "pair_id": pair_id}
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
                # Persistence guard: require funding to hold sign and threshold for recent bars
                fr_hist = panel[f"{venue}_{asset}_funding"].iloc[max(0, bar_i - self.min_funding_persistence + 1):bar_i + 1]
                if len(fr_hist) < self.min_funding_persistence:
                    continue
                if not np.all(np.sign(fr_hist.values) == np.sign(fr)):
                    continue
                if not np.all(np.abs(fr_hist.values) >= self.entry_thr):
                    continue

                # Basis filter — skip if basis is extremely stretched
                if self.use_basis:
                    basis_col = f"{venue}_{asset}_basis"
                    if basis_col in panel.columns:
                        bz = self._zscore(panel[basis_col].iloc[:bar_i+1], self.lb)
                        if abs(bz) > self.basis_z:
                            continue
                # Entry edge gate: projected funding over expected hold horizon
                # minus conservative round-trip execution + inventory costs.
                edge_bps = abs(fr) * 10_000 * max(1, int(self.expected_hold_bars))
                expected_cost_bps = (
                    2.0 * float(self.fee_bps_per_side)
                    + 2.0 * float(self.slippage_bps_per_side)
                    + float(self.inventory_risk_bps)
                )
                if edge_bps - expected_cost_bps < self.min_expected_edge_bps:
                    continue

                perp_side = Side.SHORT if fr > 0 else Side.LONG
                spot_side = Side.LONG if perp_side == Side.SHORT else Side.SHORT
                n_open += 1
                signals.append(Signal(
                    signal_id=perp_sid, strategy=self.name,
                    venue=venue, asset=asset,
                    market=Market.PERP, side=perp_side,
                    confidence=min(abs(fr) / self.entry_thr, 3.0) / 3.0,
                    metadata={"strategy": self.name, "funding_rate": fr,
                               "ann_yield": ann_yield, "pair_id": pair_id, "hedge_leg": "perp",
                               "entry_edge_bps": edge_bps - expected_cost_bps}
                ))
                if self.hedge_with_spot:
                    signals.append(Signal(
                        signal_id=spot_sid, strategy=self.name,
                        venue=venue, asset=asset,
                        market=Market.SPOT, side=spot_side,
                        confidence=min(abs(fr) / self.entry_thr, 3.0) / 3.0,
                        metadata={"strategy": self.name, "funding_rate": fr,
                                  "ann_yield": ann_yield, "pair_id": pair_id, "hedge_leg": "spot"}
                    ))

        return signals

    def compute_position_size(
        self,
        signal: Signal,
        capital: float,
        price: float,
        risk: Dict[str, Any],
    ) -> float:
        # Carry strategy sizing should be stable; confidence-scaling can underdeploy capital
        # and suppress expected funding income in normal regimes.
        pct = self.cfg.get("position_size_pct", 0.3)
        if self.hedge_with_spot and signal.metadata.get("hedge_leg") in ("perp", "spot"):
            pct *= 0.5
        notional = capital * pct
        return max(notional / max(price, 1e-9), 0.0)

    def should_exit(self, position: Position, panel: pd.DataFrame,
                     bar_i: int) -> bool:
        if position.bars_held >= self.cfg.get("max_hold_bars", 21):
            return True
        row = panel.iloc[bar_i]
        venue, asset = position.venue, position.asset
        fr = self._col(row, f"{venue}_{asset}_funding", "funding_rate")
        return (not np.isnan(fr)) and abs(fr) < self.exit_thr
