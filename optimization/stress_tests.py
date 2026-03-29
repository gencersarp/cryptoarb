"""
Stress Test Suite.

Tests:
  1. Funding flip shock — funding rates multiplied by -3x
  2. Volatility spike — realized vol multiplied by 3x
  3. Cost stress 1.5x — fees and slippage 1.5x
  4. Crisis windows — COVID crash, LUNA, FTX, BTC bear 2022
  5. Parameter perturbation robustness
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from portfolio.analytics import compute_metrics

logger = logging.getLogger(__name__)


class StressTester:

    def __init__(self, engine: BacktestEngine, stress_cfg: dict):
        self.engine     = engine
        self.cfg        = stress_cfg

    def run_all(self) -> Dict[str, Any]:
        results = {}

        if self.cfg.get("funding_flip_shock", {}).get("enabled", False):
            results["funding_flip_shock"] = self._funding_flip_shock()

        if self.cfg.get("vol_spike", {}).get("enabled", False):
            results["vol_spike"] = self._vol_spike()

        if self.cfg.get("cost_stress_15x", {}).get("enabled", False):
            results["cost_stress_15x"] = self._cost_stress()

        if self.cfg.get("crisis_windows", {}).get("enabled", False):
            results["crisis_windows"] = self._crisis_windows()

        return results

    def _funding_flip_shock(self) -> Dict:
        mult  = self.cfg["funding_flip_shock"].get("shock_multiplier", -3.0)
        panel = self.engine.panel.copy()
        fr_cols = [c for c in panel.columns if "funding" in c]
        for c in fr_cols:
            panel[c] = panel[c] * mult
        return self._run_with_panel(panel, "funding_flip_shock")

    def _vol_spike(self) -> Dict:
        mult  = self.cfg["vol_spike"].get("vol_multiplier", 3.0)
        panel = self.engine.panel.copy()
        vol_cols = [c for c in panel.columns if "vol" in c]
        for c in vol_cols:
            panel[c] = panel[c] * mult
        return self._run_with_panel(panel, "vol_spike")

    def _cost_stress(self) -> Dict:
        fee_mult  = self.cfg["cost_stress_15x"].get("fee_mult",  1.5)
        slip_mult = self.cfg["cost_stress_15x"].get("slip_mult", 1.5)
        # Patch venue cfgs
        stressed_venues = {}
        for v, cfg in self.engine.venue_cfgs.items():
            sc = copy.deepcopy(cfg)
            sc["maker_fee"] *= fee_mult
            sc["taker_fee"] *= fee_mult
            stressed_venues[v] = sc
        stressed_engine = copy.deepcopy(self.engine)
        stressed_engine.venue_cfgs = stressed_venues
        # Also patch exec_sim slippage multiplier
        stressed_engine.exec_sim.slip_vol_mult *= slip_mult
        stressed_engine.exec_sim.min_slip_bps  *= slip_mult
        n = len(self.engine.panel)
        result = stressed_engine.run_fold(
            range(0, int(n * 0.6)), range(int(n * 0.6), n), fold_id=99
        )
        return {k: v for k, v in result.items() if k != "trades"}

    def _crisis_windows(self) -> Dict:
        windows = self.cfg["crisis_windows"].get("windows", [])
        results = {}
        panel   = self.engine.panel

        ts_col  = "ts" if "ts" in panel.columns else panel.index.name
        if ts_col and ts_col in panel.columns:
            panel_ts = pd.to_datetime(panel[ts_col], utc=True)
        elif hasattr(panel.index, "tz"):
            panel_ts = pd.to_datetime(panel.index, utc=True)
        else:
            logger.warning("No timestamp column found for crisis window slicing.")
            return results

        for w in windows:
            name  = w["name"]
            start = pd.Timestamp(w["start"], tz="UTC")
            end   = pd.Timestamp(w["end"],   tz="UTC")
            mask  = (panel_ts >= start) & (panel_ts <= end)
            if mask.sum() < 10:
                results[name] = {"error": "insufficient_bars"}
                continue
            sub_panel = panel.loc[mask].reset_index(drop=True)
            sub_engine = copy.deepcopy(self.engine)
            sub_engine.panel = sub_panel
            n = len(sub_panel)
            result = sub_engine.run_fold(
                range(0, int(n * 0.5)), range(int(n * 0.5), n), fold_id=99
            )
            results[name] = {k: v for k, v in result.items() if k != "trades"}

        return results

    def _run_with_panel(self, panel: pd.DataFrame, label: str) -> Dict:
        engine = copy.deepcopy(self.engine)
        engine.panel = panel
        n = len(panel)
        result = engine.run_fold(
            range(0, int(n * 0.6)), range(int(n * 0.6), n), fold_id=99
        )
        return {k: v for k, v in result.items() if k != "trades"}
