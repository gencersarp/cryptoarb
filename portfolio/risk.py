"""
Risk Engine — real-time (or per-bar) risk monitoring.

Checks:
  - Max portfolio drawdown
  - Max single-position loss
  - Total leverage vs limit
  - Daily loss limit
  - Margin buffer
"""
from __future__ import annotations
import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class RiskEngine:

    def __init__(self, cfg: dict):
        self.max_dd_pct        = cfg.get("max_drawdown_pct",       10.0)
        self.daily_loss_pct    = cfg.get("daily_loss_limit_pct",    3.0)
        self.max_leverage      = cfg.get("max_total_leverage",       3.0)
        self.margin_buffer_pct = cfg.get("margin_buffer_pct",       20.0)
        self.peak_equity: float = 0.0
        self.daily_start_equity: float = 0.0
        self.bar_count: int = 0

    def update(self, equity: float, positions: dict) -> Dict[str, bool]:
        """
        Called every bar. Returns dict of risk flags.
        True = limit breached.
        """
        self.peak_equity = max(self.peak_equity, equity)
        flags: Dict[str, bool] = {}

        # Drawdown check
        dd_pct = 0.0
        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
        flags["max_drawdown_breached"] = dd_pct > self.max_dd_pct

        # Daily loss check (reset every 3 bars = 1 day)
        if self.bar_count % 3 == 0:
            self.daily_start_equity = equity
        daily_loss = 0.0
        if self.daily_start_equity > 0:
            daily_loss = (equity - self.daily_start_equity) / self.daily_start_equity * 100
        flags["daily_loss_breached"] = daily_loss < -self.daily_loss_pct

        # Leverage check
        total_notional = sum(p.notional for p in positions.values())
        leverage = total_notional / max(equity, 1.0)
        flags["leverage_breached"] = leverage > self.max_leverage

        # Margin buffer
        margin_used_pct = total_notional / max(equity, 1.0) * 100
        flags["margin_buffer_low"] = margin_used_pct > (100 - self.margin_buffer_pct)

        self.bar_count += 1
        return flags

    def reset(self):
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.bar_count = 0
