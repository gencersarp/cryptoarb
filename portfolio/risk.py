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
        self.unrealized_kill_switch_pct = cfg.get("unrealized_kill_switch_pct", 8.0)
        self.max_leg_divergence_bps = cfg.get("max_leg_divergence_bps", 35.0)
        self.max_exchange_exposure_pct = cfg.get("max_exchange_exposure_pct", 50.0)
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

        unrealized_total = sum(getattr(p, "pnl", 0.0) for p in positions.values())
        unrealized_pct = (-unrealized_total / max(equity, 1.0)) * 100 if unrealized_total < 0 else 0.0
        flags["kill_switch_breached"] = unrealized_pct > self.unrealized_kill_switch_pct

        # Exchange concentration control
        exposure_by_exchange = {}
        for p in positions.values():
            exposure_by_exchange[p.venue] = exposure_by_exchange.get(p.venue, 0.0) + p.notional
        max_exchange_exposure_pct = 0.0
        if exposure_by_exchange:
            max_exchange_exposure_pct = max(v / max(equity, 1.0) * 100 for v in exposure_by_exchange.values())
        flags["exchange_exposure_breached"] = max_exchange_exposure_pct > self.max_exchange_exposure_pct

        self.bar_count += 1
        return flags

    def position_risk_score(self, equity: float, positions: dict) -> float:
        """Composite 0–100 risk score for the current portfolio state.

        Aggregates drawdown severity, leverage utilisation, and unrealized
        loss fraction into a single number. Scores above 70 indicate the
        portfolio is approaching at least one hard limit.
        """
        score = 0.0

        if self.peak_equity > 0:
            dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
            score += min(40.0, (dd_pct / self.max_dd_pct) * 40.0)

        total_notional = sum(p.notional for p in positions.values())
        leverage = total_notional / max(equity, 1.0)
        score += min(30.0, (leverage / self.max_leverage) * 30.0)

        unrealized = sum(getattr(p, "pnl", 0.0) for p in positions.values())
        if unrealized < 0:
            unrealized_pct = (-unrealized / max(equity, 1.0)) * 100
            score += min(30.0, (unrealized_pct / self.unrealized_kill_switch_pct) * 30.0)

        return min(100.0, score)

    def reset(self):
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.bar_count = 0
