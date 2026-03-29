"""
Execution Simulator.

Models:
  - Maker vs taker fill probability (maker_fill_prob)
  - Slippage as function of volatility and ADV (vol_adv model)
  - Exchange downtime / missed fills
  - Funding P&L accrual (per-bar, not continuous)
  - Borrow cost for spot-short positions
  - Force-taker at end-of-window close

Slippage model (vol_adv):
  slippage_bps = slippage_vol_mult * realized_vol * sqrt(size_usd / adv_usd)
  clipped to [min_slippage_bps, max_slippage_bps]

All costs returned as absolute USD amounts.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Fill:
    signal_id:      str
    venue:          str
    asset:          str
    market:         str
    side:           str
    requested_size: float
    filled_size:    float
    avg_fill_price: float
    fee:            float       # USD
    slippage:       float       # USD (always positive = cost)
    is_maker:       bool
    missed:         bool        # True if downtime caused no fill


class ExecutionSimulator:

    def __init__(self, cfg: dict):
        self.seed            = cfg.get("seed", 42)
        self.maker_fill_prob = cfg.get("maker_fill_prob", 0.60)
        self.slip_model      = cfg.get("slippage_model", "vol_adv")
        self.slip_vol_mult   = cfg.get("slippage_vol_mult", 0.10)
        self.min_slip_bps    = cfg.get("min_slippage_bps", 0.5)
        self.max_slip_bps    = cfg.get("max_slippage_bps", 50.0)
        self.downtime_prob   = cfg.get("exchange_downtime_prob", 0.005)
        self._rng            = np.random.default_rng(self.seed)

    # ── Main fill simulation ─────────────────────────────────────────────
    def simulate_fill(
        self,
        signal_id:   str,
        venue:       str,
        asset:       str,
        market:      str,
        side:        str,
        size:        float,
        ref_price:   float,
        realized_vol: float,
        adv_usd:     float,
        venue_cfg:   dict,
        ts:          pd.Timestamp,
        force_taker: bool = False,
        fee_mult:    float = 1.0,
        slip_mult:   float = 1.0,
    ) -> Fill:
        # Downtime check
        if self._rng.random() < self.downtime_prob and not force_taker:
            return Fill(
                signal_id=signal_id, venue=venue, asset=asset,
                market=market, side=side,
                requested_size=size, filled_size=0.0,
                avg_fill_price=ref_price, fee=0.0, slippage=0.0,
                is_maker=False, missed=True
            )

        # Maker vs taker
        is_maker = (not force_taker) and (self._rng.random() < self.maker_fill_prob)
        fee_rate = venue_cfg.get("maker_fee", 0.0002) if is_maker \
                   else venue_cfg.get("taker_fee", 0.0004)
        fee_rate *= fee_mult

        # Slippage
        notional = size * ref_price
        slip_bps = self._compute_slippage(realized_vol, notional, adv_usd)
        slip_bps = np.clip(slip_bps * slip_mult,
                           self.min_slip_bps, self.max_slip_bps)
        slip_usd = notional * slip_bps / 10_000

        # Adverse price impact (slippage moves fill price)
        direction = 1 if side in ("long", "buy") else -1
        fill_price = ref_price * (1 + direction * slip_bps / 10_000)

        fee_usd = notional * fee_rate

        return Fill(
            signal_id=signal_id, venue=venue, asset=asset,
            market=market, side=side,
            requested_size=size, filled_size=size,
            avg_fill_price=fill_price, fee=fee_usd, slippage=slip_usd,
            is_maker=is_maker, missed=False
        )

    # ── Cost components ───────────────────────────────────────────────────
    def _compute_slippage(self, vol: float, notional_usd: float,
                           adv_usd: float) -> float:
        """Returns slippage in bps."""
        if self.slip_model == "zero":
            return 0.0
        if self.slip_model == "fixed":
            return self.min_slip_bps
        # vol_adv model: Almgren-Chriss style
        participation = notional_usd / max(adv_usd, 1.0)
        slip = self.slip_vol_mult * vol * np.sqrt(participation) * 10_000
        return float(np.clip(slip, self.min_slip_bps, self.max_slip_bps))

    @staticmethod
    def compute_funding_pnl(notional: float, funding_rate: float,
                             side: str) -> float:
        """
        Funding P&L per settlement period.
        Long perp pays funding when rate > 0; receives when rate < 0.
        Short perp receives funding when rate > 0; pays when rate < 0.
        """
        if side == "short":
            return notional * funding_rate   # positive rate = income for short
        else:
            return -notional * funding_rate  # positive rate = cost for long

    @staticmethod
    def compute_borrow_cost(notional: float, annual_rate: float,
                             hours: float = 8.0) -> float:
        """Annualised borrow cost pro-rated to bar length."""
        return notional * annual_rate * (hours / 8760.0)
