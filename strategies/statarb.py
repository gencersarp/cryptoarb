"""
Cointegration-based Statistical Arbitrage (BTC/ETH pair).

Approach:
  1. Over rolling coint_lookback bars (IS window), run Engle-Granger test.
  2. If p-value < min_coint_pvalue, fit OLS hedge ratio.
  3. Compute spread = log(BTC) - beta * log(ETH).
  4. Enter when spread z-score exceeds entry_z.
  5. Exit when z-score reverts to exit_z or stop_z hit.

Bias guards:
  - Cointegration test and hedge ratio fitted only on past data.
  - No in-bar recomputation of beta (fixed within the IS window).
  - Spread series computed with lagged beta.

Limitations (flagged honestly):
  - Works best in low-vol, range-bound regimes.
  - Cointegration breaks during crisis periods.
  - Capacity limited to BTC/ETH on same venue.
"""
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from .base import BaseStrategy, Signal, Position, Side, Market


class StatArbStrategy(BaseStrategy):

    def __init__(self, cfg: dict):
        super().__init__("StatArb", cfg)
        self.coint_lb       = cfg.get("coint_lookback",    90)
        self.zscore_lb      = cfg.get("zscore_lookback",   20)
        self.entry_z        = cfg.get("entry_z",           2.0)
        self.exit_z         = cfg.get("exit_z",            0.3)
        self.stop_z         = cfg.get("stop_z",            3.5)
        self.min_p          = cfg.get("min_coint_pvalue",  0.05)
        self.max_open       = cfg.get("max_open_positions", 1)
        self.venues         = cfg.get("venues",            ["binance"])
        self._beta:  Optional[float] = None
        self._last_coint_bar: int    = -999

    def generate_signals(self, panel: pd.DataFrame, bar_i: int,
                         open_positions: List[Position]) -> List[Signal]:
        if bar_i < self.coint_lb:
            return []

        signals: List[Signal] = []
        venue = self.venues[0]

        btc_col = f"{venue}_BTC_perp_close"
        eth_col = f"{venue}_ETH_perp_close"
        if btc_col not in panel.columns or eth_col not in panel.columns:
            # Try generic columns
            btc_col = "BTC_perp_close"
            eth_col = "ETH_perp_close"
            if btc_col not in panel.columns or eth_col not in panel.columns:
                return []

        btc = panel[btc_col].iloc[:bar_i+1].dropna()
        eth = panel[eth_col].iloc[:bar_i+1].dropna()
        if len(btc) < self.coint_lb or len(eth) < self.coint_lb:
            return []

        # Refit cointegration every coint_lb bars
        if bar_i - self._last_coint_bar >= self.coint_lb:
            y  = np.log(btc.values[-self.coint_lb:])
            x  = np.log(eth.values[-self.coint_lb:])
            try:
                _, pval, _ = coint(y, x)
            except Exception:
                return []
            if pval > self.min_p:
                self._beta = None
                return []
            res = OLS(y, add_constant(x)).fit()
            self._beta = float(res.params[1])
            self._last_coint_bar = bar_i

        if self._beta is None:
            return []

        # Compute spread
        log_btc = np.log(btc.values[-self.zscore_lb-1:])
        log_eth = np.log(eth.values[-self.zscore_lb-1:])
        spread  = log_btc - self._beta * log_eth
        if len(spread) < 5:
            return []

        spread_s = pd.Series(spread)
        z = self._zscore(spread_s, min(self.zscore_lb, len(spread_s)))

        sid = f"{self.name}_{venue}_BTCETH"
        existing = next((p for p in open_positions if p.signal_id == sid), None)

        # Exit
        if existing is not None:
            if abs(z) > self.stop_z or abs(z) < self.exit_z:
                reason = "stop_z" if abs(z) > self.stop_z else "z_revert"
                signals.append(Signal(
                    signal_id=sid, strategy=self.name,
                    venue=venue, asset="BTC", market=Market.PERP,
                    side=Side.FLAT, metadata={"reason": reason}
                ))
            return signals

        n_open = len([p for p in open_positions
                       if p.metadata.get("strategy") == self.name])
        if n_open >= self.max_open:
            return signals
        if abs(z) < self.entry_z:
            return signals

        # z > 0 → spread high → BTC expensive relative to ETH → short BTC
        side = Side.SHORT if z > 0 else Side.LONG
        signals.append(Signal(
            signal_id=sid, strategy=self.name,
            venue=venue, asset="BTC", market=Market.PERP, side=side,
            confidence=min(abs(z) / self.entry_z, 2.0) / 2.0,
            metadata={"strategy": self.name, "z": z, "beta": self._beta}
        ))
        return signals
