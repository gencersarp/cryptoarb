"""
Paper Trading / Dry-Run Mode.

Shares strategy logic with the backtest engine.
Differences from backtest:
  - Runs bar-by-bar on live or recent data.
  - Fills use the same ExecutionSimulator but with live prices.
  - Persists state (open positions, equity) to a JSON file between runs.
  - Logs every signal, fill, and P&L update.
  - Does NOT look ahead; each bar is processed as it arrives.

Usage (example):
    from paper_trading import PaperTrader
    trader = PaperTrader(strategies, cfg, venue_cfgs, state_file="paper_state.json")
    trader.tick(row)  # call once per new bar
    trader.report()   # print current P&L summary
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest.execution_sim import ExecutionSimulator
from strategies.base import BaseStrategy, Position, Side, Market
from portfolio.analytics import compute_metrics
from portfolio.risk import RiskEngine
from portfolio.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


class PaperTrader:

    STATE_VERSION = 1

    def __init__(
        self,
        strategies: List[BaseStrategy],
        cfg:        dict,
        venue_cfgs: dict,
        state_file: str = "paper_state.json",
    ):
        self.strategies  = strategies
        self.cfg         = cfg
        self.venue_cfgs  = venue_cfgs
        self.state_file  = state_file
        self.exec_sim    = ExecutionSimulator(cfg.get("execution", {}))
        self.risk_engine = RiskEngine(cfg.get("risk", {}))
        self.initial_cap = cfg.get("initial_capital", 100_000)
        self.last_leg_divergence_bps = 0.0
        self._halted = False
        self._halt_reason = ""
        hm_cfg = cfg.get("health_monitor", {"enabled": False})
        self.health_monitor = None
        if hm_cfg.get("enabled", False):
            self.health_monitor = HealthMonitor(
                poll_seconds=float(hm_cfg.get("poll_seconds", 2.0)),
                divergence_threshold_bps=float(cfg.get("risk", {}).get("max_leg_divergence_bps", 35.0)),
                fetch_metrics=self._health_metrics,
                on_emergency_close=self._emergency_close_callback,
            )
            self.health_monitor.start()

        self._state: Dict[str, Any] = self._load_state()
        self._bar_history: List[pd.Series] = []
        self._orphan_cfg = cfg.get("orphan_protection", {})
        self._orphan_enabled = bool(self._orphan_cfg.get("enabled", True))
        self._orphan_max_age_bars = int(self._orphan_cfg.get("max_unhedged_bars", 1))

    # ── Public API ────────────────────────────────────────────────────────

    def tick(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process one new bar. row must contain price/funding columns
        matching what strategies expect.
        """
        self._bar_history.append(row)
        panel = pd.DataFrame(self._bar_history)
        bar_i = len(panel) - 1
        ts    = pd.Timestamp.utcnow()

        capital   = self._state["capital"]
        positions = self._deserialise_positions()
        trades    = self._state.setdefault("trades", [])
        fills     = self._state.setdefault("fills", [])
        equity_h  = self._state.setdefault("equity_history", [capital])

        # Risk flags
        flags = self.risk_engine.update(capital, positions)
        if flags.get("kill_switch_breached"):
            logger.warning("PAPER: kill switch breached — closing all and halting.")
            capital = self._close_all(positions, row, capital, trades, ts, fills)
            self._halted = True
            self._halt_reason = "kill_switch"
            self._persist(capital, positions, trades, fills, equity_h)
            return {"action": "halt", "reason": "kill_switch", "flags": flags}
        if flags.get("max_drawdown_breached"):
            logger.warning("PAPER: max drawdown breached — closing all.")
            capital = self._close_all(positions, row, capital, trades, ts, fills)
            self._halt_reason = "max_drawdown"
            self._persist(capital, positions, trades, fills, equity_h)
            return {"action": "halt", "reason": "max_drawdown", "flags": flags}
        if flags.get("daily_loss_breached"):
            logger.warning("PAPER: daily loss limit breached — closing all and halting.")
            capital = self._close_all(positions, row, capital, trades, ts, fills)
            self._halted = True
            self._halt_reason = "daily_loss_limit"
            self._persist(capital, positions, trades, fills, equity_h)
            return {"action": "halt", "reason": "daily_loss_limit", "flags": flags}
        if self._halted:
            self._persist(capital, positions, trades, fills, equity_h)
            return {"action": "halt", "reason": self._halt_reason or "trading_halted", "flags": flags}

        # 1-bar delay parity with backtest: generate on previous bar, fill now.
        if bar_i >= 1:
            for strategy in self.strategies:
                open_pos = list(positions.values())
                signals  = strategy.generate_signals(panel, bar_i - 1, open_pos)

                for sig in signals:
                    if sig.side == Side.FLAT:
                        capital = self._close_position(
                            sig.signal_id, positions, row, capital, trades, ts,
                            reason="strategy_exit", fills=fills
                        )
                    elif sig.signal_id not in positions:
                        capital = self._open_position(
                            sig, strategy, row, capital, positions, ts, fills
                        )

        # Funding accrual
        for sid, pos in list(positions.items()):
            if pos.market == Market.PERP:
                fr = float(row.get(f"{pos.venue}_{pos.asset}_funding",
                                   row.get("funding_rate", 0.0)) or 0.0)
                if fr:
                    f_pnl = ExecutionSimulator.compute_funding_pnl(
                        pos.notional, fr,
                        "long" if pos.side == Side.LONG else "short"
                    )
                    pos.pnl += f_pnl
                    capital += f_pnl
                    logger.info(f"PAPER funding {sid}: {f_pnl:+.4f}")
            if pos.market == Market.SPOT and pos.side == Side.SHORT:
                venue_cfg = self.venue_cfgs.get(pos.venue, {})
                borrow = ExecutionSimulator.compute_borrow_cost(
                    pos.notional, venue_cfg.get("borrow_rate_annual", 0.05), 8.0
                )
                pos.pnl -= borrow
                capital -= borrow
            pos.bars_held += 1

        if self._orphan_enabled and positions:
            capital = self._enforce_orphan_protection(
                positions, row, capital, trades, ts, fills
            )
            if self._halted:
                self._persist(capital, positions, trades, fills, equity_h)
                return {
                    "action": "halt",
                    "reason": self._halt_reason or "orphan_leg_protection",
                    "flags": flags,
                }

        # MTM equity
        mtm = capital + sum(
            self._mark(p, row) for p in positions.values()
        )
        self.last_leg_divergence_bps = self._compute_leg_divergence_bps(positions, row)
        equity_h.append(mtm)

        self._persist(capital, positions, trades, fills, equity_h)
        logger.info(f"PAPER bar {bar_i}: equity={mtm:.2f} "
                    f"open_positions={len(positions)}")
        return {"equity": mtm, "n_positions": len(positions), "flags": flags}

    def report(self) -> Dict[str, Any]:
        equity_h = self._state.get("equity_history", [self.initial_cap])
        eq = pd.Series(equity_h)
        risk_free = float(self.cfg.get("risk_free_rate_annual", 0.0))
        metrics = compute_metrics(eq, self.initial_cap, risk_free_rate_annual=risk_free)
        print("\n=== Paper Trading Report ===")
        for k, v in metrics.items():
            print(f"  {k:<20}: {v:.4f}")
        print(f"  {'open_positions':<20}: {len(self._state.get('positions', {}))}")
        print(f"  {'total_trades':<20}: {len(self._state.get('trades', []))}")
        return metrics

    def reset(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        self._state = self._load_state()
        self._bar_history = []
        self._halted = False
        self._halt_reason = ""

    # ── Internals ───────────────────────────────────────────────────────────

    def _open_position(self, sig, strategy, row, capital, positions, ts, fills):
        price_col = (f"{sig.venue}_{sig.asset}_spot_close" if sig.market == Market.SPOT
                     else f"{sig.venue}_{sig.asset}_perp_close")
        price = float(row.get(price_col, row.get("perp_close", 0)) or 0)
        if price <= 0:
            return capital
        size = strategy.compute_position_size(sig, capital, price, {})
        if size <= 0:
            return capital
        venue_cfg = self.venue_cfgs.get(sig.venue, {})
        vol = float(row.get("realized_vol", 0.01) or 0.01)
        adv = float(row.get("adv", 1e6) or 1e6)
        fill = self.exec_sim.simulate_fill(
            sig.signal_id, sig.venue, sig.asset,
            sig.market.value, sig.side.value,
            size, price, vol, adv, venue_cfg, ts
        )
        if fill.missed:
            logger.warning(f"PAPER: missed fill for {sig.signal_id}")
            fills.append(
                {
                    "sid": sig.signal_id,
                    "action": "open",
                    "missed": True,
                    "is_maker": False,
                    "fee": 0.0,
                    "slippage": 0.0,
                    "ts": str(ts),
                }
            )
            return capital
        capital -= fill.fee + fill.slippage
        positions[sig.signal_id] = Position(
            signal_id=sig.signal_id, venue=sig.venue, asset=sig.asset,
            market=sig.market, side=sig.side,
            entry_price=fill.avg_fill_price, entry_ts=ts,
            size=fill.filled_size,
            notional=fill.filled_size * fill.avg_fill_price,
            pnl=-(fill.fee + fill.slippage), bars_held=0,
            metadata=sig.metadata
        )
        logger.info(f"PAPER OPEN {sig.signal_id}: price={fill.avg_fill_price:.2f} "
                    f"size={fill.filled_size:.6f} cost={fill.fee+fill.slippage:.4f}")
        fills.append(
            {
                "sid": sig.signal_id,
                "action": "open",
                "missed": False,
                "is_maker": bool(fill.is_maker),
                "fee": float(fill.fee),
                "slippage": float(fill.slippage),
                "ts": str(ts),
            }
        )
        return capital

    def _close_position(self, sid, positions, row, capital, trades, ts, reason, fills):
        pos = positions.get(sid)
        if pos is None:
            return capital
        price_col = (f"{pos.venue}_{pos.asset}_spot_close" if pos.market == Market.SPOT
                     else f"{pos.venue}_{pos.asset}_perp_close")
        exit_price = float(row.get(price_col, row.get("perp_close", pos.entry_price)) or pos.entry_price)
        venue_cfg = self.venue_cfgs.get(pos.venue, {})
        vol = float(row.get("realized_vol", 0.01) or 0.01)
        adv = float(row.get("adv", 1e6) or 1e6)
        exit_side = "short" if pos.side == Side.LONG else "long"
        fill = self.exec_sim.simulate_fill(
            sid, pos.venue, pos.asset, pos.market.value,
            exit_side, pos.size, exit_price, vol, adv, venue_cfg, ts,
            force_taker=True
        )
        price_pnl = (exit_price - pos.entry_price) * pos.size
        if pos.side == Side.SHORT:
            price_pnl = -price_pnl
        net_pnl = price_pnl + pos.pnl - fill.fee - fill.slippage
        # Align with backtest accounting: capital was not debited by entry notional.
        capital += net_pnl
        logger.info(f"PAPER CLOSE {sid}: net_pnl={net_pnl:.4f} reason={reason}")
        trades.append({"sid": sid, "net_pnl": net_pnl, "reason": reason,
                        "ts": str(ts)})
        fills.append(
            {
                "sid": sid,
                "action": "close",
                "missed": bool(fill.missed),
                "is_maker": bool(fill.is_maker),
                "fee": float(fill.fee),
                "slippage": float(fill.slippage),
                "ts": str(ts),
            }
        )
        del positions[sid]
        return capital

    def _close_all(self, positions, row, capital, trades, ts, fills):
        for sid in list(positions.keys()):
            capital = self._close_position(sid, positions, row, capital,
                                            trades, ts, reason="halt", fills=fills)
        return capital

    def _enforce_orphan_protection(self, positions, row, capital, trades, ts, fills):
        by_pair = {}
        for p in positions.values():
            pair = p.metadata.get("pair_id")
            if not pair:
                continue
            by_pair.setdefault(pair, []).append(p)

        for pair_id, legs in by_pair.items():
            if len(legs) >= 2:
                continue
            orphan_leg = legs[0]
            if orphan_leg.bars_held <= self._orphan_max_age_bars:
                continue
            logger.warning(
                "PAPER orphan protection: closing unhedged leg %s for pair %s",
                orphan_leg.signal_id,
                pair_id,
            )
            capital = self._close_position(
                orphan_leg.signal_id,
                positions,
                row,
                capital,
                trades,
                ts,
                reason="orphan_leg_protection",
                fills=fills,
            )
            self._halted = True
            self._halt_reason = "orphan_leg_protection"

        return capital

    @staticmethod
    def _mark(pos: Position, row: pd.Series) -> float:
        price_col = (f"{pos.venue}_{pos.asset}_spot_close" if pos.market == Market.SPOT
                     else f"{pos.venue}_{pos.asset}_perp_close")
        price = float(row.get(price_col, pos.entry_price) or pos.entry_price)
        mtm = (price - pos.entry_price) * pos.size
        if pos.side == Side.SHORT:
            mtm = -mtm
        return mtm + pos.pnl

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": self.STATE_VERSION, "capital": self.initial_cap,
                "positions": {}, "trades": [], "fills": [], "equity_history": [self.initial_cap]}

    def _persist(self, capital, positions, trades, fills, equity_h):
        serialised_pos = {}
        for sid, pos in positions.items():
            serialised_pos[sid] = {
                "signal_id": pos.signal_id, "venue": pos.venue,
                "asset": pos.asset, "market": pos.market.value,
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "entry_ts": str(pos.entry_ts),
                "size": pos.size, "notional": pos.notional,
                "pnl": pos.pnl, "bars_held": pos.bars_held,
                "metadata": pos.metadata,
            }
        state = {
            "version":        self.STATE_VERSION,
            "capital":        capital,
            "positions":      serialised_pos,
            "trades":         trades[-500:],    # keep last 500
            "fills":          fills[-2000:],
            "equity_history": equity_h[-2000:], # keep last 2000 bars
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)
        self._state = state

    def _deserialise_positions(self) -> Dict[str, Position]:
        positions = {}
        for sid, d in self._state.get("positions", {}).items():
            positions[sid] = Position(
                signal_id=d["signal_id"], venue=d["venue"],
                asset=d["asset"],
                market=Market(d["market"]), side=Side(d["side"]),
                entry_price=d["entry_price"],
                entry_ts=pd.Timestamp(d["entry_ts"]),
                size=d["size"], notional=d["notional"],
                pnl=d["pnl"], bars_held=d["bars_held"],
                metadata=d.get("metadata", {})
            )
        return positions

    def _compute_leg_divergence_bps(self, positions: Dict[str, Position], row: pd.Series) -> float:
        by_pair = {}
        for p in positions.values():
            pair = p.metadata.get("pair_id")
            if not pair:
                continue
            by_pair.setdefault(pair, []).append(p)
        worst = 0.0
        for plist in by_pair.values():
            if len(plist) < 2:
                continue
            p0 = plist[0]
            p1 = plist[1]
            c0 = float(row.get(f"{p0.venue}_{p0.asset}_{p0.market.value}_close", p0.entry_price) or p0.entry_price)
            c1 = float(row.get(f"{p1.venue}_{p1.asset}_{p1.market.value}_close", p1.entry_price) or p1.entry_price)
            d = abs(c0 - c1) / max((c0 + c1) * 0.5, 1e-9) * 10_000
            worst = max(worst, d)
        return worst

    def _health_metrics(self) -> Dict[str, Any]:
        return {"leg_divergence_bps": self.last_leg_divergence_bps}

    def _emergency_close_callback(self, reason: str) -> None:
        self._halted = True
        self._halt_reason = "health_monitor_emergency"
        logger.warning(f"PAPER emergency close triggered by health monitor: {reason}")
