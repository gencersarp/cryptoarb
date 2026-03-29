"""
Walk-forward backtest engine.

Design:
  - Strict train/test separation: optimize on IS, evaluate on OOS only
  - 1-bar execution delay enforced at signal→fill boundary
  - Funding P&L settled only at funding bars (not continuously)
  - Forced close at end of every test window
  - Full cost accounting: fee + slippage + borrow + funding
  - Mark-to-market at every bar
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from .execution_sim import ExecutionSimulator, Fill
from strategies.base import BaseStrategy, Signal, Position, Side, Market
from portfolio.analytics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    signal_id:   str
    venue:       str
    asset:       str
    market:      str
    entry_ts:    pd.Timestamp
    exit_ts:     Optional[pd.Timestamp]
    entry_price: float
    exit_price:  float
    size:        float
    gross_pnl:   float
    funding_pnl: float
    fee:         float
    slippage:    float
    borrow_cost: float
    net_pnl:     float
    bars_held:   int
    exit_reason: str
    metadata:    dict = field(default_factory=dict)


class BacktestEngine:

    def __init__(self, strategies: List[BaseStrategy], panel: pd.DataFrame,
                 cfg: dict, venue_cfgs: dict):
        self.strategies  = strategies
        self.panel       = panel
        self.cfg         = cfg
        self.venue_cfgs  = venue_cfgs
        self.exec_sim    = ExecutionSimulator(cfg.get("execution", {}))
        self.initial_cap = cfg.get("initial_capital", 100_000)
        self.risk_cfg    = cfg.get("risk", {})

    def run_fold(self, train_idx: range, test_idx: range,
                 fold_id: int) -> Dict[str, Any]:
        """
        Run one walk-forward fold.
        train_idx: bars available for parameter context (IS)
        test_idx:  bars evaluated for OOS performance
        """
        capital   = self.initial_cap
        equity    = [capital]
        trades:   List[TradeRecord] = []
        positions: Dict[str, Position] = {}

        panel_test = self.panel.iloc[list(test_idx)].reset_index(drop=False)
        if "ts" not in panel_test.columns and "index" in panel_test.columns:
            panel_test = panel_test.rename(columns={"index": "ts"})

        for bar_i in range(1, len(panel_test)):
            row      = panel_test.iloc[bar_i]
            prev_row = panel_test.iloc[bar_i - 1]
            ts       = row.get("ts", pd.Timestamp("now"))

            # ── Risk check ──────────────────────────────────────────────────
            peak = max(equity)
            current_drawdown = (peak - capital) / peak * 100 if peak > 0 else 0
            if current_drawdown > self.risk_cfg.get("max_drawdown_pct", 10.0):
                logger.warning(f"Fold {fold_id} hit max drawdown at bar {bar_i}. "
                                "Force closing all positions.")
                capital = self._close_all(positions, row, capital, trades, ts,
                                           reason="max_drawdown_stop")
                break

            # ── Generate signals from previous bar (1-bar delay) ─────────
            for strategy in self.strategies:
                open_pos = [p for p in positions.values()
                            if p.metadata.get("strategy") == strategy.name
                            or p.signal_id.startswith(strategy.name)]
                signals = strategy.generate_signals(
                    panel_test, bar_i - 1, list(open_pos)
                )

                for sig in signals:
                    if sig.side == Side.FLAT:
                        capital = self._close_position(
                            sig.signal_id, positions, row, capital,
                            trades, ts, reason="strategy_exit"
                        )
                    elif sig.signal_id not in positions:
                        capital = self._open_position(
                            sig, strategy, row, capital, positions, ts
                        )

            # ── Accrue funding P&L on open positions ────────────────────
            for sid, pos in list(positions.items()):
                # Try venue-specific funding column first
                fr_col = f"{pos.venue}_{pos.asset}_funding"
                funding_rate = float(row.get(fr_col,
                                    row.get("funding_rate", 0.0)) or 0.0)
                if not np.isnan(funding_rate) and funding_rate != 0.0:
                    f_pnl = ExecutionSimulator.compute_funding_pnl(
                        pos.notional, funding_rate,
                        "long" if pos.side == Side.LONG else "short"
                    )
                    pos.pnl += f_pnl
                    capital += f_pnl

                # Borrow cost for spot short
                if pos.market == Market.SPOT and pos.side == Side.SHORT:
                    venue_cfg = self.venue_cfgs.get(pos.venue, {})
                    borrow = ExecutionSimulator.compute_borrow_cost(
                        pos.notional,
                        venue_cfg.get("borrow_rate_annual", 0.05),
                        8.0
                    )
                    pos.pnl -= borrow
                    capital -= borrow

                pos.bars_held += 1

                # Strategy-driven exit check
                strat = next((s for s in self.strategies
                               if s.name in pos.signal_id), None)
                if strat and strat.should_exit(pos, panel_test, bar_i):
                    capital = self._close_position(
                        sid, positions, row, capital, trades, ts,
                        reason="max_hold"
                    )

            # Mark-to-market
            mtm = capital + sum(
                self._mark_position(p, row) for p in positions.values()
            )
            equity.append(mtm)

            # Daily loss limit
            if len(equity) >= 2 and equity[-2] > 0:
                daily_loss = (equity[-1] - equity[-2]) / equity[-2] * 100
                if daily_loss < -self.risk_cfg.get("daily_loss_limit_pct", 3.0):
                    logger.warning(f"Daily loss limit hit at bar {bar_i}")

        # Force close remaining at end of test window
        if positions:
            last_row = panel_test.iloc[-1]
            capital = self._close_all(positions, last_row, capital, trades,
                                       last_row.get("ts", pd.Timestamp("now")),
                                       reason="end_of_window")

        equity_series = pd.Series(equity)
        metrics = compute_metrics(equity_series, self.initial_cap)
        metrics["n_trades"]  = len(trades)
        metrics["fold_id"]   = fold_id
        metrics["trades"]    = trades

        return metrics

    # ── Position helpers ──────────────────────────────────────────────────

    def _open_position(self, sig: Signal, strategy: BaseStrategy,
                        row: pd.Series, capital: float,
                        positions: dict, ts: pd.Timestamp) -> float:
        price_col = (f"{sig.venue}_{sig.asset}_spot_close"
                     if sig.market == Market.SPOT
                     else f"{sig.venue}_{sig.asset}_perp_close")
        price = float(row.get(price_col, row.get(
            "spot_close" if sig.market == Market.SPOT else "perp_close", 0.0
        )) or 0.0)
        if price <= 0:
            return capital

        size = strategy.compute_position_size(
            sig, capital, price,
            {"per_strategy_capital_pct":
             self.cfg.get("risk", {}).get("per_strategy_capital_pct", 40.0) / 100}
        )
        if size <= 0:
            return capital

        venue_cfg = self.venue_cfgs.get(sig.venue,
                        {"maker_fee": 0.0002, "taker_fee": 0.0004})
        vol = float(row.get("realized_vol", row.get("vol", 0.01)) or 0.01)
        adv = float(row.get("adv", 1e6) or 1e6)

        fill = self.exec_sim.simulate_fill(
            sig.signal_id, sig.venue, sig.asset,
            sig.market.value, sig.side.value,
            size, price, vol, adv, venue_cfg, ts
        )
        if fill.missed:
            return capital

        cost = fill.fee + fill.slippage
        capital -= cost

        positions[sig.signal_id] = Position(
            signal_id=sig.signal_id, venue=sig.venue, asset=sig.asset,
            market=sig.market, side=sig.side,
            entry_price=fill.avg_fill_price, entry_ts=ts,
            size=fill.filled_size,
            notional=fill.filled_size * fill.avg_fill_price,
            pnl=-cost, bars_held=0, metadata=sig.metadata
        )
        return capital

    def _close_position(self, sid: str, positions: dict, row: pd.Series,
                         capital: float, trades: list, ts: pd.Timestamp,
                         reason: str) -> float:
        pos = positions.get(sid)
        if pos is None:
            return capital

        price_col = (f"{pos.venue}_{pos.asset}_spot_close"
                     if pos.market == Market.SPOT
                     else f"{pos.venue}_{pos.asset}_perp_close")
        exit_price = float(row.get(price_col, row.get(
            "spot_close" if pos.market == Market.SPOT else "perp_close",
            pos.entry_price
        )) or pos.entry_price)
        if exit_price <= 0:
            exit_price = pos.entry_price

        venue_cfg = self.venue_cfgs.get(pos.venue,
                        {"maker_fee": 0.0002, "taker_fee": 0.0004})
        vol = float(row.get("realized_vol", 0.01) or 0.01)
        adv = float(row.get("adv", 1e6) or 1e6)

        exit_side = "short" if pos.side == Side.LONG else "long"
        fill = self.exec_sim.simulate_fill(
            sid, pos.venue, pos.asset, pos.market.value,
            exit_side, pos.size, exit_price, vol, adv, venue_cfg, ts,
            force_taker=(reason in ("end_of_window", "max_drawdown_stop"))
        )

        price_pnl = (exit_price - pos.entry_price) * pos.size
        if pos.side == Side.SHORT:
            price_pnl = -price_pnl

        net_pnl = price_pnl + pos.pnl - fill.fee - fill.slippage
        capital += net_pnl + pos.size * pos.entry_price  # return notional

        trades.append(TradeRecord(
            signal_id=sid, venue=pos.venue, asset=pos.asset,
            market=pos.market.value, entry_ts=pos.entry_ts, exit_ts=ts,
            entry_price=pos.entry_price, exit_price=exit_price,
            size=pos.size, gross_pnl=price_pnl,
            funding_pnl=pos.pnl + fill.fee + fill.slippage,
            fee=fill.fee, slippage=fill.slippage, borrow_cost=0.0,
            net_pnl=net_pnl, bars_held=pos.bars_held,
            exit_reason=reason, metadata=pos.metadata
        ))
        del positions[sid]
        return capital

    def _close_all(self, positions: dict, row: pd.Series, capital: float,
                    trades: list, ts: pd.Timestamp, reason: str) -> float:
        for sid in list(positions.keys()):
            capital = self._close_position(sid, positions, row, capital,
                                            trades, ts, reason)
        return capital

    def _mark_position(self, pos: Position, row: pd.Series) -> float:
        price_col = (f"{pos.venue}_{pos.asset}_spot_close"
                     if pos.market == Market.SPOT
                     else f"{pos.venue}_{pos.asset}_perp_close")
        price = float(row.get(price_col, row.get(
            "spot_close" if pos.market == Market.SPOT else "perp_close",
            pos.entry_price
        )) or pos.entry_price)
        mtm = (price - pos.entry_price) * pos.size
        if pos.side == Side.SHORT:
            mtm = -mtm
        return mtm + pos.pnl
