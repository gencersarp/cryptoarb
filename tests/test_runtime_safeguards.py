from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from paper_trading.paper_trader import PaperTrader
from strategies.base import BaseStrategy, Signal, Side, Market


class _DailyLossStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("DailyLoss", {})
        self._opened = False

    def generate_signals(self, panel: pd.DataFrame, bar_i: int, open_positions):
        if self._opened:
            return []
        self._opened = True
        return [
            Signal(
                signal_id="DailyLoss_binance_BTC",
                strategy=self.name,
                venue="binance",
                asset="BTC",
                market=Market.PERP,
                side=Side.LONG,
                confidence=1.0,
                metadata={"strategy": self.name},
            )
        ]


class _OrphanLegStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("OrphanLeg", {})
        self._opened = False

    def generate_signals(self, panel: pd.DataFrame, bar_i: int, open_positions):
        if self._opened:
            return []
        self._opened = True
        return [
            Signal(
                signal_id="OrphanLeg_binance_BTC_PERP",
                strategy=self.name,
                venue="binance",
                asset="BTC",
                market=Market.PERP,
                side=Side.SHORT,
                confidence=1.0,
                metadata={"strategy": self.name, "pair_id": "OrphanLeg_binance_BTC"},
            )
        ]


def test_backtest_daily_loss_limit_forces_halt():
    ts = pd.date_range("2024-01-01", periods=8, freq="8h", tz="UTC")
    panel = pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": [100.0, 100.0, 100.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            "binance_BTC_spot_close": [100.0] * 8,
            "binance_BTC_funding": [0.0] * 8,
            "realized_vol": [0.01] * 8,
            "adv": [1_000_000.0] * 8,
        }
    )
    cfg = {
        "initial_capital": 100_000.0,
        "risk": {
            "max_drawdown_pct": 90.0,
            "daily_loss_limit_pct": 1.0,
            "per_strategy_capital_pct": 100.0,
        },
        "execution": {
            "seed": 7,
            "maker_fill_prob": 1.0,
            "slippage_model": "fixed",
            "min_slippage_bps": 0.0,
            "max_slippage_bps": 0.0,
            "exchange_downtime_prob": 0.0,
        },
    }
    venues = {"binance": {"maker_fee": 0.0, "taker_fee": 0.0, "borrow_rate_annual": 0.05}}
    out = BacktestEngine([_DailyLossStrategy()], panel, cfg, venues).run_fold(range(0, 1), range(1, 7), fold_id=1)
    trades = out.get("trades", [])
    assert trades, "expected forced close from daily loss stop"
    assert any(t.exit_reason == "daily_loss_limit" for t in trades)


def test_paper_orphan_leg_protection_closes_unhedged_pair():
    cfg = {
        "initial_capital": 100_000.0,
        "risk": {"max_drawdown_pct": 99.0, "daily_loss_limit_pct": 99.0},
        "orphan_protection": {"enabled": True, "max_unhedged_bars": 1},
        "execution": {
            "seed": 11,
            "maker_fill_prob": 1.0,
            "slippage_model": "fixed",
            "min_slippage_bps": 0.0,
            "max_slippage_bps": 0.0,
            "exchange_downtime_prob": 0.0,
        },
    }
    venues = {"binance": {"maker_fee": 0.0, "taker_fee": 0.0, "borrow_rate_annual": 0.05}}
    trader = PaperTrader([_OrphanLegStrategy()], cfg, venues, state_file="paper_state_test.json")
    trader.reset()

    rows = [
        pd.Series(
            {
                "binance_BTC_perp_close": 100.0,
                "binance_BTC_spot_close": 100.0,
                "binance_BTC_funding": 0.0,
                "realized_vol": 0.01,
                "adv": 1_000_000.0,
            }
        )
        for _ in range(4)
    ]
    trader.tick(rows[0])  # warmup
    trader.tick(rows[1])  # entry fill
    trader.tick(rows[2])  # orphan age -> 1
    trader.tick(rows[3])  # orphan protection should trigger
    out = trader.tick(rows[3])  # halted state is observed on next tick

    assert out.get("action") == "halt"
    assert out.get("reason") in {"trading_halted", "daily_loss_limit", "max_drawdown"}
    assert len(trader._state.get("positions", {})) == 0
    assert any(t.get("reason") == "orphan_leg_protection" for t in trader._state.get("trades", []))
