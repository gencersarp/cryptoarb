from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardRunner
from paper_trading.paper_trader import PaperTrader
from strategies.spot_perp import SpotPerpFundingStrategy


def _panel(periods: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC")
    # positive funding early, then lower for exits/reentries
    fr = [0.0004] * (periods // 2) + [0.00008] * (periods - periods // 2)
    return pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": [50_000.0] * periods,
            "binance_BTC_spot_close": [50_000.0] * periods,
            "binance_BTC_funding": fr,
            "binance_BTC_basis": [0.0] * periods,
            "realized_vol": [0.01] * periods,
            "adv": [10_000_000.0] * periods,
        }
    )


def _cfg():
    return {
        "initial_capital": 100_000.0,
        "risk": {"max_drawdown_pct": 50.0, "daily_loss_limit_pct": 50.0, "per_strategy_capital_pct": 40.0},
        "orphan_protection": {"enabled": False},
        "execution": {
            "seed": 7,
            "maker_fill_prob": 0.0,
            "slippage_model": "fixed",
            "min_slippage_bps": 1.0,
            "max_slippage_bps": 1.0,
            "exchange_downtime_prob": 0.0,
        },
    }


def _venue_cfg():
    return {"binance": {"maker_fee": 0.0002, "taker_fee": 0.0004, "borrow_rate_annual": 0.05}}


def _strategy():
    return SpotPerpFundingStrategy(
        {
            "enabled": True,
            "assets": ["BTC"],
            "venues": ["binance"],
            "entry_funding_threshold": 0.0002,
            "exit_funding_threshold": 0.0001,
            "min_annualized_yield": 0.01,
            "max_hold_bars": 21,
            "position_size_pct": 0.2,
            "max_open_positions": 1,
            "use_basis_filter": False,
            "zscore_lookback": 8,
        }
    )


def test_backtest_paper_accounting_parity():
    panel = _panel()
    cfg = _cfg()
    venues = _venue_cfg()
    strat = _strategy()

    engine = BacktestEngine([strat], panel, cfg, venues)
    bt = engine.run_fold(range(0, 40), range(40, 79), fold_id=1)

    trader = PaperTrader([_strategy()], cfg, venues, state_file="paper_state_test.json")
    trader.reset()
    for _, row in panel.iloc[40:79].iterrows():
        trader.tick(row)
    rpt = trader.report()
    # Same accounting model should keep equity return in close range.
    assert abs(bt["total_return"] - rpt["total_return"]) < 0.02
