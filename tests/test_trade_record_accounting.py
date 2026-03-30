from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from strategies.spot_perp import SpotPerpFundingStrategy


def _panel(periods: int = 90) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC")
    fr = [0.00035] * 45 + [0.00005] * 45
    return pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": [50_000.0] * periods,
            "binance_BTC_spot_close": [50_000.0] * periods,
            "binance_BTC_funding": fr,
            "binance_BTC_basis": [0.0] * periods,
            "realized_vol": [0.01] * periods,
            "adv": [12_000_000.0] * periods,
        }
    )


def test_trade_record_funding_is_not_cost_adjusted():
    cfg = {
        "initial_capital": 100_000.0,
        "risk": {"max_drawdown_pct": 50.0, "daily_loss_limit_pct": 50.0, "per_strategy_capital_pct": 40.0},
        "execution": {
            "seed": 4,
            "maker_fill_prob": 0.0,
            "slippage_model": "fixed",
            "min_slippage_bps": 1.0,
            "max_slippage_bps": 1.0,
            "exchange_downtime_prob": 0.0,
        },
    }
    venues = {"binance": {"maker_fee": 0.0002, "taker_fee": 0.0004, "borrow_rate_annual": 0.05}}
    strat = SpotPerpFundingStrategy(
        {
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
            "hedge_with_spot": True,
            "min_funding_persistence_bars": 2,
            "min_expected_edge_bps": 0.0,
        }
    )
    out = BacktestEngine([strat], _panel(), cfg, venues).run_fold(range(0, 10), range(10, 89), fold_id=1)
    trades = out.get("trades", [])
    assert trades, "expected at least one closed trade"
    for t in trades:
        expected_net = t.gross_pnl + t.funding_pnl - t.fee - t.slippage
        assert abs(t.net_pnl - expected_net) < 1e-8
