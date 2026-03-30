from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from optimization.stress_tests import StressTester
from strategies.spot_perp import SpotPerpFundingStrategy


def _panel(periods: int = 200) -> pd.DataFrame:
    ts = pd.date_range("2022-01-01", periods=periods, freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": [30_000.0] * periods,
            "binance_BTC_spot_close": [30_000.0] * periods,
            "binance_BTC_funding": [0.0002] * periods,
            "binance_BTC_basis": [0.0] * periods,
            "realized_vol": [0.01] * periods,
            "adv": [20_000_000.0] * periods,
        }
    )


def test_stress_suite_honors_enabled_flags_and_composite():
    panel = _panel()
    strategy = SpotPerpFundingStrategy(
        {
            "assets": ["BTC"],
            "venues": ["binance"],
            "entry_funding_threshold": 0.0001,
            "exit_funding_threshold": 0.00005,
            "min_annualized_yield": 0.01,
            "position_size_pct": 0.2,
            "max_open_positions": 1,
            "use_basis_filter": False,
        }
    )
    cfg = {
        "initial_capital": 100_000.0,
        "risk": {"max_drawdown_pct": 50.0, "daily_loss_limit_pct": 50.0, "per_strategy_capital_pct": 40.0},
        "execution": {"seed": 7, "maker_fill_prob": 0.0, "slippage_model": "fixed", "min_slippage_bps": 1.0, "max_slippage_bps": 1.0, "exchange_downtime_prob": 0.0},
    }
    venues = {"binance": {"maker_fee": 0.0002, "taker_fee": 0.0004, "borrow_rate_annual": 0.05}}
    engine = BacktestEngine([strategy], panel, cfg, venues)

    stress_cfg = {
        "enabled": True,
        "funding_flip_shock": {"enabled": True, "shock_multiplier": -3.0},
        "vol_spike": {"enabled": True, "vol_multiplier": 3.0},
        "cost_stress_15x": {"enabled": True, "fee_mult": 1.5, "slip_mult": 1.5},
        "worst_case_composite": {"enabled": True, "funding_mult": -3.0, "vol_mult": 3.0, "fee_mult": 1.5, "slip_mult": 1.5},
        "crisis_windows": {"enabled": False, "windows": []},
    }
    out = StressTester(engine, stress_cfg).run_all()
    assert "worst_case_composite" in out
    assert "cost_stress_15x" in out

