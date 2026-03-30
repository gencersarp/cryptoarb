from __future__ import annotations

import pandas as pd

from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardRunner
from strategies.spot_perp import SpotPerpFundingStrategy


def _panel(periods: int = 180) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=periods, freq="8h", tz="UTC")
    fr = [0.00035] * 60 + [0.0002] * 60 + [0.00005] * 60
    return pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": [50_000.0] * periods,
            "binance_BTC_spot_close": [50_000.0] * periods,
            "binance_BTC_funding": fr,
            "binance_BTC_basis": [0.0] * periods,
            "realized_vol": [0.01] * periods,
            "adv": [20_000_000.0] * periods,
        }
    )


def test_walk_forward_outputs_robustness_diagnostics():
    panel = _panel()
    strategy = SpotPerpFundingStrategy(
        {
            "assets": ["BTC"],
            "venues": ["binance"],
            "entry_funding_threshold": 0.0002,
            "exit_funding_threshold": 0.0001,
            "min_annualized_yield": 0.02,
            "position_size_pct": 0.20,
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
    wf_cfg = {
        "n_splits": 3,
        "train_ratio": 0.6,
        "test_ratio": 0.2,
        "randomize_start_offset": True,
        "offset_range_days": 2,
        "compute_is_metrics": True,
        "compute_cost_stress_metrics": True,
        "cost_stress_fee_mult": 1.5,
        "cost_stress_slip_mult": 1.5,
        "seed": 1,
    }
    val_cfg = {
        "min_oos_sharpe": 0.0,
        "max_is_oos_sharpe_gap": 10.0,
        "max_single_fold_return_pct": 1.0,
        "max_drawdown_limit_pct": 100.0,
        "min_trades_per_fold": 0,
        "perturbation_cv_threshold": 10.0,
        "require_cost_stress_profitability": False,
        "require_positive_oos_all_folds": False,
        "min_positive_oos_folds_pct": 0.0,
    }
    runner = WalkForwardRunner(BacktestEngine([strategy], panel, cfg, venues), wf_cfg, val_cfg)
    out = runner.run()
    assert out["fold_results"]
    assert "diagnostics" in out["pass_fail"]
    assert "avg_is_oos_sharpe_gap" in out["pass_fail"]["diagnostics"]

