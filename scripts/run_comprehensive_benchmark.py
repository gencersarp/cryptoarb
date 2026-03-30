#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import random
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardRunner
from scripts.run_param_matrix import load_panel
from strategies.spot_perp import SpotPerpFundingStrategy
from strategies.basis_revert import BasisMeanRevertStrategy
from strategies.perp_perp import PerpPerpDiffStrategy


def _cfg_live_like() -> Dict[str, Any]:
    return {
        "initial_capital": 100_000.0,
        "risk": {
            "max_drawdown_pct": 10.0,
            "daily_loss_limit_pct": 3.0,
            "per_strategy_capital_pct": 40.0,
            "max_total_leverage": 3.0,
            "margin_buffer_pct": 20.0,
            "unrealized_kill_switch_pct": 8.0,
            "max_exchange_exposure_pct": 50.0,
        },
        "execution": {
            "seed": 42,
            "maker_fill_prob": 0.45,
            "slippage_model": "vol_adv",
            "slippage_vol_mult": 0.10,
            "min_slippage_bps": 0.8,
            "max_slippage_bps": 18.0,
            "exchange_downtime_prob": 0.003,
            "partial_fill_prob": 0.05,
            "min_partial_fill_ratio": 0.5,
            "latency_bps": 0.5,
        },
    }


def _venue_cfg() -> Dict[str, Any]:
    return {"binance": {"maker_fee": 0.0002, "taker_fee": 0.0004, "borrow_rate_annual": 0.05}}


def _wf_cfg() -> Dict[str, Any]:
    return {
        "n_splits": 5,
        "train_ratio": 0.60,
        "test_ratio": 0.20,
        "randomize_start_offset": True,
        "offset_range_days": 7,
        "compute_is_metrics": True,
        "compute_cost_stress_metrics": True,
        "cost_stress_fee_mult": 1.5,
        "cost_stress_slip_mult": 1.5,
        "seed": 42,
    }


def _val_cfg() -> Dict[str, Any]:
    return {
        "min_oos_sharpe": 0.8,
        "max_is_oos_sharpe_gap": 0.75,
        "max_single_fold_return_pct": 0.30,
        "max_drawdown_limit_pct": 10.0,
        "min_trades_per_fold": 6,
        "perturbation_cv_threshold": 0.25,
        "require_cost_stress_profitability": True,
        "require_positive_oos_all_folds": False,
        "min_positive_oos_folds_pct": 0.8,
    }


def _strategy_param_grid(strategy: str):
    if strategy == "SpotPerpFunding":
        return [
            {
                "entry_funding_threshold": e,
                "exit_funding_threshold": x,
                "min_annualized_yield": y,
                "position_size_pct": p,
                "max_open_positions": 2,
                "use_basis_filter": bf,
                "basis_filter_z": 1.8,
                "zscore_lookback": 24,
                "max_hold_bars": h,
                "hedge_with_spot": True,
                "min_funding_persistence_bars": 2,
                "min_expected_edge_bps": 1.0,
                "inventory_risk_bps": 0.5,
            }
            for e, x, y, p, bf, h in itertools.product(
                [0.00008, 0.0001, 0.00015],
                [0.00005, 0.00008],
                [0.015, 0.03],
                [0.3, 0.4],
                [False, True],
                [21, 30],
            )
        ]
    if strategy == "BasisMeanRevert":
        return [
            {
                "zscore_lookback": lb,
                "entry_z": ez,
                "exit_z": xz,
                "stop_z": sz,
                "position_size_pct": p,
                "max_open_positions": 1,
                "max_hold_bars": 30,
            }
            for lb, ez, xz, sz, p in itertools.product([24, 36], [1.5, 2.0], [0.3, 0.5], [3.0, 3.5], [0.15, 0.2])
        ]
    if strategy == "PerpPerpDiff":
        return [
            {
                "min_funding_spread": ms,
                "zscore_lookback": lb,
                "entry_z": ez,
                "exit_z": 0.5,
                "position_size_pct": p,
                "max_open_positions": 1,
                "max_hold_bars": 12,
                "venues": ["binance", "bybit"],
            }
            for ms, lb, ez, p in itertools.product([0.00008, 0.00012], [24, 36], [1.5, 2.0], [0.15, 0.2])
        ]
    raise ValueError(strategy)


def _sample_grid(grid: List[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    if len(grid) <= n:
        return grid
    rng = random.Random(seed)
    idx = list(range(len(grid)))
    rng.shuffle(idx)
    return [grid[i] for i in idx[:n]]


def _build_strategy(strategy: str, asset: str, params: Dict[str, Any]):
    c = dict(params)
    if "venues" not in c:
        c["venues"] = ["binance"]
    c["assets"] = [asset]
    if strategy == "SpotPerpFunding":
        return SpotPerpFundingStrategy(c)
    if strategy == "BasisMeanRevert":
        return BasisMeanRevertStrategy(c)
    if strategy == "PerpPerpDiff":
        return PerpPerpDiffStrategy(c)
    raise ValueError(strategy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL", "AVAX", "LINK", "DOGE"])
    ap.add_argument("--strategies", nargs="+", default=["SpotPerpFunding", "BasisMeanRevert", "PerpPerpDiff"])
    ap.add_argument("--mode", choices=["full", "fast"], default="fast")
    ap.add_argument("--fast-grid-size", type=int, default=8)
    ap.add_argument("--grid-limit", type=int, default=0)
    ap.add_argument("--out-csv", default="../comprehensive_benchmark_results.csv")
    ap.add_argument("--out-json", default="../comprehensive_benchmark_results.json")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    all_runs: List[Dict[str, Any]] = []
    data_dir = Path(args.data_dir)

    for strategy in args.strategies:
        grid = _strategy_param_grid(strategy)
        if args.mode == "fast":
            grid = _sample_grid(grid, min(max(1, args.fast_grid_size), len(grid)), seed=42)
        if args.grid_limit and args.grid_limit > 0:
            grid = grid[: min(args.grid_limit, len(grid))]
        for params in grid:
            run_row = {"strategy": strategy, "params": params, "assets": {}}
            pass_all_assets = True
            mean_sharpes = []
            mean_returns = []
            worst_dds = []
            for asset in args.assets:
                panel = load_panel(data_dir, asset=asset)
                strat = _build_strategy(strategy, asset, params)
                out = WalkForwardRunner(
                    BacktestEngine([strat], panel, _cfg_live_like(), _venue_cfg()),
                    _wf_cfg(),
                    _val_cfg(),
                ).run()
                pf = out.get("pass_fail", {})
                agg = out.get("aggregate", {})
                asset_pass = bool(pf.get("all_pass", False))
                pass_all_assets = pass_all_assets and asset_pass
                run_row["assets"][asset] = {
                    "all_pass": asset_pass,
                    "mean_sharpe": agg.get("mean_sharpe", 0.0),
                    "mean_return": agg.get("mean_return", 0.0),
                    "worst_dd": agg.get("worst_dd", 0.0),
                    "total_n_trades": agg.get("total_n_trades", 0),
                    "checks": pf.get("checks", {}),
                    "diagnostics": pf.get("diagnostics", {}),
                }
                mean_sharpes.append(float(agg.get("mean_sharpe", 0.0)))
                mean_returns.append(float(agg.get("mean_return", 0.0)))
                worst_dds.append(float(agg.get("worst_dd", 0.0)))

            summary = {
                "strategy": strategy,
                "params": json.dumps(params, sort_keys=True),
                "pass_all_assets": pass_all_assets,
                "mean_sharpe_all_assets": float(sum(mean_sharpes) / max(1, len(mean_sharpes))),
                "mean_return_all_assets": float(sum(mean_returns) / max(1, len(mean_returns))),
                "worst_dd_all_assets": float(min(worst_dds) if worst_dds else 0.0),
            }
            rows.append(summary)
            run_row["summary"] = summary
            all_runs.append(run_row)

    out_df = pd.DataFrame(rows).sort_values(
        ["pass_all_assets", "mean_sharpe_all_assets", "mean_return_all_assets"],
        ascending=[False, False, False],
    )
    out_df.to_csv(args.out_csv, index=False)
    Path(args.out_json).write_text(json.dumps(all_runs, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o)))
    print(out_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
