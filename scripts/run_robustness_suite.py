#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestEngine
from backtest.walk_forward import WalkForwardRunner
from optimization.param_search import ParameterSearch
from optimization.robust_eval import aggregate_asset_results, hard_gate_checks, robust_objective
from optimization.stress_tests import StressTester
from scripts.run_param_matrix import load_panel
from strategies.basis_revert import BasisMeanRevertStrategy
from strategies.perp_perp import PerpPerpDiffStrategy
from strategies.spot_perp import SpotPerpFundingStrategy


def get_strategy_cls(name: str):
    if name == "SpotPerpFunding":
        return SpotPerpFundingStrategy
    if name == "BasisMeanRevert":
        return BasisMeanRevertStrategy
    if name == "PerpPerpDiff":
        return PerpPerpDiffStrategy
    raise ValueError(f"Unknown strategy: {name}")


def _base_cfg(profile: str) -> Dict[str, Any]:
    execution_cfg = {
        "balanced": {
            "seed": 42,
            "maker_fill_prob": 0.7,
            "slippage_model": "fixed",
            "min_slippage_bps": 0.6,
            "max_slippage_bps": 6.0,
            "exchange_downtime_prob": 0.0,
        },
        "live_like": {
            "seed": 42,
            "maker_fill_prob": 0.45,
            "slippage_model": "vol_adv",
            "slippage_vol_mult": 0.10,
            "min_slippage_bps": 0.8,
            "max_slippage_bps": 18.0,
            "exchange_downtime_prob": 0.003,
        },
    }
    return {
        "initial_capital": 100_000.0,
        "risk": {"max_drawdown_pct": 35.0, "daily_loss_limit_pct": 15.0, "per_strategy_capital_pct": 50.0},
        "execution": execution_cfg[profile],
    }


def _venue_cfg() -> Dict[str, Any]:
    return {"binance": {"maker_fee": 0.0002, "taker_fee": 0.0004, "borrow_rate_annual": 0.05}}


def _wf_cfg() -> Dict[str, Any]:
    return {
        "n_splits": 5,
        "train_ratio": 0.60,
        "test_ratio": 0.20,
        "randomize_start_offset": True,
        "offset_range_days": 15,
        "compute_is_metrics": True,
        "compute_cost_stress_metrics": True,
        "cost_stress_fee_mult": 1.5,
        "cost_stress_slip_mult": 1.5,
        "seed": 42,
    }


def _val_cfg() -> Dict[str, Any]:
    return {
        "min_oos_sharpe": 0.8,
        "max_is_oos_sharpe_gap": 0.3,
        "max_single_fold_return_pct": 0.30,
        "max_drawdown_limit_pct": 10.0,
        "min_trades_per_fold": 10,
        "perturbation_cv_threshold": 0.25,
        "require_cost_stress_profitability": True,
        "require_positive_oos_all_folds": True,
        "min_positive_oos_folds_pct": 1.0,
    }


def _stress_cfg() -> Dict[str, Any]:
    return {
        "enabled": True,
        "funding_flip_shock": {"enabled": True, "shock_multiplier": -3.0},
        "vol_spike": {"enabled": True, "vol_multiplier": 3.0},
        "cost_stress_15x": {"enabled": True, "fee_mult": 1.5, "slip_mult": 1.5},
        "worst_case_composite": {"enabled": True, "funding_mult": -3.0, "vol_mult": 3.0, "fee_mult": 1.5, "slip_mult": 1.5},
    }


def _param_space(strategy: str) -> Dict[str, Any]:
    if strategy == "SpotPerpFunding":
        return {
            "entry_funding_threshold": ("float", 0.00008, 0.0004),
            "exit_funding_threshold": ("float", 0.00003, 0.00015),
            "min_annualized_yield": ("float", 0.01, 0.08),
            "position_size_pct": ("float", 0.15, 0.45),
            "max_hold_bars": ("int", 12, 48),
            "use_basis_filter": ("cat", [False, True]),
            "basis_filter_z": ("float", 1.2, 2.2),
            "hedge_with_spot": ("cat", [False, True]),
        }
    if strategy == "BasisMeanRevert":
        return {
            "zscore_lookback": ("int", 24, 72),
            "entry_z": ("float", 1.2, 3.0),
            "exit_z": ("float", 0.2, 0.8),
            "stop_z": ("float", 2.5, 4.5),
            "position_size_pct": ("float", 0.1, 0.35),
            "max_hold_bars": ("int", 12, 48),
        }
    if strategy == "PerpPerpDiff":
        return {
            "min_funding_spread": ("float", 0.00005, 0.00035),
            "zscore_lookback": ("int", 12, 48),
            "entry_z": ("float", 1.0, 3.0),
            "exit_z": ("float", 0.2, 0.8),
            "position_size_pct": ("float", 0.1, 0.35),
            "max_hold_bars": ("int", 3, 24),
        }
    raise ValueError(strategy)


def _build_strategy_cfg(strategy: str, params: Dict[str, Any], asset: str) -> Dict[str, Any]:
    base = {"assets": [asset], "venues": ["binance"], "max_open_positions": 2}
    if strategy == "SpotPerpFunding":
        base.update(
            {
                "entry_funding_threshold": params["entry_funding_threshold"],
                "exit_funding_threshold": params["exit_funding_threshold"],
                "min_annualized_yield": params["min_annualized_yield"],
                "position_size_pct": params["position_size_pct"],
                "max_hold_bars": params["max_hold_bars"],
                "use_basis_filter": params["use_basis_filter"],
                "basis_filter_z": params["basis_filter_z"],
                "zscore_lookback": 24,
                "hedge_with_spot": params["hedge_with_spot"],
            }
        )
    elif strategy == "BasisMeanRevert":
        base.update(
            {
                "zscore_lookback": params["zscore_lookback"],
                "entry_z": params["entry_z"],
                "exit_z": params["exit_z"],
                "stop_z": params["stop_z"],
                "position_size_pct": params["position_size_pct"],
                "max_hold_bars": params["max_hold_bars"],
            }
        )
    else:
        base.update(
            {
                "venues": ["binance", "bybit"],
                "min_funding_spread": params["min_funding_spread"],
                "zscore_lookback": params["zscore_lookback"],
                "entry_z": params["entry_z"],
                "exit_z": params["exit_z"],
                "position_size_pct": params["position_size_pct"],
                "max_hold_bars": params["max_hold_bars"],
            }
        )
    return base


def _evaluate_params(strategy_name: str, params: Dict[str, Any], data_dir: Path, assets: List[str], profile: str) -> Dict[str, Any]:
    strat_cls = get_strategy_cls(strategy_name)
    wf_results = []
    stress_rows = []

    for asset in assets:
        panel = load_panel(data_dir, asset=asset)
        strat = strat_cls(_build_strategy_cfg(strategy_name, params, asset))
        engine = BacktestEngine([strat], panel, _base_cfg(profile), _venue_cfg())
        wf = WalkForwardRunner(engine, _wf_cfg(), _val_cfg()).run()
        stress = StressTester(engine, _stress_cfg()).run_all()
        wf_results.append({"asset": asset, **wf})
        stress_rows.append(
            {
                "asset": asset,
                "funding_flip_sharpe": stress.get("funding_flip_shock", {}).get("sharpe", 0.0),
                "vol_spike_sharpe": stress.get("vol_spike", {}).get("sharpe", 0.0),
                "cost_stress_sharpe": stress.get("cost_stress_15x", {}).get("sharpe", 0.0),
                "worst_case_sharpe": stress.get("worst_case_composite", {}).get("sharpe", 0.0),
            }
        )

    agg = aggregate_asset_results(wf_results)
    all_hard_pass = all(hard_gate_checks(r) for r in wf_results)
    perturbation_cv = float(np.mean([r.get("aggregate", {}).get("std_sharpe", 1.0) for r in wf_results]))
    score = robust_objective(agg, perturbation_cv) if all_hard_pass else -999.0
    return {
        "params": params,
        "wf_results": wf_results,
        "stress_summary": stress_rows,
        "aggregate": agg,
        "all_hard_pass": all_hard_pass,
        "perturbation_cv_proxy": perturbation_cv,
        "score": score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--strategy", choices=["SpotPerpFunding", "BasisMeanRevert", "PerpPerpDiff"], default="SpotPerpFunding")
    ap.add_argument("--execution-profile", choices=["balanced", "live_like"], default="live_like")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--output-json", default="../results_robust_eval.json")
    ap.add_argument("--output-csv", default="../results_robust_eval.csv")
    ap.add_argument("--require_hard_pass", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ps = ParameterSearch({"n_trials": args.n_trials, "n_startup_trials": max(10, args.n_trials // 4), "seed": 42})
    pspace = _param_space(args.strategy)

    def objective(params: Dict[str, Any]) -> float:
        out = _evaluate_params(args.strategy, params, data_dir, args.assets, args.execution_profile)
        return out["score"]

    search = ps.search(objective, pspace)
    best_params = search["best_params"]
    best_eval = _evaluate_params(args.strategy, best_params, data_dir, args.assets, args.execution_profile)

    if args.require_hard_pass and not best_eval.get("all_hard_pass", False):
        raise SystemExit("No hard-pass configuration found in current search budget. Increase n-trials or widen space.")

    rows = []
    for r in best_eval["wf_results"]:
        agg = r.get("aggregate", {})
        di = r.get("pass_fail", {}).get("diagnostics", {})
        rows.append(
            {
                "strategy": args.strategy,
                "asset": r["asset"],
                "mean_sharpe": agg.get("mean_sharpe", 0.0),
                "mean_return": agg.get("mean_return", 0.0),
                "worst_dd": agg.get("worst_dd", 0.0),
                "total_n_trades": agg.get("total_n_trades", 0),
                "avg_is_oos_sharpe_gap": di.get("avg_is_oos_sharpe_gap", 0.0),
                "max_fold_return_share": di.get("max_fold_return_share", 1.0),
                "positive_oos_folds_pct": di.get("positive_oos_folds_pct", 0.0),
                "all_pass": r.get("pass_fail", {}).get("all_pass", False),
                "params": json.dumps(best_params),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output_csv, index=False)

    payload = {
        "search": search,
        "best_eval": best_eval,
        "strategy": args.strategy,
        "assets": args.assets,
        "execution_profile": args.execution_profile,
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
