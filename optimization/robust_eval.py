from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def hard_gate_checks(result: Dict[str, Any]) -> bool:
    checks = result.get("pass_fail", {}).get("checks", {})
    return bool(checks) and all(bool(v) for v in checks.values())


def aggregate_asset_results(asset_results: List[Dict[str, Any]]) -> Dict[str, float]:
    if not asset_results:
        return {
            "mean_sharpe": -999.0,
            "mean_return": -999.0,
            "worst_dd": -1.0,
            "std_sharpe": 999.0,
            "total_n_trades": 0.0,
            "mean_cost_stress_return": -999.0,
        }
    mean_sharpe = float(np.mean([r["aggregate"]["mean_sharpe"] for r in asset_results]))
    mean_return = float(np.mean([r["aggregate"]["mean_return"] for r in asset_results]))
    worst_dd = float(min(r["aggregate"]["worst_dd"] for r in asset_results))
    std_sharpe = float(np.mean([r["aggregate"]["std_sharpe"] for r in asset_results]))
    total_n_trades = float(np.sum([r["aggregate"]["total_n_trades"] for r in asset_results]))
    mean_cost_stress_return = float(
        np.mean(
            [
                np.mean([f.get("cost_stress_return", 0.0) for f in r.get("fold_results", [])] or [0.0])
                for r in asset_results
            ]
        )
    )
    return {
        "mean_sharpe": mean_sharpe,
        "mean_return": mean_return,
        "worst_dd": worst_dd,
        "std_sharpe": std_sharpe,
        "total_n_trades": total_n_trades,
        "mean_cost_stress_return": mean_cost_stress_return,
    }


def robust_objective(agg: Dict[str, float], perturbation_cv: float) -> float:
    # Lexicographic-style weighted objective after hard gates are satisfied.
    return (
        2.0 * agg["mean_sharpe"]
        + 120.0 * agg["mean_return"]
        + 40.0 * agg["mean_cost_stress_return"]
        - 80.0 * abs(agg["worst_dd"])
        - 0.8 * agg["std_sharpe"]
        - 0.6 * perturbation_cv
        + 0.003 * agg["total_n_trades"]
    )

