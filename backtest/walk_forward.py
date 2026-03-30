"""
Walk-Forward Runner.

Splitting logic:
  - n_splits folds, each with train_ratio / val_ratio / test_ratio
  - Optional randomised start offset to reduce split-boundary sensitivity
  - IS (train+val) used only for parameter optimisation
  - OOS (test) used only for evaluation
  - Each fold reports per-fold metrics and pass/fail vs validation criteria
"""
from __future__ import annotations

import copy
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from .engine import BacktestEngine
from portfolio.analytics import compute_metrics, decompose_pnl

logger = logging.getLogger(__name__)


class WalkForwardRunner:

    def __init__(self, engine: BacktestEngine, wf_cfg: dict, val_cfg: dict):
        self.engine  = engine
        self.wf_cfg  = wf_cfg
        self.val_cfg = val_cfg

    def run(self) -> Dict[str, Any]:
        panel      = self.engine.panel
        n          = len(panel)
        n_splits   = self.wf_cfg.get("n_splits", 5)
        train_r    = self.wf_cfg.get("train_ratio", 0.60)
        test_r     = self.wf_cfg.get("test_ratio",  0.20)
        rng_offset = self.wf_cfg.get("randomize_start_offset", True)
        offset_rng = self.wf_cfg.get("offset_range_days", 15)
        bars_per_day = 3  # 8h bars

        fold_size   = n // n_splits
        train_bars  = int(fold_size * train_r)
        test_bars   = int(fold_size * test_r)

        fold_results: List[Dict[str, Any]] = []
        rng = np.random.default_rng(self.wf_cfg.get("seed", 42))

        for fold_id in range(n_splits):
            base_start = fold_id * fold_size

            if rng_offset:
                offset = int(rng.integers(-offset_rng * bars_per_day,
                                           offset_rng * bars_per_day))
                base_start = max(0, min(base_start + offset,
                                        n - train_bars - test_bars - 1))

            train_end = base_start + train_bars
            test_end  = min(train_end + test_bars, n)

            if test_end - train_end < 10:
                logger.warning(f"Fold {fold_id}: test window too small, skipping.")
                continue

            train_idx = range(base_start, train_end)
            test_idx  = range(train_end, test_end)

            logger.info(
                f"Fold {fold_id}: train [{base_start}:{train_end}] "
                f"test [{train_end}:{test_end}]"
            )

            metrics = self.engine.run_fold(train_idx, test_idx, fold_id)
            # IS metric for overfitting gap diagnostics.
            if self.wf_cfg.get("compute_is_metrics", True):
                is_metrics = self.engine.run_fold(train_idx, train_idx, -1000 - fold_id)
                metrics["is_sharpe"] = is_metrics.get("sharpe", 0.0)
                metrics["is_total_return"] = is_metrics.get("total_return", 0.0)

            # Optional cost-stress metric directly on OOS window.
            if self.wf_cfg.get("compute_cost_stress_metrics", True):
                fee_mult = self.wf_cfg.get("cost_stress_fee_mult", 1.5)
                slip_mult = self.wf_cfg.get("cost_stress_slip_mult", 1.5)
                stressed_engine = copy.deepcopy(self.engine)
                stressed_venues = {}
                for venue, venue_cfg in stressed_engine.venue_cfgs.items():
                    sc = copy.deepcopy(venue_cfg)
                    sc["maker_fee"] *= fee_mult
                    sc["taker_fee"] *= fee_mult
                    stressed_venues[venue] = sc
                stressed_engine.venue_cfgs = stressed_venues
                stressed_engine.exec_sim.slip_vol_mult *= slip_mult
                stressed_engine.exec_sim.min_slip_bps *= slip_mult
                stressed = stressed_engine.run_fold(train_idx, test_idx, 1000 + fold_id)
                metrics["cost_stress_return"] = stressed.get("total_return", 0.0)
                metrics["cost_stress_sharpe"] = stressed.get("sharpe", 0.0)

            metrics["fold_start"] = int(train_end)   # OOS start bar
            metrics["fold_end"]   = int(test_end)

            # PnL decomposition
            trades = metrics.get("trades", [])
            if trades:
                metrics["pnl_decomp"] = decompose_pnl(trades)
            else:
                metrics["pnl_decomp"] = {}

            fold_results.append(metrics)

        if not fold_results:
            return {"fold_results": [], "aggregate": {}, "pass_fail": {}}

        aggregate  = self._aggregate(fold_results)
        pass_fail  = self._validate(fold_results, aggregate)

        return {
            "fold_results": fold_results,
            "aggregate":    aggregate,
            "pass_fail":    pass_fail,
        }

    def _aggregate(self, folds: List[Dict]) -> Dict[str, float]:
        sharpes   = [f["sharpe"]       for f in folds]
        mdd_list  = [f["max_drawdown"] for f in folds]
        returns   = [f["total_return"] for f in folds]
        n_trades  = [f["n_trades"]     for f in folds]
        return {
            "mean_sharpe":     float(np.mean(sharpes)),
            "min_sharpe":      float(np.min(sharpes)),
            "std_sharpe":      float(np.std(sharpes)),
            "mean_max_dd":     float(np.mean(mdd_list)),
            "worst_dd":        float(np.min(mdd_list)),
            "mean_return":     float(np.mean(returns)),
            "total_n_trades":  int(np.sum(n_trades)),
            "n_folds":         len(folds),
        }

    def _validate(self, folds: List[Dict], agg: Dict) -> Dict[str, Any]:
        vc      = self.val_cfg
        sharpes = [f["sharpe"] for f in folds]
        returns = [f["total_return"] for f in folds]
        mdd     = [f["max_drawdown"] for f in folds]
        is_sharpes = [f.get("is_sharpe", f["sharpe"]) for f in folds]
        cost_stress_rets = [f.get("cost_stress_return", f["total_return"]) for f in folds]

        min_oos_sharpe  = vc.get("min_oos_sharpe",           0.8)
        max_dd_limit    = vc.get("max_drawdown_limit_pct",   10.0) / 100
        max_fold_ret_pct= vc.get("max_single_fold_return_pct", 0.30)
        min_trades_fold = vc.get("min_trades_per_fold",       10)
        perturb_cv      = vc.get("perturbation_cv_threshold", 0.25)
        max_sharpe_gap  = vc.get("max_is_oos_sharpe_gap", 0.30)
        require_cost_stress = vc.get("require_cost_stress_profitability", True)
        require_strong_oos_consistency = vc.get("require_positive_oos_all_folds", True)
        min_positive_oos_folds_pct = vc.get("min_positive_oos_folds_pct", 1.0)
        min_positive_cost_stress_folds_pct = vc.get("min_positive_cost_stress_folds_pct", 1.0)
        active_fold_trade_threshold = vc.get("active_fold_trade_threshold", 1)
        min_active_folds = vc.get("min_active_folds", 2)
        min_active_folds_for_dominance = vc.get("min_active_folds_for_dominance", 2)
        min_active_folds_for_stability = vc.get("min_active_folds_for_stability", 2)
        min_active_folds_for_is_oos_gap = vc.get("min_active_folds_for_is_oos_gap", 2)

        active_folds = [f for f in folds if f.get("n_trades", 0) >= active_fold_trade_threshold]
        active_sharpes = [f["sharpe"] for f in active_folds] if active_folds else sharpes
        active_returns = [f["total_return"] for f in active_folds] if active_folds else returns
        active_cost_stress_rets = [f.get("cost_stress_return", f["total_return"]) for f in active_folds] if active_folds else cost_stress_rets
        active_is_sharpes = [f.get("is_sharpe", f["sharpe"]) for f in active_folds] if active_folds else is_sharpes

        total_return = sum(active_returns)
        positive_returns = [max(0.0, r) for r in active_returns]
        positive_total = sum(positive_returns)
        fold_return_fracs = (
            [r / max(positive_total, 1e-9) for r in positive_returns]
            if positive_total > 0 else [1.0] * max(1, len(active_returns))
        )
        avg_is_oos_gap = float(np.mean([abs(i - o) for i, o in zip(active_is_sharpes, active_sharpes)]))
        # Consistency should be based on realized fold returns, not Sharpe sign.
        positive_oos_folds_pct = float(sum(1 for r in active_returns if r > 0) / max(1, len(active_returns)))
        positive_cost_stress_folds_pct = float(sum(1 for r in active_cost_stress_rets if r > 0) / max(1, len(active_cost_stress_rets)))

        checks = {
            "oos_sharpe_positive":    float(np.mean(active_sharpes)) > 0,
            "oos_sharpe_threshold":   float(np.mean(active_sharpes)) >= min_oos_sharpe,
            "min_active_folds":       len(active_folds) >= min_active_folds,
            "no_fold_dominates":      (
                all(f < max_fold_ret_pct for f in fold_return_fracs)
                if len(active_folds) >= min_active_folds_for_dominance else True
            ),
            "drawdown_within_limit":  all(abs(d) <= max_dd_limit for d in mdd),
            "sufficient_trades":      all(f["n_trades"] >= min_trades_fold
                                          for f in active_folds),
            "sharpe_stable":          (
                (np.std(active_sharpes) / max(np.mean(active_sharpes), 1e-9) <= perturb_cv + 0.3)
                if len(active_folds) >= min_active_folds_for_stability else True
            ),
            "small_is_oos_gap":       (
                avg_is_oos_gap <= max_sharpe_gap
                if len(active_folds) >= min_active_folds_for_is_oos_gap else True
            ),
            "cost_stress_profitable": (
                positive_cost_stress_folds_pct >= min_positive_cost_stress_folds_pct
                if require_cost_stress else True
            ),
            "oos_consistent_positive": (
                positive_oos_folds_pct >= min_positive_oos_folds_pct
                if require_strong_oos_consistency else True
            ),
        }

        all_pass = all(checks.values())
        diagnostics = {
            "active_folds": len(active_folds),
            "avg_is_oos_sharpe_gap": avg_is_oos_gap,
            "positive_oos_folds_pct": positive_oos_folds_pct,
            "positive_cost_stress_folds_pct": positive_cost_stress_folds_pct,
            "max_fold_return_share": float(max(fold_return_fracs)) if fold_return_fracs else 1.0,
        }
        return {"checks": checks, "diagnostics": diagnostics, "all_pass": all_pass}
