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

        min_oos_sharpe  = vc.get("min_oos_sharpe",           0.8)
        max_dd_limit    = vc.get("max_drawdown_limit_pct",   10.0) / 100
        max_fold_ret_pct= vc.get("max_single_fold_return_pct", 0.30)
        min_trades_fold = vc.get("min_trades_per_fold",       10)
        perturb_cv      = vc.get("perturbation_cv_threshold", 0.25)

        total_return = sum(returns)
        fold_return_fracs = (
            [abs(r) / max(abs(total_return), 1e-9) for r in returns]
            if total_return != 0 else [0.0] * len(returns)
        )

        checks = {
            "oos_sharpe_positive":    agg["mean_sharpe"] > 0,
            "oos_sharpe_threshold":   agg["mean_sharpe"] >= min_oos_sharpe,
            "no_fold_dominates":      all(f < max_fold_ret_pct
                                          for f in fold_return_fracs),
            "drawdown_within_limit":  all(abs(d) * 100 <= vc.get(
                                          "max_drawdown_limit_pct", 10.0)
                                          for d in mdd),
            "sufficient_trades":      all(f["n_trades"] >= min_trades_fold
                                          for f in folds),
            "sharpe_stable":          (np.std(sharpes) / max(np.mean(sharpes), 1e-9)
                                        <= perturb_cv + 0.3),
        }

        all_pass = all(checks.values())
        return {"checks": checks, "all_pass": all_pass}
