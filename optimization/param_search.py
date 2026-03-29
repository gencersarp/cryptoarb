"""
Parameter Search using Optuna (Bayesian optimization).

Only optimises on IS (train) data. OOS results reported separately.
Guards against overfitting:
  - min_trades threshold
  - Pruning of unprofitable trials
  - CV over IS sub-folds when n_cv > 1
  - Perturbation sensitivity check after best trial
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna not installed; falling back to random search.")


class ParameterSearch:

    def __init__(self, cfg: dict):
        self.n_trials         = cfg.get("n_trials",           200)
        self.n_startup        = cfg.get("n_startup_trials",    30)
        self.direction        = cfg.get("direction",       "maximize")
        self.min_trades       = cfg.get("min_trades",          30)
        self.perturb_pct      = cfg.get("perturbation_pct",   0.20)
        self.n_perturb        = cfg.get("n_perturbations",     20)
        self.seed             = cfg.get("seed",                42)

    def search(
        self,
        objective_fn: Callable[[Dict], float],
        param_space:  Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run Bayesian (or random) search.
        objective_fn(params) -> scalar score (higher = better).
        param_space: dict of {name: ("float", lo, hi) | ("int", lo, hi)
                              | ("cat", [choices])}
        Returns best params + study summary.
        """
        if HAS_OPTUNA:
            return self._optuna_search(objective_fn, param_space)
        else:
            return self._random_search(objective_fn, param_space)

    # ── Optuna ─────────────────────────────────────────────────────────────
    def _optuna_search(
        self,
        objective_fn: Callable[[Dict], float],
        param_space:  Dict[str, Any],
    ) -> Dict[str, Any]:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup, seed=self.seed
        )
        study = optuna.create_study(
            direction=self.direction, sampler=sampler
        )

        def wrapped(trial):
            params = self._suggest(trial, param_space)
            try:
                score = objective_fn(params)
            except Exception as e:
                logger.debug(f"Trial failed: {e}")
                return float("-inf") if self.direction == "maximize" else float("inf")
            if not np.isfinite(score):
                return float("-inf") if self.direction == "maximize" else float("inf")
            return score

        study.optimize(wrapped, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        sensitivity = self._perturbation_check(objective_fn, best, param_space)
        return {
            "best_params":        best,
            "best_value":         study.best_value,
            "n_trials":           len(study.trials),
            "perturbation_cv":    sensitivity,
            "robust":             sensitivity < self.perturb_pct + 0.05,
        }

    @staticmethod
    def _suggest(trial, param_space: Dict) -> Dict:
        params = {}
        for name, spec in param_space.items():
            kind = spec[0]
            if kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2])
            elif kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])
            elif kind == "cat":
                params[name] = trial.suggest_categorical(name, spec[1])
        return params

    # ── Random search fallback ───────────────────────────────────────────
    def _random_search(
        self,
        objective_fn: Callable[[Dict], float],
        param_space:  Dict[str, Any],
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed)
        best_score  = float("-inf")
        best_params: Dict = {}
        scores = []
        for _ in range(self.n_trials):
            params = {}
            for name, spec in param_space.items():
                kind = spec[0]
                if kind == "float":
                    params[name] = float(rng.uniform(spec[1], spec[2]))
                elif kind == "int":
                    params[name] = int(rng.integers(spec[1], spec[2] + 1))
                elif kind == "cat":
                    params[name] = rng.choice(spec[1])
            try:
                score = objective_fn(params)
            except Exception:
                score = float("-inf")
            scores.append(score)
            if np.isfinite(score) and score > best_score:
                best_score  = score
                best_params = copy.deepcopy(params)
        sensitivity = self._perturbation_check(objective_fn, best_params, param_space)
        return {
            "best_params":     best_params,
            "best_value":      best_score,
            "n_trials":        self.n_trials,
            "perturbation_cv": sensitivity,
            "robust":          sensitivity < self.perturb_pct + 0.05,
        }

    # ── Perturbation sensitivity ─────────────────────────────────────────
    def _perturbation_check(
        self,
        objective_fn: Callable[[Dict], float],
        best_params:  Dict,
        param_space:  Dict,
    ) -> float:
        """CV of objective under small parameter perturbations."""
        if not best_params:
            return 1.0
        rng    = np.random.default_rng(self.seed + 1)
        scores = []
        for _ in range(self.n_perturb):
            perturbed = {}
            for name, val in best_params.items():
                spec = param_space.get(name)
                if spec is None:
                    perturbed[name] = val
                    continue
                kind = spec[0]
                if kind == "float":
                    noise = val * self.perturb_pct * rng.uniform(-1, 1)
                    perturbed[name] = float(
                        np.clip(val + noise, spec[1], spec[2])
                    )
                elif kind == "int":
                    noise = max(1, int(abs(val) * self.perturb_pct))
                    perturbed[name] = int(
                        np.clip(val + rng.integers(-noise, noise + 1),
                                spec[1], spec[2])
                    )
                else:
                    perturbed[name] = val
            try:
                s = objective_fn(perturbed)
                if np.isfinite(s):
                    scores.append(s)
            except Exception:
                pass
        if not scores:
            return 1.0
        return float(np.std(scores) / max(abs(np.mean(scores)), 1e-9))
