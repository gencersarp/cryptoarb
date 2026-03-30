from optimization.robust_eval import aggregate_asset_results, robust_objective


def test_robust_eval_aggregate_and_score():
    rows = [
        {"aggregate": {"mean_sharpe": 1.2, "mean_return": 0.04, "worst_dd": -0.05, "std_sharpe": 0.3, "total_n_trades": 120}, "fold_results": [{"cost_stress_return": 0.01}]},
        {"aggregate": {"mean_sharpe": 1.0, "mean_return": 0.03, "worst_dd": -0.04, "std_sharpe": 0.2, "total_n_trades": 100}, "fold_results": [{"cost_stress_return": 0.02}]},
    ]
    agg = aggregate_asset_results(rows)
    score = robust_objective(agg, perturbation_cv=0.2)
    assert agg["mean_sharpe"] > 1.0
    assert score > 0.0

