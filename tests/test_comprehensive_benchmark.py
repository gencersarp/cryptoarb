from scripts.run_comprehensive_benchmark import _strategy_param_grid, _build_strategy


def test_comprehensive_benchmark_grid_and_builder():
    g = _strategy_param_grid("SpotPerpFunding")
    assert len(g) > 0
    s = _build_strategy("SpotPerpFunding", "BTC", g[0])
    assert s.name == "SpotPerpFunding"

