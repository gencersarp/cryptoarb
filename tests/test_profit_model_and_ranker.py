from portfolio.profit_model import expected_profit
from optimization.opportunity_ranker import rank_cross_exchange_opportunities
import pandas as pd


def test_expected_profit_includes_all_cost_terms():
    out = expected_profit(
        notional_usd=100000,
        funding_diff=0.0004,
        taker_fee_rate=0.0005,
        slippage_bps=2.0,
        latency_bps=0.5,
        inventory_risk_bps=1.0,
    )
    assert out.funding_edge > 0
    assert out.trading_fees > 0
    assert out.slippage_cost > 0
    assert out.latency_cost > 0
    assert out.inventory_risk_cost > 0


def test_ranker_returns_sorted_net_ev():
    row = pd.Series(
        {
            "binance_BTC_funding": 0.0005,
            "bybit_BTC_funding": 0.0001,
            "okx_BTC_funding": 0.0002,
        }
    )
    ops = rank_cross_exchange_opportunities(
        row=row,
        venues=["binance", "bybit", "okx"],
        asset="BTC",
        notional_usd=50_000,
        fee_by_venue={"binance": {"taker_fee": 0.0004}, "bybit": {"taker_fee": 0.0006}, "okx": {"taker_fee": 0.0005}},
        slippage_bps=2.0,
        latency_bps=0.5,
        inventory_risk_bps=1.0,
        dex_velocity_cfg={},
    )
    assert len(ops) > 0
    assert ops[0]["net_ev"] >= ops[-1]["net_ev"]

