from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfitBreakdown:
    funding_edge: float
    trading_fees: float
    slippage_cost: float
    latency_cost: float
    inventory_risk_cost: float
    net_expected_profit: float


def expected_profit(
    notional_usd: float,
    funding_diff: float,
    taker_fee_rate: float,
    slippage_bps: float,
    latency_bps: float,
    inventory_risk_bps: float,
) -> ProfitBreakdown:
    funding_edge = notional_usd * funding_diff
    trading_fees = notional_usd * taker_fee_rate * 2.0
    slippage_cost = notional_usd * (slippage_bps / 10_000.0) * 2.0
    latency_cost = notional_usd * (latency_bps / 10_000.0)
    inventory_risk_cost = notional_usd * (inventory_risk_bps / 10_000.0)
    net = funding_edge - trading_fees - slippage_cost - latency_cost - inventory_risk_cost
    return ProfitBreakdown(
        funding_edge=funding_edge,
        trading_fees=trading_fees,
        slippage_cost=slippage_cost,
        latency_cost=latency_cost,
        inventory_risk_cost=inventory_risk_cost,
        net_expected_profit=net,
    )

