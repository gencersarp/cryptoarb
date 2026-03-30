from __future__ import annotations

from itertools import combinations
from typing import Dict, Any, List

import pandas as pd

from portfolio.profit_model import expected_profit


def funding_velocity_adjusted_rate(base_rate: float, notional: float, depth_notional: float, impact_beta: float) -> float:
    # Market impact style model: own size pushes effective funding toward zero / adverse.
    participation = notional / max(depth_notional, 1.0)
    impact = impact_beta * participation
    if base_rate >= 0:
        return max(0.0, base_rate - impact)
    return min(0.0, base_rate + impact)


def rank_cross_exchange_opportunities(
    row: pd.Series,
    venues: List[str],
    asset: str,
    notional_usd: float,
    fee_by_venue: Dict[str, Dict[str, float]],
    slippage_bps: float,
    latency_bps: float,
    inventory_risk_bps: float,
    dex_velocity_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    out = []
    for a, b in combinations(venues, 2):
        fa = float(row.get(f"{a}_{asset}_funding", 0.0) or 0.0)
        fb = float(row.get(f"{b}_{asset}_funding", 0.0) or 0.0)
        spread = fa - fb

        for hi, lo, raw_spread in [(a, b, spread), (b, a, -spread)]:
            fr_hi = float(row.get(f"{hi}_{asset}_funding", 0.0) or 0.0)
            fr_lo = float(row.get(f"{lo}_{asset}_funding", 0.0) or 0.0)

            fr_hi_eff = fr_hi
            fr_lo_eff = fr_lo
            for venue, fr in [(hi, fr_hi), (lo, fr_lo)]:
                if dex_velocity_cfg.get(venue, {}).get("enabled", False):
                    depth = dex_velocity_cfg[venue].get("depth_notional", 1_000_000.0)
                    beta = dex_velocity_cfg[venue].get("impact_beta", 0.00005)
                    adj = funding_velocity_adjusted_rate(fr, notional_usd, depth, beta)
                    if venue == hi:
                        fr_hi_eff = adj
                    else:
                        fr_lo_eff = adj

            funding_diff = fr_hi_eff - fr_lo_eff
            fee_rate = fee_by_venue.get(hi, {}).get("taker_fee", 0.0005) + fee_by_venue.get(lo, {}).get("taker_fee", 0.0005)
            pb = expected_profit(
                notional_usd=notional_usd,
                funding_diff=funding_diff,
                taker_fee_rate=fee_rate,
                slippage_bps=slippage_bps,
                latency_bps=latency_bps,
                inventory_risk_bps=inventory_risk_bps,
            )
            out.append(
                {
                    "asset": asset,
                    "exchange_high": hi,
                    "exchange_low": lo,
                    "funding_spread_raw": raw_spread,
                    "funding_spread_effective": funding_diff,
                    "net_ev": pb.net_expected_profit,
                    "fees": pb.trading_fees,
                    "slippage": pb.slippage_cost,
                    "latency_cost": pb.latency_cost,
                    "inventory_risk_cost": pb.inventory_risk_cost,
                }
            )
    out.sort(key=lambda x: x["net_ev"], reverse=True)
    return out

