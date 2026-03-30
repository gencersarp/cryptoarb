#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.opportunity_ranker import rank_cross_exchange_opportunities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-csv", required=True)
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--venues", nargs="+", default=["binance", "bybit", "okx"])
    ap.add_argument("--notional-usd", type=float, default=50_000.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--latency-bps", type=float, default=0.5)
    ap.add_argument("--inventory-risk-bps", type=float, default=1.0)
    ap.add_argument("--out-json", default="../live_screener.json")
    args = ap.parse_args()

    df = pd.read_csv(args.panel_csv)
    row = df.iloc[-1]
    fee_by_venue = {
        "binance": {"taker_fee": 0.0004},
        "bybit": {"taker_fee": 0.0006},
        "okx": {"taker_fee": 0.0005},
        "hyperliquid": {"taker_fee": 0.00035},
        "synthetix": {"taker_fee": 0.0007},
    }
    dex_velocity_cfg = {
        "synthetix": {"enabled": True, "depth_notional": 500_000.0, "impact_beta": 0.00008},
        "hyperliquid": {"enabled": True, "depth_notional": 1_200_000.0, "impact_beta": 0.00005},
    }
    ops = rank_cross_exchange_opportunities(
        row=row,
        venues=args.venues,
        asset=args.asset,
        notional_usd=args.notional_usd,
        fee_by_venue=fee_by_venue,
        slippage_bps=args.slippage_bps,
        latency_bps=args.latency_bps,
        inventory_risk_bps=args.inventory_risk_bps,
        dex_velocity_cfg=dex_velocity_cfg,
    )
    Path(args.out_json).write_text(json.dumps(ops, indent=2))
    print(pd.DataFrame(ops).head(20).to_string(index=False))


if __name__ == "__main__":
    main()

