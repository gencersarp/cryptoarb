#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _safe_mean(series):
    return float(series.mean()) if len(series) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-state", default="paper_state.json")
    ap.add_argument("--out-json", default="../execution_calibration.json")
    args = ap.parse_args()

    state_path = Path(args.paper_state)
    if not state_path.exists():
        raise FileNotFoundError(f"Paper state not found: {state_path}")

    state = json.loads(state_path.read_text())
    fills = pd.DataFrame(state.get("fills", []))
    if fills.empty:
        payload = {
            "maker_ratio": 0.0,
            "missed_fill_ratio": 0.0,
            "avg_fee": 0.0,
            "avg_slippage": 0.0,
            "recommended_execution": {},
            "notes": "No fills found in paper state.",
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
        print(json.dumps(payload, indent=2))
        return

    maker_ratio = _safe_mean(fills["is_maker"].astype(float))
    missed_ratio = _safe_mean(fills["missed"].astype(float))
    avg_fee = _safe_mean(fills["fee"].astype(float))
    avg_slippage = _safe_mean(fills["slippage"].astype(float))

    # Conservative calibration mapping
    recommended_maker_prob = max(0.1, min(0.9, maker_ratio * 0.95))
    recommended_downtime = max(0.001, min(0.05, missed_ratio * 1.1))
    recommended_min_slip_bps = max(0.3, min(15.0, avg_slippage / 10.0))

    payload = {
        "maker_ratio": maker_ratio,
        "missed_fill_ratio": missed_ratio,
        "avg_fee": avg_fee,
        "avg_slippage": avg_slippage,
        "recommended_execution": {
            "maker_fill_prob": round(recommended_maker_prob, 4),
            "exchange_downtime_prob": round(recommended_downtime, 4),
            "min_slippage_bps": round(recommended_min_slip_bps, 4),
        },
        "notes": "Use these as calibration priors for backtest execution settings.",
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

