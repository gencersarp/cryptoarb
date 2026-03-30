#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust-csv", default="../results_robust_eval.csv")
    ap.add_argument("--ablation-json", default="../ablation_results.json")
    ap.add_argument("--out-csv", default="../final_comparison_report.csv")
    args = ap.parse_args()

    robust_path = Path(args.robust_csv)
    if not robust_path.exists():
        raise FileNotFoundError(f"Missing robust csv: {robust_path}")
    robust_df = pd.read_csv(robust_path)

    rows = []
    for _, r in robust_df.iterrows():
        rows.append(
            {
                "source": "robust_eval",
                "strategy": r.get("strategy", "unknown"),
                "asset": r.get("asset", "unknown"),
                "mean_sharpe": r.get("mean_sharpe", 0.0),
                "mean_return": r.get("mean_return", 0.0),
                "worst_dd": r.get("worst_dd", 0.0),
                "total_n_trades": r.get("total_n_trades", 0),
                "all_pass": r.get("all_pass", False),
                "params": r.get("params", "{}"),
            }
        )

    abl_path = Path(args.ablation_json)
    if abl_path.exists():
        abls = json.loads(abl_path.read_text())
        for r in abls:
            rows.append(
                {
                    "source": "ablation",
                    "strategy": "SpotPerpFunding",
                    "asset": "BTC+ETH",
                    "mean_sharpe": r.get("mean_sharpe", 0.0),
                    "mean_return": r.get("mean_return", 0.0),
                    "worst_dd": r.get("worst_dd", 0.0),
                    "total_n_trades": r.get("total_n_trades", 0),
                    "all_pass": r.get("all_hard_pass", False),
                    "params": r.get("params", "{}"),
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["all_pass", "mean_sharpe", "mean_return"], ascending=[False, False, False])
    out_df.to_csv(args.out_csv, index=False)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

