#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import BacktestEngine
from strategies.spot_perp import SpotPerpFundingStrategy
from strategies.basis_revert import BasisMeanRevertStrategy
from strategies.perp_perp import PerpPerpDiffStrategy


def load_panel(data_dir: Path, asset: str = "BTC", venue: str = "binance") -> pd.DataFrame:
    spot = pd.read_csv(data_dir / f"{asset}USDT_spot_bars.csv")
    perp = pd.read_csv(data_dir / f"{asset}USDT_perp_bars.csv")
    funding = pd.read_csv(data_dir / f"{asset}USDT_funding.csv")

    for df in (spot, perp, funding):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    panel = perp[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    panel = panel.rename(
        columns={
            "open": f"{venue}_{asset}_perp_open",
            "high": f"{venue}_{asset}_perp_high",
            "low": f"{venue}_{asset}_perp_low",
            "close": f"{venue}_{asset}_perp_close",
            "volume": f"{venue}_{asset}_perp_volume",
        }
    )
    spot2 = spot[["timestamp", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": f"{venue}_{asset}_spot_open",
            "high": f"{venue}_{asset}_spot_high",
            "low": f"{venue}_{asset}_spot_low",
            "close": f"{venue}_{asset}_spot_close",
            "volume": f"{venue}_{asset}_spot_volume",
        }
    )
    panel = panel.merge(spot2, on="timestamp", how="left")
    panel = panel.merge(
        funding[["timestamp", "rate"]].rename(columns={"rate": f"{venue}_{asset}_funding"}),
        on="timestamp",
        how="left",
    )
    panel = panel.sort_values("timestamp")
    panel[f"{venue}_{asset}_funding"] = panel[f"{venue}_{asset}_funding"].ffill().fillna(0.0)
    panel[f"{venue}_{asset}_basis"] = (
        (panel[f"{venue}_{asset}_perp_close"] - panel[f"{venue}_{asset}_spot_close"])
        / panel[f"{venue}_{asset}_spot_close"]
    ).fillna(0.0)
    ret = panel[f"{venue}_{asset}_perp_close"].pct_change().fillna(0.0)
    panel["realized_vol"] = ret.rolling(24).std().fillna(ret.std() if ret.std() > 0 else 0.01)
    panel["adv"] = (panel[f"{venue}_{asset}_perp_close"] * panel[f"{venue}_{asset}_perp_volume"]).rolling(20).mean()
    panel["adv"] = panel["adv"].fillna(panel["adv"].median() if panel["adv"].notna().any() else 1e7)
    panel = panel.rename(columns={"timestamp": "ts"}).reset_index(drop=True)
    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--asset", default="BTC")
    ap.add_argument("--output", default="../param_matrix_results.csv")
    args = ap.parse_args()

    panel = load_panel(Path(args.data_dir), asset=args.asset)
    n = len(panel)
    train_idx = range(0, int(n * 0.6))
    test_idx = range(int(n * 0.6), n)

    base_cfg = {
        "initial_capital": 100_000.0,
        "risk": {"max_drawdown_pct": 20.0, "daily_loss_limit_pct": 10.0, "per_strategy_capital_pct": 40.0},
        "execution": {
            "seed": 42,
            "maker_fill_prob": 0.6,
            "slippage_model": "fixed",
            "min_slippage_bps": 1.0,
            "max_slippage_bps": 8.0,
            "exchange_downtime_prob": 0.0,
        },
    }
    venue_cfg = {"binance": {"maker_fee": 0.0002, "taker_fee": 0.0004, "borrow_rate_annual": 0.05}}
    v = "binance"
    a = args.asset

    rows = []

    # SpotPerp sweep
    for entry, exit_, min_yield, pos in itertools.product(
        [0.0002, 0.0003, 0.0004], [0.00008, 0.0001], [0.03, 0.05], [0.2, 0.3]
    ):
        strat = SpotPerpFundingStrategy(
            {
                "assets": [a],
                "venues": [v],
                "entry_funding_threshold": entry,
                "exit_funding_threshold": exit_,
                "min_annualized_yield": min_yield,
                "position_size_pct": pos,
                "max_open_positions": 1,
                "use_basis_filter": True,
                "basis_filter_z": 1.8,
                "zscore_lookback": 24,
                "max_hold_bars": 21,
            }
        )
        res = BacktestEngine([strat], panel, base_cfg, venue_cfg).run_fold(train_idx, test_idx, 1)
        rows.append(
            {
                "strategy": "SpotPerpFunding",
                "params": json.dumps({"entry": entry, "exit": exit_, "min_yield": min_yield, "pos": pos}),
                "sharpe": res["sharpe"],
                "total_return": res["total_return"],
                "max_drawdown": res["max_drawdown"],
                "n_trades": res["n_trades"],
            }
        )

    # Basis sweep
    for entry_z, exit_z, stop_z in itertools.product([1.5, 2.0, 2.5], [0.3, 0.5], [3.0, 3.5]):
        strat = BasisMeanRevertStrategy(
            {
                "assets": [a],
                "venues": [v],
                "zscore_lookback": 36,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "position_size_pct": 0.25,
                "max_open_positions": 1,
                "max_hold_bars": 30,
            }
        )
        res = BacktestEngine([strat], panel, base_cfg, venue_cfg).run_fold(train_idx, test_idx, 1)
        rows.append(
            {
                "strategy": "BasisMeanRevert",
                "params": json.dumps({"entry_z": entry_z, "exit_z": exit_z, "stop_z": stop_z}),
                "sharpe": res["sharpe"],
                "total_return": res["total_return"],
                "max_drawdown": res["max_drawdown"],
                "n_trades": res["n_trades"],
            }
        )

    # PerpPerp on single venue data won't trade much; still include for completeness
    for min_spread, entry_z in itertools.product([0.00015, 0.0002], [1.5, 2.0]):
        strat = PerpPerpDiffStrategy(
            {
                "assets": [a],
                "venues": [v, v],  # degenerate to test config path
                "min_funding_spread": min_spread,
                "entry_z": entry_z,
                "exit_z": 0.5,
                "zscore_lookback": 30,
                "position_size_pct": 0.2,
                "max_open_positions": 1,
            }
        )
        res = BacktestEngine([strat], panel, base_cfg, venue_cfg).run_fold(train_idx, test_idx, 1)
        rows.append(
            {
                "strategy": "PerpPerpDiff",
                "params": json.dumps({"min_spread": min_spread, "entry_z": entry_z}),
                "sharpe": res["sharpe"],
                "total_return": res["total_return"],
                "max_drawdown": res["max_drawdown"],
                "n_trades": res["n_trades"],
            }
        )

    out = pd.DataFrame(rows).sort_values(["sharpe", "total_return"], ascending=False)
    out.to_csv(args.output, index=False)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
