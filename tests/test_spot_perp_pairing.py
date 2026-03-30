from __future__ import annotations

import pandas as pd

from strategies.spot_perp import SpotPerpFundingStrategy
from strategies.base import Position, Side, Market


def _panel(periods: int = 40) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": [50_000.0] * periods,
            "binance_BTC_spot_close": [50_000.0] * periods,
            "binance_BTC_funding": [0.0003] * 20 + [0.00005] * 20,
            "binance_BTC_basis": [0.0] * periods,
        }
    )


def test_spot_perp_generates_paired_entry_and_exit_signals():
    panel = _panel()
    strat = SpotPerpFundingStrategy(
        {
            "assets": ["BTC"],
            "venues": ["binance"],
            "entry_funding_threshold": 0.0002,
            "exit_funding_threshold": 0.0001,
            "min_annualized_yield": 0.01,
            "max_open_positions": 1,
            "use_basis_filter": False,
            "zscore_lookback": 8,
            "hedge_with_spot": True,
            "min_funding_persistence_bars": 2,
            "min_expected_edge_bps": 0.0,
        }
    )

    entry_signals = strat.generate_signals(panel, 10, [])
    assert len(entry_signals) == 2
    assert {s.market for s in entry_signals} == {Market.PERP, Market.SPOT}

    open_positions = [
        Position(
            signal_id="SpotPerpFunding_binance_BTC_PERP",
            venue="binance",
            asset="BTC",
            market=Market.PERP,
            side=Side.SHORT,
            entry_price=50_000.0,
            entry_ts=panel.iloc[10]["ts"],
            size=1.0,
            notional=50_000.0,
            metadata={"strategy": "SpotPerpFunding"},
        ),
        Position(
            signal_id="SpotPerpFunding_binance_BTC_SPOT",
            venue="binance",
            asset="BTC",
            market=Market.SPOT,
            side=Side.LONG,
            entry_price=50_000.0,
            entry_ts=panel.iloc[10]["ts"],
            size=1.0,
            notional=50_000.0,
            metadata={"strategy": "SpotPerpFunding"},
        ),
    ]
    exit_signals = strat.generate_signals(panel, 30, open_positions)
    assert len(exit_signals) == 2
    assert all(s.side == Side.FLAT for s in exit_signals)
