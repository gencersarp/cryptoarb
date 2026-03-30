from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.statarb import StatArbStrategy
from strategies.base import Market, Position, Side


def _panel(periods: int = 180) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=periods, freq="8h", tz="UTC")
    x = np.linspace(0, 8 * np.pi, periods)
    eth = 2_000 + 40 * np.sin(x) + 8 * np.sin(3 * x)
    btc = 25.0 * eth + 500.0 + 12 * np.cos(2 * x)
    return pd.DataFrame(
        {
            "ts": ts,
            "binance_BTC_perp_close": btc,
            "binance_ETH_perp_close": eth,
        }
    )


def test_statarb_generates_paired_entry_and_exit_signals():
    panel = _panel()
    strat = StatArbStrategy(
        {
            "venues": ["binance"],
            "coint_lookback": 90,
            "zscore_lookback": 20,
            "entry_z": 0.2,
            "exit_z": 10.0,
            "stop_z": 3.0,
            "min_coint_pvalue": 0.2,
            "position_size_pct": 0.2,
            "max_open_positions": 1,
        }
    )
    # Force a stable fitted-beta context so this unit test focuses on pairing logic,
    # not on coint test sensitivity for synthetic fixtures.
    strat._beta = 1.2
    bar_i = 140
    strat._last_coint_bar = bar_i
    panel.loc[bar_i - 20 : bar_i, "binance_BTC_perp_close"] = (
        panel.loc[bar_i - 20 : bar_i, "binance_ETH_perp_close"] * 25.0 + 1200.0
    )
    panel.loc[bar_i, "binance_BTC_perp_close"] *= 1.03
    entry = strat.generate_signals(panel, bar_i, [])
    assert len(entry) == 2
    assert {s.asset for s in entry} == {"BTC", "ETH"}
    assert all(s.market == Market.PERP for s in entry)
    assert {s.side for s in entry} == {Side.LONG, Side.SHORT}

    open_positions = [
        Position(
            signal_id=entry[0].signal_id,
            venue=entry[0].venue,
            asset=entry[0].asset,
            market=entry[0].market,
            side=entry[0].side,
            entry_price=float(panel.iloc[bar_i][f"binance_{entry[0].asset}_perp_close"]),
            entry_ts=panel.iloc[bar_i]["ts"],
            size=1.0,
            notional=10_000.0,
            metadata={"strategy": "StatArb"},
        ),
        Position(
            signal_id=entry[1].signal_id,
            venue=entry[1].venue,
            asset=entry[1].asset,
            market=entry[1].market,
            side=entry[1].side,
            entry_price=float(panel.iloc[bar_i][f"binance_{entry[1].asset}_perp_close"]),
            entry_ts=panel.iloc[bar_i]["ts"],
            size=1.0,
            notional=10_000.0,
            metadata={"strategy": "StatArb"},
        ),
    ]

    exit_signals = strat.generate_signals(panel, bar_i + 1, open_positions)
    assert len(exit_signals) == 2
    assert all(s.side == Side.FLAT for s in exit_signals)
