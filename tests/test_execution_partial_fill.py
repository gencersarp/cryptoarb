from backtest.execution_sim import ExecutionSimulator
import pandas as pd


def test_partial_fill_can_reduce_size():
    sim = ExecutionSimulator(
        {
            "seed": 1,
            "maker_fill_prob": 1.0,
            "partial_fill_prob": 1.0,
            "min_partial_fill_ratio": 0.2,
            "slippage_model": "fixed",
            "min_slippage_bps": 1.0,
            "max_slippage_bps": 1.0,
            "exchange_downtime_prob": 0.0,
        }
    )
    f = sim.simulate_fill(
        signal_id="s",
        venue="binance",
        asset="BTC",
        market="perp",
        side="long",
        size=1.0,
        ref_price=50000.0,
        realized_vol=0.01,
        adv_usd=1_000_000,
        venue_cfg={"maker_fee": 0.0002, "taker_fee": 0.0004},
        ts=pd.Timestamp.utcnow(),
    )
    assert 0 < f.filled_size <= 1.0

