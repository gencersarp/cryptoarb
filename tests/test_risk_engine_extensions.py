from portfolio.risk import RiskEngine
from types import SimpleNamespace


def test_risk_engine_kill_switch_and_exchange_exposure():
    r = RiskEngine(
        {
            "max_drawdown_pct": 50,
            "daily_loss_limit_pct": 50,
            "max_total_leverage": 10,
            "margin_buffer_pct": 5,
            "unrealized_kill_switch_pct": 1.0,
            "max_exchange_exposure_pct": 10.0,
        }
    )
    positions = {
        "a": SimpleNamespace(notional=20_000, pnl=-2_000, venue="binance"),
        "b": SimpleNamespace(notional=5_000, pnl=100, venue="bybit"),
    }
    flags = r.update(100_000, positions)
    assert flags["kill_switch_breached"] is True
    assert flags["exchange_exposure_breached"] is True

