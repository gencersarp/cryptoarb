"""
Portfolio analytics and metrics.

All metrics computed from equity curve. No lookahead.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import scipy.stats as stats


def compute_metrics(equity: pd.Series, initial_capital: float,
                    freq_hours: int = 8) -> Dict[str, float]:
    """
    Compute comprehensive risk-adjusted metrics from equity curve.
    freq_hours: bar frequency. 8h → 3 bars/day → 1095 bars/year.
    """
    bars_per_year = int(365 * 24 / freq_hours)

    if len(equity) < 5:
        return _empty_metrics()

    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        return _empty_metrics()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0
    n_years      = len(equity) / bars_per_year
    cagr         = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0.0

    ann_ret  = returns.mean() * bars_per_year
    ann_vol  = returns.std()  * np.sqrt(bars_per_year)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0.0

    neg_ret  = returns[returns < 0]
    downside = neg_ret.std() * np.sqrt(bars_per_year) if len(neg_ret) > 0 else 1e-9
    sortino  = ann_ret / downside if downside > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd   = drawdown.min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Max drawdown duration (bars)
    in_dd = drawdown < 0
    dd_dur, cur_dur = 0, 0
    for v in in_dd:
        if v:
            cur_dur += 1
            dd_dur = max(dd_dur, cur_dur)
        else:
            cur_dur = 0

    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "total_return":    float(total_return),
        "cagr":            float(cagr),
        "ann_return":      float(ann_ret),
        "ann_vol":         float(ann_vol),
        "sharpe":          float(sharpe),
        "sortino":         float(sortino),
        "calmar":          float(calmar),
        "max_drawdown":    float(max_dd),
        "max_dd_duration": int(dd_dur),
        "win_rate":        float((returns > 0).mean()),
        "skewness":        float(stats.skew(clean))   if len(clean) > 3 else 0.0,
        "kurtosis":        float(stats.kurtosis(clean)) if len(clean) > 3 else 0.0,
        "var_95":          float(returns.quantile(0.05)),
        "cvar_95":         float(returns[returns <= returns.quantile(0.05)].mean()),
    }


def _empty_metrics() -> Dict[str, float]:
    return {k: 0.0 for k in ["total_return", "cagr", "ann_return", "ann_vol",
                               "sharpe", "sortino", "calmar", "max_drawdown",
                               "max_dd_duration", "win_rate", "skewness",
                               "kurtosis", "var_95", "cvar_95"]}


def decompose_pnl(trades: list) -> Dict[str, float]:
    """Break P&L into funding, basis, execution loss, fees."""
    total_funding = sum(t.funding_pnl for t in trades)
    total_fee     = sum(t.fee         for t in trades)
    total_slip    = sum(t.slippage    for t in trades)
    total_gross   = sum(t.gross_pnl   for t in trades)
    total_net     = sum(t.net_pnl     for t in trades)
    return {
        "gross_pnl":         total_gross,
        "funding_pnl":       total_funding,
        "fee_drag":         -total_fee,
        "slippage_drag":    -total_slip,
        "net_pnl":           total_net,
        "cost_as_pct_gross": (total_fee + total_slip) / max(abs(total_gross), 1e-9),
    }


def run_monte_carlo(equity: pd.Series, n_sims: int = 1000,
                    seed: int = 42) -> Dict[str, Any]:
    """Bootstrap returns to estimate metric distribution."""
    rng     = np.random.default_rng(seed)
    returns = equity.pct_change().dropna().values
    sharpes, mdd_list = [], []
    for _ in range(n_sims):
        sampled = rng.choice(returns, size=len(returns), replace=True)
        eq = pd.Series(np.cumprod(1 + sampled) * equity.iloc[0])
        m  = compute_metrics(eq, equity.iloc[0])
        sharpes.append(m["sharpe"])
        mdd_list.append(m["max_drawdown"])
    return {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_p5":   float(np.percentile(sharpes, 5)),
        "sharpe_p95":  float(np.percentile(sharpes, 95)),
        "mdd_mean":    float(np.mean(mdd_list)),
        "mdd_p95":     float(np.percentile(mdd_list, 95)),
    }


def regime_breakdown(equity: pd.Series, panel: pd.DataFrame,
                      price_col: str = "BTC_perp_close") -> Dict[str, Any]:
    """
    Segment equity curve into bull/bear/sideways/high-vol regimes
    based on a rolling price trend and volatility percentile.
    """
    results = {}
    if price_col not in panel.columns or len(panel) != len(equity):
        return results

    prices  = panel[price_col].values
    returns = pd.Series(prices).pct_change().fillna(0)
    roll_ret = returns.rolling(90).mean()    # 30-day trend (3 bars/day)
    roll_vol = returns.rolling(90).std()
    vol_75   = roll_vol.quantile(0.75)

    regimes = pd.Series("sideways", index=range(len(prices)))
    regimes[roll_ret > 0.001]  = "bull"
    regimes[roll_ret < -0.001] = "bear"
    regimes[roll_vol > vol_75] = "high_vol"

    eq_returns = equity.pct_change().fillna(0)
    for regime in ["bull", "bear", "sideways", "high_vol"]:
        mask = (regimes == regime).values
        if mask.sum() < 5:
            continue
        eq_slice = equity.iloc[mask]
        results[regime] = compute_metrics(
            pd.Series(eq_slice.values), eq_slice.iloc[0]
        )
    return results
