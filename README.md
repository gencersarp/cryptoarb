# CryptoArb — Production-Grade Crypto Market-Neutral Arbitrage Framework

> Built for robust research, not backtest decoration.

## What This Is

A clean-room, production-quality framework for researching, backtesting, validating, and paper-trading crypto market-neutral strategies:

- **Spot–Perp Funding Arbitrage** (BTC, ETH, majors)
- **Perp–Perp Funding Differential** (cross-venue)
- **Basis Mean-Reversion** (convergence logic, z-score regimes)
- **Statistical Arbitrage** (cointegration-based spread trading)

Every strategy survives adversarial validation: walk-forward OOS Sharpe, IS/OOS gap check, cost stress (1.5×), drawdown limits, and perturbation stability before a single result is trusted.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/gencersarp/cryptoarb.git
cd cryptoarb

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Download real data (Binance, Bybit, OKX)
python scripts/download_data.py --assets BTC ETH --start 2020-01-01 --end 2024-12-31

# 4. Run full walk-forward backtest
python scripts/run_backtest.py --config config/main.yaml

# 5. Parameter search (Bayesian, Optuna)
python scripts/optimize.py --strategy SpotPerpFunding --config config/main.yaml

# 6. Stress tests
python scripts/stress_test.py --config config/main.yaml

# 7. Paper trading (testnet)
python scripts/paper_trade.py --strategy SpotPerpFunding --config config/live.yaml

# 8. Unit tests
pytest tests/ -v
```

---

## Project Structure

```
cryptoarb/
├── config/
│   ├── main.yaml            # Master config: venues, fees, risk, backtest params
│   └── live.yaml            # Paper/live trading config
├── data/
│   ├── downloader.py        # Binance/Bybit/OKX historical OHLCV + funding
│   ├── normalizer.py        # Normalize to unified panel format
│   └── raw/                 # Downloaded CSVs (gitignored)
├── strategies/
│   ├── base.py              # BaseStrategy, Signal, Position, Side, Market
│   ├── spot_perp_funding.py # Spot-perp funding capture
│   ├── perp_perp_diff.py    # Cross-venue perp funding differential
│   ├── basis_mean_revert.py # Basis convergence + z-score regime
│   ├── stat_arb.py          # Cointegration-based spread trading
│   └── __init__.py          # Strategy registry
├── backtest/
│   ├── engine.py            # Walk-forward backtest engine (1-bar delay, full costs)
│   ├── walk_forward.py      # WalkForwardSplitter with randomized offsets
│   └── execution_sim.py     # Fill simulator: slippage, fees, maker/taker uncertainty
├── portfolio/
│   └── analytics.py         # Sharpe, Sortino, Calmar, MDD, PnL decomp, Monte Carlo
├── research/
│   ├── param_search.py      # Optuna Bayesian search + perturbation stability
│   └── validation.py        # All stop criteria: OOS Sharpe, IS/OOS gap, DD, fold dominance
├── scripts/
│   ├── download_data.py     # CLI data downloader
│   ├── run_backtest.py      # Full backtest runner
│   ├── optimize.py          # Parameter optimization runner
│   ├── stress_test.py       # Stress test suite
│   └── paper_trade.py       # Paper trading / dry-run mode
├── tests/
│   └── test_strategies.py   # Unit tests: signals, no-lookahead, execution sim
├── notebooks/
│   └── research_report.ipynb # Analysis notebook
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Validation Stop Criteria (All Required)

| Criterion | Threshold | What It Catches |
|---|---|---|
| OOS Sharpe ≥ threshold on ALL folds | ≥ 0.8 | Consistently weak strategies |
| IS–OOS Sharpe gap | ≤ 0.3 | Overfit |
| No single fold > 30% of total PnL | ≤ 30% | Fragile, regime-specific |
| Profitable under 1.5× cost stress | Sharpe > 0 | Cost-sensitive edge |
| Max drawdown within declared limit | ≤ 10% | Uncontrolled risk |
| Perturbation stability (CV of Sharpe) | ≤ 0.25 | Parameter sensitivity |

---

## Realistic Cost Model

- **Fees**: maker/taker per venue (Binance: 2/4 bps, Bybit: 1/6 bps, OKX: 2/5 bps)
- **Slippage**: volatility × ADV-adjusted square-root market impact
- **Borrow cost**: spot short financing at venue-specific annual rate
- **Funding timing**: exchange-specific intervals (Binance 8h, Bybit 8h, OKX 8h)
- **1-bar execution delay**: signals generated at bar t, filled at bar t+1
- **Forced close** at end of every test window (no carry-forward)
- **Maker fill uncertainty**: probabilistic maker vs taker based on limit price distance

---

## Key Research References

- Cong et al. (2021) — Crypto funding rate predictability and arbitrage limits
- Avellaneda & Lee (2010) — Statistical arbitrage in equity markets (adapted for crypto)
- Makarov & Schoar (2020) — Trading and arbitrage in cryptocurrency markets
- Prado (2018) — Advances in Financial Machine Learning (walk-forward methodology)
- Bybit/Binance funding rate documentation for exchange-specific conventions
- Public repos surveyed: `nicehash/funding-rate-scanner`, `freqtrade/freqtrade`, `nautechsystems/nautilus_trader` (architecture patterns only — code written from scratch)

---

## Go / No-Go Assessment

See `results/validation_report.json` after running backtests. A strategy is **GO** only if:
1. All six validation criteria pass
2. PnL decomposition shows funding/basis as primary source (not noise)
3. Cost stress (1.5×) leaves Sharpe > 0.5
4. Live divergence estimate < 40% of backtest Sharpe

---

## Disclaimer

This framework is for research. Past backtest performance does not guarantee live profitability. Crypto markets are adversarial, and edge decays. Use at your own risk.
