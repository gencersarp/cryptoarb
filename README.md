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

# 9. Multi-strategy parameter matrix sweep
python scripts/run_param_matrix.py --data-dir ../data --asset BTC --output ../param_matrix_results.csv

# 10. Live-like sweep (more realistic execution assumptions)
python scripts/run_param_matrix.py --mode fast --execution-profile live_like --data-dir ../data --asset BTC --output ../param_matrix_results_live_like_btc.csv

# 11. Canonical robustness suite (walk-forward + stress + robust selection)
python scripts/run_robustness_suite.py --data-dir ../data --assets BTC ETH --strategy SpotPerpFunding --execution-profile live_like --n-trials 40 --output-json ../results_robust_eval.json --output-csv ../results_robust_eval.csv

# 12. Go / no-go gate on robustness outputs
python scripts/run_go_no_go.py --input-json ../results_robust_eval.json

# 13. Calibrate execution assumptions from paper trading fills
python scripts/calibrate_execution.py --paper-state paper_state.json --out-json ../execution_calibration.json

# 14. Final comparison report (robust eval + ablations)
python scripts/generate_comparison_report.py --robust-csv ../results_robust_eval.csv --ablation-json ../ablation_results.json --out-csv ../final_comparison_report.csv

# 15. Async downloader (compatible with existing raw layout)
python scripts/async_download_data.py --assets BTC ETH --start 2021-01-01 --end 2024-12-31 --raw-dir data/raw --max-concurrency 8

# 16. Live screener for cross-exchange EV ranking
python scripts/run_live_screener.py --panel-csv ../data/BTCUSDT_perp_bars.csv --asset BTC --venues binance bybit okx --notional-usd 50000

# 17. Comprehensive multi-asset strategy benchmark (anti-overfit gates)
python scripts/run_comprehensive_benchmark.py --data-dir ../data --assets BTC ETH SOL AVAX LINK DOGE --strategies SpotPerpFunding BasisMeanRevert PerpPerpDiff --out-csv ../comprehensive_benchmark_results.csv --out-json ../comprehensive_benchmark_results.json
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
| OOS Sharpe threshold (mean fold Sharpe) | ≥ 0.7 | Consistently weak strategies |
| Active OOS folds (non-trivial trading activity) | ≥ 1 fold | Prevents degenerate no-trade winners |
| IS–OOS Sharpe gap | ≤ 1.5 | Overfit |
| No single fold > 75% of positive PnL | ≤ 75% | Fragile, regime-specific |
| Profitable under 1.5× cost stress | Positive return on ≥60% folds | Cost-sensitive edge |
| Max drawdown within declared limit | ≤ 10% | Uncontrolled risk |
| Perturbation stability (CV of Sharpe) | ≤ 0.6 | Parameter sensitivity |

For low-activity samples, fold-dominance/stability/IS-OOS-gap checks are enforced once at least 2 active folds exist.

---

## Notes on Live Readiness

- The current runtime includes a robust **paper trading** loop and realistic execution simulation.
- A full **exchange order-routing executor** (placing real orders via exchange APIs) is not yet wired in this repo.
- CCXT is included in dependencies and `.env.example` includes API key placeholders, but you should treat current code as research + paper execution until a dedicated live order router is added and validated.

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
