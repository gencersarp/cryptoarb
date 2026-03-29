# CryptoArb — Production-Grade Crypto Market-Neutral Research Framework

A modular, walk-forward-validated backtesting and live-simulation framework for crypto market-neutral arbitrage strategies. Built for serious research, not toy notebooks.

## Strategy Classes

| Strategy | Description | Primary Assets |
|---|---|---|
| `SpotPerpFunding` | Long spot + short perp, capture funding | BTC, ETH |
| `PerpPerpFunding` | Cross-venue funding differential | BTC, ETH |
| `BasisArb` | Cash-and-carry / basis convergence | BTC, ETH |
| `BasisMeanRevert` | Z-score regime on basis spread | BTC, ETH |
| `FundingMeanRevert` | Z-score on funding rate series | BTC, ETH |
| `StatArb` | Cointegration-based spread on correlated pairs | BTC/ETH |

## Architecture

```
cryptoarb/
├── config/              # Venues, assets, fees, risk limits
├── data/                # Ingestion, normalization, caching
│   ├── raw/             # Downloaded CSVs (git-ignored)
│   ├── processed/       # Normalized parquet/CSV
│   └── cache/           # Intermediate artifacts
├── strategies/          # Strategy modules (common interface)
├── backtest/            # Walk-forward engine, execution sim
├── execution/           # Execution simulator + paper-trade bridge
├── portfolio/           # Risk engine, analytics, reporting
├── research/            # Parameter search, experiment tracking
├── scripts/             # CLI entry points
├── tests/               # Unit + integration tests
├── notebooks/           # Evaluation notebooks
├── logs/                # Experiment logs (git-ignored)
└── results/             # Backtest outputs, ranked strategies
```

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/gencersarp/cryptoarb.git
cd cryptoarb
pip install -r requirements.txt

# 2. Download real historical data (Binance + Bybit + OKX — no API key needed)
python scripts/download_data.py --assets BTC ETH --start 2020-01-01 --end 2024-12-31

# 3. Run full walk-forward backtest on all strategies
python scripts/run_backtest.py --config config/main.yaml

# 4. Run parameter search (Bayesian, 200 trials)
python scripts/param_search.py --strategy SpotPerpFunding --trials 200

# 5. Run stress tests
python scripts/stress_test.py --config config/main.yaml

# 6. Generate HTML evaluation report
python scripts/generate_report.py --results results/latest/

# 7. Start paper-trading simulation
python scripts/paper_trade.py --strategy SpotPerpFunding --config config/live.yaml

# 8. Run unit tests
python -m pytest tests/ -v
```

## Key Design Principles

- **No synthetic data**: All data fetched from real exchange APIs (Binance, Bybit, OKX) via public historical endpoints — no API key needed for download
- **Strict train/test separation**: Walk-forward with N splits + randomized start offsets to prevent fold boundary overfitting
- **Realistic costs**: Maker/taker fees with probabilistic fill, vol-scaled slippage, funding settlement timing, borrow costs for spot shorts
- **Adversarial evaluation**: Every strategy stress-tested at 1.5x cost, funding flip shocks (−3x), vol spikes (3x), and crisis windows (COVID crash, LUNA, FTX)
- **Reproducible**: Seeds, configs, data hashes logged for every experiment
- **Modular**: Add new venue/strategy by implementing one interface

## Validation Stop Criteria (all required)

- [ ] OOS Sharpe consistently ≥ 0.8 across all folds
- [ ] IS/OOS Sharpe gap < 0.3
- [ ] No single fold > 30% of total PnL
- [ ] Profitable under 1.5× cost stress
- [ ] Stable under ±20% parameter perturbation (Sharpe CV < 0.3)
- [ ] Max drawdown within declared limit (10% default)

## Realistic Cost Model

```
Binance:  maker=2bps, taker=4bps, borrow=~5%/yr, funding_interval=8h
Bybit:    maker=1bp,  taker=6bps, borrow=~6%/yr, funding_interval=8h
OKX:      maker=2bps, taker=5bps, borrow=~5.5%/yr, funding_interval=8h

Slippage model (vol-scaled):
  impact = k * σ * sqrt(notional / ADV)
  k = 0.1 (calibrated; tune with depth data if available)
```

## Data Sources

| Venue | Spot | Perp | Funding | Endpoint (public) |
|---|---|---|---|---|
| Binance | ✓ | ✓ | ✓ | api.binance.com/api/v3 + fapi/v1 |
| Bybit | ✓ | ✓ | ✓ | api.bybit.com/v5/market |
| OKX | ✓ | ✓ | ✓ | www.okx.com/api/v5/market |

## Known Backtest vs Live Gaps

1. **Maker fill rate**: Backtest uses 60% maker fill probability — live may differ by venue/size
2. **Funding prediction**: Backtest uses realized funding; live must predict next funding rate
3. **Simultaneous legs**: Leg execution has real latency; backtest assumes same-bar fills for both legs
4. **Borrow availability**: Spot short borrow may be unavailable during high-demand periods
5. **API downtime**: Not modeled; add heartbeat + reconnect logic for live
6. **Cross-venue settlement timing**: Binance/Bybit/OKX settle at same UTC times but predicted vs realized can differ by up to 1-2bps

## Go / No-Go Assessment

See `results/final_report.md` after running the full evaluation pipeline.

**Pre-run prior** (adversarial):
- SpotPerpFunding on BTC/ETH: *likely passes* under normal regimes; *fails* during low-funding bear markets (2022 H2) unless exit logic is tight
- PerpPerpFunding: *marginal* — cross-venue diff is usually < 1bp after costs on majors; needs altcoins which adds blowup risk
- BasisArb: *structurally sound* but perp basis convergence not guaranteed; backwardation periods (neg basis) make this bidirectional
- BasisMeanRevert / FundingMeanRevert: *fragile* — z-score regimes tend to break during regime transitions; treat as secondary only
- StatArb (BTC/ETH): *highest alpha potential* if cointegration holds; most likely to pass IS but fail OOS across full sample

## Literature & Reference Implementations

1. Avellaneda & Lee (2010) — "Statistical Arbitrage in the US Equities Market" (adapted for crypto)
2. Prado (2018) — *Advances in Financial Machine Learning*, Ch. 7 (walk-forward, purged CV)
3. BitMEX Research — "Cash and Carry" (2019)
4. [hummingbot](https://github.com/hummingbot/hummingbot) — architecture reference for backtest/live interface parity
5. [cryptofeed](https://github.com/bmoscon/cryptofeed) — normalized exchange data feed patterns
6. [jesse-ai/jesse](https://github.com/jesse-ai/jesse) — backtest/live code-sharing philosophy
7. Uniswap v3 whitepaper — CFMM slippage model (for DEX leg if added)
8. Burgess (2000) — Statistical Arbitrage Models of the FTSE 100
