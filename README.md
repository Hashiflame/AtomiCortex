# AtomiCortex

AI-powered crypto futures trading system.  
4H/Daily timeframe | Binance USDT-M Perpetual | Python 3.11.9 + Nautilus Trader 1.221.0

## Status

| Phase | Description | Status | Tests |
|---|---|---|---|
| **1** | Data Pipeline | ✅ Completed | 62/62 |
| **2.1** | Data Catalog (Parquet → Nautilus) | ✅ Completed | 26/26 |
| **2.2** | BacktestEngine + BuyAndHold strategy | ✅ Completed | 26/26 |
| **2.3** | Cost Model (fees + slippage + funding) | ✅ Completed | 28/28 |
| **2.4** | Metrics (Sharpe, Calmar, MaxDD, WinRate, PF) | ✅ Completed | 16/16 |
| **2.5** | Walk-Forward + Purged K-Fold CV | ✅ Completed | 20/20 |
| **2.6** | MLflow Experiment Tracker | ✅ Completed | 6/6 |
| **2.7** | Feature Engineering | 🔲 Planned | — |
| **2.8** | ML Models (LightGBM) | 🔲 Planned | — |
| **3** | Live Trading (Nautilus LiveNode) | 🔲 Planned | — |

**Total: 158/158 tests passing**

### Data

|  Symbol | Intervals | Volume | Period      |
|---      |---        |---     |---          |
| BTCUSDT | 4h, 1d, agg_trades | ~1.0B строк | 2020–2024 |
| ETHUSDT | 4h, 1d, agg_trades | ~0.9B строк | 2020–2024 |
| SOLUSDT | 4h, 1d, agg_trades | ~0.9B строк | 2020–2024 |


## Stack

- **Execution:** Nautilus Trader 1.221.0 (Rust/Cython core)
- **Data:** Cryptofeed + Binance Data Portal
- **Storage:** Parquet (ZSTD) + DuckDB + QuestDB
- **ML:** LightGBM (trend/range/highvol models)
- **Experiment Tracking:** MLflow (filesystem backend)
- **Bot:** Telegram signals

## Structure

```
src/
├── ingestion/          # Phase 1 — data pipeline
│   ├── binance_downloader.py
│   ├── parquet_converter.py
│   ├── data_quality.py
│   └── live_feed.py
├── execution/          # Phase 2 — backtest & live trading
│   ├── data_catalog.py           # Parquet → Nautilus Bar/TradeTick
│   ├── backtest_runner.py        # BacktestEngine wrapper + cost analytics
│   ├── cost_model.py             # FeeConfig, CostModel, RoundTripCost
│   ├── metrics.py                # MetricsResult, Sharpe, Calmar, MaxDD
│   ├── walk_forward.py           # WalkForwardValidator, PurgedKFoldCV
│   ├── experiment_tracker.py     # MLflow integration
│   └── strategies/
│       ├── baseline_strategy.py      # BuyAndHoldStrategy
│       └── random_entry_strategy.py  # RandomEntryStrategy (cost validation)
├── features/           # Phase 2.7 — feature engineering
├── models/             # Phase 2.8 — LightGBM
├── risk/               # Phase 3 — risk engine
└── telegram_bot/       # Phase 3 — signal bot

scripts/
├── download_historical.py
├── convert_to_parquet.py
├── check_data_quality.py
├── run_backtest.py               # backtesting CLI
├── validate_cost_model.py        # cost table display
└── run_walk_forward.py           # walk-forward validation CLI
```

## Setup

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Start Backtest

```bash
python scripts/run_backtest.py \
    --symbol BTCUSDT \
    --interval 4h \
    --start 2024-01-01 \
    --end 2024-06-30 \
    --capital 10000 \
    --strategy buy_and_hold
```

## Walk-Forward Validation

```bash
python scripts/run_walk_forward.py \
    --symbol BTCUSDT \
    --interval 4h \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --strategy buy_and_hold \
    --train-months 6 \
    --test-months 2

# Log results to MLflow
python scripts/run_walk_forward.py --symbol BTCUSDT --mlflow
```

## Cost Model Validation

```bash
python scripts/validate_cost_model.py
```

## Tests

```bash
pytest tests/ -v
pytest tests/test_backtest_engine.py -v
pytest tests/test_cost_model.py -v
pytest tests/test_walk_forward.py -v
```
