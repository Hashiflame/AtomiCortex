# AtomiCortex

AI-powered crypto futures trading system.  
4H/Daily timeframe | Binance USDT-M Perpetual | Python 3.11.9 + Nautilus Trader 1.221.0

## Status

| Phase | Описание | Статус | Тесты |
|---|---|---|---|
| **1** | Data Pipeline | ✅ Completed | 62/62 |
| **2.1** | Data Catalog (Parquet → Nautilus) | ✅ Completed | 26/26 |
| **2.2** | BacktestEngine + BuyAndHold strategy | ✅ Completed | 26/26 |
| **2.3** | Feature Engineering | 🔲 Planned | — |
| **2.4** | ML Models (LightGBM) | 🔲 Planned | — |
| **3** | Live Trading (Nautilus LiveNode) | 🔲 Planned | — |


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
│   ├── data_catalog.py       # Parquet → Nautilus Bar/TradeTick
│   ├── backtest_runner.py    # BacktestEngine wrapper
│   └── strategies/
│       └── baseline_strategy.py   # BuyAndHoldStrategy
├── features/           # Phase 2.3 — feature engineering
├── models/             # Phase 2.4 — LightGBM
├── risk/               # Phase 3 — risk engine
└── telegram_bot/       # Phase 3 — signal bot

scripts/
├── download_historical.py
├── convert_to_parquet.py
├── check_data_quality.py
└── run_backtest.py           # backtesting CLI
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

## Tests

```bash
pytest tests/ -v          
pytest tests/test_backtest_engine.py -v   
```
