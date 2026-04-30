# AtomiCortex 

AI-powered crypto futures trading system.
4H/Daily timeframe | Binance & Bybit USDT-M Perpetual

## Stack
- **Execution:** Nautilus Trader (Rust core)
- **Data:** Cryptofeed + Binance Data Portal
- **Storage:** Parquet (ZSTD) + DuckDB + QuestDB
- **ML:** LightGBM (trend/range/highvol models)
- **Bot:** Telegram signals

## Structure
src/
├── ingestion/    # Data pipeline
├── features/     # Feature engineering
├── models/       # ML models
├── risk/         # Risk engine
├── execution/    # Nautilus strategies
└── telegram_bot/ # Signal bot

## Setup
```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Status
Phase 1: Data Pipeline (in progress)