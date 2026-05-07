# AtomiCortex

AI-powered crypto futures trading system.  
BTC / ETH / SOL | 4H timeframe | Binance USDT-M Perpetual  
Python 3.11.9 · Nautilus Trader 1.221.0

---

## Status

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| **1** | Data Pipeline (download, Parquet, quality) | ✅ Completed | 37 |
| **2** | Backtest Engine (catalog, cost model, metrics, WF, MLflow) | ✅ Completed | 99 |
| **3** | Feature Engineering (microstructure, derivatives, regime) | ✅ Completed | 41 |
| **4** | ML Models (LightGBM, Optuna tuning, validation) | ✅ Completed | 70 |
| **5** | Risk Engine (pre-trade filters, circuit breaker, portfolio tracker) | ✅ Completed | 22 |
| **6** | Live Execution (ML strategy, paper trader, live trader, preload) | ✅ Completed | 78 |
| **7** | Telegram Bot (commands, payments, keyboards, signal bridge) | ✅ Completed | 162 |
| **8** | Production (deploy, watchdog, chaos tests) | ✅ Completed | 27 |

**Total: 536 tests passing**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AtomiCortex                              │
├──────────────────────┬──────────────────────────────────────────┤
│   Trading Bot        │   Telegram Bot                          │
│   (Nautilus)         │   (python-telegram-bot v21)             │
│                      │                                         │
│   MLTradingStrategy  │   SignalPoller (30s)                    │
│        │             │        │                                │
│        ▼             │        ▼                                │
│   SignalBridge ──────┼──► shared SQLite ◄── read               │
│   (sync, safe)       │    atomicortex.db      │                │
│                      │                   Broadcaster           │
│                      │                   │       │             │
│                      │              Premium    Free            │
│                      │              (full)    (teaser)         │
└──────────────────────┴──────────────────────────────────────────┘
```

### Signal Flow

1. **ML Strategy** detects regime → selects LightGBM model → generates signal
2. **Risk Engine** validates (drawdown, consecutive losses, funding, leverage)
3. **Signal Bridge** writes to `signals_log` table in shared SQLite
4. **Signal Poller** reads new records every 30 seconds
5. **Broadcaster** sends formatted alerts to Telegram subscribers

---

## Data

| Symbol  | Intervals           | Volume       | Period    |
|---------|---------------------|--------------|-----------|
| BTCUSDT | 4h, 1d, agg_trades  | ~1.0B строк  | 2020–2024 |
| ETHUSDT | 4h, 1d, agg_trades  | ~0.9B строк  | 2020–2024 |
| SOLUSDT | 4h, 1d, agg_trades  | ~0.9B строк  | 2020–2024 |

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Execution** | Nautilus Trader 1.221.0 (Rust/Cython core) |
| **Data** | Cryptofeed · Binance Data Portal · DuckDB |
| **Storage** | Parquet (ZSTD) · SQLite (WAL) · QuestDB |
| **ML** | LightGBM (trend + high_vol models) · Optuna HPO |
| **Features** | ADX regime detector · Microstructure · Derivatives |
| **Risk** | RiskEngine · CircuitBreaker · PortfolioTracker |
| **Experiment** | MLflow (filesystem backend) |
| **Bot** | python-telegram-bot v21 · Telegram Stars · CryptoBot USDT |
| **Deploy** | systemd · Watchdog · Heartbeat |

---

## Project Structure

```
AtomiCortex/
├── src/
│   ├── config.py                          # Centralized settings (pydantic-settings)
│   ├── logger.py                          # Structured logging (loguru)
│   │
│   ├── ingestion/                         # Phase 1 — Data Pipeline
│   │   ├── binance_downloader.py          #   Binance historical data downloader
│   │   ├── parquet_converter.py           #   CSV/JSON → Parquet (ZSTD)
│   │   ├── data_quality.py               #   Gap detection, outlier checks
│   │   ├── data_store.py                 #   Unified Parquet read/write interface
│   │   └── live_feed.py                  #   Real-time WebSocket feed (Cryptofeed)
│   │
│   ├── features/                          # Phase 3 — Feature Engineering
│   │   ├── feature_pipeline.py           #   End-to-end feature matrix builder
│   │   ├── microstructure.py             #   VWAP, CVD, OI, spread features
│   │   ├── derivatives.py               #   Funding rate, basis, OI delta
│   │   ├── regime_detector.py            #   ADX-first market regime classification
│   │   └── utils.py                      #   Rolling windows, normalization helpers
│   │
│   ├── models/                            # Phase 4 — ML Models
│   │   ├── lgbm_trainer.py               #   LightGBM training + signal generation
│   │   ├── training_pipeline.py          #   Automated train/validate pipeline
│   │   ├── dataset_builder.py            #   Feature matrix → train/test splits
│   │   ├── ml_validator.py               #   Statistical validation of models
│   │   └── statistical_tests.py          #   Significance tests for trading metrics
│   │
│   ├── risk/                              # Phase 5 — Risk Management
│   │   ├── risk_engine.py                #   Pre-trade risk filters + position sizing
│   │   ├── circuit_breaker.py            #   Emergency stop (drawdown, loss streaks)
│   │   └── portfolio_tracker.py          #   Real-time equity, PnL, drawdown tracking
│   │
│   ├── execution/                         # Phase 2 + 6 — Backtest & Live Trading
│   │   ├── data_catalog.py              #   Parquet → Nautilus Bar/TradeTick
│   │   ├── backtest_runner.py           #   BacktestEngine wrapper + cost analytics
│   │   ├── cost_model.py                #   FeeConfig, slippage, funding costs
│   │   ├── metrics.py                   #   Sharpe, Calmar, MaxDD, WinRate, PF
│   │   ├── walk_forward.py              #   Walk-Forward + Purged K-Fold CV
│   │   ├── experiment_tracker.py        #   MLflow integration
│   │   ├── live_trader.py               #   Nautilus LiveNode launcher
│   │   ├── paper_trader.py              #   Paper trading (simulated fills)
│   │   ├── signal_bridge.py             #   Trading→Telegram shared SQLite writer
│   │   ├── reconciler.py                #   Exchange position reconciliation
│   │   ├── heartbeat.py                 #   Process health heartbeat
│   │   ├── watchdog.py                  #   Process supervisor
│   │   └── strategies/
│   │       ├── ml_strategy.py           #   Main ML trading strategy (live/backtest)
│   │       ├── paper_strategy.py        #   Paper trading strategy
│   │       ├── baseline_strategy.py     #   BuyAndHold benchmark
│   │       └── random_entry_strategy.py #   Random entry (cost validation)
│   │
│   ├── telegram_bot/                      # Phase 7 — Telegram Bot
│   │   ├── bot.py                       #   Application orchestrator + handler routing
│   │   ├── broadcaster.py               #   Role-aware signal/event broadcasting
│   │   ├── database.py                  #   Users, signals, payments SQLite DB
│   │   ├── roles.py                     #   Role hierarchy + access decorators
│   │   ├── keyboards.py                 #   ReplyKeyboard + InlineKeyboard builders
│   │   ├── handlers_free.py             #   /start, /help, /stats, /subscribe, /mystatus
│   │   ├── handlers_premium.py          #   /signal, /history, /regime, /funding, /risk
│   │   ├── handlers_owner.py            #   /users, /health, /grant, /ban, /broadcast
│   │   ├── payments_stars.py            #   Telegram Stars (XTR) payment flow
│   │   ├── payments_crypto.py           #   CryptoBot USDT/TON payment flow
│   │   └── signal_poller.py             #   Polls shared SQLite for new signals/events
│   │
│   └── monitoring/                        # Phase 8 — Monitoring
│       ├── metrics_collector.py          #   System metrics collection
│       └── telegram_reporter.py          #   Automated reporting
│
├── scripts/
│   ├── download_historical.py            # Download historical klines
│   ├── download_funding_rate.py          # Download funding rate history
│   ├── convert_to_parquet.py             # Batch Parquet conversion
│   ├── check_data_quality.py             # Data quality validation CLI
│   ├── build_features.py                 # Feature matrix generation
│   ├── train_models.py                   # LightGBM training pipeline
│   ├── tune_models.py                    # Optuna hyperparameter tuning
│   ├── validate_ml_models.py             # Model validation suite
│   ├── analyze_regimes.py                # Regime distribution analysis
│   ├── run_backtest.py                   # Backtesting CLI
│   ├── run_walk_forward.py               # Walk-forward validation CLI
│   ├── validate_cost_model.py            # Cost model display
│   ├── run_paper.py                      # Paper trading launcher
│   ├── run_live.py                       # Live trading launcher
│   ├── run_live_feed.py                  # WebSocket feed launcher
│   ├── run_telegram_bot.py               # Telegram bot launcher
│   ├── run_watchdog.py                   # Watchdog supervisor
│   ├── verify_setup.py                   # Environment verification
│   └── create_env.sh                     # .env template generator
│
├── deploy/
│   ├── atomicortex-bot.service           # systemd — trading bot
│   ├── atomicortex-telegram.service      # systemd — Telegram bot
│   └── atomicortex-watchdog.service      # systemd — process watchdog
│
├── tests/                                 # 536 tests
│   ├── test_binance_downloader.py        #   5 tests
│   ├── test_parquet_converter.py         #  12 tests
│   ├── test_data_quality.py              #  20 tests
│   ├── test_live_feed.py                 #  26 tests
│   ├── test_backtest_engine.py           #  26 tests
│   ├── test_cost_model.py                #  28 tests
│   ├── test_walk_forward.py              #  45 tests
│   ├── test_feature_pipeline.py          #  20 tests
│   ├── test_regime_detector.py           #  21 tests
│   ├── test_lgbm_trainer.py              #  25 tests
│   ├── test_optuna_trainer.py            #  20 tests
│   ├── test_ml_validator.py              #  25 tests
│   ├── test_risk_engine.py               #  22 tests
│   ├── test_ml_strategy.py               #  49 tests
│   ├── test_paper_trader.py              #  29 tests
│   ├── test_telegram_db_roles.py         #  32 tests
│   ├── test_telegram_handlers.py         #  30 tests
│   ├── test_payments.py                  #  34 tests
│   ├── test_keyboards.py                 #  28 tests
│   ├── test_signal_bridge.py             #  19 tests
│   ├── test_chaos.py                     #  20 tests
│   └── conftest.py
│
├── data/                                  # Data storage (gitignored)
│   ├── raw/                              #   Raw CSV/JSON downloads
│   └── features/                         #   ML feature matrices + models
│
├── logs/                                  # Trading logs (gitignored)
├── notebooks/                             # Jupyter analysis notebooks
├── docs/                                  # Documentation
├── .env                                   # Environment variables (gitignored)
├── .env.example                           # Environment template
└── pytest.ini                             # Test configuration
```

---

## Telegram Bot

### Role-based Menu

| Button | Free | Premium | Owner |
|--------|:----:|:-------:|:-----:|
| 📊 Статистика | ✅ | ✅ | ✅ |
| ⭐ Подписка | ✅ | — | — |
| ❓ Помощь | ✅ | — | — |
| 🟢 Сигнал | 🔒 | ✅ | ✅ |
| 📈 История | 🔒 | ✅ | ✅ |
| 🌡 Режим рынка | 🔒 | ✅ | ✅ |
| 💰 Funding | 🔒 | ✅ | ✅ |
| 👥 Юзеры | — | — | ✅ |
| 🖥 Здоровье | — | — | ✅ |

### Payment Methods

- **Telegram Stars** (XTR) — 30 / 90 дней
- **CryptoBot USDT** — 30 / 90 дней
- **Manual** — через владельца

### Signal Alerts

Premium/Owner получают полный сигнал:
```
══════════════════════════════
🟢 LONG BTCUSDT-PERP PERP
══════════════════════════════
⏰ 2026-05-07 12:00 UTC
💵 Вход:    $103,500
🛑 Стоп:    $101,400 (-2.0%)
🎯 Тейк:    $106,600 (+3.0%)
⚖️ R:R:     1:1.5
🤖 Режим:   TREND
📊 Conf:    73%
💸 Funding: +0.010%
📏 Размер:  0.0500 BTC ($5,175)
⚡ Leverage: 5.0x
══════════════════════════════
```

Free получают тизер с `/subscribe`.

---

## Setup

```bash
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Training

```bash
# Build feature matrices
python scripts/build_features.py

# Train LightGBM models
python scripts/train_models.py

# Optuna hyperparameter tuning
python scripts/tune_models.py

# Validate trained models
python scripts/validate_ml_models.py
```

### Backtesting

```bash
python scripts/run_backtest.py \
    --symbol BTCUSDT \
    --interval 4h \
    --start 2024-01-01 \
    --end 2024-06-30 \
    --capital 10000

# Walk-forward validation
python scripts/run_walk_forward.py \
    --symbol BTCUSDT \
    --interval 4h \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --train-months 6 \
    --test-months 2 \
    --mlflow
```

### Live Trading

```bash
# Paper trading (simulated)
python scripts/run_paper.py

# Live trading (Binance testnet/mainnet)
python scripts/run_live.py

# Telegram bot
python scripts/run_telegram_bot.py
```

### systemd (Production)

```bash
sudo cp deploy/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable atomicortex-bot atomicortex-telegram atomicortex-watchdog
sudo systemctl start atomicortex-bot atomicortex-telegram atomicortex-watchdog
```

## Tests

```bash
# All tests
pytest tests/ -v

# By module
pytest tests/test_ml_strategy.py -v       # 49 tests
pytest tests/test_signal_bridge.py -v      # 19 tests
pytest tests/test_keyboards.py -v          # 28 tests
pytest tests/test_payments.py -v           # 34 tests
```

---

## Roadmap

### 🚧 В разработке

| Feature | Timeframes | Status |
|---------|-----------|--------|
| Скальпинг сигналы | **1m, 5m** | 🔨 В разработке |
| Интрадей сигналы | **15m, 1h** | 🔨 В разработке |
| Multi-timeframe анализ | 1m + 5m + 15m + 1h + 4h | 📋 Планируется |
| Расширенный ML пайплайн | Отдельные модели per timeframe | 📋 Планируется |

> Текущая версия работает на **4H таймфрейме**. Поддержка 1m, 5m, 15m и 1h
> потребует адаптации feature pipeline, отдельных LightGBM моделей для каждого
> таймфрейма, и оптимизации SignalBridge для высокочастотных сигналов.

---

## License

Private repository. All rights reserved.
