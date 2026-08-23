# АУДИТ-2 / ПРОХОД 0 — Инвентаризация: точки входа и карта модулей

HEAD: `f4af5fd210b32db8af4478f0e2f440e2eb504ccc` (2026-08-22 16:18:08 +0530)
Рабочее дерево чисто. Всё ниже — из свежего вывода команд, не из памяти.

Протокол аудита — в [`00_method.md`](00_method.md).

---

## 1. ТОЧКИ ВХОДА

### 1.1 Исполняемые точки

#### 1.1.1 `scripts/*.py` — все 41 файла имеют `if __name__`

Проверено: `grep -c "if __name__" scripts/*.py` → у каждого `1`.
Описание — первая содержательная строка докстринга (`ast.get_docstring`).

| Скрипт | Что делает (из докстринга) |
|---|---|
| `analyze_regimes.py` | Анализ рыночных режимов на исторических данных, печать сводной статистики |
| `build_15m_dataset.py` | Строит feature+target датасет для обучения 15m LightGBM |
| `build_1h_dataset.py` | Строит feature+target датасет для обучения 1H LightGBM; читает `klines_1h` Parquet, пайплайн с `timeframe='1h'` |
| `build_features_15m_agg.py` | Дополняет существующие 4H feature-паркеты 4 колонками `agg_15m_*` (Block-3 Step 3); пишет в соседний каталог |
| `build_features.py` | Строит ML feature-матрицу для одного символа и диапазона дат |
| `build_meta_dataset.py` | Block 4 / Step 2 — строит датасет мета-разметки (López de Prado AFML §3.6) |
| `build_mtf_features.py` | Строит feature-матрицы для 1H и 15m из сырого OHLCV Parquet |
| `check_data_quality.py` | Отчёт о качестве данных по Parquet feature store |
| `check_mtf_data_quality.py` | Валидация качества MTF-данных: полнота, дубли, нулевые цены, непрерывность меток, читаемость DuckDB |
| `check_signal_freshness.py` | Проверка свежести сигналов; источник 1 — heartbeat в Redis, поле `last_signal_ts` |
| `convert_mtf_to_parquet.py` | Конвертирует скачанные MTF CSV klines в Parquet (ZSTD) для 1m/5m/15m/1h |
| `convert_to_parquet.py` | CLI конвертации сырых Binance CSV в Parquet + инспекция стора |
| `download_funding_rate.py` | Скачивает исторический funding rate с Binance USDT-M Futures REST (`GET /fapi/v1/fundingRate`) |
| `download_historical.py` | CLI скачивания исторических USDT-M futures c Binance Data Portal |
| `download_mtf_data.py` | Мультитаймфреймовый загрузчик klines (1m/5m/15m/1h) с Binance Data Portal |
| `feature_selection_v3.py` | Block 3 / Step 1 — Clustered-MDA отбор фич для v3-моделей |
| `migrate_db_v3.py` | Идемпотентная миграция БД v3: схема stats-engine, колонки в `signals_log` |
| `retrain_v3.py` | Block 2 — vol-scaled симметричные triple-barrier метки + AFML sample-uniqueness веса + переобучение LightGBM |
| `run_api.py` | Запуск REST API (uvicorn, строковый импорт `"src.api.main:app"`) |
| `run_backtest.py` | CLI запуска бэктестов |
| `run_live_15m.py` | Лаунчер 15m live-трейдера (ISOLATED), может ставить РЕАЛЬНЫЕ ордера |
| `run_live_feed.py` | Запуск live Binance Futures WebSocket-фида |
| `run_live.py` | Лаунчер live-торговли (основная прод-точка) |
| `run_paper_15m.py` | Лаунчер 15m paper-торговли (ISOLATED): своя SQLite `data/atomicortex_15m.db`, свой heartbeat, свои модели |
| `run_paper.py` | Paper-торговля на live Nautilus WebSocket-данных с симулированным исполнением |
| `run_reconciler.py` | Запуск сверки сигналов |
| `run_telegram_bot.py` | Точка входа Telegram-бота |
| `run_walk_forward.py` | CLI walk-forward валидации |
| `run_watchdog.py` | Запуск Watchdog отдельным процессом |
| `train_15m_models.py` | Обучение LightGBM для 15m (бэкенд — `lgbm_trainer.py`) |
| `train_1h_models.py` | Обучение LightGBM для 1H (бэкенд — `lgbm_trainer.py`) |
| `train_meta_model.py` | Block 4 / Step 3 — обучение take/skip мета-бустера |
| `train_models.py` | Обучение LightGBM по каждому рыночному режиму |
| `tune_models.py` | Optuna-тюнинг гиперпараметров LightGBM по режимам |
| `validate_15m_models.py` | Валидация 15m-моделей по go/no-go критериям, включая DSR, PBO, t-stat |
| `validate_1h_models.py` | Валидация 1H-моделей по go/no-go критериям, включая DSR, PBO, t-stat |
| `validate_cost_model.py` | Печать разбивки round-trip издержек для разных размеров позиции |
| `validate_ml_models.py` | Purged K-Fold CV + Walk-Forward + DSR/PBO/t-stat |
| `verify_phase1.py` | Проверка завершённости Phase 1: наличие MTF-файлов, прохождение quality-чеков |
| `verify_setup.py` | Верификация окружения |

Плюс 2 shell-скрипта: `scripts/create_env.sh`, `scripts/run_phase1_download.sh`.

#### 1.1.2 systemd-юниты (`deploy/`)

Дословные `ExecStart` (с продолжениями строк):

| Юнит | Description | Type / Restart | ExecStart |
|---|---|---|---|
| `atomicortex-bot.service` | `AtomiCortex Trading Bot` | simple / on-failure | `scripts/run_live.py --mode paper --dry-run --symbols BTCUSDT-PERP …` |
| `atomicortex-telegram.service` | `AtomiCortex Telegram Bot` | simple / on-failure | `scripts/run_telegram_bot.py` |
| `atomicortex-signal-check.service` | `AtomiCortex Signal Starvation Check (oneshot)` | oneshot | `scripts/check_signal_freshness.py --log-level INFO` |
| `atomicortex-signal-check.timer` | `AtomiCortex Signal Freshness Check` | — | `OnBootSec=5min`, `OnUnitActiveSec=1h`, `Unit=atomicortex-signal-check.service` |
| `atomicortex-bot-15m.service` | `AtomiCortex Trading Bot 15m (ISOLATED)` | simple / on-failure | `scripts/run_paper_15m.py --symbol BTCUSDT-PERP --capital 10000 --trading-mode testnet …` |
| `atomicortex-watchdog.service` | `AtomiCortex External Watchdog` | simple / always | `scripts/run_watchdog.py --redis-host localhost --trading-mode testnet --check-interval 15 …` |
| `atomicortex-watchdog-15m.service` | `AtomiCortex External Watchdog 15m (ISOLATED)` | simple / always | `scripts/run_watchdog.py --service-name 15m --heartbeat-key bot_15m_heartbeat --symbol BTCUSDT …` |
| `atomicortex-reconciler.service` | `AtomiCortex Signal Reconciler (oneshot)` | oneshot | `scripts/run_reconciler.py --all --trading-mode ${TRADING_MODE} --log-level INFO` |
| `atomicortex-reconciler.timer` | `Run AtomiCortex Signal Reconciler every 15 minutes` | — | `OnBootSec=2min`, `OnUnitActiveSec=15min`, `Unit=atomicortex-reconciler.service` |
| `atomicortex-api.service` | `AtomiCortex REST API` | simple / on-failure | `scripts/run_api.py --host 0.0.0.0 --port 8080` |

Пути в юнитах абсолютные и указывают на `/home/hashiflame/AtomiCortex/.venv/bin/python`
(локальный путь разработки — `/home/asus/Desktop/AtomiCortex`; юниты рассчитаны на VM).

#### 1.1.3 CI

Один воркфлоу: `.github/workflows/deploy.yml`, `name: Deploy to Cloud VM`,
триггер `on: push: branches: [master]`.

Джоб `test`: python 3.11.9, `pip install -r requirements.txt`, затем дословно:

```
pytest tests/ -x --ignore=tests/test_live_feed.py -q
```

Джоб `deploy` (`needs: test`, `if: github.ref == 'refs/heads/master'`): ssh на VM,
`git pull origin master`, `pip install -r requirements.txt --quiet`, затем
`sudo -n /usr/local/sbin/atomicortex-install-units`, если он исполняемый; иначе —
предупреждение и пропуск systemd-деплоя.

### 1.2 Что реально запускается

**В проде** — определяется файлом-манифестом `deploy/units.enabled`.
`deploy/install_units.sh:41` — `MANIFEST="$REPO_DIR/units.enabled"`, читает его
построчно (`done < "$MANIFEST"`, строка 78) и, `install_units.sh:98`, печатает
`==> Installing ${#UNITS[@]} unit(s) listed in units.enabled...`.

Содержимое `deploy/units.enabled` (дословно, без комментариев):

```
atomicortex-bot.service
atomicortex-telegram.service
atomicortex-signal-check.service
atomicortex-signal-check.timer
```

То есть в проде исполняются ровно **три** точки входа Python:
`scripts/run_live.py` (paper, `--dry-run`), `scripts/run_telegram_bot.py`,
`scripts/check_signal_freshness.py` (по таймеру раз в час).

Тот же файл прямо перечисляет, что НЕ деплоится (дословная цитата шапки
`units.enabled`):

> «Everything else in deploy/ stays in the repository but is not touched on the
> VM: the 15m bot, the watchdogs, the REST API and the reconciler are kept for
> reference and for the unit linters, and are not part of the current paper
> --dry-run deployment.»

**В CI** — запускается только `pytest tests/` (см. 1.1.3). Никакие `scripts/*`
в CI не вызываются.

**Вручную** — всё остальное. Референс — README.md и `docs/troubleshooting.md`
(таблица ссылок в §3.3).

### 1.3 Что не запускается нигде

Не в `units.enabled`, не в CI:

- `deploy/atomicortex-bot-15m.service` → `run_paper_15m.py`
- `deploy/atomicortex-watchdog.service` → `run_watchdog.py`
- `deploy/atomicortex-watchdog-15m.service` → `run_watchdog.py`
- `deploy/atomicortex-reconciler.service` + `.timer` → `run_reconciler.py`
- `deploy/atomicortex-api.service` → `run_api.py`

Пять юнитов из десяти файлов в `deploy/` присутствуют в репозитории, но
манифестом не выбраны. Это **не** дефект сам по себе — файл `units.enabled`
объявляет это намеренно; проверка на противоречие (напр. код, который считает
reconciler работающим) — предмет прохода 8.

---

## 2. КАРТА МОДУЛЕЙ

### 2.1 `find src -name '*.py' -exec wc -l {} + | sort -rn`

Дословный вывод:

```
  27800 total
   2262 src/execution/strategies/ml_strategy.py
   1533 src/models/lgbm_trainer.py
    979 src/telegram_bot/database.py
    946 src/execution/watchdog.py
    792 src/telegram_bot/bot.py
    705 src/features/derivatives.py
    665 src/features/regime_detector.py
    625 src/execution/strategies/ml_strategy_15m.py
    613 src/features/feature_pipeline.py
    580 src/telegram_bot/handlers_owner.py
    549 src/ingestion/data_quality.py
    544 src/ingestion/binance_downloader.py
    522 src/telegram_bot/handlers_premium.py
    521 src/risk/portfolio_tracker.py
    511 src/analytics/stats_engine.py
    504 src/ingestion/parquet_converter.py
    490 src/features/session_features.py
    487 src/telegram_bot/signal_poller.py
    465 src/models/dataset_builder.py
    462 src/models/statistical_tests.py
    449 src/risk/risk_engine.py
    446 src/telegram_bot/handlers_free.py
    446 src/execution/walk_forward.py
    439 src/models/ml_validator.py
    419 src/execution/reconciler_signals.py
    417 src/features/live_feature_state.py
    408 src/config.py
    395 src/telegram_bot/payments_crypto.py
    389 src/telegram_bot/payments_stars.py
    379 src/execution/backtest_runner.py
    367 src/execution/live_trader.py
    366 src/features/microstructure.py
    366 src/execution/signal_bridge.py
    346 src/features/mtf_context.py
    332 src/monitoring/metrics_collector.py
    331 src/execution/strategies/meta_strategy.py
    331 src/api/main.py
    326 src/telegram_bot/keyboards.py
    317 src/ingestion/live_feed.py
    309 src/execution/paper_trader.py
    306 src/risk/circuit_breaker.py
    305 src/features/orb_features.py
    286 src/telegram_bot/broadcaster.py
    275 src/execution/reconciler.py
    268 src/ingestion/data_store.py
    254 src/execution/metrics.py
    232 src/execution/data_catalog.py
    227 src/execution/pending_orders_store.py
    221 src/execution/heartbeat.py
    220 src/telegram_bot/roles.py
    209 src/features/triple_barrier.py
    203 src/execution/binance_rate_limiter.py
    183 src/monitoring/telegram_reporter.py
    178 src/monitoring/signal_alert_state.py
    175 src/execution/strategies/paper_strategy.py
    173 src/logger.py
    162 src/risk/risk_state_store.py
    162 src/features/agg_15m.py
    156 src/models/training_pipeline.py
    146 src/execution/experiment_tracker.py
    138 src/telegram_bot/signal_formatter.py
    137 src/execution/startup_check.py
    132 src/execution/cost_model.py
    113 src/models/temporal_split.py
     91 src/execution/strategies/random_entry_strategy.py
     90 src/configs/strategy_15m.py
     83 src/execution/strategies/baseline_strategy.py
     78 src/configs/strategy_1h.py
     70 src/telegram_bot/timeframes.py
     56 src/features/utils.py
     45 src/features/window_sizes.py
     40 src/patches/nautilus_enums.py
     23 src/telegram_bot/__init__.py
     22 src/risk/__init__.py
      5 src/features/__init__.py
      1 src/configs/__init__.py
      1 src/api/__init__.py
      1 src/analytics/__init__.py
      0 src/patches/__init__.py
      0 src/monitoring/__init__.py
      0 src/models/__init__.py
      0 src/__init__.py
      0 src/ingestion/__init__.py
      0 src/execution/strategies/__init__.py
      0 src/execution/__init__.py
```

### 2.2 Ответственность модулей

Выведена из докстринга (`ast.get_docstring`, первые содержательные строки) или,
где докстринг отсутствует/пуст, отмечена как отсутствующая.

**`src/` корень**

| Модуль | Ответственность |
|---|---|
| `config.py` | Централизованная конфигурация из `.env` через pydantic-settings; доступ только через `get_settings()` |
| `logger.py` | Централизованное логирование на loguru; намеренно НЕ импортирует `src.config` на уровне модуля во избежание циклов |

**`src/ingestion/`**

| Модуль | Ответственность |
|---|---|
| `binance_downloader.py` | Асинхронное (aiohttp) скачивание USDT-M futures с публичного Binance Data Portal |
| `parquet_converter.py` | Конвертация сырых Binance CSV в Parquet (ZSTD), партиционирование по дате |
| `data_store.py` | DuckDB-интерфейс к Parquet; соединение in-memory, чтение прямо с диска |
| `data_quality.py` | Пять проверок качества Parquet feature store |
| `live_feed.py` | Живые данные через Cryptofeed + Binance Futures WebSocket: TRADES, L2_BOOK, FUNDING |

**`src/features/`**

| Модуль | Ответственность |
|---|---|
| `window_sizes.py` | Единственный источник истины по длинам lookback-окон; листовой модуль, ничего из `src` не импортирует |
| `utils.py` | Общие Polars-выражения для feature-модулей |
| `microstructure.py` | Микроструктурные фичи: CVD, объём, ценовые паттерны |
| `derivatives.py` | Деривативные фичи: funding rate, open interest, базис |
| `regime_detector.py` | Детекция режима: Hurst (R/S), ADX, ATR-перцентиль + композитный классификатор (TREND_UP/TREND_DOWN/…) |
| `session_features.py` | Внутридневная структура рынка (сессии) для 1H и 15m |
| `orb_features.py` | Opening Range Breakout: high/low первых 4 баров (1 час) после открытия сессии |
| `mtf_context.py` | Контекст старшего таймфрейма (HTF) как фичи |
| `agg_15m.py` | Агрегация 16 дочерних 15m-баров в микроструктурные фичи 4H-бара (Block-3 Step 3) |
| `triple_barrier.py` | Triple-Barrier разметка (AFML гл. 3) вместо fixed-horizon `sign(return)` |
| `feature_pipeline.py` | Главный пайплайн: OHLCV + деривативы из DataStore → единый Parquet-ready DataFrame |
| `live_feature_state.py` | Состояние live-фич: скользящие буферы и последние деривативные данные для `build_from_buffer()` |

**`src/models/`**

| Модуль | Ответственность |
|---|---|
| `dataset_builder.py` | Подготовка данных к LightGBM: загрузка и объединение мультисимвольных feature-паркетов |
| `temporal_split.py` | Честное разбиение train/test для одно- и мультисимвольных датасетов (корень ML-018) |
| `lgbm_trainer.py` | Бинарный LightGBM-тренер по режимам (ML-017: переход с 3-классов на бинарную UP/DOWN) |
| `training_pipeline.py` | End-to-end: итерация по режимам, обучение, оценка, сводный отчёт |
| `ml_validator.py` | Валидация: Purged K-Fold CV и Walk-Forward |
| `statistical_tests.py` | Deflated Sharpe Ratio (López de Prado 2014), PBO, t-stat |

**`src/execution/`**

| Модуль | Ответственность |
|---|---|
| `data_catalog.py` | Загрузка Parquet и конвертация в объекты Nautilus |
| `backtest_runner.py` | Запуск Nautilus `BacktestEngine` |
| `walk_forward.py` | Walk-forward валидация и purged K-fold CV |
| `metrics.py` | Метрики производительности для бэктестов и walk-forward окон |
| `experiment_tracker.py` | MLflow-трекинг; mlflow импортируется лениво внутри методов, чтобы не тянуть matplotlib при импорте модуля |
| `cost_model.py` | Модель издержек: комиссии, проскальзывание, funding |
| `live_trader.py` | Live/testnet через Nautilus TradingNode: Binance USDT-futures data + execution клиенты |
| `paper_trader.py` | Симуляция исполнения без реального капитала на live WebSocket-ценах; модель slippage (bps) + комиссии |
| `heartbeat.py` | Dead-man's switch: периодический heartbeat в Redis; ключ истекает через `heartbeat_ttl` |
| `watchdog.py` | Внешний watchdog отдельным процессом: читает heartbeat-ключ каждые `check_interval` секунд |
| `startup_check.py` | Fail-fast проверка (fix A1): демон-поток, проверяющий подключение DataEngine и ExecEngine после grace period |
| `signal_bridge.py` | Мост бот↔Telegram через общую SQLite: бот пишет сигналы и события |
| `pending_orders_store.py` | Crash-safe дисковый стор параметров pending stop-loss (`RiskDecision` + `TradeSignal`) |
| `binance_rate_limiter.py` | Централизованный лимитер Binance Futures REST (H22): 2400 weight/мин на IP; 429→бан 2 мин, 418→до 3 суток |
| `reconciler.py` | Сверка внутреннего состояния позиций с реальными позициями биржи через Binance REST |
| `reconciler_signals.py` | Закрытие «осиротевших» `open`-сигналов переигрыванием исторических цен (`SignalBridge.close_signal` срабатывает только из `on_position_closed`) |

**`src/execution/strategies/`**

| Модуль | Ответственность |
|---|---|
| `ml_strategy.py` | Главная live/backtest стратегия: ML-сигналы по режимам (trend + high_vol LightGBM) |
| `ml_strategy_15m.py` | 15m стратегия, полностью изолированная от 4H: своя SQLite `data/atomicortex_15m.db` |
| `meta_strategy.py` | Block 4 — слой мета-разметки (две класса: гейт + стратегия) |
| `paper_strategy.py` | Тонкая обёртка над `MLTradingStrategy`: перехват отправки ордеров, исполнение через `PaperTrader` |
| `baseline_strategy.py` | Buy-and-hold бейзлайн для валидации бэктест-движка |
| `random_entry_strategy.py` | Случайные входы/выходы для валидации cost-модели |

**`src/risk/`**

| Модуль | Ответственность |
|---|---|
| `risk_engine.py` | Пред-трейд фильтры, ATR-based сайзинг позиции, расчёт stop/take-profit |
| `portfolio_tracker.py` | Реальное состояние портфеля: equity, позиции, P&L, просадка, счётчик подряд-убытков — вход для RiskEngine |
| `circuit_breaker.py` | Многоуровневый прерыватель: уменьшает размер позиции либо полностью останавливает торговлю |
| `risk_state_store.py` | Crash-safe персистенция состояния `PortfolioTracker` + `CircuitBreaker` |

**`src/monitoring/`**

| Модуль | Ответственность |
|---|---|
| `metrics_collector.py` | Сбор метрик из `PortfolioTracker` в SQLite для Grafana и Telegram-отчётов |
| `telegram_reporter.py` | Отправка трейд-алертов, дневных/недельных отчётов и critical-алертов через Telegram Bot API (aiohttp) |
| `signal_alert_state.py` | Персистентный ledger дедупликации алертов о голоде по сигналам |

**`src/telegram_bot/`**

| Модуль | Ответственность |
|---|---|
| `bot.py` | Сборка Application (PTB v21): БД, хендлеры, broadcaster, платежи |
| `database.py` | SQLite-персистенция пользователей, логов сигналов и событий; connection-per-call |
| `handlers_free.py` / `handlers_premium.py` / `handlers_owner.py` | Команды free / premium / owner уровня |
| `roles.py` | RBAC-декораторы для PTB v21; читает `OWNER_ID` из `TELEGRAM_ADMIN_ID` |
| `keyboards.py` | Ролевые ReplyKeyboard-меню и InlineKeyboard-билдеры |
| `broadcaster.py` | Автоматические уведомления по ролям: сигналы, смены режима, circuit breaker, дневные отчёты |
| `signal_poller.py` | Опрос общей SQLite на новые сигналы/события от бот-процесса и пересылка их дальше |
| `signal_formatter.py` | Единственный источник истины рендера сигнала (full / compact / open); чистые функции |
| `timeframes.py` | Единый реестр таймфреймов: добавление ТФ = одна строка в `TIMEFRAMES` |
| `payments_crypto.py` | Платежи USDT/TON через @CryptoBot, поллинг вместо вебхуков |
| `payments_stars.py` | Нативные Telegram Stars (XTR) через invoice API PTB v21 |

**`src/analytics/`, `src/api/`, `src/patches/`, `src/configs/`**

| Модуль | Ответственность |
|---|---|
| `analytics/stats_engine.py` | Расчёт торговой статистики из `signals_log` по нескольким изолированным БД, кэш в `performance_cache` |
| `api/main.py` | FastAPI (read-only) поверх изолированных сигнальных БД; статистика из `StatsEngine` (cache-first) |
| `patches/nautilus_enums.py` | Monkey-patch: добавляет `TRADING_HALT` в `BinanceFuturesContractStatus` — msgspec-декодеру не хватает значения |
| `configs/strategy_1h.py`, `configs/strategy_15m.py` | Параметры стратегий 1H и 15M |

Без докстринга: `src/__init__.py`, `src/execution/__init__.py`,
`src/execution/strategies/__init__.py`, `src/ingestion/__init__.py`,
`src/models/__init__.py`, `src/monitoring/__init__.py`, `src/patches/__init__.py`
(все нулевой длины, кроме перечисленных в 2.1 непустых `__init__`).

### 2.3 Граф импортов внутри `src/`

Построен AST-разбором всех `.py` в `src/`, `scripts/`, `tests/`; учитываются
`from src... import` и `import src...`. Импортёры в `scripts/` показаны с
префиксом `!`, пути в `src/` — без префикса `src/`. Колонка «тестов» — число
тестовых файлов, ссылающихся на модуль импортом.

| модуль | импортёры в src/ | импортёры в scripts/ (`!`) | тестов |
|---|---|---|---|
| `src/analytics/__init__.py` | — | — | 0 |
| `src/analytics/stats_engine.py` | api/main.py, execution/reconciler_signals.py, telegram_bot/handlers_free.py, telegram_bot/handlers_premium.py | — | 5 |
| `src/api/__init__.py` | — | — | 0 |
| `src/api/main.py` | — | — | 2 |
| `src/config.py` | analytics/stats_engine.py, execution/live_trader.py, telegram_bot/bot.py | !check_signal_freshness.py, !run_live.py, !run_paper.py, !run_telegram_bot.py, !run_watchdog.py, !verify_setup.py | 3 |
| `src/configs/__init__.py` | — | — | 0 |
| `src/configs/strategy_15m.py` | — | !build_15m_dataset.py, !train_15m_models.py, !validate_15m_models.py | 2 |
| `src/configs/strategy_1h.py` | — | !build_1h_dataset.py, !train_1h_models.py, !validate_1h_models.py | 2 |
| `src/execution/__init__.py` | — | — | 0 |
| `src/execution/backtest_runner.py` | execution/experiment_tracker.py, execution/walk_forward.py | !run_backtest.py, !run_walk_forward.py | 4 |
| `src/execution/binance_rate_limiter.py` | execution/reconciler.py, execution/watchdog.py | — | 1 |
| `src/execution/cost_model.py` | execution/backtest_runner.py, risk/risk_engine.py | !validate_cost_model.py | 1 |
| `src/execution/data_catalog.py` | execution/backtest_runner.py | — | 3 |
| `src/execution/experiment_tracker.py` | — | !run_walk_forward.py | 1 |
| `src/execution/heartbeat.py` | execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py | — | 2 |
| `src/execution/live_trader.py` | — | !run_live.py, !run_live_15m.py, !run_paper.py, !run_paper_15m.py | 3 |
| `src/execution/metrics.py` | analytics/stats_engine.py, execution/backtest_runner.py, execution/experiment_tracker.py, execution/walk_forward.py | — | 4 |
| `src/execution/paper_trader.py` | execution/strategies/paper_strategy.py | — | 1 |
| `src/execution/pending_orders_store.py` | execution/strategies/ml_strategy.py | — | 1 |
| `src/execution/reconciler.py` | — | — | 1 |
| `src/execution/reconciler_signals.py` | — | !run_reconciler.py | 1 |
| `src/execution/signal_bridge.py` | execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py | — | 11 |
| `src/execution/startup_check.py` | execution/live_trader.py | — | 1 |
| `src/execution/strategies/__init__.py` | — | — | 1 |
| `src/execution/strategies/baseline_strategy.py` | — | !run_backtest.py, !run_walk_forward.py | 2 |
| `src/execution/strategies/meta_strategy.py` | — | — | 5 |
| `src/execution/strategies/ml_strategy.py` | execution/live_trader.py, execution/strategies/meta_strategy.py, execution/strategies/ml_strategy_15m.py, execution/strategies/paper_strategy.py | — | 19 |
| `src/execution/strategies/ml_strategy_15m.py` | — | !run_live_15m.py, !run_paper_15m.py | 7 |
| `src/execution/strategies/paper_strategy.py` | — | — | 1 |
| `src/execution/strategies/random_entry_strategy.py` | — | !run_backtest.py, !run_walk_forward.py | 1 |
| `src/execution/walk_forward.py` | execution/experiment_tracker.py, models/ml_validator.py | !run_walk_forward.py, !train_15m_models.py, !train_1h_models.py, !validate_15m_models.py, !validate_1h_models.py | 7 |
| `src/execution/watchdog.py` | — | !run_watchdog.py | 11 |
| `src/features/__init__.py` | — | — | 0 |
| `src/features/agg_15m.py` | — | !build_features_15m_agg.py | 0 |
| `src/features/derivatives.py` | features/feature_pipeline.py | !build_15m_dataset.py, !build_1h_dataset.py | 7 |
| `src/features/feature_pipeline.py` | execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py, features/__init__.py, models/dataset_builder.py | !analyze_regimes.py, !build_15m_dataset.py, !build_1h_dataset.py, !build_features.py, !build_mtf_features.py | 7 |
| `src/features/live_feature_state.py` | execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py | — | 5 |
| `src/features/microstructure.py` | features/feature_pipeline.py | !build_15m_dataset.py, !build_1h_dataset.py, !build_mtf_features.py | 6 |
| `src/features/mtf_context.py` | features/feature_pipeline.py | — | 1 |
| `src/features/orb_features.py` | features/feature_pipeline.py | — | 2 |
| `src/features/regime_detector.py` | execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py, features/feature_pipeline.py | !analyze_regimes.py, !build_15m_dataset.py, !build_1h_dataset.py, !build_mtf_features.py | 5 |
| `src/features/session_features.py` | features/feature_pipeline.py | !build_15m_dataset.py | 2 |
| `src/features/triple_barrier.py` | models/dataset_builder.py | !build_15m_dataset.py, !build_1h_dataset.py | 4 |
| `src/features/utils.py` | features/derivatives.py, features/microstructure.py, features/mtf_context.py, features/orb_features.py, features/session_features.py | — | 1 |
| `src/features/window_sizes.py` | execution/strategies/ml_strategy.py, features/feature_pipeline.py, features/live_feature_state.py, features/regime_detector.py | — | 5 |
| `src/ingestion/__init__.py` | — | — | 0 |
| `src/ingestion/binance_downloader.py` | — | !download_historical.py | 1 |
| `src/ingestion/data_quality.py` | — | !check_data_quality.py | 2 |
| `src/ingestion/data_store.py` | execution/reconciler_signals.py, execution/strategies/ml_strategy.py, features/feature_pipeline.py, models/dataset_builder.py | !analyze_regimes.py, !build_15m_dataset.py, !build_1h_dataset.py, !build_features.py, !build_features_15m_agg.py, !convert_to_parquet.py | 1 |
| `src/ingestion/live_feed.py` | — | !run_live_feed.py | 1 |
| `src/ingestion/parquet_converter.py` | — | !check_data_quality.py, !convert_to_parquet.py | 1 |
| `src/logger.py` | analytics/stats_engine.py, execution/binance_rate_limiter.py, execution/experiment_tracker.py, execution/heartbeat.py, execution/live_trader.py, execution/metrics.py, execution/paper_trader.py, execution/pending_orders_store.py, execution/reconciler.py, execution/reconciler_signals.py, execution/signal_bridge.py, execution/startup_check.py, execution/strategies/meta_strategy.py, execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py, execution/strategies/paper_strategy.py, execution/walk_forward.py, execution/watchdog.py, features/agg_15m.py, features/derivatives.py, features/feature_pipeline.py, features/live_feature_state.py, features/microstructure.py, features/mtf_context.py, features/orb_features.py, features/regime_detector.py, features/session_features.py, features/triple_barrier.py, ingestion/binance_downloader.py, ingestion/data_quality.py, ingestion/data_store.py, ingestion/live_feed.py, ingestion/parquet_converter.py, models/dataset_builder.py, models/lgbm_trainer.py, models/ml_validator.py, models/statistical_tests.py, models/training_pipeline.py, monitoring/metrics_collector.py, monitoring/signal_alert_state.py, monitoring/telegram_reporter.py, patches/nautilus_enums.py, risk/circuit_breaker.py, risk/portfolio_tracker.py, risk/risk_engine.py, risk/risk_state_store.py, telegram_bot/bot.py, telegram_bot/broadcaster.py, telegram_bot/database.py, telegram_bot/handlers_free.py, telegram_bot/handlers_premium.py, telegram_bot/payments_crypto.py, telegram_bot/payments_stars.py, telegram_bot/roles.py, telegram_bot/signal_poller.py | !analyze_regimes.py, !build_15m_dataset.py, !build_1h_dataset.py, !build_features.py, !build_features_15m_agg.py, !build_meta_dataset.py, !build_mtf_features.py, !check_data_quality.py, !check_mtf_data_quality.py, !check_signal_freshness.py, !convert_mtf_to_parquet.py, !convert_to_parquet.py, !download_funding_rate.py, !download_historical.py, !download_mtf_data.py, !feature_selection_v3.py, !retrain_v3.py, !run_live.py, !run_live_15m.py, !run_live_feed.py, !run_paper.py, !run_paper_15m.py, !run_reconciler.py, !run_telegram_bot.py, !run_watchdog.py, !train_15m_models.py, !train_1h_models.py, !train_meta_model.py, !train_models.py, !tune_models.py, !validate_15m_models.py, !validate_1h_models.py, !validate_ml_models.py | 0 |
| `src/models/__init__.py` | — | — | 0 |
| `src/models/dataset_builder.py` | models/lgbm_trainer.py | !build_15m_dataset.py, !build_1h_dataset.py, !build_meta_dataset.py, !feature_selection_v3.py, !retrain_v3.py | 5 |
| `src/models/lgbm_trainer.py` | execution/strategies/meta_strategy.py, execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py, models/ml_validator.py, models/statistical_tests.py, models/training_pipeline.py | !build_meta_dataset.py, !feature_selection_v3.py, !retrain_v3.py, !train_15m_models.py, !train_1h_models.py, !tune_models.py, !validate_15m_models.py, !validate_1h_models.py, !validate_ml_models.py | 12 |
| `src/models/ml_validator.py` | — | !train_15m_models.py, !train_1h_models.py, !validate_15m_models.py, !validate_1h_models.py, !validate_ml_models.py | 3 |
| `src/models/statistical_tests.py` | — | !retrain_v3.py, !validate_15m_models.py, !validate_1h_models.py, !validate_ml_models.py | 2 |
| `src/models/temporal_split.py` | models/lgbm_trainer.py | !train_15m_models.py, !train_1h_models.py | 1 |
| `src/models/training_pipeline.py` | — | !train_models.py | 2 |
| `src/monitoring/__init__.py` | — | — | 0 |
| `src/monitoring/metrics_collector.py` | execution/strategies/paper_strategy.py | — | 1 |
| `src/monitoring/signal_alert_state.py` | — | !check_signal_freshness.py | 2 |
| `src/monitoring/telegram_reporter.py` | execution/live_trader.py | !check_signal_freshness.py | 2 |
| `src/patches/__init__.py` | — | — | 0 |
| `src/patches/nautilus_enums.py` | execution/live_trader.py | !run_live.py, !run_paper_15m.py | 1 |
| `src/risk/__init__.py` | — | — | 0 |
| `src/risk/circuit_breaker.py` | execution/strategies/ml_strategy.py, risk/__init__.py | — | 4 |
| `src/risk/portfolio_tracker.py` | execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py, monitoring/metrics_collector.py, risk/__init__.py | — | 9 |
| `src/risk/risk_engine.py` | execution/pending_orders_store.py, execution/strategies/meta_strategy.py, execution/strategies/ml_strategy.py, execution/strategies/ml_strategy_15m.py, execution/strategies/paper_strategy.py, risk/__init__.py, risk/circuit_breaker.py, risk/portfolio_tracker.py | — | 10 |
| `src/risk/risk_state_store.py` | risk/circuit_breaker.py, risk/portfolio_tracker.py | — | 1 |
| `src/telegram_bot/__init__.py` | — | — | 2 |
| `src/telegram_bot/bot.py` | telegram_bot/__init__.py | !run_telegram_bot.py | 2 |
| `src/telegram_bot/broadcaster.py` | telegram_bot/__init__.py, telegram_bot/bot.py, telegram_bot/signal_poller.py | — | 1 |
| `src/telegram_bot/database.py` | telegram_bot/__init__.py, telegram_bot/bot.py, telegram_bot/broadcaster.py, telegram_bot/handlers_free.py, telegram_bot/handlers_owner.py, telegram_bot/handlers_premium.py, telegram_bot/payments_crypto.py, telegram_bot/payments_stars.py, telegram_bot/roles.py | — | 8 |
| `src/telegram_bot/handlers_free.py` | telegram_bot/bot.py, telegram_bot/handlers_owner.py, telegram_bot/handlers_premium.py | — | 4 |
| `src/telegram_bot/handlers_owner.py` | telegram_bot/bot.py | — | 2 |
| `src/telegram_bot/handlers_premium.py` | telegram_bot/bot.py | — | 3 |
| `src/telegram_bot/keyboards.py` | telegram_bot/bot.py, telegram_bot/handlers_free.py, telegram_bot/handlers_owner.py, telegram_bot/handlers_premium.py | — | 2 |
| `src/telegram_bot/payments_crypto.py` | telegram_bot/__init__.py, telegram_bot/bot.py | — | 2 |
| `src/telegram_bot/payments_stars.py` | telegram_bot/__init__.py, telegram_bot/bot.py | — | 2 |
| `src/telegram_bot/roles.py` | telegram_bot/bot.py, telegram_bot/broadcaster.py, telegram_bot/handlers_free.py, telegram_bot/handlers_owner.py, telegram_bot/handlers_premium.py, telegram_bot/payments_crypto.py, telegram_bot/payments_stars.py | !run_telegram_bot.py | 5 |
| `src/telegram_bot/signal_formatter.py` | telegram_bot/bot.py, telegram_bot/handlers_premium.py | — | 0 |
| `src/telegram_bot/signal_poller.py` | telegram_bot/bot.py | — | 3 |
| `src/telegram_bot/timeframes.py` | telegram_bot/handlers_premium.py, telegram_bot/keyboards.py, telegram_bot/signal_formatter.py | — | 0 |

---

## 3. МЁРТВЫЕ ВЕТКИ — ПЕРВЫЙ ПРОХОД (кандидаты, не выводы)

Ниже — **черновой список для прохода 1**. Метод: AST-разбор импортов и
regex-подсчёт упоминаний идентификаторов. Метод даёт ложноположительные там,
где вызов делает фреймворк (FastAPI-роут, pydantic-валидатор, коллбэк Nautilus,
`getattr`, строковый импорт). Такие случаи размечены явно.

### 3.1 Модули `src/`, которых не импортирует никто, кроме тестов

Полный список модулей без импортёра в `src/` и без импортёра в `scripts/`
(исключены пакетные `__init__.py`):

| Модуль | Строк | Импортёров в тестах | Комментарий |
|---|---|---|---|
| `src/api/main.py` | 331 | 2 | **ложноположительное**: `scripts/run_api.py:31` содержит строковый импорт `"src.api.main:app"` для uvicorn — AST его не видит. Юнит `atomicortex-api.service` при этом не в `units.enabled`. |
| `src/execution/reconciler.py` | 275 | 1 | **кандидат в мёртвые**. Grep по `src`/`scripts` на `execution.reconciler` (за вычетом `reconciler_signals`) даёт пустой вывод. `run_reconciler.py` импортирует `reconciler_signals`, а не `reconciler`. |
| `src/execution/strategies/meta_strategy.py` | 331 | 5 | **кандидат в мёртвые**. Единственное упоминание вне тестов — собственная первая строка докстринга `src/execution/strategies/meta_strategy.py:2`. Аудит-1 это уже фиксировал: `docs/audit/01_architecture_review.md:43` — «Мёртвый/несвязанный код: meta_strategy (гейт починен, но стратегия не…»; `docs/code_review_v3.md:988` — «meta_strategy.py: `_apply_meta_gate` never called». Проверить в проходе 1, закрыто ли. |
| `src/execution/strategies/paper_strategy.py` | 175 | 1 | **кандидат в мёртвые**. Единственное упоминание вне тестов — README.md:132 (дерево каталогов). |

### 3.2 Публичные функции/классы, вызываемые только из тестов

Критерий: 0 упоминаний идентификатора в `src/` и `scripts/` вне собственного
файла, при ≥1 упоминании в `tests/`. Публичные top-level `def`/`class` и
публичные методы классов.

Найдено **90** таких символов. Полный машинный вывод сохранён в `/tmp/test_only.txt`
(артефакт временный). Наиболее содержательные группы:

- `src/execution/watchdog.py:48` `HeartbeatVerdict` (36 упоминаний в 5 тестовых
  файлах), `watchdog.py:275` `Watchdog.emergency_close_all` (39 в 5),
  `watchdog.py:428` `Watchdog.send_telegram_alert` (16 в 3) — watchdog не в
  `units.enabled` (см. §1.3), поэтому «только тесты» ожидаемо.
- `src/execution/walk_forward.py:182` `WindowResult`, `:192`
  `WalkForwardGateConfig`, `:245` `worst_sharpe`, `:252` `aggregate_return_pct`,
  `:263` `passes_gate` — гейт walk-forward вызывается только из тестов.
- `src/execution/strategies/meta_strategy.py:50` `MetaDecision`, `:116`
  `MetaSignalGate.build_feature_vector`, `:180` `MetaMLStrategyConfig`, `:191`
  `MetaMLTradingStrategy` — согласуется с §3.1.
- `src/features/derivatives.py:274` `compute_liquidation_proximity`, `:412`
  `compute_basis_annualized`, `:458` `compute_oi_velocity`, `:525`
  `compute_sentiment_features`; `src/features/microstructure.py:130`
  `compute_vpin` — фичевые функции без потребителя в пайплайне. Требует
  проверки в проходе 3: возможно, вызываются через список имён/`getattr`.
- `src/models/lgbm_trainer.py:1202` `LGBMTrainer.permutation_importance`,
  `src/models/statistical_tests.py:328` `StatTestResult.passes_all_thresholds`.
- `src/risk/risk_engine.py:233` `calculate_position_size`, `:264`
  `calculate_stop_loss`, `:281` `calculate_take_profit` — публичные методы
  риск-движка без внешнего вызывателя; вероятно, вызываются приватными
  обёртками внутри файла. Проверить в проходе 7.
- `src/monitoring/metrics_collector.py` — 5 публичных методов (`record_trade`,
  `save_to_db`, `get_daily_report`, `get_weekly_report`) + класс
  `TradingMetrics` только из тестов.
- `src/telegram_bot/database.py` — 7 публичных методов только из тестов
  (`unban_user`, `get_users_by_role`, `set_notes`, `get_signals_paginated`,
  `get_events`, `get_latest_metrics`, `get_pending_payments`).

### 3.3 Символы без единой ссылки (включая тесты)

Уточнённый список: разделён на «вызывается фреймворком / внутри своего файла»
и «действительно без потребителя».

**Ложноположительные — вызываются фреймворком или используются внутри файла:**

| Где | Символ | Почему не мёртвый |
|---|---|---|
| `src/api/main.py:225`, `:248`, `:267` | `stats_by_tf`, `signals_open`, `monthly_stats` | декораторы `app.get(...)` |
| `src/api/main.py:46`, `:145` | `get_db_paths`, `require_api_key` | 4 и 7 использований внутри файла |
| `src/config.py:191`, `:195` | `coerce_path`, `create_directories` | `@field_validator`, `@model_validator` — вызывает pydantic |
| `src/execution/strategies/ml_strategy.py:943` | `on_position_opened` | коллбэк Nautilus |
| `src/execution/reconciler_signals.py:46,60,94,132` | `normalize_symbol`, `PriceSource`, `BinanceRESTPriceSource`, `CompositePriceSource` | используются внутри файла (`:289`, `:168`, `:177`, `:141`, `:142`) |
| `src/logger.py:34`, `:39` | `set_correlation_id`, `get_correlation_id` | 3 и 2 использования внутри файла, экспортированы в `__all__` (`:171`, `:172`) |
| `src/telegram_bot/bot.py:790`, `:736` | `re_escape`, `crypto_payment` | используются внутри файла (`bot.py:248`) |
| `src/execution/strategies/random_entry_strategy.py:15`, `src/risk/portfolio_tracker.py:345`, `src/telegram_bot/roles.py:30`, `src/config.py:368` | `RandomEntryConfig`, `get_weekly_pnl`, `get_owner_id`, `safe_dict` | 1 использование внутри своего файла |

**Кандидаты в действительно мёртвые (0 ссылок вообще, включая свой файл):**

| Где | Символ | Строк-контекст |
|---|---|---|
| `src/config.py:229` | `Settings.bybit_api_key` (`@property`) | Bybit в системе не используется — стек Binance |
| `src/config.py:234` | `Settings.bybit_api_secret` (`@property`) | то же |
| `src/features/utils.py:33` | `rolling_correlation` | единственный, кроме определения, файл `utils.py` содержит 56 строк |
| `src/ingestion/binance_downloader.py:362` | `BinanceDataDownloader.download_metrics` | |
| `src/ingestion/binance_downloader.py:385` | `BinanceDataDownloader.download_agg_trades` | |
| `src/ingestion/live_feed.py:155` | `LiveFeedManager.last_tick_time` (`@property`) | |
| `src/models/dataset_builder.py:94` | `DatasetBuilder.build_features_all_symbols` | |
| `src/monitoring/telegram_reporter.py:42` | `TelegramReporter.send_trade_alert` | весь публичный API репортера мёртв |
| `src/monitoring/telegram_reporter.py:91` | `TelegramReporter.send_daily_report` | |
| `src/monitoring/telegram_reporter.py:99` | `TelegramReporter.send_weekly_report` | |
| `src/telegram_bot/payments_crypto.py:159` | `CryptoBotPayment.stop_polling` | поллер не останавливается никем |
| `src/telegram_bot/timeframes.py:48` | `db_path_for` | при том что модуль объявлен «единственным источником истины» по ТФ |

### 3.4 `scripts/`, не упомянутые нигде

Метод: для каждого файла `scripts/*` — `grep -rl <basename>` по `deploy/`,
`.github/`, остальным `scripts/`, `docs/` + `README.md`, `src/`.

**Не упомянуты нигде вообще (ни юнит, ни CI, ни другой скрипт, ни документация, ни src):**

| Скрипт | Строк докстринга-назначения |
|---|---|
| `scripts/build_features_15m_agg.py` | Block-3 Step 3, дополняет 4H-паркеты колонками `agg_15m_*` |
| `scripts/build_mtf_features.py` | Feature-матрицы 1H и 15m |
| `scripts/feature_selection_v3.py` | Block 3 / Step 1, Clustered-MDA отбор фич |
| `scripts/migrate_db_v3.py` | Миграция БД v3 (идемпотентная) |
| `scripts/retrain_v3.py` | Block 2, переобучение v3-линии |
| `scripts/run_live_15m.py` | 15m live-лаунчер, может ставить реальные ордера |
| `scripts/train_15m_models.py` | Обучение 15m |
| `scripts/train_1h_models.py` | Обучение 1H |
| `scripts/run_phase1_download.sh` | (shell; сам вызывает `download_mtf_data.py`, `convert_mtf_to_parquet.py`, `check_mtf_data_quality.py`, `verify_phase1.py`) |

Отдельно: `scripts/train_meta_model.py` и `scripts/build_meta_dataset.py`
упомянуты только внутри `src/execution/strategies/meta_strategy.py` — модуля,
который сам является кандидатом в мёртвые (§3.1). Замкнутая на себя подсистема.

**Упомянуты только в документации (не в юнитах, не в CI, не в других скриптах):**
`analyze_regimes.py`, `build_features.py`, `check_data_quality.py`,
`convert_to_parquet.py`, `download_funding_rate.py`, `download_historical.py`,
`run_backtest.py`, `run_live_feed.py`, `run_paper.py`, `run_walk_forward.py`,
`train_models.py`, `tune_models.py`, `validate_cost_model.py`,
`validate_ml_models.py`, `validate_15m_models.py`, `validate_1h_models.py`,
`verify_setup.py`, `create_env.sh`.

**Упомянуты в юнитах (§1.1.2):** `run_live.py`, `run_telegram_bot.py`,
`check_signal_freshness.py`, `run_paper_15m.py`, `run_watchdog.py`,
`run_reconciler.py`, `run_api.py`.

**Упомянуты в CI:** ни один. CI запускает только `pytest`.

---

## 4. КОНТРАКТЫ, КОТОРЫЕ ПРЕДСТОИТ ПРОВЕРИТЬ (проход 6)

Ниже — только адреса. Проверка «кто пишет / кто читает / совпадают ли
допущения» — предмет прохода 6. Списки получены `grep -rl --include='*.py'`
по `src` и `scripts`.

### 4.1 `open_time` — 30 файлов

Пишут (кандидаты): `src/ingestion/parquet_converter.py`,
`src/features/feature_pipeline.py`, `src/features/agg_15m.py`,
`scripts/convert_mtf_to_parquet.py`, `scripts/build_features*.py`.
Читают: `src/models/{temporal_split,dataset_builder,ml_validator,lgbm_trainer}.py`,
`src/ingestion/{data_quality,data_store}.py`,
`src/features/{microstructure,session_features,derivatives,live_feature_state,orb_features,mtf_context}.py`,
`src/execution/{paper_trader,reconciler_signals,walk_forward,data_catalog}.py`,
`src/execution/strategies/{ml_strategy,ml_strategy_15m}.py`,
`src/risk/portfolio_tracker.py`,
`scripts/{train_1h_models,train_meta_model,validate_1h_models,build_meta_dataset,build_mtf_features}.py`.
**Вопрос:** единицы (ms/с/datetime), это открытие или закрытие бара, tz.
Смежный факт: HEAD-1 `4d08696` — «fix(15m): take preload ts_event from kline
close time and derive open_time on the bar grid».

### 4.2 `ts_event` — 7 файлов

`src/features/live_feature_state.py`, `src/execution/data_catalog.py`,
`src/execution/strategies/{paper_strategy,random_entry_strategy,baseline_strategy,ml_strategy,ml_strategy_15m}.py`.
**Вопрос:** конвенция Nautilus (наносекунды) против `open_time` (мс?) — стык
`data_catalog` ↔ `live_feature_state` ↔ стратегии.

### 4.3 `confidence` / порог — 30 файлов

Пишут: `src/models/lgbm_trainer.py`, `src/execution/strategies/ml_strategy*.py`,
`src/execution/strategies/meta_strategy.py`.
Порог задают: `src/config.py`, `src/configs/strategy_1h.py`,
`src/configs/strategy_15m.py`.
Читают: `src/execution/signal_bridge.py`, `src/risk/risk_engine.py`,
`src/analytics/stats_engine.py`, `src/telegram_bot/{database,signal_formatter,broadcaster,handlers_free,handlers_premium}.py`,
`src/monitoring/{metrics_collector,telegram_reporter}.py`,
`scripts/{validate_ml_models,validate_1h_models,validate_15m_models,train_1h_models,train_15m_models,build_meta_dataset,analyze_regimes}.py`,
`src/features/{feature_pipeline,regime_detector}.py`, `src/execution/live_trader.py`.
**Вопрос:** `confidence` = `p(UP)` или `max(p, 1-p)`; сравнимо ли то, что
пишет тренер, с тем, с чем сравнивает порог стратегия. Ключевой вход:
live confidence 0.50–0.54 против порога 0.65.

### 4.4 `atr`, `atr_pct` — 15 файлов

`src/features/{triple_barrier,regime_detector,orb_features,microstructure,feature_pipeline,mtf_context}.py`,
`src/models/dataset_builder.py`, `src/risk/risk_engine.py`,
`src/execution/strategies/{meta_strategy,paper_strategy,ml_strategy,ml_strategy_15m}.py`,
`scripts/{build_meta_dataset,analyze_regimes,feature_selection_v3}.py`.
**Вопрос:** `atr_pct` — доля (0.01) или проценты (1.0); одинаково ли это
понимают `risk_engine` (сайзинг) и `triple_barrier` (барьеры).

### 4.5 `regime` — 25 файлов

Пишет: `src/features/regime_detector.py` (лейблы TREND_UP / TREND_DOWN / …).
Читают: `src/features/{feature_pipeline,window_sizes,mtf_context}.py`,
`src/models/{training_pipeline,dataset_builder,ml_validator,lgbm_trainer}.py`,
`src/risk/risk_engine.py`, `src/execution/signal_bridge.py`,
`src/execution/strategies/{meta_strategy,paper_strategy,ml_strategy,ml_strategy_15m}.py`,
`src/monitoring/telegram_reporter.py`, `src/api/main.py`,
`src/telegram_bot/{handlers_premium,handlers_free,handlers_owner,bot,signal_formatter,database,broadcaster}.py`,
`src/configs/strategy_15m.py`.
**Вопрос:** множество лейблов детектора против множества имён моделей
(`trend` / `high_vol` / `range`) — совпадает ли; куда деградирует неизвестный
режим. Смежный факт: HEAD-9 `e67a451` — «fix(features): trim the offline warmup
head by the detector's own window so training rows carry real regime values».

### 4.6 `equity` / `peak_equity` — equity 26 файлов, peak_equity 4

`peak_equity`: `src/risk/portfolio_tracker.py`, `src/risk/risk_engine.py`,
`src/execution/strategies/ml_strategy.py`, `src/monitoring/metrics_collector.py`.
`equity` дополнительно: `src/execution/{live_trader,signal_bridge,paper_trader,metrics,reconciler_signals,backtest_runner,walk_forward}.py`,
`src/execution/strategies/{random_entry_strategy,paper_strategy,ml_strategy_15m,baseline_strategy}.py`,
`src/analytics/stats_engine.py`, `src/api/main.py`,
`src/telegram_bot/{handlers_premium,handlers_free,broadcaster,database}.py`,
`scripts/{migrate_db_v3,run_paper,run_live,run_live_15m,run_paper_15m}.py`.
**Вопрос:** база просадки; кто первым устанавливает `peak_equity`.
Смежный факт: HEAD-2 `b2cbd40` — «fix(risk): seed the drawdown baseline from
the first authoritative exchange balance».

### 4.7 `funding_rate` — 25 файлов

Пишут: `src/ingestion/{binance_downloader,parquet_converter}.py`,
`scripts/download_funding_rate.py`.
Читают: `src/features/{derivatives,feature_pipeline,live_feature_state}.py`,
`src/execution/{cost_model,signal_bridge,backtest_runner}.py`,
`src/execution/strategies/*`, `src/risk/risk_engine.py`,
`src/models/lgbm_trainer.py`, `src/ingestion/{data_quality,data_store}.py`,
`src/monitoring/telegram_reporter.py`, `src/telegram_bot/broadcaster.py`,
`scripts/{build_meta_dataset,check_data_quality,validate_cost_model,build_1h_dataset,build_15m_dataset,download_historical}.py`.
**Вопрос:** ставка за 8 часов или годовая; знак; выравнивание по времени с
4H-баром (funding приходит раз в 8 ч).

### 4.8 `oi_value` — 5 файлов

`src/features/{feature_pipeline,derivatives,live_feature_state}.py`,
`src/execution/strategies/ml_strategy.py`, `scripts/feature_selection_v3.py`.
**Вопрос:** контракты или USD; согласованность offline (`feature_pipeline`) и
live (`live_feature_state`).

### 4.9 `signal_id` — 5 файлов

Пишет: `src/execution/signal_bridge.py` (и `src/execution/strategies/ml_strategy.py`).
Читают: `src/telegram_bot/{database,signal_poller,keyboards}.py`.
**Вопрос:** уникальность между изолированными БД (4H / 15m); что кладётся в
callback_data клавиатур.

### 4.10 `position_size` — 14 файлов

Пишет: `src/risk/risk_engine.py` (+ множитель от `src/risk/circuit_breaker.py`).
Читают: `src/execution/{cost_model,signal_bridge,backtest_runner}.py`,
`src/execution/strategies/{meta_strategy,paper_strategy,ml_strategy,ml_strategy_15m}.py`,
`src/monitoring/telegram_reporter.py`, `src/telegram_bot/{handlers_premium,broadcaster}.py`,
`src/features/regime_detector.py`, `scripts/analyze_regimes.py`.
**Вопрос:** единицы — базовый актив, котируемая валюта или доля equity;
одинаково ли их понимают cost_model и стратегия.

### 4.11 Дополнительно найденные сквозные значения

| Значение | Пишет | Читает | Вопрос |
|---|---|---|---|
| `last_signal_ts` | `src/execution/heartbeat.py` | `scripts/check_signal_freshness.py` | Единицы и смысл `null`. Смежные коммиты: `d60e55e`, `7593989`, `c4cc3e0`. Контракт держится ровно между двумя файлами — узкое место. |
| `heartbeat` (ключ + TTL) | `src/execution/heartbeat.py`, `src/execution/strategies/ml_strategy*.py` | `src/execution/watchdog.py`, `scripts/{run_watchdog,check_signal_freshness}.py`, `src/configs/strategy_{1h,15m}.py`, `src/config.py`, `src/execution/live_trader.py` | Имя ключа (`bot_15m_heartbeat` в юните `atomicortex-watchdog-15m.service`) против дефолта в коде; TTL против `check_interval`. Смежный коммит `5f12e98` (UNKNOWN-вердикт). |
| `trading_mode` | CLI-аргументы лаунчеров, `src/config.py` | `src/execution/{live_trader,reconciler_signals,reconciler,watchdog}.py`, `src/execution/strategies/ml_strategy*.py`, `src/logger.py`, `scripts/*` | Прод-юнит `atomicortex-bot.service` передаёт `--mode paper --dry-run`, а `atomicortex-watchdog.service` — `--trading-mode testnet`. Разные словари значений (`paper`/`testnet`/`live`?) — проверить, кто как их сопоставляет. |
| путь к БД (`data/atomicortex*.db`) | `src/execution/signal_bridge.py`, лаунчеры | `src/analytics/stats_engine.py`, `src/api/main.py` (`ATOMICORTEX_DB_PATHS`), `src/telegram_bot/{database,timeframes}.py` | Изоляция 4H / 15m: `src/telegram_bot/timeframes.py:48` `db_path_for` — мёртвая функция (§3.3) при том что модуль объявлен единственным источником истины. |

---

## 5. СОСТОЯНИЕ ДЕРЕВА

### 5.1 `git log --oneline -10`

```
f4af5fd chore: ignore local snapshot directory
5f12e98 fix(watchdog): add an UNKNOWN verdict so a blind check is no longer read as proof of life
b2cbd40 fix(risk): seed the drawdown baseline from the first authoritative exchange balance
4d08696 fix(15m): take preload ts_event from kline close time and derive open_time on the bar grid
d60e55e fix(ops): treat a null last_signal_ts as the bot's own claim of silence, not a gap
7593989 feat(ops): publish signal freshness in the heartbeat and read it before the ledger
c4cc3e0 fix(ops): treat an unreadable database as its own alert and fail the unit when freshness cannot be established
348a243 feat(ops): install only the units listed in a manifest and de-duplicate starvation alerts across timer runs
9d8bfbc fix(trainer): split train/test on one wall-clock boundary and embargo by time so test never overlaps train
e67a451 fix(features): trim the offline warmup head by the detector's own window so training rows carry real regime values
```

### 5.2 `git status --short`

Пустой вывод — рабочее дерево чисто.

### 5.3 `git tag --sort=-creatordate`

```
sprint0-complete-2026-08-20
pre-audit-fixes-2026-07-02
```

### 5.4 `ls docs/audit/`

```
00_index.md
01_architecture_review.md
02_math_and_validation.md
03_data_and_features.md
04_models_and_training.md
05_execution_risk_security.md
06_test_plan.md
07_roadmap.md
08_references.md
atomicortex_20260702.db.bak
models_snapshot_20260702.tgz
scratch
```

`docs/audit/scratch/` содержит `dsr_synthetic_check.py`, `parity_4h_check.py`.

Прочее в `docs/`: `atomicortex_v3_research.md`, `code_review_v3.md`,
`retrain_v3_results.txt`, `retrain_v3_selected_results.txt`, `troubleshooting.md`.

### 5.5 Тесты

```
1923 tests collected in 9.28s
```

(`pytest tests/ --collect-only -q`, ошибок сбора нет.)
Тестовых файлов: 96. CI прогоняет их с `-x --ignore=tests/test_live_feed.py`.

---

## НЕ ИССЛЕДОВАНО (передаётся в проходы 1 и 6)

1. Фактическая закрытость находок A1…A20 Аудита-1 — проход 8.
   Косвенный признак: `src/execution/startup_check.py` докстринг содержит
   «fail-fast, fix A1»; `src/execution/binance_rate_limiter.py` — «(H22)».
2. Кандидаты §3.1–§3.4 не подтверждены динамическим замером — только статикой.
   Динамические вызовы (`getattr`, реестры имён фич, строковые импорты) могут
   оживить часть кандидатов. Особенно вероятно для `compute_*` функций
   `src/features/`.
3. `deploy/install_units.sh` прочитан только на предмет чтения манифеста
   (строки 13, 41, 64, 65, 78, 81, 98). Health-check логика не разобрана.
4. `.env` / `.env.example` не сопоставлены с `src/config.py` — проход 8.
5. Дубли между `ml_strategy.py` (2262 строки) и `ml_strategy_15m.py` (625) не
   измерены — проход 1.
6. Дубли между `train_models.py` / `train_1h_models.py` / `train_15m_models.py`
   и между `validate_*_models.py` не измерены — проход 1.
7. Контракты §4 — только адреса, ни один не проверен. Проход 6.
