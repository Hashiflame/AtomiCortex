# 01 — Архитектура

> TL;DR: архитектура в целом здравая и НЕ нуждается в перестройке. Слои разделены
> правильно (ingestion → features → labeling → training → validation → execution →
> risk → monitoring → telegram), Nautilus как исполняющий каркас — адекватный выбор.
> Проблемы — не в форме, а в трёх швах: (1) валидационный слой не связан с деплоем,
> (2) live-фичевый путь дублирует офлайн с другими глубинами истории,
> (3) отсутствует operational-контур (fail-fast, watchdog, алертинг).

## Карта зависимостей (критический путь)
```
binance_downloader → parquet_converter → DataStore(DuckDB) 
      → FeaturePipeline.build (offline)  ──┐
                                           ├─ dataset_builder(create_target_triple_barrier,
LiveFeatureState + build_from_buffer (live)┘   uniqueness) → LGBMTrainer → *.pkl
                                                    ↓
                    ml_validator(PurgedKFoldCV) + walk_forward + statistical_tests(DSR/PBO)
                                                    ↓        (НЕ СВЯЗАНО с деплоем — A7)
run_live.py → Nautilus TradingNode → MLTradingStrategy{,15M}
      → RegimeDetector → _select_model → get_signal → RiskEngine → orders
      → SignalBridge → atomicortex.db → signal_poller → Telegram bot
```

## Что оставить как есть
- **LightGBM + regime-специфичные модели** — правильный класс для 2.4k–10k строк (см. 04 §4.7).
- **Polars-пайплайн фичей** и единый `build_from_buffer` для live — правильная идея
  (одна кодовая база фичей); чинить надо только контракт глубины буфера, не архитектуру.
- **SQLite + WAL** для сигналов/подписок на текущем масштабе — достаточно.
- **Nautilus** — не менять; добавить только fail-fast обвязку в run_live.

## Архитектурные долги (по убыванию)
1. **Нет контракта «модель ↔ стратегия»** (A6, A7): бандл не несёт манифеста
   (барьеры, порог, eval, даты данных), стратегия не проверяет пригодность.
   Решение — манифест в бандле + проверка при загрузке (2 маленьких PR).
2. **Два источника правды для live-фичей 4H**: `self._bars` (deque в стратегии)
   и `LiveFeatureState.bar_buffer_4h` с разными maxlen — регим считается по одному,
   фичи по другому. Свести к одному буферу с одной константой глубины.
3. **Конфигурационный разброс порогов** (0.55/0.58/0.60/0.65 в 5 местах) —
   единая точка истины в манифесте модели.
4. **Мониторинг для ML отсутствует**: нет drift-мониторинга фичей, нет алерта
   «N дней без сигналов», нет данных feed-watchdog (June: месяц молчания незамечен).
   Минимум: cron-проверка `MAX(created_at)` в signals_log + Telegram-алерт.
5. **Мёртвый/несвязанный код**: meta_strategy (гейт починен, но стратегия не
   запускается ни одним run-скриптом), LiveFeedManager (орфан, отмечен ещё в
   code_review_v3), circuit_breaker.get_position_size_multiplier (не вызывается).
   Удалить или подключить — сейчас они создают ложное чувство защищённости.

## Замечание про DR/устойчивость VM
Единственная VM (Tokyo), SQLite-файлы и модели на локальном диске, без бэкап-джоба
в репо. Для paper некритично; до live — снапшоты диска + выгрузка atomicortex.db.
