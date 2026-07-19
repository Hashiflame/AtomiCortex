# Velontir / AtomiCortex — Аудит 2026-07-02 (индекс)

> Аудитор: Claude (Fable 5), read-only сессия на prod-VM.
> Метод: построчный проход критического пути + прогон scratch-скриптов на синтетике
> (`docs/audit/scratch/`) + сверка формул с первоисточниками + анализ живых логов/БД.
> Ни один боевой файл, процесс, БД не изменялись. Все числа получены командами (указаны в тексте).

---

## TL;DR — три вещи важнее всего остального

1. **🔴 STOP-SHIP (доступность).** Сегодня в 06:36 UTC оба торговых бота после рестарта
   не смогли подключиться к Binance (транзиентная сетевая ошибка при старте),
   Nautilus `TradingNode` по таймауту 30с перешёл в `RUNNING` **с неподключёнными
   DataEngine/ExecEngine** и висит так до сих пор: данных нет, стратегия не стартовала,
   процесс не падает → systemd не рестартует. Watchdog (`atomicortex-watchdog.service`)
   **enabled, но inactive** — dead-man's switch не работает и это никто не заметил.
   Это и есть механика «stale prices»: не WS-реконнект и не кэш, а отсутствие
   fail-fast/retry при неудачном коннекте + неработающий watchdog.
   Доказательство: `journalctl -u atomicortex-bot --since "2026-07-02 06:36"` →
   `DataEngine.check_connected() == False` → `TradingNode: RUNNING` и тишина;
   `systemctl is-active atomicortex-watchdog` → `inactive`.

2. **🔴 Paper trading не производит доказательств.** За 2 месяца — **8 сигналов**
   (все BTCUSDT 4H; WR 4/8), последний — **2026-06-02**, месяц тишины.
   Причина тишины — не мёртвая лента (бары шли, 6/день весь июнь), а то, что после
   переобучения 24 мая live-уверенность модели рухнула к **0.50–0.53 почти на каждом
   баре** при пороге 0.65 (см. п.3). Собственный gate проекта требует ≥300 OOS-сигналов —
   на этом темпе он недостижим за годы. Никакой вывод о DSR/PF по paper сейчас невозможен —
   «реальный Sharpe» системы **не измерен**.
   Доказательство: `sqlite3 -readonly data/atomicortex.db "SELECT ... FROM signals_log"` (8 строк);
   `journalctl ... | grep "dir=0 confidence="` (июнь: 0.500–0.536 на ~95% баров).

3. **🔴 Разрыв train↔serve, доказанный экспериментом.** Одни и те же исторические бары,
   прогнанные offline-пайплайном и live-путём (`build_from_buffer`, буфер 400 баров как в проде):
   **regime-лейбл расходится на 14/40 (35%) баров**, `atr_percentile` — средняя относительная
   ошибка 16.5% (макс 100%). То есть треть времени live выбирает не ту модель и не тот порог,
   чем видел трейнинг. Причина: `bar_buffer_4h=400` < `atr_lookback=540`; на 15m ещё хуже —
   HTF-4H фичи в live **всегда** дефолтные (буфер даёт ~100 4H-баров при `min_bars=300`).
   Скрипт: `docs/audit/scratch/parity_4h_check.py`.

**Ответ на главный вопрос владельца** («упускаю ли я огромный потенциал?»):
пока невозможно сказать, есть ли edge вообще, потому что (а) измерительный слой
(DSR/PBO) математически сломан в обе стороны, (б) live-фичи не совпадают с трейнинговыми,
(в) исполняемая сделка геометрически не совпадает с тем событием, которое предсказывает
модель (см. 02 и 05). Сначала P0/P1 — потом разговор о «потенциале». Единственный
реально валидированный конфиг (`trend`, pt=1.0/sl=0.8/h=6, WR=59.01%, PF=1.3786 **gross,
без комиссий**, на 610 тестовых строках) — это слабое, но не нулевое основание; после
вычета издержек и метрических поправок он может оказаться и ниже воды.

---

## Карта проекта (числа — командами, 2026-07-02)

| Метрика | Значение | Команда |
|---|---|---|
| Python LOC всего | 62 408 | `find . -name '*.py' … \| xargs wc -l` |
| src/ | 25 763 | то же |
| tests/ | 25 875 (86 файлов, 1443 test-функций) | `grep -rn "def test_" tests/ \| wc -l` |
| scripts/ | 10 665 | — |
| Задеплоенные модели | trend=23 фичи (whitelist), high_vol=45, range=45 | scratch-unpickle |
| Прод-порог confidence | **0.65** (`.env: CONFIDENCE_THRESHOLD=0.65`) | `grep .env` |
| Прод-конфиг лейблов | trend: pt=1.0, sl=0.8, h=6 (passes=True, WR 59.01/PF 1.3786) | `docs/retrain_v3_selected_results.txt:49` |
| Сервисы | api, bot, bot-15m, telegram: running; **watchdog: inactive**; reconciler.timer: active | `systemctl` |

Файлы `VELONTIR_AUDIT_0*.md` и `AtomiCortex_Deep_Audit_Report.md` **в репозитории отсутствуют**
(`find . -iname '*VELONTIR*'` → пусто). Сверка со старыми находками сделана по
`docs/code_review_v3.md` и чек-листу владельца.

---

## Сводная таблица severity (новые + ключевые подтверждённые)

| # | Находка | Sev | Файл | Док |
|---|---|---|---|---|
| A1 | Zombie-RUNNING после неудачного коннекта; watchdog не работает | 🔴 | run_live.py / systemd | 05 |
| A2 | Paper: 8 сигналов/2 мес; live-уверенность ≈0.5 с 24 мая; evidence-поток мёртв | 🔴 | — | 00, 03 |
| A3 | Train/serve skew: 35% regime-mismatch (доказано), HTF-4H фичи в 15m live — константы | 🔴 | live_feature_state.py:50-52, ml_strategy_15m.py:120,326 | 03 |
| A4 | DSR: E[SR_max] не умножен на √V[SR]; SR аннуализирован внутри формулы (→ и крэш `math domain error`, молча глотаемый в DSR=0); (γ4−3)/4 вместо канонич. (γ4−1)/4; proxy (WR−0.5)·PF·10 — единственный реально используемый путь; n_trials=10 в 4H-валидаторе | 🔴 | statistical_tests.py:97-122,110-116,427-441; validate_ml_models.py:186 | 02 |
| A5 | PBO — LOO-эвристика, не CSCV; результат неинтерпретируем | 🔴 | statistical_tests.py:177-255 | 02 |
| A6 | Геометрия сделки ≠ геометрия лейбла: модель предсказывает «+1.0·ATR раньше −0.8·ATR за ≤6 баров (по close)», торгуется SL=1.5·ATR/TP=2.25·ATR без time-exit, с интрабарными срабатываниями | 🔴 | risk_engine.py:37,106,281-300; ml_strategy.py (нет time-exit) | 05 |
| A7 | high_vol-модель задеплоена и торгует при **всех** passes=False (PF 0.90–0.95 в собственном eval); порог 0.65 в проде никогда не валидировался (eval на 0.55) | 🔴 | retrain_v3_selected_results.txt; .env; _select_model | 04 |
| A8 | Eval-гейт (WR/PF) считается **gross**, без комиссий/слиппеджа | 🟠 | lgbm_trainer.py:507-548 | 02 |
| A9 | Sqrt-импакт с σ_annual вместо σ_daily → издержки завышены ~в 19 раз → фильтр expected_return душит сигналы | 🟠 | cost_model.py:75 | 05 |
| A10 | Reconciler в стратегии молча падает каждые 15 мин («no running event loop»); джун-2: ордер отвергнут биржей (Margin insufficient), но сигнал записан как исполненный | 🟠 | ml_strategy.py:493-533; журнал | 05 |
| A11 | Барьеры лейблинга по close, SL/TP исполняются по интрабарному пути → систематический оптимизм лейблов | 🟠 | triple_barrier.py:129-141 | 02 |
| A12 | val→test embargo отсутствует (train→val сделан) | 🟠 | lgbm_trainer.py:344-357 | 04 |
| A13 | regime-фильтр до временного сплита; сплит по строкам, не по времени | 🟠 | lgbm_trainer.py:273-287 | 04 |
| A14 | `_WARMUP_ROWS=200` < min_bars=300/atr_lookback=540 → ~100 строк дефолтных regime-фич в трейне | 🟠 | feature_pipeline.py:43,287 | 03 |
| A15 | t-stat: взвешенное среднее с невзвешенным std, деление на √n_windows | 🟡 | statistical_tests.py:299-306 | 02 |
| A16 | PF капается 999; вин-рейты малых окон без биномиальных CI | 🟡 | lgbm_trainer.py:857-864 | 02 |
| A17 | walk_forward Sharpe без bar_duration → daily-collapse (3–5× inflation по собств. докстрингу) | 🟡 | walk_forward.py:438 vs metrics.py:71 | 02 |
| A18 | Все прод-модели — полный pickle; `use_native_save` никем не включён | 🟡 | lgbm_trainer.py:125,456-474 | 05 |
| A19 | Regime size-multiplier реализован, но не подключён к сайзингу; vol-фильтр с hardcoded 1% | 🟡 | risk_engine.py:396; regime_detector.py:181-213 | 05 |
| A20 | VPIN по stride-сэмплу — не VPIN (bucket-состав искажается); сейчас безвредно (фича не в моделях) | 🟡 | data_catalog.py:207-213 | 03 |

## Статус находок предыдущих аудитов (чек-лист владельца, 20 гипотез)

| # | Гипотеза | Статус | Где |
|---|---|---|---|
| 1 | Kurtosis-константа DSR | **Открыто, наоборот**: код «исправили» на (γ4−3)/4, но канон Bailey–LdP 2014 — (γ4−1)/4 (включает 0.5·SR² Мертенса). Текущий код теряет базовый член | statistical_tests.py:114 |
| 2 | E[SR_max] без √V(trials) | **Открыто** (std_sr считается на :84 и не используется в z) | statistical_tests.py:84,122 |
| 3 | SR аннуализирован в DSR | **Открыто** + вызывает крэш sqrt (доказано синтетикой) | :403 |
| 4 | n_experiments=10 | **Открыто** для 4H (`validate_ml_models.py:186`); 1h/15m — 100 (default) | — |
| 5 | Proxy (WR−0.5)·PF·10 | **Открыто**: `per_fold_daily_returns` не передаёт ни один вызывающий | grep по репо |
| 6 | PBO = LOO, не CSCV | **Открыто** | :177-255 |
| 7 | 15m-буфер 600 < 672 | **Открыто**; плюс 4H-буфер 400 < 540 (доказан 35% regime-mismatch) | live_feature_state.py:50-52 |
| 8 | Warmup < lookback | **Открыто** (200 < 300/540/720) | feature_pipeline.py:43 |
| 9 | Regime-фильтр до сплита | **Открыто** | lgbm_trainer.py:273-287 |
| 10 | 15m HTF из ресемпла | **Открыто и хуже**: 4H-фичи в live — всегда дефолты (100 баров < min_bars=300) | ml_strategy_15m.py:120,295-331 |
| 11 | live_enrichment не в трейнинге | **Подтверждено**: ни в одном из 3 прод-бандлов (unpickle-проверка) | — |
| 12 | vol-фильтр hardcoded 0.01 | **Открыто** | risk_engine.py:396 |
| 13 | Regime size-mult не подключён | **Открыто** | risk_engine.py:233-262 |
| 14 | confidence 0.55 vs 0.65 | **Подтверждено**: прод=0.65 (env), eval=0.55, 15m=0.58 — рассинхрон живой | .env, config.py:90, ml_strategy*.py |
| 15 | Sharpe без bar_duration | **Частично**: metrics.py исправлен; walk_forward.py:438 не передаёт | — |
| 16 | Асимметричные барьеры | Прод-конфиг = pt1.0/sl0.8/h6 (класс 46/54, умеренно). Экстремальные асимметрии — только в grid, не в проде | retrain_v3_selected_results.txt |
| 17 | Feed-bug stale prices | **Переквалифицировано**: лента в июне жила; реальный баг — zombie-RUNNING без данных после неудачного коннекта (A1) + молчание модели (A2) | 05 |
| 18 | pickle RCE/версии | **Открыто**: use_native_save нигде не включён; 15m грузит pickle напрямую (ml_strategy_15m.py:261) | — |
| 19 | PF cap 999 | **Открыто** | lgbm_trainer.py:864 |
| 20 | Регрессии прошлых фиксов | **Не найдено**: nan_to_num→NaN сохранён (:775), train→val embargo есть, ORB cum_max/after-window корректен, walk-forward таргет ветвится по use_triple_barrier, PurgedKFoldCV стал time-based, uniqueness на реальных t1_bar в исходных координатах | — |

Из top-5 `code_review_v3.md`: PurgedKFold ✅ закрыто, WF-таргет ✅ закрыто, nan_to_num ✅ закрыто,
DSR ❌ «исправлено» неверно (см. #1–3), val-embargo ⚠️ наполовину (val→test нет).

## Файлы отчёта

- `01_architecture_review.md` — архитектура, зависимости, что менять/не менять
- `02_math_and_validation.md` — DSR/PBO/t-stat/Sharpe/triple-barrier/cost — формулы против кода
- `03_data_and_features.md` — parity-эксперимент, буферы, warmup, VPIN
- `04_models_and_training.md` — тренер, сплиты, деплой-гейтинг, объём данных vs класс модели
- `05_execution_risk_security.md` — исполнение, риск, доступность, безопасность
- `06_test_plan.md` — failing-тесты на каждый критический дефект, таксономия
- `07_roadmap.md` — P0→P3 с честными диапазонами
- `08_references.md` — библиография с привязкой к находкам
