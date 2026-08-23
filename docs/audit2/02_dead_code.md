# АУДИТ-2 / ПРОХОД 1 — Мёртвый код, недоделки, дубли

HEAD: `f4af5fd210b32db8af4478f0e2f440e2eb504ccc`. Рабочее дерево чисто.
Протокол — [`00_method.md`](00_method.md). Кандидаты — §3 [`01_inventory.md`](01_inventory.md).

Всё ниже — из свежего вывода команд. Ни одной правки в `src/ scripts/ tests/ deploy/`.

---

## 1. ВЕРДИКТЫ ПО КАНДИДАТАМ ПРОХОДА 0

### 1.1 `reconciler.py` против `reconciler_signals.py` — РАЗНЫЕ ЗАДАЧИ, один МЁРТВ

Это **не** две реализации одного. Установлено чтением структуры обоих файлов:

```
$ grep -n 'def \|class ' src/execution/reconciler.py
34:class ReconciliationResult:
49:class InternalPosition:
61:class PositionReconciler:
82:    def __init__(
100:    async def reconcile(
236:    async def _fetch_exchange_positions(self) -> list[dict[str, Any]] | None:
=====SIGNALS=====
157:class SignalReconciler:
189:    def _bar_for(self, signal: dict) -> tuple[str, int, float]:
202:    def _evaluate(
254:    def reconcile(self) -> dict:
375:    def _update_daily_stats(self) -> None:
412:    def _refresh_performance_cache(self) -> None:
```

- `reconciler.py` / `PositionReconciler` — сверка **позиций** с биржей через Binance
  REST: orphan / ghost / mismatched sizes.
- `reconciler_signals.py` / `SignalReconciler` — закрытие **осиротевших строк
  `signals_log`** переигрыванием исторических цен. Работает с SQLite, не с
  позициями биржи.

Пересечения логики нет. Юнит `atomicortex-reconciler.service` вызывает
`scripts/run_reconciler.py`, который импортирует `reconciler_signals`.

**Вердикт `src/execution/reconciler.py`: МЁРТВЫЙ, причём удалён из графа намеренно.**

### A2-001 `PositionReconciler` мёртв, но докстринг утверждает обратное

**Севирити:** MEDIUM
**Тип:** мёртвый код / недоделка
**Где:** `src/execution/reconciler.py:1-13`

**Что в коде** (`sed -n '1,13p' src/execution/reconciler.py`):

```
"""
AtomiCortex — Position Reconciler.

Compares internal position state with the real exchange positions
(via Binance REST API) and detects:
- **Orphan positions**: exist on exchange but not in our state.
- **Ghost positions**: exist in our state but not on exchange.
- **Mismatched sizes**: direction or quantity disagree.

Runs at every reconnect to ensure internal state is correct.

Phase 4 — Step 4.6.
"""
```

**В чём дефект:** строка «Runs at every reconnect to ensure internal state is
correct» ложна. Модуль не импортирует никто:

```
$ grep -rn --include='*.py' 'execution.reconciler' src scripts | grep -v reconciler_signals
(пустой вывод)
```

Более того, отвязка закреплена тестом — `tests/test_reconciler_cleanup.py:6-12`:

```
def test_no_reconciler_methods_on_strategy():
    for name in ("_schedule_reconciliation", "_run_reconciliation", "_reconcile_async"):
        assert not hasattr(MLTradingStrategy, name)

def test_no_position_reconciler_import():
    content = Path("src/execution/strategies/ml_strategy.py").read_text()
    assert "PositionReconciler" not in content
```

**Как проявляется:** 275 строк с докстрингом, описывающим работающую защиту от
рассинхрона позиций после краха. Читатель кода (и оператор) считает orphan/ghost
detection действующим. Он не действует и по решению не должен.

**Кто ещё это читает:** только `tests/test_chaos.py` (импорт `InternalPosition`,
`PositionReconciler`, `ReconciliationResult`, строки 27-29) и
`tests/test_reconciler_cleanup.py`, который проверяет **отсутствие** связи.

**Как установлено:** замером (два grep выше + чтение теста).
**Уверенность:** доказано.

**Отношение к Аудиту-1:** находка не новая по факту, а по статусу. Аудит-1
зафиксировал `PositionReconciler` как мёртвый (`docs/code_review_v3.md:928` —
«CRITICAL: `PositionReconciler` is never called from any strategy»;
`:2092` — «Подключить `PositionReconciler.reconcile` на startup + периодически»).
Рекомендация Аудита-1 **не выполнена и выполнена не будет** — принято обратное
решение, закреплённое тестом. Не закрыто расхождение докстринга с этим решением.

---

### 1.2 Meta-подсистема — НЕДОСТИЖИМА, артефакт отсутствует

### A2-002 Meta-подсистема замкнута на себя: нет ни импортёра, ни модели

**Севирити:** HIGH
**Тип:** мёртвый код / недоделка
**Где:** `src/execution/strategies/meta_strategy.py`, `scripts/train_meta_model.py`,
`scripts/build_meta_dataset.py`

**Что в коде** (`src/execution/strategies/meta_strategy.py:183`):

```
    meta_model_path: str = "./data/features/models/v3/meta_model_v3.pkl"
```

`scripts/train_meta_model.py:256`:

```
    out_path = models_dir / "meta_model_v3.pkl"
```

**В чём дефект:** три звена, ни одно не подключено, артефакта нет.

1. **Импортёра нет.** `MetaMLTradingStrategy` не импортирует ни один лаунчер:
   ```
   $ grep -rn 'meta_strategy\|MetaStrategy' src scripts deploy .github README.md
   src/execution/strategies/meta_strategy.py:2:src/execution/strategies/meta_strategy.py
   docs/audit/01_architecture_review.md:43:…
   docs/code_review_v3.md:636:…
   ```
   Единственное упоминание вне `docs/` — собственная строка докстринга.

2. **Модели нет.** Поиск по файловой системе:
   ```
   $ find / -maxdepth 6 -name '*meta_model*' 2>/dev/null | grep -v '\.venv\|proc'
   /home/asus/Desktop/AtomiCortex/scripts/train_meta_model.py
   ```
   Найден только скрипт обучения. Файл `meta_model_v3.pkl` **отсутствует**.
   На диске в `data/features/models/v3/` лежат только:
   ```
   $ ls -la data/features/models/v3/
   drwxrwxr-x 2 asus asus   4096 Aug 14 22:39 _grid
   -rw-rw-r-- 1 asus asus 172614 Aug 14 22:39 high_vol_model_v3.pkl
   -rw-rw-r-- 1 asus asus 363142 Aug 14 22:39 trend_model_v3.pkl
   ```

3. **Скрипты подсистемы не упомянуты нигде,** кроме самой подсистемы
   (§3.4 `01_inventory.md`): `train_meta_model.py` и `build_meta_dataset.py`
   встречаются только внутри `meta_strategy.py`.

**Как проявляется:** заявленный слой мета-разметки (López de Prado AFML §3.6),
который в мастер-документе отвечает за фильтр take/skip, не влияет ни на один
торговый вывод. Ни в проде, ни в бэктесте, ни в валидации.

**Отношение к Аудиту-1:** `docs/code_review_v3.md:988` — «meta_strategy.py:
`_apply_meta_gate` never called». **Не закрыто.** Более того, состояние
ухудшилось в измеримом смысле: `_apply_meta_gate` теперь вызывается
(`meta_strategy.py:307`) внутри `_open_position`, то есть внутрифайловая связь
восстановлена, но сам класс по-прежнему никем не инстанцируется, а модель, без
которой гейт бессмыслен, отсутствует.

**Как установлено:** замером (три вывода выше).
**Уверенность:** доказано.

**Дополнительно — опасность оживления (см. §5).** `on_start` гейта
фейл-софтный (`meta_strategy.py:225-243`):

```
        try:
            self._gate = MetaSignalGate(
                bundle_path=Path(self._meta_config.meta_model_path),
                threshold=self._meta_config.meta_threshold,
                min_size=self._meta_config.meta_min_size,
            )
```
```
        except Exception as exc:
            # Fail-soft: degrade to base strategy. No silent corruption —
            # log loudly so the operator can fix it.
            self.log.error(
                f"Meta gate failed to load ({exc}); "
                f"continuing as base MLTradingStrategy"
            )
            self._gate = None
```

При включении подсистемы без файла модели она молча деградирует до базовой
стратегии, оставив в логах ERROR. Это **правильное** поведение — деградация
громкая, не тихая. Отмечено в §5 как «безвредна при оживлении».

---

### 1.3 `timeframes.py::db_path_for` — МЁРТВ; путей к БД пять независимых

### A2-003 Пять независимых определений «где лежат торговые БД»

**Севирити:** MEDIUM
**Тип:** архитектура / дубль
**Где:** `src/telegram_bot/timeframes.py:29-33`, `src/telegram_bot/bot.py:747-782`,
`src/api/main.py:46-57`, `src/telegram_bot/signal_poller.py:64`,
`scripts/run_reconciler.py:31-32`

**Что в коде.** Модуль, объявленный единственным источником истины
(`src/telegram_bot/timeframes.py:1-5`):

```
"""
Unified timeframe registry for the AtomiCortex Telegram bot.

Single source of truth: adding a new timeframe = one line in
``TIMEFRAMES``; keyboards, formatter and filters adapt automatically.
"""
```

Его карта (`timeframes.py:29-33`):

```
# 4H is the canonical DB; 15m/1h are appended when their isolated DBs
# exist (mirrors TelegramBot._get_shared_db_paths discovery order).
_DB_MAP: dict[str, str] = {
    "4h": "data/atomicortex.db",
    "15m": "data/atomicortex_15m.db",
    "1h": "data/atomicortex_1h.db",
}
```

Его аксессор (`timeframes.py:48-51`):

```
def db_path_for(timeframe: str) -> str | None:
    """Absolute DB path for a timeframe, or None if unknown."""
    rel = _DB_MAP.get(timeframe)
    return str(_ROOT / rel) if rel else None
```

**`db_path_for` не вызывает никто — ни `src/`, ни `scripts/`, ни `tests/`:**

```
$ grep -rn 'db_path_for' src scripts tests
src/telegram_bot/timeframes.py:48:def db_path_for(timeframe: str) -> str | None:
```

Фактически путь определяют четыре других места, каждое своим способом:

| Место | Способ |
|---|---|
| `src/telegram_bot/bot.py:750-753` | список кандидатов, включая **захардкоженный абсолютный путь другой машины**: `Path("/home/hashiflame/AtomiCortex/data/atomicortex.db")` |
| `src/telegram_bot/bot.py:778` | сиблинги перебором кортежа `("atomicortex_15m.db", "atomicortex_1h.db")` |
| `src/api/main.py:53` | свой кортеж `("atomicortex.db", "atomicortex_15m.db", "atomicortex_1h.db")` + env `ATOMICORTEX_DB_PATHS` |
| `src/telegram_bot/signal_poller.py:64` | фолбэк-литерал `paths = ["data/atomicortex.db"]` |
| `scripts/run_reconciler.py:31-32` | своя карта, ещё и с длительностью бара: `_BAR_HOURS = {"atomicortex.db": 4.0, "atomicortex_1h.db": 1.0, "atomicortex_15m.db": 0.25}` |

**В чём дефект:** контракт «имя файла ⇄ таймфрейм» размножен в пяти местах, при
том что существует модуль, объявленный его единственным владельцем, и его
аксессор мёртв. Комментарий в `timeframes.py:29` («mirrors
`TelegramBot._get_shared_db_paths` discovery order») прямо признаёт, что карта —
копия, а не источник.

**Как проявляется:** добавление таймфрейма требует правки пяти мест вместо
объявленной одной. Расхождение любого из них даёт частичную видимость: бот
Telegram видит БД, API — нет, реконсилятор считает длительность бара по имени
файла, которого нет в его карте (тогда `_BAR_HOURS.get(...)` вернёт `None` или
дефолт — предмет прохода 8).

**Кто ещё читает `timeframes`:** живы `active_timeframes()`
(`src/telegram_bot/keyboards.py:296`, `handlers_premium.py:55,136`) и
`tf_for_db_path()` (`handlers_premium.py:21`). То есть модуль живой, мёртв
именно тот его элемент, который отвечает за адрес БД.

**Как установлено:** замером (grep выше + чтение пяти реализаций).
**Уверенность:** доказано.

---

### 1.4 Сводная таблица вердиктов по §3.1 `01_inventory.md`

| Кандидат | Вердикт | Обоснование |
|---|---|---|
| `src/api/main.py` | **ЖИВОЙ, но не в проде** | `scripts/run_api.py:31` — строковый импорт `"src.api.main:app"`. Юнит `atomicortex-api.service` не в `units.enabled` → УСЛОВНО ЖИВОЙ |
| `src/execution/reconciler.py` | **МЁРТВЫЙ** | A2-001 |
| `src/execution/strategies/meta_strategy.py` | **МЁРТВЫЙ** | A2-002 |
| `src/execution/strategies/paper_strategy.py` | **МЁРТВЫЙ** | единственное упоминание вне тестов — `README.md:132` (дерево каталогов); ни один лаунчер его не импортирует (§2.3 `01_inventory.md`: колонки src/scripts пусты) |
| `src/telegram_bot/timeframes.py::db_path_for` | **МЁРТВЫЙ** | A2-003 |

### 1.5 Вердикты по §3.3 `01_inventory.md` (символы без ссылок)

Подтверждены как мёртвые (0 ссылок в `src/`, `scripts/`, `tests/`, включая
собственный файл; метод — AST + подсчёт вхождений идентификатора):

| Где | Символ |
|---|---|
| `src/config.py:229` | `Settings.bybit_api_key` |
| `src/config.py:234` | `Settings.bybit_api_secret` |
| `src/features/utils.py:33` | `rolling_correlation` |
| `src/ingestion/binance_downloader.py:362` | `download_metrics` |
| `src/ingestion/binance_downloader.py:385` | `download_agg_trades` |
| `src/ingestion/live_feed.py:155` | `LiveFeedManager.last_tick_time` |
| `src/models/dataset_builder.py:94` | `DatasetBuilder.build_features_all_symbols` |
| `src/monitoring/telegram_reporter.py:42,91,99` | `send_trade_alert`, `send_daily_report`, `send_weekly_report` |
| `src/telegram_bot/payments_crypto.py:159` | `CryptoBotPayment.stop_polling` |
| `src/telegram_bot/timeframes.py:48` | `db_path_for` |

`src/monitoring/telegram_reporter.py` — **весь публичный отправляющий API мёртв**
(три метода из трёх). См. A2-013.

---

## 2. ДУБЛИ РЕАЛИЗАЦИЙ

### 2.1 `ml_strategy.py` против `ml_strategy_15m.py`

Измерено AST-разбором обоих классов.

**Что переопределено** (7 методов из 34 родительских):

```
  __init__: parent ml_strategy.py:178 (84 lines) -> 15m ml_strategy_15m.py:111 (13 lines)
  _load_models: parent ml_strategy.py:1716 (22 lines) -> 15m ml_strategy_15m.py:251 (23 lines)
  _preload_historical_bars: parent ml_strategy.py:1788 (114 lines) -> 15m ml_strategy_15m.py:522 (104 lines)
  _start_heartbeat: parent ml_strategy.py:507 (36 lines) -> 15m ml_strategy_15m.py:210 (36 lines)
  on_bar: parent ml_strategy.py:641 (238 lines) -> 15m ml_strategy_15m.py:380 (137 lines)
  on_start: parent ml_strategy.py:267 (213 lines) -> 15m ml_strategy_15m.py:129 (62 lines)
  on_stop: parent ml_strategy.py:481 (21 lines) -> 15m ml_strategy_15m.py:192 (13 lines)
```

**Расхождения только по параметрам** (не дефект, зафиксировано для полноты):

```
  bar_type: 4H='BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL'  ->  15m='...-15-MINUTE-...'
  confidence_threshold: 4H=0.55  ->  15m=0.58
  heartbeat_key: 4H='atomicortex:heartbeat'  ->  15m='bot_15m_heartbeat'
  interval: 4H='4h'  ->  15m='15m'
  max_open_positions: 4H=3  ->  15m=1
  signal_db_path: 4H='data/atomicortex.db'  ->  15m='data/atomicortex_15m.db'
  warmup_bars: 4H=300  ->  15m=200
```

Ниже — два расхождения **по смыслу**.

### A2-004 15m-стратегия работает без circuit breaker

**Севирити:** HIGH
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy_15m.py:129-190`

**Что в коде** (`ml_strategy_15m.py:129-136`):

```
    def on_start(self) -> None:
        """Initialise 15m components and subscribe to 15m bars.

        Mirrors the parent's on_start structure but with the 15m regime
        detector, 15m models and 15m preload (parent's on_start would
        load 4H models and preload 4H bars — wrong for this bot — so it
        is intentionally not called).
        """
```

Родитель создаёт прерыватель именно в `on_start` (`ml_strategy.py:304`):

```
        self._breaker = CircuitBreaker(state_path=_risk_state_path)
```

`__init__` родителя оставляет его `None` (`ml_strategy.py:246`):

```
        self._breaker: Any = None
```

**Замер — единственный атрибут, который родительский `on_start` создаёт, а 15m
не создаёт** (AST-диф присваиваний `self._*` внутри обоих `on_start`):

```
Set by 4H on_start but NOT by 15m on_start:
   self._breaker

Set by 15m only:
   (пусто)
```

Во всём файле 15m-стратегии нет ни одного упоминания прерывателя:

```
$ grep -n '_breaker\|update_price\|_circuit\|CircuitBreaker' src/execution/strategies/ml_strategy_15m.py
(пустой вывод)
```

Родитель его проверяет (`ml_strategy.py:716`, `:724`):

```
            if self._breaker is not None and self._tracker is not None:
```
```
                    breaker_state = self._breaker.check(
```

**В чём дефект:** докстринг обещает «Mirrors the parent's on_start structure».
Зеркало неполно ровно в одном месте — и это место есть многоуровневый
прерыватель, отвечающий за каскадное уменьшение размера позиции и полную
остановку торговли. `self._breaker` остаётся `None`, и родительская проверка
`if self._breaker is not None` тихо пропускается.

**Как проявляется:** 15m-бот торгует без предохранителя по просадке и серии
убытков. Ограничения `-2%/-3%/-15%` (по описанию модуля
`src/risk/circuit_breaker.py`) на нём не действуют вообще. Молчаливо: ветка
`if self._breaker is not None` не логирует своё несрабатывание.

**Кто ещё это читает:** `scripts/run_paper_15m.py` (юнит
`atomicortex-bot-15m.service`) и `scripts/run_live_15m.py`. Второй, по
собственному докстрингу, «can place REAL orders». Ни один из этих юнитов не
входит в `units.enabled`, поэтому сегодня дефект не реализуется. Он реализуется
в тот момент, когда 15m-юнит включают.

**Как установлено:** замером (AST-диф + grep выше).
**Уверенность:** доказано.

### A2-005 15m `on_bar` не переоценивает открытые позиции по рынку

**Севирити:** MEDIUM
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy_15m.py:380-516`

**Что в коде.** AST-диф множеств вызовов `self.*` внутри обоих `on_bar`:

```
=== in 4H on_bar, ABSENT from 15m on_bar (self.* only) ===
   self._bars[-1].close.as_double
   self._bars[-2].close.as_double
   self._breaker.check
   self._compute_features_unified
   self._fetch_taker_buy_volume_for_bar
   self._live_state.add_bar
   self._select_model
   self._signal_bridge.log_regime_change
   self._signal_bridge.update_metrics
   self._tracker._positions.keys
   self._tracker.update_price
   self.log.debug
=== in 15m on_bar, absent from 4H ===
   self._build_feature_row
   self._vector
```

Родитель (`ml_strategy.py:687`):

```
                    self._tracker.update_price(_sym, close_px)
```

**В чём дефект:** `PortfolioTracker.update_price` — единственный путь, которым
открытая позиция переоценивается по текущей цене. Без него нереализованный P&L
не обновляется, а значит `get_state().equity`, просадка и счётчики, на которые
опирается риск-движок, отражают только закрытые сделки.

**Как проявляется:** 15m-бот принимает риск-решения по equity, не учитывающему
плавающий убыток по открытой позиции. Так как `max_open_positions: int = 1`
(`ml_strategy_15m.py:88`), эффект ограничен одной позицией, но по направлению
всегда одинаков: риск недооценён.

**Кто ещё это читает:** `self._risk_engine.evaluate` и `self._tracker.get_state`
вызываются в 15m `on_bar` (они в списке `=== common ===`), то есть потребители
искажённого equity присутствуют.

**Как установлено:** замером (AST-диф + grep `update_price` по файлу 15m — пустой).
**Уверенность:** доказано (что вызова нет); последствие — вероятно, точная
величина зависит от `PortfolioTracker`, разбор которого отнесён к проходу 7.

---

### 2.2 `train_models` / `train_1h_models` / `train_15m_models`

**Замер объёма дубля:**

```
$ wc -l scripts/train_models.py scripts/train_1h_models.py scripts/train_15m_models.py
   87 scripts/train_models.py
  433 scripts/train_1h_models.py
  435 scripts/train_15m_models.py
```
```
$ diff --unchanged-group-format='%=' --old-group-format='' --new-group-format='' \
       --changed-group-format='' scripts/train_1h_models.py scripts/train_15m_models.py | wc -l
363
```

**363 из 433 строк (83.8%) идентичны** между `train_1h_models.py` и
`train_15m_models.py`. `train_models.py` (87 строк) — иной, тонкая обёртка над
`src/models/training_pipeline.py`.

**Расхождения по математике** (дословно из `diff scripts/train_1h_models.py
scripts/train_15m_models.py`) — все обоснованы таймфреймом, дефекта не найдено:

```
< _WF_TRAIN_MONTHS = 12   # vs 18 for 4H — fewer months, more bars per month
< _WF_TEST_MONTHS = 4     # vs 6 for 4H
---
> _WF_TRAIN_MONTHS = 10   # less than 1H (12) — 15m has 4× more bars per month
> _WF_TEST_MONTHS = 3     # vs 4 for 1H
```
```
< _EMBARGO_BARS = 48       # 2 days × 24 bars/day on 1H
---
> _EMBARGO_BARS = 16       # 4 hours × 4 bars/hour on 15m
```

**Вердикт по 2.2:** дубль кода велик (83.8%), но расхождения параметров
осмысленны и задокументированы в самих комментариях. Математического
расхождения между 1H и 15m в сплитах и эмбарго **не обнаружено**. Гейт (критерий
остановки / go-no-go) в этих скриптах не задаётся — он в `validate_*`, см. 2.3.

---

### 2.3 `validate_ml_models` / `validate_1h_models` / `validate_15m_models`

**Замер объёма дубля:**

```
$ wc -l scripts/validate_ml_models.py scripts/validate_1h_models.py scripts/validate_15m_models.py
  196 scripts/validate_ml_models.py
  792 scripts/validate_1h_models.py
  798 scripts/validate_15m_models.py
```
```
$ diff --unchanged-group-format='%=' ... scripts/validate_1h_models.py scripts/validate_15m_models.py | wc -l
728
```

**728 из 792 строк (91.9%) идентичны.**

### A2-006 `--n-experiments` печатается в отчёте, но в расчёт DSR не попадает

**Севирити:** HIGH
**Тип:** математика / недоделка
**Где:** `scripts/validate_1h_models.py:562` и `:745`, `:763`;
`scripts/validate_15m_models.py:568` и `:751`, `:769`

**Что в коде.** Полный перечень вхождений идентификатора в файле:

```
$ grep -n 'n_experiments' scripts/validate_1h_models.py
562:            n_experiments=_N_EXPERIMENTS_DEFAULT,
745:    n_experiments = args.n_experiments
763:    print(f"  N experiments : {n_experiments}")
```
```
$ grep -n 'n_experiments' scripts/validate_15m_models.py
568:            n_experiments=_N_EXPERIMENTS_DEFAULT,
751:    n_experiments = args.n_experiments
769:    print(f"  N experiments : {n_experiments}")
```

Аргумент объявлен и документирован (`validate_1h_models.py:713-721`):

```
    p.add_argument(
        "--n-experiments",
        default=_N_EXPERIMENTS_DEFAULT,
        type=int,
        help=(
            "Number of strategy configurations tested (for DSR). "
            f"Default={_N_EXPERIMENTS_DEFAULT} (honest project estimate). "
            "See DSR-sensitivity table in the report."
        ),
    )
```

Единственный расчётный вызов (`validate_1h_models.py:559-562`):

```
    if cv_results and wf_result.windows:
        stat_result = run_all_tests(
            cv_results=cv_results,
            wf_result=wf_result,
            n_experiments=_N_EXPERIMENTS_DEFAULT,
```

**В чём дефект:** значение из CLI кладётся в локальную переменную
`n_experiments`, печатается в шапке отчёта как «N experiments : …», и на этом
его жизнь заканчивается. DSR всегда считается с модульной константой
`_N_EXPERIMENTS_DEFAULT = 100` (`validate_1h_models.py:104`,
`validate_15m_models.py:105`).

**Как проявляется:** запуск `--n-experiments 500` выдаёт отчёт, где написано
`N experiments : 500`, а Deflated Sharpe Ratio посчитан при N=100. DSR монотонно
убывает с ростом числа испытаний — значит отчёт систематически **завышает**
DSR относительно того, что сам же декларирует. Отчёт лжёт о собственных
параметрах, и лжёт в сторону оптимизма. Ровно то, ради чего DSR и вводится.

Отдельно: `dsr_sensitivity(cv_results, wf_result)` (строка 566) строит таблицу
чувствительности, а строка 673 помечает в ней строку
`"  ← current assumption" if n == _N_EXPERIMENTS_DEFAULT` — то есть маркер
«текущее допущение» тоже указывает на константу, а не на то, что напечатано в
шапке. Два места отчёта расходятся между собой при любом CLI-переопределении.

**Кто ещё это читает:** `src/models/statistical_tests.py:357` `run_all_tests`
(параметр `n_experiments: int = 10`), далее `calculate_dsr`. Контракт «число
испытаний» до места расчёта не доходит.

**Как установлено:** замером (два grep выше — полный перечень вхождений).
**Уверенность:** доказано.

### A2-007 Три валидатора — три разных конфигурации DSR; аннуализация расходится внутри одного отчёта

**Севирити:** HIGH
**Тип:** математика
**Где:** `scripts/validate_ml_models.py:183-187`, `scripts/validate_1h_models.py:559-562`,
`scripts/validate_15m_models.py:565-568`; `src/models/statistical_tests.py:357-362`

**Что в коде.** Сигнатура (`src/models/statistical_tests.py:357-362`):

```
def run_all_tests(
    cv_results: list[EvaluationResult],
    wf_result: "WalkForwardMLResult",  # forward ref to avoid circular import
    n_experiments: int = 10,
    per_fold_daily_returns: list[np.ndarray] | None = None,
    annualization_factor: float = 365.0,
) -> StatTestResult:
```

Её же докстринг (`statistical_tests.py:377-384`):

```
    per_fold_daily_returns:
        Optional list of daily P&L arrays (one per fold/window).
        If provided, real Sharpe ratios are computed from these returns
        along with their true skewness and kurtosis.
        If ``None``, a proxy SR is computed from win_rate and profit_factor
        (backward-compatible fallback).
    annualization_factor:
        Number of trading days per year for SR annualization.
        Default 365 (crypto trades 24/7).
```

Три вызывающих места:

`scripts/validate_ml_models.py:183-187`:
```
        stat_result = run_all_tests(
            cv_results=cv_results,
            wf_result=wf_result,
            n_experiments=10,
        )
```

`scripts/validate_1h_models.py:559-562` и `scripts/validate_15m_models.py:565-568`:
```
        stat_result = run_all_tests(
            cv_results=cv_results,
            wf_result=wf_result,
            n_experiments=_N_EXPERIMENTS_DEFAULT,
```

**В чём дефект — три расхождения сразу:**

1. **`per_fold_daily_returns` не передаёт никто.** Во всех трёх вызовах аргумент
   отсутствует ⇒ `None` ⇒ DSR считается по **прокси-Sharpe из win_rate и
   profit_factor**, а не по реальным доходностям. При этом те же скрипты
   реальный Sharpe считают — и печатают его в том же отчёте
   (`validate_1h_models.py:464`):
   ```
        # Sharpe: daily P&L aggregation with costs, annualized by sqrt(252)
   ```
   ```
            sharpe = (
                mean_daily / std_daily * np.sqrt(252)
            ) if std_daily > 0 else 0.0
   ```
   То есть в отчёте соседствуют `OOS Sharpe`, посчитанный из дневного P&L с
   издержками, и `DSR`, посчитанный из прокси WR×PF по тем же данным. Это две
   разные величины, представленные как один непротиворечивый набор.

2. **Аннуализация расходится.** Собственный Sharpe скрипта аннуализирован
   `np.sqrt(252)` (`validate_1h_models.py:484`, `validate_15m_models.py:485`);
   `run_all_tests` по умолчанию использует `annualization_factor: float = 365.0`
   и переопределения не получает ни от одного вызывающего:
   ```
   $ grep -n '252\|365' scripts/validate_1h_models.py scripts/validate_15m_models.py scripts/validate_ml_models.py
   scripts/validate_15m_models.py:465:        # Sharpe: daily P&L aggregation with costs, annualized by sqrt(252)
   scripts/validate_15m_models.py:485:                mean_daily / std_daily * np.sqrt(252)
   scripts/validate_1h_models.py:464:        # Sharpe: daily P&L aggregation with costs, annualized by sqrt(252)
   scripts/validate_1h_models.py:484:                mean_daily / std_daily * np.sqrt(252)
   ```
   `365` в этих файлах **отсутствует**. Крипта торгуется 24/7 — 365 корректно, 252
   унаследовано от акций. Внутри одного отчёта используются обе конвенции:
   отношение √(365/252) ≈ 1.204, то есть Sharpe в шапке занижен относительно
   базы, на которой строится DSR, примерно на 20%.

3. **`n_experiments` различается между валидаторами:** 10 в
   `validate_ml_models.py`, 100 в 1H и 15m. Один и тот же проект, одно и то же
   семейство моделей — три разных допущения о числе испытаний, влияющих на DSR.

**Как проявляется:** порог go/no-go по DSR применяется к величине, которая
(а) построена на прокси-Sharpe, а не на измеренном, (б) неявно предполагает
365 торговых дней, тогда как отображаемый Sharpe предполагает 252, (в) зависит
от того, каким из трёх скриптов запущена валидация. Сравнивать результаты
1H/15m с 4H по DSR нельзя — они посчитаны в разных системах допущений.

**Кто ещё это читает:** `StatTestResult.passes_all_thresholds`
(`src/models/statistical_tests.py:328`) — по §3.2 `01_inventory.md`
вызывается только из тестов; то есть гейт по порогам к тому же не применяется
автоматически.

**Как установлено:** замером (три вызова, сигнатура, grep по константам).
**Уверенность:** доказано (расхождение); влияние на конкретные числа —
вероятно, точная величина требует прогона валидации, который в этом проходе
не разрешён.

**Отношение к бэклогу:** пункт D-2 («1h/15m считают DSR/PBO не на
прод-параметрах») **подтверждён и расширен**: дело не только в параметрах, но и
в источнике Sharpe и в конвенции аннуализации.

---

### 2.4 Другие найденные дубли

### A2-008 Шесть копий отображения «символ → код», от которого зависит модель

**Севирити:** MEDIUM
**Тип:** дубль / архитектура
**Где:** `src/models/lgbm_trainer.py:141-145` (канонический),
`src/execution/strategies/ml_strategy.py:1344` и `:1506`,
`src/execution/strategies/ml_strategy_15m.py:351`,
`scripts/validate_1h_models.py:409`, `scripts/validate_15m_models.py:409`

**Что в коде.** Источник (`src/models/lgbm_trainer.py:141-145`):

```
SYMBOL_ENCODING: dict[str, int] = {
    "BTCUSDT": 0,
    "ETHUSDT": 1,
    "SOLUSDT": 2,
}
```

Пять инлайн-копий:

```
$ grep -rn --include='*.py' '"BTCUSDT": 0' src scripts
src/models/lgbm_trainer.py:142:    "BTCUSDT": 0,
src/execution/strategies/ml_strategy.py:1344:            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
src/execution/strategies/ml_strategy.py:1506:            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
scripts/validate_1h_models.py:409:        sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
scripts/validate_15m_models.py:409:        sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
src/execution/strategies/ml_strategy_15m.py:351:            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
```

Живой прод-путь (`ml_strategy.py:1341-1345`):

```
            # Add symbol_encoded (not emitted by pipeline)
            sym_str = str(self._instrument_id)
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            base = sym_str.split("-")[0] if "-" in sym_str else sym_str.split(".")[0]
            rd["symbol_encoded"] = float(sym_map.get(base, -1))
```

**В чём дефект:** `symbol_encoded` — обучающая фича (подтверждено замером, см.
A2-009). Её кодировка на обучении задаётся `SYMBOL_ENCODING`, а на инференсе —
литералом, скопированным в пять мест. Ни одна копия не импортирует канонический
словарь. Дефолт при промахе — `-1`, а не исключение.

**Как проявляется:** добавление четвёртого символа или перестановка кодов
требует синхронной правки шести мест. Пропуск любого даёт инференс с кодом `-1`
на фиче, на которой модель обучалась значениями `{0,1,2}` — не падение, а
молчаливо неверное предсказание. `_safe_float` (`ml_strategy.py:140-148`) такую
величину не отличит от корректной: `-1` конечен.

**Кто ещё это читает:** `src/models/lgbm_trainer.py:1369-1377` (обучение),
все шесть перечисленных мест (инференс и валидация). Контракт держится только
до тех пор, пока копии не разошлись; механизма, который бы это заметил, нет.

**Как установлено:** замером (grep выше).
**Уверенность:** доказано.

### A2-009 Исключение `symbol_encoded` из обучения не действует — фича добавляется обратно ниже по потоку

**Севирити:** MEDIUM
**Тип:** логика / мёртвый код
**Где:** `src/models/dataset_builder.py:58-63` и `:455`;
`src/models/lgbm_trainer.py:1366-1377`

**Что в коде.** Объявленное исключение (`src/models/dataset_builder.py:58-63`):

```
# Features excluded from training due to zero importance on BTCUSDT-only
# datasets.  Kept in the feature pipeline so they remain available if
# multi-symbol training is added later.
_TRAINING_EXCLUDE: set[str] = {
    "mtf_alignment_score",   # zero importance in both 1H models
    "symbol_encoded",        # only one symbol = no information
}
```

Единственное применение (`src/models/dataset_builder.py:455`):

```
        exclude = _EXCLUDE_COLUMNS | _TRAINING_EXCLUDE
```

Безусловное возвращение той же колонки ниже по потоку
(`src/models/lgbm_trainer.py:1366-1377`):

```
        # Always append symbol_encoded as the last feature
        all_cols = list(df_cols)
        if "symbol" in df.columns:
            symbol_encoded = (
                df["symbol"]
                .replace(SYMBOL_ENCODING, default=-1)
                .cast(pl.Float64)
                .to_numpy()
                .reshape(-1, 1)
            )
            X = np.hstack([X, symbol_encoded])
            all_cols.append("symbol_encoded")
```

**Замер — что реально лежит в бандлах:**

```
$ python - <<'EOF'   (polars + pickle, чтение data/features/ml_features и data/features/models)
ml_features columns: 64 | rows: 5038
  data/features/models/v3/_grid/high_vol_model_v3.pkl: 46 feature_columns
  data/features/models/v3/_grid/trend_model_v3.pkl: 46 feature_columns
  data/features/models/v3/high_vol_model_v3.pkl: 46 feature_columns
  data/features/models/v3/trend_model_v3.pkl: 46 feature_columns

union of model feature_columns: 46
features required by models but ABSENT from ml_features: 1 ['symbol_encoded']
```

**В чём дефект:** `dataset_builder` исключает `symbol_encoded` с явным
обоснованием «only one symbol = no information», а `lgbm_trainer._prepare`
добавляет её обратно безусловно — комментарием «Always append». Исключение
недостижимо: оно применяется к списку колонок DataFrame, а добавление
происходит после отбора, прямым `np.hstack`. Замер подтверждает: все четыре
бандла на диске содержат `symbol_encoded` в списке фич.

**Как проявляется:** два следствия.
1. Заявленное решение «эта фича бесполезна на одном символе» не исполнено —
   в модели она есть. На однасимвольном датасете это константный столбец,
   поданный в LightGBM. На мультисимвольном — порядковое кодирование
   категориальной переменной (`BTCUSDT`<`ETHUSDT`<`SOLUSDT`), по которому дерево
   будет строить пороги, а порядок в нём смысла не несёт.
2. `symbol_encoded` — **единственная** фича из 46, которой нет в
   `ml_features` (замер выше). Значит любой потребитель, читающий фичи прямо из
   Parquet, обязан достроить её сам. Это и порождает шесть копий кодировки из
   A2-008.

**Кто ещё это читает:** `src/execution/strategies/ml_strategy.py:1341-1345` и
`ml_strategy_15m.py:351-355` (достраивают на инференсе),
`scripts/validate_1h_models.py:409`, `scripts/validate_15m_models.py:409`.
Комментарий `lgbm_trainer.py:247` («order); symbol_encoded is still
auto-appended downstream») показывает, что автор знал о расхождении.

**Как установлено:** замером (чтение бандлов + parquet, вывод выше) и чтением.
**Уверенность:** доказано.

### A2-010 Настройки API объявлены в `Settings`, но читаются мимо неё через `os.getenv`

**Севирити:** LOW
**Тип:** архитектура / дубль
**Где:** `src/config.py:136`, `:137`, `:141`; `src/api/main.py:116`, `:131`, `:135`

**Что в коде.** Модуль конфигурации объявляет единственный путь
(`src/config.py`, докстринг):

```
AtomiCortex — centralized configuration module.
Reads all settings from a .env file via pydantic-settings.
Use get_settings() everywhere in the application — never instantiate
```

Три поля в `Settings` не читает никто, ни в `src/`, ни в `scripts/`, ни в
`tests/` (замер AST + подсчёт вхождений):

```
--- src/config.py :: Settings — поля без единого чтения вне определения ---
   src/config.py:136  atomicortex_api_key   (упоминаний в tests: 0)
   src/config.py:137  api_cors_origins   (упоминаний в tests: 0)
   src/config.py:141  api_rate_limit_per_minute   (упоминаний в tests: 0)
```

Тот же смысл читается напрямую из окружения (`src/api/main.py:116`, `:131`):

```
    key = os.getenv("ATOMICORTEX_API_KEY", "").strip()
```
```
    raw = os.getenv("API_CORS_ORIGINS", "http://localhost,http://127.0.0.1")
```

**В чём дефект:** два независимых пути чтения одних и тех же настроек. Значение,
попавшее в `.env` и разобранное pydantic-settings, не обязано присутствовать в
`os.environ` процесса API.

**Как проявляется:** при загрузке `.env` только через pydantic API получит
дефолты: сгенерированный эфемерный ключ (`main.py:120-127`) и CORS
`http://localhost,http://127.0.0.1`. Юнит `atomicortex-api.service` не в
`units.enabled`, поэтому сегодня не реализуется.

**Как установлено:** замером (AST-подсчёт + grep).
**Уверенность:** доказано (расхождение путей); реализация — гипотеза, зависит
от способа загрузки `.env` на VM, куда ходить запрещено.

### A2-011 `MLStrategyConfig1H.high_vol_model_path` не читает никто

**Севирити:** LOW
**Тип:** мёртвый код
**Где:** `src/configs/strategy_1h.py:78`

**Что в коде:**

```
    trend_model_path: str = "data/models/1h/trend_model_1h.pkl"
    high_vol_model_path: str = "data/models/1h/high_vol_model_1h.pkl"
```

Замер (AST + подсчёт вхождений по `src/`, `scripts/`, `tests/`):

```
--- src/configs/strategy_1h.py :: MLStrategyConfig1H — поля без единого чтения вне определения ---
   src/configs/strategy_1h.py:78  high_vol_model_path   (упоминаний в tests: 0)
```

Путь строится потребителями самостоятельно (`scripts/validate_1h_models.py:777`):

```
        model_path = models_dir / f"{regime}_model_1h.pkl"
```

**В чём дефект:** соседнее поле `trend_model_path` читается, `high_vol_model_path` —
нет; конвенция имени продублирована f-строкой в потребителе.
**Как проявляется:** правка конфига для high_vol не даёт эффекта.
**Как установлено:** замером. **Уверенность:** доказано.

Аналогично мёртво `src/configs/strategy_15m.py:68 session_trap_bars`
(тот же замер, 0 упоминаний вне определения).

---

## 3. НЕДОДЕЛКИ

### 3.1 TODO / FIXME / XXX / HACK / NotImplementedError

Полный вывод по `src/` и `scripts/`:

```
$ grep -rn --include='*.py' -E 'TODO|FIXME|XXX|HACK|NotImplementedError' src scripts
src/ingestion/data_quality.py:123:        """Return sorted list of ``date=XXXX`` directory names."""
src/execution/strategies/ml_strategy.py:1207:                # TODO(PR-0.2): Route this to Telegram via a watchdog-channel (no asyncio.run here)
src/execution/strategies/ml_strategy.py:1391:        TODO: replace with ``build_from_buffer()`` like
```

Три вхождения, из них одно (`data_quality.py:123`) — ложное срабатывание на
`XXXX` в докстринге. `NotImplementedError`, `FIXME`, `HACK` в кодовой базе
**отсутствуют**.

Второй TODO указывает на мёртвый метод: `ml_strategy.py:1369` — это
`_compute_features`, чей собственный докстринг (`:1373`) гласит
«DEPRECATED — replaced by ``_compute_features_unified()`` in». Именно там
находится вторая инлайн-копия `sym_map` (`:1506`, см. A2-008).

### 3.4 `except`, гасящие исключение без лога

**Замер:** AST-обход всех `ExceptHandler` в `src/` и `scripts/`; отобраны те,
чьё тело не содержит ни одного из `log/_log/logger/raise/print/warn/alert/notify/sys.exit`.

```
--- всего: 121 ---
```

Большинство — безобидные парсеры, возвращающие дефолт (`except (TypeError,
ValueError): return None`). Ниже — те, что гасят отказ в торговом или
риск-контуре.

### A2-012 Посев базы просадки и синхронизация equity целиком под `except Exception: pass`

**Севирити:** HIGH
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:2232-2262`

**Что в коде** (`sed -n '2232,2262p'`):

```
        try:
            self._equity_curve.append((ts_ns, equity))
            # S0-2: the first authoritative read seeds the drawdown and
            # percent baselines, which __init__ could only take from the
            # configured capital. Strictly before sync_equity: the seed
            # must fix day_start_equity at the balance as read, not at a
            # figure sync has already moved.
```
```
            if abs(equity) >= _ZERO_BALANCE_EPSILON:
                seeded = self._tracker.seed_from_authoritative_equity(equity)
```
```
            # H6: Nautilus is the authoritative source for cash balance
            # (the exchange confirms it). Sync the tracker so risk
            # decisions (sizing / drawdown / circuit breaker) see the
            # same equity, not a tracker-local drift.
            self._tracker.sync_equity(equity)
        except Exception:
            pass
```

**В чём дефект:** блок, который сам же в комментариях объявлен основанием для
sizing, drawdown и circuit breaker, обёрнут в перехват всех исключений без
единой строки лога. Любой отказ — `seed_from_authoritative_equity`,
`sync_equity`, `_equity_curve.append` — исчезает бесследно.

**Как проявляется:** риск-решения продолжают приниматься по
`initial_equity` из конфига (10 000 по умолчанию), а не по балансу счёта.
Наблюдаемо это ничем не отличается от нормальной работы: тот же лог, те же
сообщения, тот же поток баров. Тихая деградация к дефолту, который
потребитель считает данными — ровно тот класс, что описан в сквозной линзе
`00_method.md`.

Соседний код в том же методе тщательно разбирает похожую ситуацию громко
(`ml_strategy.py:2228-2231`):

```
            self.log.error(
                "exchange futures balance is 0 USDT — risk decisions will "
                "see 100% drawdown; check API keys / venue / account funding"
            )
```

То есть проектное намерение — сообщать оператору. Внешний `except` его
перекрывает.

**Кто ещё это читает:** `PortfolioTracker.get_state().equity` →
`RiskEngine.evaluate` (`ml_strategy.py`, on_bar), `CircuitBreaker.check`
(`ml_strategy.py:724`), `get_drawdown()`. Все они получат необновлённое equity
и не узнают об этом.

**Важное уточнение — сегодня эта ветка не исполняется.** Блок находится за
ранним возвратом по `dry_run` (`ml_strategy.py:2196`):

```
        if self._config.dry_run:
```

Прод-юнит `atomicortex-bot.service` запускает `run_live.py --mode paper
--dry-run`. Следовательно код с `except: pass` относится к категории §5:
мёртвый в текущей конфигурации и молча считающий не то при оживлении.

**Как установлено:** замером (AST-обход except-блоков) и чтением.
**Уверенность:** доказано.

Прочие гасящие `except` в горячем пути, зафиксированные для прохода 7
(не разворачиваю здесь):

```
src/execution/strategies/ml_strategy.py:858  except Exception:  ->  pass      (log_regime_change)
src/execution/strategies/ml_strategy.py:872  except Exception:  ->  pass      (update_metrics)
src/execution/strategies/ml_strategy.py:1203 except Exception:  ->  order = None   (cache.order при разборе SL)
src/execution/strategies/ml_strategy.py:2215 except Exception:  ->  pass      (_equity_curve.append в dry-run)
src/execution/strategies/ml_strategy.py:405  except (KeyError, TypeError, ValueError):  ->  continue
src/execution/strategies/ml_strategy.py:447  except (KeyError, TypeError, ValueError):  ->  continue
src/risk/circuit_breaker.py:265              except Exception:  ->  existing = {}
src/risk/portfolio_tracker.py:488/:497/:506  except ValueError:  ->  pass
src/risk/risk_state_store.py:134             except OSError:  ->  pass
src/execution/heartbeat.py:104/:108          except Exception:  ->  pass
```

### 3.5 Ветки, недостижимые при текущей конфигурации

Прод-юнит (§1.1.2 `01_inventory.md`) запускает
`scripts/run_live.py --mode paper --dry-run --symbols BTCUSDT-PERP …`.
Флаг доходит до конфига стратегии (`scripts/run_live.py:172`):

```
        dry_run=args.dry_run,
```

Все вхождения флага в 4H-стратегии:

```
$ grep -n 'dry_run' src/execution/strategies/ml_strategy.py
118:    dry_run: bool = False
317:            f"dry_run={self._config.dry_run} | "
495:        if not self._config.dry_run:
842:            if not self._config.dry_run:
2178:        PR-C: ``dry_run`` is the switch. It is the same flag that gates
2196:        if self._config.dry_run:
```

| Ветка | Условие | Исполнялась ли в проде |
|---|---|---|
| `ml_strategy.py:495` — `cancel_all_orders` + `close_all_positions` при остановке | `dry_run=False` | **нет** |
| `ml_strategy.py:842` — `self._open_position(decision, signal)` | `dry_run=False` | **нет** |
| `ml_strategy.py:2232-2262` — посев базы просадки, `sync_equity` (A2-012) | `dry_run=False` | **нет** |
| `ml_strategy.py:2196` — «Equity is SIMULATED» | `dry_run=True` | да |

### A2-013 В `--dry-run` сигнал не может попасть в журнал: у прод-конфигурации нет производителя сигналов

**Севирити:** CRITICAL
**Тип:** архитектура / логика
**Где:** `src/execution/strategies/ml_strategy.py:842`, `:1133`, `:992-1018`;
`deploy/atomicortex-bot.service`

**Что в коде.** Ветка входа (`ml_strategy.py:838-848`):

```
                f"Signal APPROVED | {regime_label} dir={direction} "
                f"conf={confidence:.3f} | size={decision.position_size:.6f} "
                f"notional=${decision.notional:.2f}"
            )
            if not self._config.dry_run:
                self._open_position(decision, signal)
            else:
                self.log.info(
                    f"[DRY RUN] Would open {signal.direction} "
                    f"{decision.position_size:.6f} @ ${signal.entry_price:.2f} | "
                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
```

Единственная запись в журнал сигналов (`ml_strategy.py:1016-1018`):

```
        try:
            signal_id = self._signal_bridge.log_signal(
```

Она находится в `_emit_signal`, и `_emit_signal` вызывается ровно из одного места:

```
$ grep -n '_emit_signal' src/execution/strategies/ml_strategy.py
992:    def _emit_signal(
1133:        self._emit_signal(decision, signal)
```

Строка 1133 лежит внутри `_open_position` (замер AST: «enclosing of 1133:
_open_position 1079 - 1155»), контекст (`ml_strategy.py:1131-1133`):

```
        # Log signal in bridge for Telegram notification BEFORE submitting order
        # to ensure signal_id exists if submit_order fails synchronously.
        self._emit_signal(decision, signal)
```

Собственный докстринг `_emit_signal` (`ml_strategy.py:997-999`):

```
        The single place where a signal becomes visible outside this
        process, and the reason it is a method rather than a block
        inlined in ``_open_position``:
```

**В чём дефект:** цепочка `dry_run=True → _open_position не вызывается →
_emit_signal не вызывается → log_signal не вызывается`. Флаг `--dry-run`
задуман как «не отправлять ордера» (`ml_strategy.py:2178`: «It is the same flag
that gates order submission in ``on_bar``»), но фактически он же отключает
**запись сигнала в журнал** — совершенно другое действие. Два разных решения
навешены на один флаг.

Та же конструкция в 15m (`ml_strategy_15m.py:504-505`):

```
            if not self._config.dry_run:
                self._open_position(decision, signal)
```

**Как проявляется:** прод-бот структурно не способен произвести ни одной строки
`signals_log` — независимо от confidence, режима и модели. При этом:

- в `units.enabled` (§1.2 `01_inventory.md`) включён
  `atomicortex-signal-check.service` + `.timer`, который каждый час проверяет
  **свежесть сигналов** — то есть мониторит производителя, отключённого
  конструктивно;
- `SignalBridge` → Telegram → `signal_poller` → `broadcaster` — вся доставка
  сигналов подписчикам не имеет источника;
- `StatsEngine` и REST API читают `signals_log`, который не наполняется.

**Локальный замер состояния журналов** (только локальное дерево; на VM
протокол ходить запрещает):

```
$ ls data/*.db
data/metrics.db
data/mlflow.db
```
```
=== data/atomicortex.db ===
  отсутствует
=== data/metrics.db ===
  metrics: 0 rows
  signals: 0 rows
  sqlite_sequence: 0 rows
```

**Кто ещё это читает:** `scripts/check_signal_freshness.py` (прод-юнит),
`src/telegram_bot/signal_poller.py`, `src/telegram_bot/broadcaster.py`,
`src/analytics/stats_engine.py`, `src/api/main.py`,
`src/execution/reconciler_signals.py`. Шесть потребителей у контракта, у
которого в текущей конфигурации нет писателя.

**Отношение к входным фактам:** установленный факт «live confidence 0.50-0.54
против порога 0.65, сигналов нет» объясняет отсутствие сигналов низкой
уверенностью. Находка показывает, что это объяснение **не полно**: даже при
confidence выше порога строка `signals_log` не появилась бы. Диагностика
системы о самой себе неверна.

**Как установлено:** замером (три grep + AST-определение объемлющей функции +
чтение юнита из §1.1.2) и чтением.
**Уверенность:** доказано для механики; полнота вывода — доказано в пределах
кода, поскольку `log_signal` вызывается ровно в одном месте.

---

---

## УТОЧНЕНИЕ A2-013 — ВЕРИФИКАЦИЯ (вставка, 2026-08-22)

Находка A2-013 отменяет объяснение, на котором построена работа Спринта 0,
поэтому проверена отдельно, по шести пунктам. Ниже жёстко разделено:
**[ЗАМЕР]** — вывод реально выполненной команды; **[ЧТЕНИЕ]** — вывод из
прочитанного кода.

### V1. Полная цепочка `on_bar → log_signal`, дословно [ЗАМЕР]

**Звено 1 — проверка `dry_run` в `on_bar`.** `sed -n '820,850p'
src/execution/strategies/ml_strategy.py`:

```
                timestamp=now_utc,
            )

            # 7. Risk evaluation
            self.log.info("on_bar step 7: risk evaluation")
            portfolio_state = self._tracker.get_state()
            decision = self._risk_engine.evaluate(signal, portfolio_state)

            if not decision.approved:
                self.log.info(
                    f"Signal BLOCKED | {regime_label} | "
                    f"dir={direction} conf={confidence:.3f} | "
                    f"reason={decision.reason}"
                )
                return

            # 8. Execute
            self.log.info(
                f"Signal APPROVED | {regime_label} dir={direction} "
                f"conf={confidence:.3f} | size={decision.position_size:.6f} "
                f"notional=${decision.notional:.2f}"
            )
            if not self._config.dry_run:
                self._open_position(decision, signal)
            else:
                self.log.info(
                    f"[DRY RUN] Would open {signal.direction} "
                    f"{decision.position_size:.6f} @ ${signal.entry_price:.2f} | "
                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
                )
```

Между `if not self._config.dry_run:` (строка 842) и `self._open_position(...)`
(строка 843) пропусков нет: вызов — единственная инструкция ветки.

**Звено 2 — `_open_position` до вызова `_emit_signal`.** `sed -n '1079,1135p'`:

```
    def _open_position(
        self,
        decision: RiskDecision,
        signal: TradeSignal,
    ) -> None:
        """Submit market entry order.

        PROD-005 fix: stop-loss is now submitted in on_order_filled()
        after the entry fill is confirmed, not here.  This eliminates
        the crash-between-entry-and-SL window.
        """
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self.log.error(f"Instrument {self._instrument_id} not found in cache")
            return
```
```
        # Log signal in bridge for Telegram notification BEFORE submitting order
        # to ensure signal_id exists if submit_order fails synchronously.
        self._emit_signal(decision, signal)
```

Между ними — сборка ордера и запись pending-SL, ни одного `return` кроме
`instrument is None` (строка 1092). То есть при отсутствии инструмента в кэше
сигнал тоже не пишется — второй, независимый путь потери записи.

**Звено 3 — `_emit_signal` до `log_signal`.** `sed -n '992,1020p'`:

```
    def _emit_signal(
        self,
        decision: RiskDecision,
        signal: TradeSignal,
    ) -> int:
```
```
        if not self._signal_bridge:
            return 0

        try:
            signal_id = self._signal_bridge.log_signal(
```

**Звено 4 — сам INSERT.** `src/execution/signal_bridge.py:200`, `:213`:

```
        """Write a new open signal to signals_log. Returns signal_id.
```
```
                        """INSERT INTO signals_log (
```

Цепочка замкнута: `on_bar:842 → on_bar:843 → _open_position:1133 →
_emit_signal:1018 → signal_bridge.py:213 (INSERT)`.

### V2. Все вхождения трёх идентификаторов [ЗАМЕР]

`grep -rn --include='*.py' 'log_signal\|_emit_signal\|_open_position' src/ scripts/`
— релевантные строки (вхождения `max_open_positions` отброшены как совпадение
подстроки, они к делу не относятся):

| Строка | Роль |
|---|---|
| `src/execution/signal_bridge.py:184` | **определение** `SignalBridge.log_signal` |
| `src/telegram_bot/database.py:355` | **определение** `Database.log_signal` (второй, см. V3) |
| `src/execution/strategies/ml_strategy.py:992` | **определение** `_emit_signal` |
| `src/execution/strategies/ml_strategy.py:1018` | **вызов** `self._signal_bridge.log_signal(` — единственный в `src/` и `scripts/` |
| `src/execution/strategies/ml_strategy.py:1079` | **определение** `_open_position` (родитель) |
| `src/execution/strategies/ml_strategy.py:1133` | **вызов** `self._emit_signal(decision, signal)` — единственный |
| `src/execution/strategies/ml_strategy.py:843` | **вызов** `self._open_position(decision, signal)` (4H, под `not dry_run`) |
| `src/execution/strategies/ml_strategy_15m.py:505` | **вызов** `self._open_position(decision, signal)` (15m, под `not dry_run`) |
| `src/execution/strategies/meta_strategy.py:297` | **определение** override `_open_position` (мёртвый класс, A2-002) |
| `src/execution/strategies/meta_strategy.py:303`, `:331` | **вызовы** `super()._open_position(decision, signal)` — переопределение не обходит цепочку, а делегирует в неё |
| `src/execution/signal_bridge.py:48`, `ml_strategy.py:221/890/905/1001/1002/1007`, `ml_strategy_15m.py:175`, `paper_strategy.py:38` | упоминания в комментариях и докстрингах, не вызовы |

Вызовов `_open_position` в `src/` и `scripts/` — **три**: `ml_strategy.py:843`,
`ml_strategy_15m.py:505`, `meta_strategy.py:303/331` (последние два —
`super()`-делегирование изнутри его же override). Первые два оба стоят под
`if not self._config.dry_run:`.

**Побочно обнаружено:** докстринг `src/execution/strategies/paper_strategy.py:34-40`
утверждает «Overrides only \_open\_position to route through PaperTrader», но
AST-разбор класса даёт:

```
PaperTradingStrategy : ['__init__', 'on_bar']
```

`_open_position` он не переопределяет. Класс мёртв (§1.4), на прод-путь не
влияет; зафиксировано как расхождение докстринга с кодом.

### V3. Второй путь записи в `signals_log`? [ЗАМЕР]

`grep -rn 'signals_log' --include='*.py' src scripts` — все **INSERT**:

```
src/execution/signal_bridge.py:213:                        """INSERT INTO signals_log (
src/telegram_bot/database.py:360:                """INSERT INTO signals_log
```

Остальные вхождения — `CREATE TABLE IF NOT EXISTS` (`signal_bridge.py:116`,
`database.py:103`), индексы, идемпотентные `ALTER`, `UPDATE`
(`signal_bridge.py:255`, `:281`; `reconciler_signals.py:336`) и `SELECT`.
`UPDATE` строк не создаёт.

**Итого два INSERT. Второй в прод-пути не участвует:**

```
$ grep -rn --include='*.py' '\.log_signal(' src scripts
src/execution/strategies/ml_strategy.py:1018:            signal_id = self._signal_bridge.log_signal(
```

`Database.log_signal` (`src/telegram_bot/database.py:355`) не вызывается ни из
`src/`, ни из `scripts/` — только из тестов:

```
$ grep -rn --include='*.py' '\.log_signal(' tests
tests/test_order_rejection.py:166:    sid = bridge.log_signal(
tests/test_signal_bridge_concurrency.py:137:                    br.log_signal(
tests/test_signal_bridge_concurrency.py:187:                br.log_signal(
tests/test_strategy_isolation.py:109:    sid = bridge.log_signal(
tests/test_telegram_db_roles.py:105:        sid = db.log_signal({"symbol": "BTCUSDT", "direction": "long",
…
```

`SignalBridge` инстанцируется ровно дважды, оба раза — стратегиями:

```
$ grep -rn --include='*.py' 'SignalBridge(' src scripts
src/execution/strategies/ml_strategy_15m.py:177:            self._signal_bridge = SignalBridge(
src/execution/strategies/ml_strategy.py:330:            self._signal_bridge = SignalBridge(db_path=db_path)
```

Прочие `INSERT` в дереве идут в другие таблицы и `signals_log` не касаются:
`metrics`/`signals` в `metrics_collector.py:159/:204`, `bot_events`
(`signal_bridge.py:320`, `database.py:749`), `daily_stats`
(`reconciler_signals.py:394`), `performance_cache` (`stats_engine.py:416`),
`users` (`database.py:250`), `payments` (`database.py:824`).

**Вывод V3:** второго пути записи в `signals_log` в прод-конфигурации нет.

### V4. 15m — тот же вывод [ЗАМЕР]

`sed -n '488,516p' src/execution/strategies/ml_strategy_15m.py`:

```
            self.log.info(
                f"15m signal APPROVED | {model_kind} dir={direction} "
                f"conf={confidence:.3f} | size={decision.position_size:.6f}"
            )
            if not self._config.dry_run:
                self._open_position(decision, signal)
            else:
                self.log.info(
                    f"[DRY RUN 15m] {signal.direction} "
                    f"{decision.position_size:.6f} @ ${current_price:.2f} "
                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
                )
```

Собственных `_open_position` / `_emit_signal` / `log_signal` у 15m нет:

```
$ grep -n '_open_position\|_emit_signal\|log_signal' src/execution/strategies/ml_strategy_15m.py
88:    max_open_positions: int = 1
147:            max_open_positions=self._config.max_open_positions,
175:            # inherited _open_position (parent code untouched) so the
505:                self._open_position(decision, signal)
```

Строки 88 и 147 — совпадение подстроки `max_open_positions`, строка 175 —
комментарий. 15m наследует ту же цепочку и гейтит её собственной проверкой
`dry_run`. **Вывод для 15m идентичен выводу для 4H.**

### V5. Писались ли сигналы когда-либо — ФАКТ [ЗАМЕР]

Живых торговых БД в локальном дереве нет:

```
$ ls -la data/*.db
-rw-r--r-- 1 asus asus    16384 May  2 23:32 data/metrics.db
-rw-r--r-- 1 asus asus 25260032 Aug 20 08:19 data/mlflow.db
```

Но в репозитории лежит снимок прод-БД, сделанный Аудитом-1:

```
$ find . -name 'atomicortex*.db*' -not -path './.venv/*'
./docs/audit/atomicortex_20260702.db.bak
```

Чтение снимка (режим `mode=ro`):

```
tables:
  bot_events: 3 rows
  bot_metrics: 1 rows
  daily_stats: 7 rows
  payments: 0 rows
  performance_cache: 12 rows
  signals_log: 8 rows
  sqlite_sequence: 3 rows
  users: 0 rows

signals_log rows: 8
   (1, 'BTCUSDT-PERP.BINANCE', 'short', 0.7365677149031172, 'trend_up', 'win', '2026-05-13T04:00:00.437763+00:00')
   (2, 'BTCUSDT-PERP.BINANCE', 'short', 0.6526180776604393, 'trend_up', 'loss', '2026-05-14T08:00:00.370716+00:00')
   (3, 'BTCUSDT-PERP.BINANCE', 'short', 0.6542212366155612, 'trend_up', 'win', '2026-05-17T20:00:00.762898+00:00')
   (4, 'BTCUSDT-PERP.BINANCE', 'short', 0.6671981233601358, 'trend_down', 'loss', '2026-05-19T04:00:01.171901+00:00')
   (5, 'BTCUSDT-PERP.BINANCE', 'short', 0.6714433739228134, 'trend_up', 'win', '2026-05-19T08:00:00.275929+00:00')
   (6, 'BTCUSDT-PERP.BINANCE', 'short', 0.6608271193764372, 'trend_down', 'win', '2026-05-22T00:00:00.685035+00:00')
   (7, 'BTCUSDT-PERP.BINANCE', 'long', 0.6655427027106472, 'trend_down', 'loss', '2026-05-22T20:00:00.384903+00:00')
   (8, 'BTCUSDT-PERP.BINANCE', 'long', 0.6556943351460622, 'high_vol', 'loss', '2026-06-02T20:00:00.673006+00:00')
```

**Сигналы писались.** Восемь строк, 2026-05-13 … 2026-06-02, confidence
0.6527 … 0.7366. Это наблюдение — и по протоколу (`00_method.md`, стандарт
доказательства: «Если находка противоречит наблюдаемому поведению системы —
наблюдение сильнее») оно требовало перепроверки находки. Перепроверка дана в V6:
наблюдение находке **не противоречит**, оно её датирует. Все восемь строк
относятся к периоду, когда прод-юнит работал **без** `--dry-run`.

Текущее состояние `signals_log` на VM этим замером не устанавливается: в
локальном дереве `data/atomicortex.db` **отсутствует**, `data/` в `.gitignore`,
на VM протокол ходить запрещает. Вопрос «сколько строк в `signals_log` сегодня»
решается владельцем на VM.

### V6. ПРЯМОЙ ОТВЕТ [ЗАМЕР + ЧТЕНИЕ]

> **`dry_run=True` → возможна ли запись в `signals_log`?**
>
> **НЕТ.**

Обоснование: единственный `INSERT INTO signals_log`, достижимый из прод-кода
(`signal_bridge.py:213`), вызывается только из `SignalBridge.log_signal`, та —
только из `_emit_signal` (`ml_strategy.py:1018`), та — только из
`_open_position` (`ml_strategy.py:1133`), а тот — только из ветки
`if not self._config.dry_run:` (`ml_strategy.py:843` для 4H,
`ml_strategy_15m.py:505` для 15m). Второй `INSERT`
(`telegram_bot/database.py:360`) вне тестов не вызывается (V3).

**С какого коммита это так — два уровня, оба нужны:**

**1. Уровень кода — `26acae9`, 2026-05-06 01:01:56 +0530**
«feat: signal bridge — cross-process trading→Telegram integration via shared SQLite».

```
$ git log --format='%h %ci %s' -S 'self._signal_bridge.log_signal(' -- src/execution/strategies/ml_strategy.py
26acae9 2026-05-06 01:01:56 +0530 feat: signal bridge — cross-process trading→Telegram integration via shared SQLite
```

Проверка расположения на самом `26acae9` (AST по `git show`):

```
dry_run-проверка(325) внутри: on_bar 226 - 356
log_signal(512) внутри: _open_position 461 - 528
```

и дословно `git show 26acae9:…ml_strategy.py | sed -n '320,330p'`:

```
        if not self._config.dry_run:
            self._open_position(decision, signal)
        else:
            self.log.info(
                f"[DRY RUN] Would open {signal.direction} "
```

То есть запись в журнал была помещена под гейт `dry_run` **в том же коммите,
которым появилась**. Позже она была вынесена в отдельный метод — `7593989`,
2026-08-20 00:16:48 +0530, «feat(ops): publish signal freshness in the heartbeat
and read it before the ledger» (`git log -S 'def _emit_signal'`) — но точка
вызова осталась внутри `_open_position`.

Коммит `71f163b` (2026-05-02 22:32:24 +0530), который выдаёт
`git log -S 'if not self._config.dry_run:'`, ввёл **саму проверку**, но на тот
момент `log_signal` в файле ещё не было — проверено `git show
71f163b:…ml_strategy.py | grep log_signal` (пустой результат). Поэтому он
датирует гейт, а не сцепку гейта с записью.

**2. Уровень прод-конфигурации — `e444eac`, 2026-08-14 11:45:19 +0530**
«fix(deploy): run the 4H unit as paper --dry-run and refuse order-capable
configs without it at startup».

```
$ git log --format='%h %ci %s' -S '--dry-run' -- deploy/atomicortex-bot.service
e444eac 2026-08-14 11:45:19 +0530 fix(deploy): run the 4H unit as paper --dry-run and refuse order-capable configs without it at startup
```

Диф `ExecStart` в этом коммите:

```
-    --mode testnet \
+    --mode paper \
+    --dry-run \
```

До `e444eac` прод-юнит работал `--mode testnet` **без** `--dry-run`, то есть
`_open_position` вызывался и сигналы писались. Это ровно объясняет восемь строк
из V5 (2026-05-13 … 2026-06-02) и снимает кажущееся противоречие.

**Итоговая формулировка A2-013, уточнённая:**

Механика (запись в журнал достижима только через `_open_position`, а тот —
только при `dry_run=False`) существует с **`26acae9` (2026-05-06)**.
Прод-развёртывание вошло в это состояние с **`e444eac` (2026-08-14 11:45:19
+0530)**. С этой даты прод-бот структурно не может записать ни одной строки
`signals_log` — независимо от confidence, режима и модели.

**Что доказано замером:** цепочка вызовов и её единственность (V1, V2, V3, V4);
факт прошлых записей и их даты (V5); даты обоих коммитов и содержимое дифа (V6).

**Что установлено чтением:** интерпретация — что `--dry-run` задуман как «не
отправлять ордера» (докстринг `ml_strategy.py:2178`: «It is the same flag that
gates order submission in ``on_bar``»), а фактически выключает и запись в
журнал; и что это два разных решения на одном флаге. Само расхождение
намерения и следствия — вывод из текста, не замер.

**Что НЕ установлено и решается владельцем на VM:** сколько строк в
`signals_log` сегодня и что показывает `atomicortex-signal-check.service` с
2026-08-14. Локальных копий живой БД нет.

---

## 4. МЁРТВЫЕ ДАННЫЕ И АРТЕФАКТЫ

### 4.1 Файлы в `data/`

```
$ find data/models -type f | sort
data/models/1h/high_vol_model_1h.pkl
data/models/1h/high_vol_model.pkl
data/models/1h/trend_model_1h.pkl
data/models/1h/trend_model.pkl
```
```
$ ls -la data/features/models/
drwxrwxr-x 3 asus asus 4096 Aug 14 22:39 v3
```
```
$ ls -la data/features/models/v3/
drwxrwxr-x 2 asus asus   4096 Aug 14 22:39 _grid
-rw-rw-r-- 1 asus asus 172614 Aug 14 22:39 high_vol_model_v3.pkl
-rw-rw-r-- 1 asus asus 363142 Aug 14 22:39 trend_model_v3.pkl
```

**Кто что читает:**

| Артефакт | Читатель | Статус |
|---|---|---|
| `data/features/models/{trend,high_vol}_model.pkl` | `ml_strategy.py:1721` — `path = models_dir / f"{regime}_model.pkl"` при `models_dir: str = "./data/features/models"` (`:105`) | **в локальном дереве ОТСУТСТВУЮТ** — есть только подкаталог `v3/` |
| `data/features/models/v3/*_v3.pkl` | никем из `src/` по имени не читается: имя строится как `{regime}_model.pkl`, без суффикса `_v3` | не читаются прод-путём |
| `data/features/models/v3/_grid/*` | не читается ничем | мёртвый |
| `data/models/1h/*` | `scripts/validate_1h_models.py:777` — `models_dir / f"{regime}_model_1h.pkl"` | `*_model_1h.pkl` читаются; `trend_model.pkl` и `high_vol_model.pkl` в том же каталоге — **не читаются** (не тот шаблон имени) |
| `data/models/15m/…` | `src/configs/strategy_15m.py:93-94` (`trend_model_15m.pkl`, `orb_model_15m.pkl`) | каталог `data/models/15m` **отсутствует** |
| `data/metrics.db` | `src/monitoring/metrics_collector.py` | существует, обе таблицы пусты (0 строк) — писателя нет, см. A2-014 |
| `data/atomicortex.db` | 6 потребителей (A2-013) | **отсутствует** в локальном дереве |

**Оговорка.** `data/` в `.gitignore` (строка `data/`), поэтому содержимое
локального дерева — локальный артефакт, а не слепок VM. На VM протокол ходить
запрещает. Утверждение «прод-бандл — не тот артефакт, для которого считались
метрики» дано во входных фактах и здесь не перепроверяется; локальный замер ему
не противоречит: в локальном дереве по прод-пути `./data/features/models/` нет
ни одного файла с ожидаемым именем, а `_load_models` на такой случай
предусматривает только предупреждение (`ml_strategy.py:1736`):

```
                self.log.warning(f"Model not found: {path}")
```

то есть стратегия стартует с `self._trend_model = None` и молча.

### 4.2 Колонки `ml_features`, не используемые ни одной моделью

Замер (polars + pickle по `data/features/ml_features/BTCUSDT_4h_features.parquet`
и всем бандлам `data/features/models/**/*.pkl`):

```
ml_features columns: 64 | rows: 5038
union of model feature_columns: 46
features required by models but ABSENT from ml_features: 1 ['symbol_encoded']

columns in ml_features NOT used by any model bundle: 19
['_f_time', '_m_time', 'close', 'close_time', 'date', 'datetime', 'exchange',
 'high', 'ignore', 'low', 'open', 'open_time', 'quote_volume', 'regime',
 'symbol', 'taker_buy_quote_volume', 'taker_buy_volume', 'trade_count', 'volume']
```

Из 19 неиспользуемых колонок большинство — служебные и сырые OHLCV, нужные для
разметки и джойнов, дефектом не являются. Выделяются:

- **`ignore`** — колонка из формата Binance klines CSV, буквально
  зарезервированная спецификацией как неиспользуемая; протащена через
  конвертацию в Parquet и дожила до `ml_features`. Мёртвые данные.
- **`_f_time`, `_m_time`** — служебные метки времени джойна деривативов;
  проверка их семантики отнесена к проходу 6 (контракт `open_time`/`ts_event`).
- **`regime`** — не входит ни в один список фич (модели раздельные по режимам),
  но читается маршрутизацией `_select_model`. Не мёртвая.

Число строк `5038` совпадает с входным фактом «`ml_features` 5038 строк ×
3 символа».

### 4.3 Таблицы SQLite

```
=== data/metrics.db ===
  metrics: 0 rows
  signals: 0 rows
  sqlite_sequence: 0 rows
```

### A2-014 `MetricsCollector` и `TelegramReporter`: писателя нет, таблицы пусты

**Севирити:** LOW
**Тип:** мёртвый код
**Где:** `src/monitoring/metrics_collector.py`, `src/monitoring/telegram_reporter.py`,
`data/metrics.db`

**Что в коде.** По §3.2 `01_inventory.md` весь публичный API обоих модулей
вызывается только из тестов: `MetricsCollector.record_trade` (`:143`),
`save_to_db` (`:153`), `get_daily_report` (`:224`), `get_weekly_report` (`:247`),
класс `TradingMetrics` (`:36`); `TelegramReporter.send_trade_alert` (`:42`),
`send_daily_report` (`:91`), `send_weekly_report` (`:99`) — последние три без
единой ссылки вообще (§1.5 выше).

**Замер, подтверждающий:** обе таблицы `data/metrics.db` содержат 0 строк при
том, что файл создан 2026-05-02 (`ls -la data/`: `16384 May  2 23:32 metrics.db`).

**В чём дефект:** заявленный контур наблюдаемости (по докстрингу
`metrics_collector.py` — «Collects trading performance metrics from
PortfolioTracker and stores them in a SQLite database for Grafana dashboards and
Telegram reports») не подключён. Grafana-дашборды и отчёты питаются из пустых
таблиц.

**Как проявляется:** отчётность о результатах отсутствует; при этом
`src/telegram_bot/broadcaster.py` имеет собственный `broadcast_daily_report`
(`:177`, тоже только из тестов) — вторая нереализованная реализация той же
задачи.

**Как установлено:** замером (SQLite-подсчёт строк) и §3.2 прохода 0.
**Уверенность:** доказано.

---

## 5. ОПАСНОСТЬ МЁРТВОГО КОДА ПРИ ОЖИВЛЕНИИ

Оценка для каждой подтверждённой мёртвой единицы: что произойдёт, если её
случайно подключить.

| Единица | Что будет при оживлении | Класс |
|---|---|---|
| `ml_strategy.py:2232-2262` (посев equity под `except: pass`) — оживает снятием `--dry-run` | **Тихо неверный результат.** Любой отказ внутри блока проглатывается; риск-решения продолжат идти по `initial_equity` без единого признака в логе. Ветка **никогда не исполнялась в проде** (§3.5), то есть её отказоустойчивость ни разу не проверялась на реальном балансе. | **опасен** |
| `ml_strategy.py:842` / `_open_position` / `_emit_signal` — оживает снятием `--dry-run` | **Тихо меняет поведение системы качественно:** одновременно включается и отправка ордеров, и запись в журнал сигналов. Оператор, снимающий флаг ради «пусть хотя бы сигналы пишутся», немедленно получает боевую отправку ордеров. | **опасен** |
| `src/execution/reconciler.py` / `PositionReconciler` | **Тихо неверный результат при неверной конфигурации.** `__init__` (`:96-98`): `self._base_url = self._URLS.get(trading_mode.lower(), self._URLS["testnet"])` — неизвестный режим молча даёт **testnet**-эндпоинт. При `auto_fix=True` (`:88`) реконсилятор закрывает «orphan»-позиции: сверка боевого состояния с testnet-ответом при auto_fix означает закрытие живых позиций по данным другого счёта. | **опасен** |
| `meta_strategy.py` / `MetaSignalGate` | **Безвредна.** Отсутствие `meta_model_v3.pkl` перехватывается фейл-софтом (`:239-243`) с `self.log.error(...)` и деградацией до базовой стратегии. Деградация громкая. | безвредна |
| `timeframes.py::db_path_for` | **Безвредна.** Возвращает `None` на неизвестном ТФ; карта совпадает с четырьмя другими копиями на сегодняшний день. Риск — в будущем расхождении, не в оживлении. | безвредна |
| `paper_strategy.py` / `PaperTradingStrategy` | **Неизвестно — не проверено.** Класс перехватывает отправку ордеров и маршрутизирует через `PaperTrader`; его совместимость с текущим `_open_position` (изменившимся с момента написания) не измерялась. Отнесено в «НЕ ИССЛЕДОВАНО». | не установлено |
| `dataset_builder._TRAINING_EXCLUDE` (A2-009) | **Уже даёт тихо неверный результат** — не при оживлении, а сейчас: исключение объявлено, не действует, фича в модели есть. | **опасен, активен** |
| `Settings.bybit_api_key` / `bybit_api_secret` | **Безвредны.** Bybit в системе не используется; поля не читает никто. | безвредна |
| `rolling_correlation`, `download_metrics`, `download_agg_trades`, `last_tick_time`, `build_features_all_symbols`, `stop_polling` | **Безвредны при оживлении** в смысле «упадут или сработают явно»; ни одна не участвует в расчёте торгового вывода. | безвредна |
| `TelegramReporter.send_*` (A2-014) | **Безвредна.** Отправка в Telegram; при неверном токене — явная ошибка сети. | безвредна |
| `symbol_encoded`-копии (A2-008) | **Опасны по построению:** промах карты даёт `-1` вместо исключения. Это не «оживление», а постоянно действующий тихий дефолт. | **опасен, активен** |

**Общий вывод по §5.** Три единицы из перечисленных при оживлении дают тихо
неверный результат в торговом или риск-контуре: блок посева equity, снятие
`--dry-run` как единый рубильник двух разных решений, и `PositionReconciler`
с молчаливым фолбэком на testnet при `auto_fix`. Две (`_TRAINING_EXCLUDE`,
копии `symbol_encoded`) вредят уже сейчас, не дожидаясь оживления.

---

## РЕЕСТР НАХОДОК ПРОХОДА 1

| ID | Имя | Севирити | Тип |
|---|---|---|---|
| A2-001 | `PositionReconciler` мёртв, докстринг утверждает обратное | MEDIUM | мёртвый код / недоделка |
| A2-002 | Meta-подсистема недостижима, артефакт отсутствует | HIGH | мёртвый код / недоделка |
| A2-003 | Пять независимых определений путей к торговым БД | MEDIUM | архитектура / дубль |
| A2-004 | 15m-стратегия работает без circuit breaker | HIGH | логика |
| A2-005 | 15m `on_bar` не переоценивает открытые позиции | MEDIUM | логика |
| A2-006 | `--n-experiments` печатается, но в DSR не попадает | HIGH | математика / недоделка |
| A2-007 | Три валидатора — три конфигурации DSR; 252 против 365 | HIGH | математика |
| A2-008 | Шесть копий отображения «символ → код» | MEDIUM | дубль / архитектура |
| A2-009 | Исключение `symbol_encoded` не действует | MEDIUM | логика / мёртвый код |
| A2-010 | Настройки API читаются мимо `Settings` | LOW | архитектура / дубль |
| A2-011 | `high_vol_model_path` (1H) не читает никто | LOW | мёртвый код |
| A2-012 | Посев базы просадки под `except Exception: pass` | HIGH | логика |
| A2-013 | В `--dry-run` у журнала сигналов нет производителя | CRITICAL | архитектура / логика |
| A2-014 | `MetricsCollector` / `TelegramReporter` без писателя | LOW | мёртвый код |

**Сводка:** CRITICAL 1, HIGH 5, MEDIUM 5, LOW 3. Всего 14.

---

## НЕ ИССЛЕДОВАНО

1. **§3.2 задания — функции, всегда возвращающие константу или `None`
   независимо от аргументов.** Не выполнено: требует символьного анализа тел
   функций, не сводимого к AST-обходу за разумное время. Передаётся в проход 7.
2. **§3.3 задания — неиспользуемые *параметры функций*.** Выполнена только
   часть про поля конфигов (A2-010, A2-011). Параметры функций не проверялись.
3. **`paper_strategy.py`**: подтверждён как мёртвый, но совместимость с текущим
   `_open_position` не измерялась (см. §5).
4. **Прочие 121 гасящих `except`**: развёрнута одна находка (A2-012), десять
   помечены адресами для прохода 7, остальные ~110 не классифицированы.
5. **Флаги, кроме `dry_run`**: `use_native_save`, `feature_whitelist`,
   `meta_enabled`, `preload_enabled` не проверялись на достижимость.
   `feature_whitelist` упоминается в `lgbm_trainer.py:1360` — предмет прохода 4.
6. **`scripts/`-кандидаты из §3.4 `01_inventory.md`**: вердикты не выносились
   индивидуально — они мёртвы по критерию «нет упоминаний», но их внутренняя
   корректность (важная, если их запускают вручную) не проверялась.
7. **`data/features/exchange=BINANCE_UM/…` и `data/features/symbol=…/interval=1h`**:
   8763 файла `part-0.parquet` и 1128 `klines.parquet` не сопоставлены с
   читателями — только каталоги `models/` и `ml_features/`.
8. **`_BAR_HOURS` в `scripts/run_reconciler.py:31-32`**: поведение при
   отсутствующем имени файла в карте не проверено (проход 8).
9. **Локальное дерево против VM**: все утверждения об артефактах в `data/`
   относятся только к локальной машине. `data/` в `.gitignore`; на VM протокол
   ходить запрещает.
