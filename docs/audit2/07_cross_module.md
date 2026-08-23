# Аудит-2, проход 6 — МЕЖМОДУЛЬНЫЕ КОНТРАКТЫ

Файл: `docs/audit2/07_cross_module.md`
Дата прохода: 2026-08-23
HEAD на момент прохода: `f4af5fd210b32db8af4478f0e2f440e2eb504ccc`
(`chore: ignore local snapshot directory`, 2026-08-22 16:18:08 +0530)
Рабочее дерево: `?? docs/audit2/` — единственное изменение, правок в `src/` нет.

Предмет: контракты §4 файла `01_inventory.md`. Метод для каждого —
**писатель → значение и единица → все читатели → допущение каждого**.
Расхождение допущений = находка; совпадение записано одной строкой,
чтобы следующий проход не перепроверял.

> Замечание о нумерации файлов. `00_method.md` §4 предполагал имена
> `07_contracts.md` / `08_execution_risk.md`; фактически на диске
> `06_execution_risk.md` (проход 5) уже занят. Этот файл — проход 6 по
> предмету, `07_cross_module.md` по имени. Расхождение фиксируется, не
> исправляется (протокол: запись только в `docs/audit2/`).

## Как получены числа

| Инструмент | Команда / путь |
|---|---|
| Интерпретатор | `/home/asus/Desktop/AtomiCortex/.venv/bin/python` (системный `python3` не имеет `loguru`) |
| Сырые бары | `data/features/exchange=BINANCE_UM/symbol=BTCUSDT/klines_4h/**/*.parquet` (963 файла, 5778 строк) |
| Funding | `.../funding_rate/**/*.parquet` (32 файла, 2889 строк) |
| Метрики (OI/LS) | `.../metrics/**/*.parquet` (963 файла, 277211 строк) |
| Обучающая матрица | `data/features/ml_features/BTCUSDT_4h_features.parquet` (5038 × 64) |
| Прод-бандл | `data/features/models/v3/trend_model_v3.pkl` (46 фич) |
| Снимок БД сигналов | `docs/audit/atomicortex_20260702.db.bak` → копия в `/tmp/audit2_p6/snap.db` |
| Артефакты замеров | `/tmp/audit2_p6/{parity.py,combined.py,funding_stale.py,regime_parity.py}` + `out_*.txt` |

`pytest` не запускался. Бот, watchdog, обучение не запускались. На VM не ходили.

---

## 1. КОНТРАКТЫ ИЗ РЕЕСТРА

### 1.1 `open_time` / `ts_event` / `close_time` — три конвенции времени

**Что установлено замером.**

```
$ .venv/bin/python -c "...read_parquet('.../BTCUSDT_4h_features.parquet')..."
┌───────────────┬───────────────┬─────────────────────┬────────────┐
│ open_time     ┆ close_time    ┆ datetime            ┆ date       │
│ i64           ┆ i64           ┆ datetime[μs]        ┆ date       │
╞═══════════════╪═══════════════╪═════════════════════╪════════════╡
│ 1714723200000 ┆ 1714737599999 ┆ 2024-05-03 08:00:00 ┆ 2024-05-03 │
│ 1714737600000 ┆ 1714751999999 ┆ 2024-05-03 12:00:00 ┆ 2024-05-03 │
└───────────────┴───────────────┴─────────────────────┴────────────┘
```

`close_time = open_time + 4h − 1 ms`. `open_time` кратен 4 ч в абсолютном
UTC. `datetime` — tz-naive UTC.

**Таблица контракта.**

| Место | Величина | Единица | Смысл |
|---|---|---|---|
| `parquet_converter.py:33,102` | `open_time` | ms, Int64 | открытие бара; ключ сортировки и партиционирования |
| `data_store.get_klines` (`data_store.py:107`) | `open_time` | ms | фильтр `open_time >= start_ms AND <= end_ms` |
| `feature_pipeline.build` → `add_funding_features` (`derivatives.py:117`) | `open_time` | ms | левый ключ `join_asof(..., strategy="backward")` |
| `add_oi_features` (`derivatives.py:181`) | `open_time` | ms | тот же |
| Nautilus `Bar.ts_event` | ns | **закрытие** бара | конвенция Nautilus |
| `live_feature_state.bar_open_time_ms` (`live_feature_state.py:39`) | ms | приводит close → open по сетке | `((ts+1)//d − 1)*d` |
| `ml_strategy._preload_from_parquet` | `ts_event = close_time * 1e6` | ns | берёт **`close_time`** из parquet, не `open_time` |
| `ml_strategy._preload_from_binance_api` | `ts_event` = kline[6] | ns | close_time от Binance |
| `ml_strategy_15m` (`:298`) | `bar_open_time_ms(ts_event//1e6, interval)` | ms | тот же преобразователь |
| `signal_bridge.log_signal` | `created_at` | ISO-8601 с `T` и `+00:00` | **момент записи**, не время бара |
| `heartbeat._heartbeat_loop` | `last_bar_ts` | секунды float (`bar.ts_event/1e9`) | закрытие бара |

**Вердикт: контракт держится.** Это доказано косвенно, но сильно:
в замере паритета (§3.2) признаки `returns_1/3/6/12/24`, `gap`,
`body_ratio`, `upper_wick`, `lower_wick`, `volume_sma_20`, `volume_ratio`
совпали **побитово** (`max_rel = 0`) между обучающей матрицей и
live-путём. Эти признаки — чистые позиционные сдвиги по буферу; любой
сдвиг сетки на один бар сделал бы их различными. Значит `open_time`,
восстановленный `bar_open_time_ms` из `ts_event`, попадает ровно в тот
же слот, что и `open_time` в parquet.

Два места, где время бара **не** сводится к общей сетке, и это осознанно:

* `signals_log.created_at` — это wall-clock записи, а не `open_time`
  бара. Проверено на снимке БД: `2026-06-02T20:00:00.673006+00:00` —
  20:00:00.673 против границы 4H-бара 20:00:00.000, т.е. запись через
  0.67 с после закрытия. `_publish_signal_ts` (`ml_strategy.py:1056`)
  явно документирует, что публикуется `time.time()`, а не
  `signal.timestamp`, «so the external checker compares like with like».
  Контракт с `check_signal_freshness.py` держится.
* `agg_15m.aggregate_15m_to_4h` (`agg_15m.py:130-133`) бакетирует по
  `floor(open_time_15m, 4h)` и джойнит на `open_time` 4H-кадра. Ключ —
  открытие в обоих кадрах. Согласовано.

**Отставание derivatives относительно точки решения.** `join_asof` идёт по
`open_time`, т.е. `funding_rate` и `oi_value` в строке бара — это
последнее значение **на момент открытия** бара, тогда как решение
принимается на его закрытии, через 4 часа. Это отсутствие заглядывания
вперёд, а не дефект; и **offline, и live** используют один и тот же
`join_asof` по `open_time`, поэтому смещение одинаково с обеих сторон.
Записано как «держится» (§6).

### 1.2 `confidence` и порог

**Определение у писателя.** `LGBMTrainer.get_signal`
(`lgbm_trainer.py:1263-1302`):

```
        p_up = float(model.predict(features)[0])
        direction = 1 if p_up > 0.5 else -1
        confidence = p_up if direction == 1 else 1.0 - p_up
```

то есть `confidence = max(p, 1−p) ∈ [0.5, 1.0]`.

**Тренер на обучении считает то же самое** (`lgbm_trainer.py:1116-1117`):

```
        max_proba = np.maximum(proba_up, 1.0 - proba_up)
        signal_mask = max_proba >= confidence_threshold
```

и в разрезе по символам (`lgbm_trainer.py:1514-1516`) — идентично.
**Определение совпадает: обучающая метрика `signal_rate` и live-гейт
меряют одну и ту же величину.**

**Где заданы пороги.**

| Место | Значение | Что ограничивает | Читает ли его прод-путь |
|---|---|---|---|
| `src/config.py:90` `CONFIDENCE_THRESHOLD` | 0.65 (и в `.env`, и в `.env.example`) | попадает в `Settings.confidence_threshold` | **да** — единственный из семи торговых параметров `.env`, который доходит (см. §5) |
| `live_trader.py:73` | `None` | сентинел, чтобы взять значение из `Settings` | да |
| `ml_strategy.py:104` `MLStrategyConfig.confidence_threshold` | 0.55 | дефолт dataclass; в проде перекрыт 0.65 | перекрывается |
| `ml_strategy.py:1755-1766` `_select_model` | `base` либо `max(base, 0.60)` | режимы `range` и неизвестный | да |
| `risk_engine.py:46` `RiskConfig.confidence_threshold` | 0.55 | второй гейт, `_check_confidence` (`:308`) | в проде получает 0.65 из `on_start` (`ml_strategy.py:276`) |
| `configs/strategy_1h.py:31` | 0.55 (комментарий: «was 0.63») | 1H-бот | 1H-бот не в проде |
| `configs/strategy_15m.py:41` | 0.58 (комментарий: «was 0.67») | 15m-бот | да, для 15m |
| `ml_validator.py:112` | 0.35 | валидатор | нет |
| `scripts/train_1h_models.py:107,189`, `train_15m_models.py:107,189` | 0.35 | «eval threshold (3-class baseline ~0.33)» | нет |
| `manifest` прод-бандла | `confidence_threshold: 0.55` | записано в артефакт | **никто не читает** — `_load_models` берёт только `booster` и `feature_columns` (`ml_strategy.py:1724-1725`) |

**Порог 0.35 в тренерах 1H/15m — реликт 3-классовой постановки.** В
докстринге прямо: «3-class baseline ~0.33». Модели давно бинарные
(`get_signal` возвращает `max(p,1−p) ≥ 0.5` по построению), поэтому
порог 0.35 в бинарной шкале **всегда истинен** — `signal_rate` в отчётах
`train_1h_models` / `train_15m_models` тождественно 1.0, а `avg_confidence`
считается по всей выборке. Это не порог, а отсутствие порога.

**Довожу до конца вопрос из прохода 2 (A2-022).** Две прод-модели —
`trend_model_v3.pkl` и `high_vol_model_v3.pkl` — записаны в один и тот же
момент (manifest `created_at` 2026-08-14T17:09:29 и 17:09:41), обе с
`n_features: 46`, `feature_columns_hash:
3429c552be803bcd5c52ef747c01e6cf1dbe65285d55257c71586b1c6db9b036` — хеш
**один и тот же**, т.е. пространство признаков идентично. Различаются
только обучающие подвыборки (`_filter_by_regime`, `lgbm_trainer.py:1310+`:
`trend` = `trend_up|trend_down`, `high_vol` = `high_vol`).

Сопоставимы ли их выходы под одним порогом 0.65? Из manifest
`trend_model_v3`: `signal_rate: 0.629`, `avg_confidence: 0.6132` при
пороге, на котором эти метрики считались, — **0.55**. Порог, с которым
живёт прод, — 0.65. То есть заявленный `signal_rate` относится к другому
порогу, чем работающий гейт, и переносить его на прод нельзя. Ни один
артефакт не содержит `signal_rate` при 0.65, и калибровки (Platt/isotonic)
нет ни у одной модели — `_load_models` грузит сырой `Booster`. Две модели
делят порог, не будучи калиброванными к общей шкале; A2-022 не закрыта.

**Наблюдение на реальных данных.** В снимке БД 8 сигналов, `confidence`
от 0.6556 до 0.6714 — все выше 0.65. Значит гейт когда-то пропускал.
Текущее «live confidence 0.50–0.54» (вход §3 `00_method.md`) согласуется с
§3.2: вектор, который получает модель сегодня, отличается от обучающего
по 14 признакам из 46, и распределение выхода смещено.

### 1.3 `atr`, `atr_pct`, `atr_percentile` — полная таблица

Три величины, два имени. Разбор по местам:

| # | Величина | Формула | Единица | Где вычисляется |
|---|---|---|---|---|
| A | `atr` (абсолютный) | Wilder ATR(14) по H/L/C | **цена** (USDT) | `regime_detector.detect_all:392-400`, `calculate_atr_percentile:138` |
| B | `atr_pct` | `A / close` | **доля** (0.0143 ≈ 1.43 %) | `regime_detector.detect_all:459`, `detect:329` |
| C | `atr_percentile` | `searchsorted(sort(ATR[t−540:t]), A) / len` | **доля ранга** [0, 1] | `detect_all:438-447`, `calculate_atr_percentile:169-170` |

Замер на последнем баре BTCUSDT: `atr_pct = 0.014307574`,
`atr_percentile = 0.87222222` — величины различаются на два порядка,
спутать их численно нельзя, только по имени.

**Кто как их читает.**

| Читатель | Читает | Ожидание | Верно? |
|---|---|---|---|
| `triple_barrier.py:53,62` | `atr_col="atr_pct"` | «ATR / price (dimensionless)», барьеры `entry × (1 ± k·atr_pct)` | ✅ |
| `risk_engine.py:75-76` | `TradeSignal.atr` (`$`), `TradeSignal.atr_pct` (доля) | два отдельных поля с явными комментариями `# current ATR in $` / `# ATR / price` | ✅ |
| `risk_engine.py:246` | `stop_distance = signal.atr * atr_stop_multiplier` | доллары | ✅ |
| `risk_engine.py:354` | `expected_return_bps = signal.atr_pct * 10_000` | доля → bps | ✅ |
| `ml_strategy.py:804` | `atr_dollar = regime_state.atr_pct * current_price` | доля × цена → доллары | ✅ |
| `signals_log.atr` | `atr_dollar` | доллары; в снимке `1032.6489849` при цене BTC ≈ 67 000 → 1.54 % | ✅ |
| `mtf_context.py:324-325` | `atr_pct` → `htf_4h_atr_pct` | доля | ✅ |
| `orb_features` | `orb_range_*_atr_pct` | доля | ✅ (отдельная величина: диапазон ORB / ATR) |
| **`ml_strategy.py:761`** | `regime_state.atr_percentile` | **печатается под именем `atr_pct`** | ❌ |
| `ml_strategy.py:1704` | `state.atr_pct` под именем `ATR%` | доля | ✅ |
| `ml_strategy.py:1503` (мёртвый `_compute_features`) | `atr_pct = (high[-1]−low[-1])/close[-1]` | **не ATR(14), а диапазон одного бара** | ❌ (в мёртвой ветке) |

Единственная живая путаница — строка 761: имя одного контракта носит
значение другого. Это тот пример, ради которого затевался проход; см.
находку A2-067. Числовых последствий нет (только журнал), но именно эта
строка — то, по чему оператор судит о волатильности.

Замечание о ветке `_compute_features` (`ml_strategy.py:1369-1516`):
она помечена `.. deprecated::` и не вызывается из `on_bar` (вызывается
`_compute_features_unified`, `ml_strategy.py:786`→`:768`). Её версия
`atr_pct` — истинный диапазон одного бара, а не ATR(14); её `regime_confidence`
= `min(adx/50,1)` вместо `0.4·|H−0.5|·2 + 0.6·min(adx/50,1)`. Ветка мёртвая,
но она — вторая, несовместимая реализация того же контракта в том же файле.
Проход 1 (A2-014 и соседи) уже фиксировал мёртвый код; здесь важно другое:
если её когда-нибудь оживят, семь признаков поменяют определение молча.

### 1.4 `regime` — строка, не enum, на границе модулей

**Писатель.** `RegimeDetector.detect_all` (`regime_detector.py:466`)
кладёт `pl.Series("regime", regimes, dtype=pl.Utf8)`. Значения — из
`MarketRegime` (`regime_detector.py:177+`), строчными:
`trend_up`, `trend_down`, `range`, `high_vol`.

`_classify` (`regime_detector.py:545`) в докстринге: «Deterministic regime
classification — **never returns UNKNOWN**». Проверено замером по всем
трём символам:

```
BTCUSDT {'trend_up': 1531, 'trend_down': 1323, 'range': 1237, 'high_vol': 947}
ETHUSDT {'range': 1578, 'trend_up': 1262, 'trend_down': 1374, 'high_vol': 824}
SOLUSDT {'trend_up': 1353, 'trend_down': 1433, 'range': 1443, 'high_vol': 809}
TOTAL {'trend_up': 4146, 'trend_down': 4130, 'range': 4258, 'high_vol': 2580} rows 15114
```

Метки `unknown` в матрице нет ни разу. Множество из четырёх значений
устойчиво.

**Сравнения со строковым литералом.**

| Место | Сравнение | Регистр важен | Замечание |
|---|---|---|---|
| `ml_strategy.py:1756` | `regime_label in ("trend_up","trend_down")` | да | совпадает с писателем |
| `ml_strategy.py:1759` | `== "range"` | да | совпадает |
| `ml_strategy.py:1761` | `== "high_vol"` | да | совпадает |
| `ml_strategy.py:1763-1766` | всё остальное → trend-модель, порог `max(base,0.60)` + WARNING | — | недостижимо при текущем `_classify` |
| `lgbm_trainer._filter_by_regime:1310+` | `is_in(["trend_up","trend_down"])` / `== "range"` / `== "high_vol"` | да | совпадает |
| `_load_models` (`ml_strategy.py:1720`) | `for regime in ["trend","high_vol"]` | — | **имена моделей**, не метки детектора |

**Расхождение множеств.** Детектор выдаёт 4 метки; моделей — 2
(`trend`, `high_vol`). Отображение `range → trend-модель` объявлено в
докстринге `_select_model` и реализовано. Но `_filter_by_regime("trend")`
отбирает **только** `trend_up|trend_down`: trend-модель **никогда не
видела** ни одной строки с `regime == "range"`. По замеру выше это
4258 из 15114 строк = **28.2 %** баров. В live эти бары идут в модель,
для которой они вне обучающего распределения, — с порогом, поднятым до
0.60, но с той же самой моделью. Находка A2-065.

**Куда деградирует «неизвестный» режим.** `RegimeDetector._unknown()`
(`regime_detector.py:594-606`) возвращает **не** `UNKNOWN`, а
`MarketRegime.RANGE` с `hurst=0.5, adx=0.0, atr_pct=0.0,
atr_percentile=0.5, trend_strength=0.0, confidence=0.0`. Вызывается при
пустом или некорректном `df` в `detect()`. Для потребителя
(`_select_model`) результат неотличим от настоящего `range` — только по
`adx=0.0`, чего никто не проверяет. Это молчаливая деградация; см. §2.

**Живучесть `e67a451`.** Коммит HEAD-9 «trim the offline warmup head by
the detector's own window». Проверено: `feature_pipeline.build` шаг 11
(`feature_pipeline.py:296-318`) режет `required_history_bars(detector.atr_lookback)`
= 200 + 540 = 740 строк. Первая строка `ml_features` BTCUSDT —
`2024-05-03 08:00`, сырые данные начинаются `1704067200000` =
2024-01-01 00:00; 740 баров × 4 ч = 123.3 суток; 2024-01-01 + 123.3 сут =
2024-05-03. Сходится. Строк 5778 − 740 = 5038 = ровно длина `ml_features`.
**Фикс жив.**

### 1.5 `equity` / `peak_equity` / `initial_equity` / `day_start_equity`

| Значение | Писатель | Смысл | В `--dry-run` |
|---|---|---|---|
| `equity` | `PortfolioTracker._get_equity` = `_cash + Σ unrealized_pnl` | текущий капитал | **константа 10 000**: позиции не открываются (`ml_strategy.py:841-842`: `if not self._config.dry_run: self._open_position(...)`), `_cash` не меняется |
| `peak_equity` | `__init__` ← `initial_equity`; поднимается в `update_price:172`, `sync_equity:201`, `close_position:301`; сеется `seed_from_authoritative_equity:247` | знаменатель просадки; **только растёт** | 10 000 навсегда |
| `initial_equity` | `__init__`; переписывается `seed_from_authoritative_equity:249` | знаменатель недельного % | 10 000 |
| `day_start_equity` | `__init__`; `_roll_periods:391`; `seed_...:248` | знаменатель дневного % | 10 000 |

**Кто первым устанавливает `peak_equity`** (вопрос из реестра). Порядок
строго определён и, в отличие от того, что было до `b2cbd40`, теперь
однозначен:

1. `PortfolioTracker.__init__` ставит `_peak_equity = initial_equity`
   (конфигурационные 10 000).
2. `_restore_from_store` (`portfolio_tracker.py:462-464`) — если в
   `risk_state_4h.json` есть ключ `peak_equity`, он побеждает и ставится
   флаг `_peak_restored = True`.
3. `_record_equity` → `seed_from_authoritative_equity` (`ml_strategy.py:2246`)
   — первое авторитетное чтение баланса биржи; **отказывается**, если
   `_peak_restored or _peak_seeded` (`portfolio_tracker.py:236-237`).

Приоритет: файл состояния > биржа > конфиг. Комментарий в коде это и
объясняет: «A peak restored from the state file is history the exchange
balance cannot reconstruct». **Контракт держится.**

**В `--dry-run` весь риск-контур инертен.** `_record_equity`
(`ml_strategy.py:2196-2216`) при `dry_run` не читает биржу вовсе, а
`_open_position` не вызывается — значит `_positions` всегда пуст,
`equity ≡ 10 000`, `drawdown ≡ 0`, `daily_pnl_pct ≡ 0`. Circuit breaker,
`_check_daily_loss`, `_check_max_drawdown` в прод-конфигурации
(`--mode paper --dry-run`) **не могут сработать никогда**. Это не дефект
сам по себе (нечему просаживаться), но означает: наблюдение «бот работает,
breaker не срабатывает» не является свидетельством того, что breaker
работает. Прямое следствие A2-013.

### 1.6 `funding_rate` — знак, единица, частота

**Единица и частота — замер:**

```
funding settlement hours (UTC) -> {0: 963, 8: 963, 16: 963}
minutes -> {0: 2889}
n = 2889
```

Ставка за 8 часов (не годовая), расчёты ровно в 00:00 / 08:00 / 16:00 UTC,
963 дня × 3 = 2889 записей. Величина порядка `9.4e-05` = 0.0094 % за 8 ч.

**Знак.** Конвенция Binance: `funding_rate > 0` → лонги платят шортам.

| Читатель | Трактовка | Совпадает? |
|---|---|---|
| `derivatives.add_funding_features:117-140` | сырое значение → `funding_rate`, `funding_abs`, `funding_positive = (rate > 0)`, `funding_cum_24h = rolling_sum(6)` | признаки, знак сохранён |
| `add_basis_features:221-233` | `basis_approx = funding_cum_24h`, `basis_extreme = |basis| > 0.001` | производная от суммы за 24 ч (3 расчёта); знак сохранён |
| `cost_model.calculate_funding_cost:78-92` | `gross = position_size * funding_rate * num_payments`; докстринг: «Positive → net cost for the caller (long pays when funding > 0)» | ✅ совпадает |
| `risk_engine._check_funding_rate:367-383` | блок при `abs(rate) > 0.001` **или** `rate is None` | ✅ по модулю, знак не важен |
| `ml_strategy._get_funding_rate:1267-1289` | берёт `funding_rate` **из вектора признаков**, `None` = «нет чтения» → жёсткий блок | ✅ |

**Трактовка знака и единицы совпадает во всех трёх местах** (признаки,
риск-фильтр, P&L). Контракт держится.

**Но частота в live сломана.** `ml_strategy.on_data:626`:

```
                if dt.hour in (1, 9, 17) and dt.minute == 0:
```

Замер выше даёт часы расчётов `{0, 8, 16}`. Условие **никогда не
истинно**, поэтому `funding_rate_history` в live не пополняется ни разу
после старта. Последствия измерены — см. §2.2 и находку A2-060.

`cost_model.calculate_funding_cost` имеет параметр с именем
`position_size`, но единственный вызывающий (`cost_model.py:114-116`)
передаёт `position_size=notional`, а докстринг говорит «Return funding
cost in USDT». То есть параметр называется как контракт §1.9
(контракты базового актива), а означает нотионал. Компенсируется на
стороне вызова; см. A2-073.

### 1.7 `oi_value` и `oi_*` — конвенция после D-9

**Единица.** Замер последнего бара BTCUSDT: `oi_value = 7.8375884e+09` —
это USD (`sum_open_interest_value` из parquet-метрик), а не контракты.
`add_oi_features` (`derivatives.py:164-170`) явно выбирает
`sum_open_interest_value → oi_value`; колонка `sum_open_interest`
(в контрактах) в конвейер не попадает вовсе.

**Конвенция сдвига.** `join_asof(left_on="open_time",
right_on="_m_time", strategy="backward")` — последний сэмпл OI на
момент **открытия** бара, без заглядывания вперёд. `oi_delta_4h =
(oi[t] − oi[t−1]) / oi[t−1]`, `oi_delta_12h` — сдвиг на 3 бара,
`oi_zscore` — окно 180 баров (30 сут), `oi_quadrant` — знак изменения
цены × знак `oi_delta_4h`.

**Ожидают ли читатели текущую конвенцию.** Замером (§3.2, вариант L1 —
идеальные входы): `oi_value`, `oi_delta_4h`, `oi_delta_12h`,
`oi_quadrant` совпали с обучающей матрицей **побитово** (`max_rel = 0`),
`oi_zscore` — `1.85e-12` (машинная точность). Значит live-путь ожидает
ровно ту конвенцию, которая в матрице. **Разметка после D-9 согласована.**

Расхождение возникает не от конвенции, а от того, **какие данные live
подаёт на вход** — см. §2.1 и A2-059.

### 1.8 `signal_id` — 0 как «не записано»

**Писатель.** `signal_bridge.log_signal:229` — `signal_id = cursor.lastrowid or 0`;
в `except` (`:238-240`) возвращает `0`. SQLite `AUTOINCREMENT` начинается
с 1, поэтому 0 — валидный сентинел.

**Знают ли читатели.**

| Читатель | Знает про 0 | Как |
|---|---|---|
| `ml_strategy._emit_signal:1035-1041` | **да** | явная ветка: ERROR в журнал, `_pending_signal_ids` не заполняется, `_publish_signal_ts` не вызывается |
| `signal_poller._last_signal_ids` (`:73, :203, :303`) | **да, но иначе** | 0 здесь — это «high-water mark ещё не установлен», начальное значение словаря; `WHERE id > 0` корректно отбирает всё |
| `keyboards.build_signal_keyboard:307-320` | **нет** | кладёт `signal_id` в `callback_data` без проверки; но получает id только из строки БД, где он ≥ 1 |
| `telegram_bot/database.add_signal:375` | своя копия `lastrowid or 0` | параллельный писатель в ту же таблицу (см. §4.1) |
| `signal_bridge.close_signal:243`, `mark_rejected:273` | UPDATE по id | вызываются только из `_pending_signal_ids`, куда 0 не попадает |

**Контракт держится.** Единственное место, где 0 мог утечь дальше
(`_pending_signal_ids`), закрыто явной веткой с ERROR.

Уникальность между изолированными БД (`data/atomicortex.db` 4H и
`data/atomicortex_15m.db` 15m) **не обеспечивается**: обе последовательности
начинаются с 1. Это учтено: `signal_poller` держит
`_last_signal_ids: dict[str, int]` **по пути БД** (`:73`), а
`keyboards.py:318` кладёт в callback `signal_detail:{timeframe}:{signal_id}`,
т.е. пара (timeframe, id) уникальна. Контракт держится за счёт
составного ключа.

### 1.9 `position_size` / `quantity` / `notional`

**Писатель.** `risk_engine.calculate_position_size:233-262`:

```
        dollar_risk = equity * self._config.risk_per_trade
        stop_distance = signal.atr * self._config.atr_stop_multiplier
        contracts = dollar_risk / stop_distance
        notional = contracts * signal.entry_price
        leverage = notional / equity if equity > 0 else 0.0
```

`position_size` = **контракты базового актива** (BTC), `notional` = USDT,
`leverage` = безразмерное. `signal.atr` обязан быть в долларах — так и есть
(`ml_strategy.py:804`).

**Проверка на реальных данных.** Снимок БД, строка id=8:
`position_size = 0.06455888461762499`, `notional = 4343.954301600747`,
`atr = 1032.6489849`. `notional / position_size = 67 289` — цена BTC.
✅ контракты, не доля капитала и не USDT.

| Читатель | Ожидание | Верно? |
|---|---|---|
| `signal_bridge.log_signal(position_size=, notional=)` | оба отдельными колонками | ✅ |
| `ml_strategy._open_position` → Nautilus `quantity` | контракты | ✅ (Nautilus ожидает базовый актив) |
| `cost_model.calculate_fee/slippage(notional=)` | USDT | ✅ |
| `cost_model.calculate_funding_cost(position_size=)` | **нотионал**, имя вводит в заблуждение | ⚠ компенсируется вызовом `position_size=notional` (`cost_model.py:115`) |
| `telegram_bot`, `broadcaster` | отображение | ✅ |

**Округления.** В коде риска округления нет вовсе — `contracts` уходит
как float с полной точностью (`0.06455888461762499` в БД это
подтверждает). Округление к шагу инструмента делает Nautilus при
создании `Quantity`. Значит **число в `signals_log.position_size`
и число в ордере могут отличаться** на величину лота; для BTCUSDT-PERP
шаг 0.001, т.е. 0.0645588… → 0.064, расхождение до 1.5 %. В `--dry-run`
ордера нет, расхождение ненаблюдаемо. Записано как открытый вопрос
(§8), не как находка: подтвердить можно только на живом ордере.

### 1.10 `symbol_encoded` — что получает модель в live

Из прохода 1: A2-008 (шесть копий отображения), A2-009 (исключение не
действует). Здесь — ответ на вопрос «что получает модель».

**Признака нет в `ml_features`.** Матрица — 64 колонки, `symbol_encoded`
среди них нет (замер: список колонок в §«Как получены числа»). Он
добавляется `dataset_builder` при сборке обучающего набора.

**В бандле он есть.** `trend_model_v3.pkl` → `feature_columns` содержит
`symbol_encoded` (46-й признак, алфавитно между `returns_6` и
`taker_buy_ratio`).

**В live.** `ml_strategy._compute_features_unified:1340-1344`:

```
            sym_str = str(self._instrument_id)
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            base = sym_str.split("-")[0] if "-" in sym_str else sym_str.split(".")[0]
            rd["symbol_encoded"] = float(sym_map.get(base, -1))
```

Прод-юнит `deploy/atomicortex-bot.service` передаёт `--symbols BTCUSDT-PERP`;
`live_trader.build_node:226-227` строит `instrument_id = "BTCUSDT-PERP.BINANCE"`.
`"-" in sym_str` → истина → `split("-")[0]` → `"BTCUSDT"` → `sym_map` → **0**.
`symbols_in_train` бандла — `['BTCUSDT','ETHUSDT','SOLUSDT']`, значит
кодировка 0 соответствует обучению. **В проде признак корректен.**

Дефолт `-1` достижим только при другом формате символа. Такой формат в
дереве есть: `src/config.py:149-152` объявляет
`SYMBOLS = "BTC-USDT-PERP,ETH-USDT-PERP,SOL-USDT-PERP"`, и то же значение
стоит в `.env` и `.env.example`. Для `"BTC-USDT-PERP.BINANCE"`
`split("-")[0]` даёт `"BTC"` → `sym_map.get("BTC", -1)` → **−1**, кода,
которого модель не видела ни разу. Спасает только то, что
`settings.symbols` читается ровно в одном месте — `scripts/verify_setup.py:172`
(проверка «символы распарсились»), — и никогда не доходит до
`LiveTraderConfig.symbols`, который приходит из CLI. Находка A2-069:
заряженное ружьё, сегодня не выстрелившее.

---

## 2. МОЛЧАЛИВЫЕ ДЕГРАДАЦИИ

Критерий: модуль при ошибке/отсутствии данных подставляет значение,
которое потребитель не может отличить от измерения.

### 2.1 `get_metrics_df` отдаёт 100 записей и две колонки из четырёх

`live_feature_state.get_metrics_df:255-270`:

```
    def get_metrics_df(self, n_bars: int = 100) -> pl.DataFrame:
        ...
        records = list(self.oi_history)[-n_bars:]
        return pl.DataFrame(records).rename({
            "timestamp": "create_time",
            "oi_value": "sum_open_interest_value",
        })
```

Два независимых обрезания:

**(а) Колонки.** `oi_history` хранит только `{timestamp, oi_value}`
(`update_oi:139-143`, `preload_oi:196`). `add_oi_features:164-172`
ищет `count_long_short_ratio` и `sum_taker_long_short_vol_ratio`; не
найдя, подставляет `pl.lit(0.0)`. Следовательно **в live
`ls_ratio ≡ 0.0`, `taker_vol_ratio ≡ 0.0`, и `ls_ratio_zscore ≡ 0.0`
всегда**. В обучении это реальные величины (`ls_ratio = 0.96265293`,
`taker_vol_ratio = 0.910171` на последнем баре). Три признака из 46 —
константный ноль, неотличимый от «рынок сбалансирован».

**(б) Длина.** Буфер баров — 740 (`HISTORY_BARS_4H`), а метрик отдаётся
100 записей. Источники в `oi_history` смешаны: preload берёт
`openInterestHist?period=4h&limit=500` (`ml_strategy.py:428-436`), таймер
доливает по одному сэмплу каждые 5 минут (`ml_strategy.py:371-376`).
После ~8.3 ч работы **последние 100 записей — сплошь 5-минутные поллы**,
покрывающие 8 часов. `join_asof` не находит записи для 738 баров из 740
→ `fill_null(0.0)` → `oi_value = 0` на всей истории кроме последних двух
баров. `oi_zscore` (окно 180 баров) считается по ряду из нулей и двух
реальных значений; `oi_delta_12h = (oi[t] − oi[t−3])/oi[t−3]` → `oi[t−3] = 0`
→ `safe_divide` → **0.0**.

Замерено (`/tmp/audit2_p6/out_L234.txt`):

```
===== L2 LIVE-AS-CODED at t0: metrics=last100 @4h preload, funding=last100, real tbv =====
oi_zscore                   1.903      1.107        4.5929158        1.1305567 X
ls_ratio                        1          1       0.96265293                0 X
ls_ratio_zscore                 1          1       -1.7535203                0 X
taker_vol_ratio                 1          1         0.910171                0 X

===== L3 LIVE after >8h uptime: metrics=last100 @5min polls, funding=last100 =====
oi_zscore                   1.159      0.787        4.5929158        7.6701589 X
ls_ratio                        1          1       0.96265293                0 X
ls_ratio_zscore                 1          1       -1.7535203                0 X
oi_delta_12h                    1          1      0.015059392                0 X
taker_vol_ratio                 1          1         0.910171                0 X
```

`oi_zscore` при обучающем значении 4.59 даёт 1.13 сразу после старта и
7.67 после 8 часов работы — то есть **величина зависит от времени
работы процесса**, а не от рынка. Для модели это выглядит как валидное
число: NaN не появляется, знак правдоподобен. Находка A2-059.

### 2.2 История funding не пополняется — 6 признаков деградируют со временем работы

Механизм разобран в §1.6: `on_data:626` проверяет часы `{1, 9, 17}`,
расчёты происходят в `{0, 8, 16}`. Следствие: `funding_rate_history`
после `preload_funding` (200 записей, `ml_strategy.py:394`) не растёт.
Чем дольше бот работает, тем сильнее `join_asof` промахивается.

Замер (`/tmp/audit2_p6/funding_stale.py`, последний бар BTCUSDT,
offline-значение — из обучающей матрицы):

```
--- bot uptime = 0 days ---
   funding_rate         offline=     9.422e-05   live=     9.422e-05   MATCH
   funding_zscore_7d    offline=     1.2541473   live=     1.2541473   MATCH
   funding_zscore_30d   offline=     1.4299232   live=     1.4299232   MATCH
   funding_cum_24h      offline=    0.00059077   live=    0.00059077   MATCH

--- bot uptime = 7 days ---
   funding_rate         offline=     9.422e-05   live=     7.841e-05   DIVERGES
   funding_zscore_7d    offline=     1.2541473   live=             0   DIVERGES
   funding_zscore_30d   offline=     1.4299232   live=    0.71242407   DIVERGES
   funding_cum_24h      offline=    0.00059077   live=    0.00047046   DIVERGES

--- bot uptime = 30 days ---
   funding_zscore_7d    offline=     1.2541473   live= 4.7706993e-08   DIVERGES
   funding_zscore_30d   offline=     1.4299232   live=             0   DIVERGES

--- bot uptime = 90 days ---
   funding_zscore_7d    offline=     1.2541473   live=             0   DIVERGES
   funding_zscore_30d   offline=     1.4299232   live=             0   DIVERGES
```

Затронуто 6 признаков: `funding_rate`, `funding_abs`,
`funding_zscore_7d`, `funding_zscore_30d`, `funding_cum_24h`,
`basis_approx`. Механизм схлопывания z-score: окно заполняется одной и
той же константой → `rolling_std = 0` → `safe_divide` → `0.0`.
`funding_extreme` и `funding_positive` остаются верными случайно (знак
ставки не менялся).

**Отличимо ли от валидных значений.** Нет. `funding_zscore_7d = 0.0` —
совершенно нормальное значение («ставка на своём среднем»); в обучающей
матрице оно встречается. Модель не может отличить «ставка ровно на
среднем» от «истории нет». Находка A2-060.

Побочно: даже если починить часы, ветка `on_data:626-632` добавляет в
историю **сырым `append`**, минуя `preload_funding`'s дедуп, а поток
`BinanceFuturesMarkPriceUpdate` идёт раз в секунду. Условие
`dt.minute == 0` истинно для всех 60 секунд минуты расчёта → до 60
дубликатов на один расчёт при `maxlen=300`. Пять расчётов — и буфер
состоит из дублей. Находка A2-070.

### 2.3 `taker_buy_volume` → `volume × 0.5`

Два уровня подстановки:

* `FeaturePipeline._ensure_taker_buy_volume:388-398` — `volume * 0.5`,
  если колонки нет;
* `LiveFeatureState.get_bar_df:373-388` — то же, если `add_bar` был
  вызван без реального значения, с одноразовым WARNING.

`cvd = 2·taker_buy_volume − volume` → при подстановке **тождественно 0**.
Замер (`out_L234.txt`, вариант L4):

```
===== L4 LIVE, tbv fallback volume*0.5 (no real taker_buy_volume) =====
cvd                             1          1         2455.778                0 X
cvd_rolling_24                  1          1        60789.726                0 X
cvd_rolling_96                  1          1        59702.391                0 X
cvd_slope_12                    1          1        203.64108                0 X
cvd_slope_3                     1          1       -3253.6437                0 X
cvd_slope_6                     1          1         -743.219                0 X
taker_buy_ratio            0.1252    0.05528       0.54610986              0.5 X
```

Семь признаков. Отличимо ли? `cvd = 0` — валидное значение
(сбалансированный бар); `taker_buy_ratio = 0.5` — тоже. Модель этого не
видит.

**Насколько это реально в проде.** Оба preload-пути кладут реальный
`taker_buy_volume` в `_preload_tbv` (parquet — `ml_strategy.py:2031-2042`;
REST — kline[9], `ml_strategy.py:2062-2065`), а `on_bar` тянет его REST-ом
на каждый бар (`_fetch_taker_buy_volume_for_bar`, `ml_strategy.py:663-665`).
Значит нормальный режим — реальные значения. Но `_fetch_...` возвращает
`None` при любом сбое сети, и тогда **один** бар получает `volume*0.5`,
что портит `cvd` этого бара и `cvd_rolling_24/96`, `cvd_slope_3/6/12` на
следующие 96 баров (16 суток). Одноразовый WARNING в `get_bar_df`
взводится только один раз за жизнь процесса, поэтому второй и
последующие сбои молчат.

Код честен насчёт этого: в `_preload_historical_bars:1897-1904` есть
WARNING при `real_tbv == 0`. Дефект не в отсутствии диагностики, а в том,
что деградация одного бара живёт 96 баров и невидима для модели.

### 2.4 15m HTF-признаки в live (A3)

`build_from_buffer` для `interval='15m'` требует `df_htf_4h` и
`df_htf_1h`; при `None` колонки `htf_4h_*` / `mtf_*` остаются на
дефолтах `build_mtf` (`mtf_context.py:341`: `pl.lit(0.0).alias("htf_4h_atr_pct")`
и соседние). Докстринг `build_from_buffer` это признаёт дословно:
«When ``None`` those columns stay at their ``build_mtf`` defaults
(train/serve skew — caller must supply)».

`ml_strategy_15m` документирует в шапке (`:37`): «with a ~1500-bar 15m
preload the htf_4h_* features start partially». 15m-бот не является
предметом этого прохода по существу (модель 4H — прод), поэтому здесь
только фиксируется: **механизм деградации тот же — константа вместо
данных, неотличимая для модели**, и он не закрыт.

### 2.5 Режим-константы при коротком буфере

`RegimeDetector.detect_all:422-424`: для `i < min_bars` строки получают
`regime="range"`, `hurst=0.5`, `atr_percentile=0.5`, `trend_strength=0`,
`regime_confidence=0` — и **только** `adx` реальный. Это те самые
«нейтральные константы».

`build_from_buffer` вызывает `detect_all(df, min_bars=detector.hurst_window)`
= 300 для 4H, `build()` — `detect_all(df)` с дефолтом `min_bars=300`.
**Совпадает.** Буфер live — 740 баров (`HISTORY_BARS_4H`, PR-A), т.е.
последняя строка имеет `i = 739 > 300` и полный 540-барный хвост для
`atr_percentile`. Порог `_warn_if_buffer_too_short` (`feature_pipeline.py:236-263`)
даёт WARNING при `< 740`. **В нормальном режиме деградации нет** — что и
подтверждает §3.2 (L1: `adx`, `atr_pct`, `atr_percentile` совпали
побитово).

Деградация остаётся достижимой на старте, если preload вернул меньше 740
баров: `_preload_from_binance_api` просит `min(n_bars, 1500)` = 740, но
принимает и меньше — порог годности всего 50 баров
(`ml_strategy.py:1823`: `if len(bars) >= 50`). При 300 ≤ N < 740 бот
торгует с `atr_percentile`, посчитанным по неполному окну, — с одним
WARNING в журнале.

### 2.6 `RegimeDetector._unknown()` → `range` с нулевой уверенностью

`regime_detector.py:594-606`. Возвращается при пустом `df` в `detect()`.
Потребитель — `ml_strategy._detect_regime:1690-1710`, который на исключение
возвращает `None` (бар пропускается — честно), но на **пустой** `df`
получает валидный `RegimeState(regime=RANGE, adx=0.0, confidence=0.0)`.
`_select_model("range")` → trend-модель с порогом 0.60. Признаки при этом
считает **другой** путь (`_compute_features_unified`), который на пустом
буфере вернёт `None` и бар пропустится. То есть до модели такой бар не
дойдёт. **Деградация есть, последствий нет** — потому что два независимых
пути к одному бару спасают друг друга. Записано, чтобы следующий проход
не искал.

---

## 3. ГРАНИЦА OFFLINE / LIVE

### 3.1 Признаки, где `build()` и `build_from_buffer()` расходятся операциями

Сопоставление тел (`feature_pipeline.py:233-343` против `:406-524`) для
`interval='4h'`:

| Шаг | `build()` | `build_from_buffer()` | Расхождение |
|---|---|---|---|
| taker_buy_volume | **не вызывает** `_ensure_taker_buy_volume` | вызывает | ✅ безразлично: parquet всегда несёт колонку |
| источник данных | `DataStore.get_klines/get_funding_rate/get_metrics` | кадры от вызывающего | ⚠ **вся разница в данных, а не в коде** |
| CVD / volume / price | `add_cvd_features → add_volume_features → add_price_features` | тот же порядок | нет |
| funding | `add_funding_features(df, funding_df, bar_min)` | то же | нет |
| OI | `add_oi_features(df, metrics_df, bar_min)` | то же | нет |
| basis | `add_basis_features` | то же | нет |
| детектор | `RegimeDetector()` | `RegimeDetector()` для `'4h'` | нет |
| `min_bars` | `detect_all(df)` → 300 | `detect_all(df, min_bars=detector.hurst_window)` → 300 | нет (совпадает численно) |
| обрезка головы | шаг 11: `slice(740)` | нет; возвращает `tail(1)` | ⚠ **влияет на фазу амортизации Hurst** |
| длина кадра | вся история (5778 баров) | ровно 740 баров | ⚠ **источник трёх расхождений** |

**Единственная операция, расходящаяся по существу — амортизация Hurst.**
`detect_all:427-430`:

```
            if (i - min_bars) % _RECOMPUTE_EVERY == 0:
                start = max(0, i + 1 - self.hurst_window)
                last_hurst = calculate_hurst_exponent(close[start : i + 1])
            hursts[i] = last_hurst
```

`_RECOMPUTE_EVERY = 6`. Фаза пересчёта зависит от `i` — индекса **внутри
переданного кадра**. Offline `i` считается от начала всей истории;
live — от начала 740-барного буфера. Их разность по модулю 6 в общем
случае не ноль, поэтому последняя строка получает Hurst, пересчитанный
на баре, отстоящем от live-варианта на 0…5 баров. Через
`trend_strength = 0.4·|H−0.5|·2 + 0.6·min(adx/50,1)` и
`regime_confidence = trend_strength` расхождение размножается на три
признака.

Прочие расхождения — не в коде, а в том, **чем кормят**
`build_from_buffer` (§2.1, §2.2, §2.3).

### 3.2 Замер паритета — ключевой замер прохода

**Постановка.** Окно: последние 20 строк `ml_features` BTCUSDT
(`open_time` от 1787011200000 до 1787256000000, т.е. 2026-08-17 20:00 …
2026-08-20 20:00 UTC). Эталон — сама обучающая матрица (то, на чём
модель училась). Live-путь воспроизведён покомпонентно: буфер баров с
теми же колонками, что кладёт `LiveFeatureState.add_bar`
(`open_time, open, high, low, close, volume, taker_buy_volume`), длиной
`HISTORY_BARS_4H = 740`; `funding_df` / `metrics_df` — в форме, которую
отдают `get_funding_df` / `get_metrics_df`; вызов
`FeaturePipeline(interval='4h').build_from_buffer(single_row=True)`.
Сравниваются 45 признаков бандла (46-й, `symbol_encoded`, конвейером не
производится и разобран в §1.10).

**Допуск объявлен заранее: относительная ошибка
`|a−b| / max(|a|,|b|,1e-12) > 1e-6` считается расхождением.** Порог
выбран так, чтобы двойная точность и разный порядок суммирования не
считались расхождением, а любое содержательное отличие — считалось.

**Четыре варианта входа.**

| Вариант | funding_df | metrics_df | taker_buy_volume |
|---|---|---|---|
| L1 «идеал» | вся история | вся история, все колонки | реальный |
| L2 «как в коде, t=0» | последние 100 | последние 100 сэмплов 4H-preload, 2 колонки | реальный |
| L3 «после 8 ч работы» | последние 100 | последние 100 5-минутных поллов, 2 колонки | реальный |
| L4 «+ сбой tbv» | как L3 | как L3 | `volume × 0.5` |

**Результат L1 — изолирует чистое расхождение алгоритмов:**

```
===== L1 IDEAL: full funding+metrics, real taker_buy_volume =====
feature                   max_rel   mean_rel         ref_last        live_last
regime_confidence         0.01642   0.005446       0.78600515       0.78361794 X
trend_strength            0.01642   0.005446       0.78600515       0.78361794 X
hurst                     0.01525   0.004995       0.73250643       0.72952242 X
oi_zscore               1.851e-12  7.067e-13        4.5929158        4.5929158 .
price_to_vwap           5.462e-13  1.238e-13      0.021564938      0.021564938 .
funding_zscore_30d       1.12e-13  4.064e-14        1.4299232        1.4299232 .
...
adx                             0          0        51.875293        51.875293 .
atr_pct                         0          0      0.014307574      0.014307574 .
atr_percentile                  0          0       0.87222222       0.87222222 .
...
-- features exceeding tol 1e-6: 3/45
```

**При идеальных входах расходятся ровно три признака — и все три от
одной причины (фаза амортизации Hurst, §3.1).** Это сильный
положительный результат: сам конвейер `build_from_buffer` воспроизводит
`build` с точностью до двойной. 42 признака из 45 — либо побитово, либо
на уровне 1e-12.

**Результат в рабочем режиме** (`/tmp/audit2_p6/out_combined.txt`;
uptime 7 суток — минимальный срок, на котором проявляется §2.2):

```
===== STEADY STATE: uptime 7d, real taker_buy_volume =====
   funding_zscore_30d     max_rel_err=1.831
   oi_zscore              max_rel_err=1.159
   funding_zscore_7d      max_rel_err=1
   oi_delta_12h           max_rel_err=1
   ls_ratio               max_rel_err=1
   ls_ratio_zscore        max_rel_err=1
   taker_vol_ratio        max_rel_err=1
   funding_abs            max_rel_err=0.8383
   funding_rate           max_rel_err=0.8383
   funding_cum_24h        max_rel_err=0.7121
   basis_approx           max_rel_err=0.7121
   regime_confidence      max_rel_err=0.01642
   trend_strength         max_rel_err=0.01642
   hurst                  max_rel_err=0.01525
   >>> features exceeding 1e-6: 14/45  (+symbol_encoded ok = 14/46)

===== STEADY STATE + taker_buy_volume outage =====
   [те же 14] + cvd, cvd_rolling_24, cvd_rolling_96,
               cvd_slope_3, cvd_slope_6, cvd_slope_12, taker_buy_ratio
   >>> features exceeding 1e-6: 21/45  (+symbol_encoded ok = 21/46)
```

**Сводка замера.**

| Режим | Расходится признаков из 46 | Доля |
|---|---|---|
| Идеальные входы (нижняя граница, только алгоритм) | 3 | 6.5 % |
| Первые часы после старта (L2) | 7 | 15.2 % |
| После ~8 ч работы (L3) | 8 | 17.4 % |
| **Рабочий режим, ≥ 7 суток аптайма** | **14** | **30.4 %** |
| То же + сбой `taker_buy_volume` | 21 | 45.7 % |

Из 14 расходящихся в рабочем режиме: 11 расходятся **порядково**
(`max_rel ≥ 0.7`, т.е. значение либо ноль вместо величины, либо
отличается кратно), 3 — на 1.5–1.6 % (`hurst` и производные).

### 3.3 Жив ли 35 %-расхождение regime-лейбла из A3

Замер: метка, которую даёт live-путь (`RegimeDetector.detect(df, idx=-1)`
на 540 барах — ровно то, что делает `ml_strategy._detect_regime:1695-1700`),
против колонки `regime` в `ml_features` (`detect_all` на полной матрице).
400 последних баров, все три символа.

```
### BTCUSDT: bars=400 regime mismatch=1 (0.2%)
    offline=trend_up    -> live=high_vol     1
    |atr_percentile offline - live| max=0.0389 mean=0.0221
    |hurst offline - live|          max=0.0260 mean=0.0059

### ETHUSDT: bars=400 regime mismatch=3 (0.8%)
    offline=trend_down  -> live=high_vol     3
    |atr_percentile offline - live| max=0.0408 mean=0.0223
    |hurst offline - live|          max=0.0253 mean=0.0052

### SOLUSDT: bars=400 regime mismatch=4 (1.0%)
    offline=trend_down  -> live=high_vol     2
    offline=range       -> live=high_vol     2
    |atr_percentile offline - live| max=0.0371 mean=0.0196
    |hurst offline - live|          max=0.0247 mean=0.0054
```

**A3 в прежнем виде мёртв: 0.2–1.0 % вместо 35 %.** PR-A (буфер 740) и
`e67a451` сделали своё дело.

Но замер вскрыл другое, чего A3 не описывал. Расхождения **не
симметричны**: во всех восьми случаях из 1200 live-метка — `high_vol`,
а offline — что-то другое. Причина видна в той же таблице:
`atr_percentile`, посчитанный live-путём, систематически выше на
0.020–0.022 (среднее), максимум 0.041, а порог `high_vol` —
`atr_percentile > atr_vol_threshold` (первое правило `_classify`). Смещение
одностороннее, поэтому и ошибки односторонние.

Источник смещения — **третья реализация того же `atr_percentile`**:

* `detect_all:438-447` (даёт признак модели) — окно
  `atr_arr[i+1−540 : i+1]`, фильтр `history > 0`, ATR посчитан по
  **всему** кадру;
* `calculate_atr_percentile:138-171` (даёт метку через `detect()`) —
  `valid = atr_series[~isnan]`, затем `valid[-540:]`; ATR посчитан по
  **540-барному срезу**, из которого первые 13 значений NaN, т.е.
  фактическое окно 527, и warm-up Wilder-сглаживания короче.

Итог: **в одном и том же баре метка режима и признак `atr_percentile`,
уходящий в модель, посчитаны разными кодовыми путями по разным окнам.**
Метка выбирает модель, признак идёт в неё же — и они не согласованы.
Находка A2-064.

---

## 4. ГРАНИЦА КОД / ХРАНИЛИЩЕ

### 4.1 `signals_log` — схема против писателей и шести потребителей

**Два независимых `CREATE TABLE` для одного имени.**

| Колонка | `signal_bridge.py:116-136` | `telegram_bot/database.py:103-117` |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | то же |
| `symbol, direction, entry_price, stop_loss, take_profit, confidence, regime` | есть | есть |
| `timeframe` | TEXT DEFAULT '4h' | TEXT DEFAULT '4h' |
| `atr` | REAL | **отсутствует** |
| `funding_rate` | REAL | **отсутствует** |
| `position_size` | REAL | **отсутствует** |
| `notional` | REAL | **отсутствует** |
| `leverage` | REAL | **отсутствует** |
| `close_price` | REAL | **отсутствует** |
| `created_at` | TIMESTAMP DEFAULT CURRENT_TIMESTAMP | то же |
| `closed_at, pnl_pct` | есть | есть |
| `result` | TEXT DEFAULT 'open' | TEXT (**без DEFAULT**) |

Оба используют `CREATE TABLE IF NOT EXISTS`, поэтому побеждает тот, кто
запустился первым. Если первым поднялся телеграм-бот, у таблицы нет
шести колонок, и `SignalBridge.log_signal` упадёт на `INSERT` —
поймает своё исключение (`signal_bridge.py:237-240`) и вернёт `0`.
`_emit_signal` это увидит и напишет ERROR (§1.8) — то есть отказ
диагностируется, но сигнал теряется. Идемпотентная миграция есть только
для `timeframe` (`signal_bridge.py:167-176`), для остальных пяти — нет.

Проверено на снимке: колонки на месте (бот поднялся первым), 8 строк.

**Единицы и nullable по факту записи** (`snap.db`, `id=8`):

```
(8, 'BTCUSDT-PERP.BINANCE', 'long', 0.6556943351460622, 'high_vol',
 1032.6489849, 6.29e-05, 0.06455888461762499, 4343.954301600747,
 '2026-06-02T20:00:00.673006+00:00', 'loss')
```

| Колонка | Тип по схеме | Что реально | Согласовано с читателями |
|---|---|---|---|
| `symbol` | TEXT | `'BTCUSDT-PERP.BINANCE'` — Nautilus instrument_id, не тикер Binance | ✅ компенсировано: `stats_engine.py:198` фильтрует `symbol LIKE '%{symbol}%'`, `signal_formatter._clean_symbol:46` нормализует к отображаемому виду |
| `confidence` | REAL | `max(p,1−p) ∈ [0.5,1]` | ✅ единая шкала (§1.2) |
| `regime` | TEXT | одна из четырёх меток детектора | ✅ |
| `atr` | REAL | **доллары** (1032.65 при цене 67 289) | ✅ |
| `funding_rate` | REAL, nullable в сигнатуре (`float | None = 0.0`) | `6.29e-05`, но в 3 строках из 5 — `0.0` | ⚠ `0.0` неотличим от «нет данных»; сама сигнатура `log_signal` допускает `None`, а колонка не `NOT NULL` |
| `position_size` | REAL | контракты базового актива | ✅ (§1.9) |
| `notional`, `leverage` | REAL | USDT, безразмерное | ✅ |
| `created_at` | TIMESTAMP | ISO-8601 с `T` и `+00:00` из `datetime.now(timezone.utc).isoformat()` | ⚠ **два формата в одной колонке** |
| `result` | TEXT | `'open' / 'win' / 'loss'` | ✅ |

**Про `created_at`.** Писатель всегда кладёт ISO-8601 с `T`; но у колонки
`DEFAULT CURRENT_TIMESTAMP`, который SQLite пишет как
`'YYYY-MM-DD HH:MM:SS'` (пробел, без зоны). Любая вставка без явного
`created_at` — например, через `telegram_bot/database.add_signal:360-364`,
которая перечисляет колонки и `created_at` среди них **не** упоминает, —
попадёт в формат с пробелом. Тогда:

* `MAX(created_at)` (`check_signal_freshness.py:384`) сравнивает строки
  лексикографически: `'2026-08-23T…'` > `'2026-08-23 …'`, т.к. `'T'`
  (0x54) > `' '` (0x20). При смешанных форматах максимум **всегда**
  достаётся строке с `T`, независимо от того, какая из них позже. Ложное
  «свежо» при новой записи из телеграм-пути.
* `datetime.fromisoformat(last_time_str)` (`:394`) корректно съест оба;
  для формата без зоны срабатывает `replace(tzinfo=utc)` (`:395-396`) —
  и это верно, т.к. `CURRENT_TIMESTAMP` в SQLite — UTC. ✅
* `datetime(created_at)` (`signal_poller.py:195`) — SQLite понимает оба. ✅
* `created_at >= date(?, '-30 days')` (`database.py:569,575`) — сравнение
  строк с `'2026-07-24'`; работает для обоих. ✅

Сегодня в БД один формат, поэтому дефект латентный. Записан в §8.

### 4.2 Heartbeat JSON — пять ключей, три читателя

Писатель, `heartbeat.py:156-162`:

```
                payload = json.dumps({
                    "process_ts": ts,
                    "started_ts": self._started_ts,
                    "last_bar_ts": self._last_bar_ts,
                    "bars_seen": self._bars_seen,
                    "last_signal_ts": self._last_signal_ts,
                })
```

| Ключ | Единица | Watchdog | `check_signal_freshness.py` | Сам бот |
|---|---|---|---|---|
| `process_ts` | сек., `time.time()` | `:660-665` — отсутствие → UNKNOWN; старее `max_silence_seconds` → DEAD | не читает | `is_alive()` использует локальный `_last_beat_ts`, не ключ |
| `started_ts` | сек. | `:670` — grace-период до первого бара | `:556` `_epoch_to_dt(...)` — с какого момента считать молчание | — |
| `last_bar_ts` | сек., `bar.ts_event/1e9` = **закрытие бара** | `:668-673` — старее `max_bar_silence_seconds` → DEAD (zombie) | не читает | — |
| `bars_seen` | счётчик | **не читает** | **не читает** | **не читает** |
| `last_signal_ts` | сек., `time.time()` момента записи | **не читает** | `:313, :530, :583-609` — четыре различённых состояния (`_MISSING`, `None`, не-число, число) | — |

**Совпадают ли ожидания.** По четырём читаемым ключам — да, единицы
(секунды epoch) и смысл согласованы; `d60e55e` («treat a null
last_signal_ts as the bot's own claim of silence») и `5f12e98`
(UNKNOWN-вердикт) закрыли ровно те места, где трактовка расходилась.

Одно расхождение по клиенту Redis, безобидное: watchdog создаёт клиент с
`decode_responses=True` (`watchdog.py:697`) и делает `json.loads(val)` над
`str`; `check_signal_freshness._read_heartbeat:257` делает
`json.loads(raw.decode("utf-8"))`, т.е. ожидает `bytes`. Два разных
клиента, каждый согласован сам с собой.

`bars_seen` — ключ без читателя. `reconciler_signals.py:304` пишет своё,
одноимённое, в другую структуру. Находка A2-072 (LOW).

**Имя ключа.** `_STRATEGY_HEARTBEAT_KEYS` (`watchdog.py:78-82`):
`4h → "atomicortex:heartbeat"`, `1h → "bot_1h_heartbeat"`,
`15m → "bot_15m_heartbeat"`. Юнит `atomicortex-watchdog-15m.service:22`
передаёт `--heartbeat-key bot_15m_heartbeat` — совпадает с картой.
`MLStrategyConfig.heartbeat_key` (`ml_strategy.py:127`) =
`"atomicortex:heartbeat"` — совпадает с `4h`. **Контракт держится.**

### 4.3 `risk_state_4h.json` — два писателя, общий файл

Файл резолвится один раз в `ml_strategy.on_start:288-292` и передаётся
**обоим**: `PortfolioTracker(..., state_path=_risk_state_path)` (`:297-299`)
и `CircuitBreaker(state_path=_risk_state_path)` (`:307`).

**Асимметрия записи.**

* `CircuitBreaker._persist` (`circuit_breaker.py:258-284`) — **сливает**:
  `existing = self._store.load()`, добавляет свои два ключа, сохраняет.
  Комментарий прямо: «Merge — preserve unrelated keys (the same file is
  shared with PortfolioTracker in production)».
* `PortfolioTracker._persist` (`portfolio_tracker.py:431-441`) —
  **перезаписывает**: `self._store.save(self._snapshot())`, где
  `_snapshot()` (`:409-429`) содержит только 11 своих ключей. Чужие
  ключи не читаются и не сохраняются.

**Замер** (`/tmp`, живого файла не касались):

```
after breaker persist : ['breaker_daily_trigger_reason', 'breaker_daily_triggered', 'day_start']
  breaker_daily_triggered = True
after tracker persist : ['cash', 'consecutive_losses', 'daily_realized_pnl', 'day_start',
                         'day_start_equity', 'initial_equity', 'last_loss_time',
                         'peak_equity', 'total_realized_pnl', 'week_start', 'weekly_realized_pnl']
  breaker_daily_triggered = <KEY GONE>
restart -> breaker._daily_triggered = False
```

Флаг сработавшего дневного breaker'а стирается первой же записью
трекера. Гонки в смысле параллельных потоков нет (Nautilus —
однопоточный цикл), но есть безусловная перезапись чужих ключей.

**Насколько быстро это происходит.** В `on_bar` порядок такой
(`ml_strategy.py:686-716`): `update_price` → `_record_equity` →
проверка breaker. `_record_equity` в реальном режиме вызывает
`sync_equity`, а `seed_from_authoritative_equity` вызывает `_persist`
(`portfolio_tracker.py:259`); `close_position` (`:314`), `record_loss`
(`:370`), `_roll_periods` (`:403`) — тоже. То есть флаг стирается на
ближайшем следующем событии трекера, обычно в пределах одного бара.

Что при этом всё-таки удерживает halt: `_daily_triggered` живёт **в
памяти** процесса и переживает бары. Теряется он только при
рестарте — и это ровно тот случай, ради которого модуль
`risk_state_store.py` написан (его докстринг: «a bot that lost -2.9 %
over the morning and was restarted at noon could still lose another
-3 %»). Находка A2-062.

`RiskStateStore.load` (`risk_state_store.py:85-115`) применяет дневной и
недельный сброс до возврата; оба потребителя это знают
(`portfolio_tracker.py:442-444`, `circuit_breaker.py:288-289`). Эта часть
контракта держится.

---

## 5. ГРАНИЦА КОД / КОНФИГ

### 5.1 Пять уровней и фактический приоритет

| Значение | `.env` | `.env.example` | `Settings` | dataclass-дефолт | CLI | systemd | Фактический приоритет |
|---|---|---|---|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.65 | 0.65 | `config.py:90` = 0.65 | `MLStrategyConfig` 0.55, `LiveTraderConfig` **None** | нет | нет | `.env` → `Settings` → `build_node:232-236` → strategy. **Работает** (H12) |
| `TRADING_MODE` | testnet | testnet | `config.py`, валидатор `{testnet,paper,live}` | `"testnet"` | `--mode` | `--mode paper` | **CLI побеждает**: `.env` = testnet, юнит = paper |
| `INITIAL_CAPITAL` | 10000.0 | 10000.0 | `config.py:87` | 10 000 | `--capital 10000` | `--capital 10000` | CLI; `Settings.initial_capital` **никем не читается** |
| `MAX_LEVERAGE` | 10 | 10 | `config.py:88` | `RiskConfig` 10, `LiveTraderConfig` 10 | нет | нет | dataclass; `.env` **не действует** |
| `RISK_PER_TRADE` | 0.01 | 0.01 | `config.py:89` | 0.01 ×3 | нет | нет | dataclass; `.env` **не действует** |
| `MAX_OPEN_POSITIONS` | 3 | 3 | `config.py:91` | 3 | нет | нет | dataclass; `.env` **не действует** |
| `DAILY_LOSS_LIMIT` | −0.03 | −0.03 | `config.py:92` | `RiskConfig` −0.03; `CircuitBreaker.DAILY_LOSS_HARD` = −0.03 (**константа класса**) | нет | нет | dataclass/константа; `.env` **не действует** |
| `WEEKLY_LOSS_LIMIT` | −0.08 | −0.08 | `config.py:93` | −0.08 / `WEEKLY_LOSS` | нет | нет | то же |
| `MAX_DRAWDOWN_KILL` | −0.15 | −0.15 | `config.py:94` | −0.15 / `MAX_DRAWDOWN_KILL` | нет | нет | то же |
| `SYMBOLS` | `BTC-USDT-PERP,…` | то же | `config.py:149` + `@property symbols` | `["BTCUSDT-PERP"]` | `--symbols` | `--symbols BTCUSDT-PERP` | CLI; `Settings.symbols` читается только `verify_setup.py:172` |
| `DATA_DIR` | `/mnt/hdd/AtomiCortex/data` | `./data` | `config.py` | — | нет | нет | `Settings.data_dir` читается 2 раза; конвейер фич берёт путь **не отсюда** (`features_dir` из `MLStrategyConfig`) |
| `heartbeat_key` | нет | нет | нет | `"atomicortex:heartbeat"` | `--heartbeat-key` | 15m: явно | dataclass / CLI, согласованы (§4.2) |

**Замер: какие поля `Settings` вообще кто-то читает.**

```
$ grep -roE "settings\.[a-z_]+" --include='*.py' src scripts | sort | uniq -c | sort -rn
      6 settings.telegram_bot_token
      6 settings.telegram_admin_id
      5 settings.trading_mode
      3 settings.cryptobot_token
      2 settings.symbols
      2 settings.signal_stale_hours_
      2 settings.redis_password
      2 settings.premium_price_usdt_
      2 settings.premium_price_stars_
      2 settings.data_dir
      2 settings.binance_testnet_api_secret
      2 settings.binance_testnet_api_key
      2 settings.binance_mainnet_api_secret
      2 settings.binance_mainnet_api_key
      1 settings.startup_grace_sec
      1 settings.signal_stale_hours_default
      1 settings.signal_bridge_lag_tolerance_sec
      1 settings.signal_alert_cooldown_hours
      1 settings.redis_port
      1 settings.redis_host
      1 settings.confidence_threshold
```

`initial_capital`, `max_leverage`, `risk_per_trade`, `max_open_positions`,
`daily_loss_limit`, `weekly_loss_limit`, `max_drawdown_kill` — в списке
**отсутствуют**. Семь риск-критичных значений объявлены в `.env`,
объявлены в `.env.example`, разобраны pydantic'ом — и не читаются никем.
Находка A2-068.

Особый случай — `CircuitBreaker`: его пороги не просто не берутся из
`Settings`, они **константы класса** (`circuit_breaker.py:59-63`), т.е.
их нельзя переопределить даже из кода без наследования. Численно они
совпадают с `.env` (−0.02/−0.03/−0.08/−0.10/−0.15), поэтому расхождения
поведения сегодня нет — есть расхождение **ожидания**: правка
`DAILY_LOSS_LIMIT` в `.env` не изменит ничего.

**Расхождение `TRADING_MODE`.** `.env` = `testnet`; юнит
`atomicortex-bot.service` = `--mode paper --dry-run`. Комментарий в
юните это признаёт и объясняет («testnet is unusable here —
BINANCE_TESTNET_* are empty by design … it matches TRADING_MODE=paper in
.env»), но в `.env` стоит `testnet`, а не `paper` — комментарий
описывает состояние, которого нет. `Settings.trading_mode` читается
пятью местами; для 4H-бота фактический режим приходит из CLI, поэтому
последствий нет. Для **watchdog'а** — есть, см. §5.4 и A2-063.

### 5.2 Значения в `.env.example`, не читаемые кодом

```
$ comm -13 code_env.txt example_env.txt     # в примере, но нет alias в Settings
<пусто>
```

Формально таких нет: у каждого ключа `.env.example` есть `Field(alias=…)`
в `src/config.py`. Но «объявлено в `Settings`» ≠ «читается». По замеру
§5.1 фактически не влияют на поведение:

`INITIAL_CAPITAL`, `MAX_LEVERAGE`, `RISK_PER_TRADE`, `MAX_OPEN_POSITIONS`,
`DAILY_LOSS_LIMIT`, `WEEKLY_LOSS_LIMIT`, `MAX_DRAWDOWN_KILL`,
`QUESTDB_HOST`, `QUESTDB_PORT`, `QUESTDB_HTTP_PORT`, `BYBIT_*` (4 ключа),
`LOGS_DIR` (читается один раз, в `logger`), `SYMBOLS` (только
`verify_setup.py`).

QuestDB и Bybit — остатки нереализованных интеграций: `grep -r questdb`
и `grep -r bybit` по `src/` дают только `config.py`.

### 5.3 Значения, читаемые кодом и отсутствующие в `.env.example`

```
$ comm -23 code_env.txt example_env.txt
API_CORS_ORIGINS
API_RATE_LIMIT_PER_MINUTE
ATOMICORTEX_API_KEY
STARTUP_GRACE_SEC
```

Плюс ключ, читаемый **мимо** `Settings` (нет `Field` вовсе):

```
$ grep -roE "(os\.environ(\.get)?\[?\(?[\"'][A-Z_0-9]+|getenv\([\"'][A-Z_0-9]+)" --include='*.py' src scripts | ...
API_CORS_ORIGINS
API_RATE_LIMIT_PER_MINUTE
ATOMICORTEX_API_KEY
ATOMICORTEX_DB_PATHS      <-- нет ни в Settings, ни в .env.example
REDIS_HOST
REDIS_PASSWORD
REDIS_PORT
TELEGRAM_ADMIN_ID
```

`ATOMICORTEX_DB_PATHS` — путь к торговым БД для `src/api/main.py`;
это пятое независимое определение путей к БД из A2-003, и оно **вообще не
документировано**: ни `.env`, ни `.env.example`, ни `Settings`.
Развёртывание из `.env.example` даст API дефолтный путь молча.

`ATOMICORTEX_API_KEY` отсутствует в `.env.example` — это ключ
аутентификации REST API (A2-010). Развёртывание по примеру поднимет API
без него.

### 5.4 Расхождение venue между ботом и его watchdog'ом

Это не отдельная переменная, а расхождение словарей, обнаруженное при
разборе `trading_mode`.

* `Settings._ALLOWED_TRADING_MODES` (`config.py:21`) = `("testnet","paper","live")`.
* `scripts/run_watchdog.py:106-108`: `choices=["testnet","live"]` —
  **`paper` не существует** для watchdog'а.
* `deploy/atomicortex-bot.service` запускает бота с `--mode paper`;
  `live_trader.build_node` для `paper` резолвит **mainnet** ключи и
  **mainnet** endpoints (комментарий юнита: «paper resolves keys through
  the mainnet branch»).
* `deploy/atomicortex-watchdog.service:19` запускает watchdog с
  `--trading-mode testnet`.
* `watchdog.py:179-181`: `urls = _BINANCE_URLS.get(mode, ...["testnet"])`
  → `base = "https://testnet.binancefuture.com"`.
* `run_watchdog.py:192, 209-216`: `is_testnet = True` → ключи берутся из
  `settings.binance_testnet_api_key/secret`, про которые юнит бота сам
  пишет: «BINANCE_TESTNET_* are empty by design».

Итог: **аварийное закрытие 4H-watchdog'а идёт на testnet, пустыми
ключами, тогда как бот работает на mainnet.** `_run_emergency_close`
запросит `positionRisk` у testnet, получит пусто или 401 и отчитается,
что закрывать нечего. В `--dry-run` позиций нет и последствий нет; но
конфигурация, объявленная как боевая, не может выполнить свою функцию.
Находка A2-063.

15m-пара согласована иначе: `atomicortex-bot-15m.service:25` тоже
`--trading-mode testnet`, и watchdog `--trading-mode testnet` — оба на
testnet. Там расхождения нет.

---

## 6. ЧТО ДЕРЖИТСЯ

Проверено, дефекта нет. Следующим проходам и ремонту перепроверять не нужно.

| # | Контракт | Как проверено |
|---|---|---|
| 1 | `open_time` = ms, открытие бара, UTC, кратно длительности — во всех parquet, DataStore, конвейере фич | замер схемы + побитовое совпадение 12 позиционных признаков в §3.2 |
| 2 | `ts_event` (Nautilus) = закрытие бара в нс; `bar_open_time_ms` возвращает на сетку открытий корректно для обеих конвенций close | чтение `live_feature_state.py:39-64` + §3.2 |
| 3 | `close_time = open_time + duration − 1 ms` в parquet, и оба preload-пути берут `ts_event` именно из `close_time` | замер + `_log_preload_timestamp_check` |
| 4 | `confidence = max(p, 1−p)` одинаково у тренера (обучение и eval, включая per-symbol) и у `get_signal` в live | чтение `lgbm_trainer.py:1117, 1290-1291, 1515` |
| 5 | `atr_pct` — безразмерная доля `ATR(14)/close` во всех вычислениях и у всех потребителей (`triple_barrier`, `risk_engine`, `mtf_context`) | таблица §1.3 |
| 6 | `atr` в `TradeSignal` и в `signals_log` — доллары; согласовано с сайзингом и SL | замер БД: `notional/position_size = 67 289` |
| 7 | `regime` — строка нижним регистром из четырёх значений; все сравнения совпадают по регистру и по множеству; `unknown` не производится | замер распределения по 15 114 строкам |
| 8 | `atr_percentile` ∈ [0,1] и `atr_pct` различаются на два порядка — численно спутать нельзя | замер: 0.872 против 0.0143 |
| 9 | Приоритет `peak_equity`: файл состояния > биржа > конфиг; `_peak_restored` / `_peak_seeded` защищают от повторного посева | чтение `portfolio_tracker.py:236-249, 462-464` |
| 10 | `RiskStateStore.load` применяет дневной/недельный сброс, и оба потребителя на это рассчитывают | чтение `risk_state_store.py:85-115` |
| 11 | `funding_rate`: единица (за 8 ч), знак (Binance) и трактовка совпадают в признаках, риск-фильтре и P&L | замер расчётов + чтение трёх потребителей |
| 12 | `funding_rate is None` ≠ `0.0`: `_get_funding_rate` возвращает `None`, `_check_funding_rate` блокирует — fail-safe целостен | чтение `ml_strategy.py:1276-1289`, `risk_engine.py:367-379` |
| 13 | `oi_value` = USD (`sum_open_interest_value`), не контракты; разметка после D-9 согласована offline/live | замер L1: `oi_value`, `oi_delta_4h`, `oi_delta_12h`, `oi_quadrant` — `max_rel = 0` |
| 14 | `join_asof` по `open_time` не заглядывает вперёд, и смещение одинаково offline и live | чтение `derivatives.py:117-121, 175-179` + L1 |
| 15 | `signal_id = 0` = «не записано»; единственный путь утечки закрыт явной веткой с ERROR | чтение `_emit_signal:1035-1041` |
| 16 | Уникальность `signal_id` между 4H и 15m БД обеспечена составным ключом (timeframe, id) | чтение `signal_poller.py:73`, `keyboards.py:318` |
| 17 | `position_size` = контракты, `notional` = USDT, `leverage` = безразмерное; всеми потребителями читаются верно | замер БД |
| 18 | `symbol_encoded` в проде = 0 для BTCUSDT, что соответствует `symbols_in_train` бандла | разбор `instrument_id` + manifest |
| 19 | `signals_log.symbol` несёт Nautilus instrument_id, и оба читателя это компенсируют (`LIKE '%…%'`, `_clean_symbol`) | чтение `stats_engine.py:198`, `signal_formatter.py:46` |
| 20 | Heartbeat: 4 из 5 ключей читаются, единицы (секунды epoch) и смысл согласованы у обоих читателей | таблица §4.2 |
| 21 | Имена heartbeat-ключей: карта `watchdog._STRATEGY_HEARTBEAT_KEYS` совпадает с дефолтами конфигов и с юнитами | сверка `watchdog.py:78-82` с юнитами |
| 22 | `min_bars` детектора одинаков в `build()` (дефолт 300) и `build_from_buffer()` (`hurst_window` = 300) для 4H | чтение + L1 (`adx`, `atr_pct`, `atr_percentile` — `max_rel = 0`) |
| 23 | `e67a451` (обрезка головы по окну детектора) жив: 5778 − 740 = 5038 строк, ровно длина `ml_features` | замер |
| 24 | PR-A (буфер 740) жив: `HISTORY_BARS_4H = required_history_bars(540) = 740`, `deque(maxlen=740)` | замер импорта |
| 25 | `build_from_buffer` воспроизводит `build` алгоритмически: при идеальных входах 42 признака из 45 совпадают с точностью двойной | §3.2, L1 |
| 26 | A3 (35 % расхождения regime-лейбла) закрыт: 0.2–1.0 % | §3.3 |
| 27 | `CircuitBreaker._persist` корректно сливает чужие ключи (проблема — на стороне трекера, не его) | замер §4.3 |
| 28 | `CONFIDENCE_THRESHOLD` — единственное значение `.env`, реально доходящее до стратегии и риск-движка (H12) | §5.1 + `live_trader.py:232-236` |
| 29 | Валидатор `TRADING_MODE` строгий: отвергает пробелы, регистр и опечатки | чтение `config.py:158-186` |
| 30 | 15m-пара «бот + watchdog» согласована по venue (оба testnet) | сверка юнитов |

---

## 7. НАХОДКИ

### A2-059 `get_metrics_df` подаёт модели 100 записей и две колонки из четырёх
**Севирити:** HIGH
**Тип:** логика / семантика
**Где:** `src/features/live_feature_state.py:255-270`; потребитель —
`src/features/derivatives.py:164-172`, `:201-203`
**Что в коде:**
```
    def get_metrics_df(self, n_bars: int = 100) -> pl.DataFrame:
        ...
        records = list(self.oi_history)[-n_bars:]
        return pl.DataFrame(records).rename({
            "timestamp": "create_time",
            "oi_value": "sum_open_interest_value",
        })
```
**В чём дефект:** два обрезания сразу. (а) `oi_history` не хранит
`count_long_short_ratio` и `sum_taker_long_short_vol_ratio`, поэтому
`add_oi_features` подставляет `pl.lit(0.0)` — три признака становятся
константным нулём. (б) 100 записей против 740-барного буфера: после
~8 ч работы это 8 часов 5-минутных поллов, и `join_asof` не находит
значений для 738 баров из 740 → `fill_null(0.0)`.
**Как проявляется:** `ls_ratio`, `ls_ratio_zscore`, `taker_vol_ratio` ≡ 0
всегда; `oi_zscore` при обучающем 4.5929 даёт 1.1306 сразу после старта
и 7.6702 после 8 часов — величина зависит от аптайма, а не от рынка;
`oi_delta_12h` → 0 (делитель `oi[t−3] = 0`). Пять признаков из 46.
**Кто ещё это читает:** только модель. Ни один потребитель не проверяет
`oi_value` на ноль; `feature_selection_v3.py` работает по offline-матрице,
где значения реальные.
**Как установлено:** замером — `/tmp/audit2_p6/parity.py`, варианты L2/L3,
вывод в `/tmp/audit2_p6/out_L234.txt`.
**Уверенность:** доказано.

### A2-060 История funding в live не пополняется: расчёты в 00/08/16, код ждёт 01/09/17
**Севирити:** CRITICAL
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:626`
**Что в коде:**
```
                if dt.hour in (1, 9, 17) and dt.minute == 0:
```
**В чём дефект:** Binance USDⓈ-M рассчитывает funding в 00:00 / 08:00 /
16:00 UTC. Замер по 2889 записям `funding_rate` в parquet:
`{0: 963, 8: 963, 16: 963}`, минуты — все `0`. Условие никогда не
истинно, поэтому `funding_rate_history` после `preload_funding` на
старте не растёт ни разу.
**Как проявляется:** `join_asof` по `open_time` промахивается тем
сильнее, чем дольше живёт процесс. Замер по последнему бару BTCUSDT:
при аптайме 7 суток `funding_zscore_7d` = 0 против обучающего 1.2541,
`funding_zscore_30d` = 0.7124 против 1.4299, `funding_rate` = 7.841e-05
против 9.422e-05; при 30 сутках `funding_zscore_30d` = 0. Затронуто
6 признаков: `funding_rate`, `funding_abs`, `funding_zscore_7d`,
`funding_zscore_30d`, `funding_cum_24h`, `basis_approx`. Z-score
схлопывается в 0, потому что окно заполняется одной константой и
`rolling_std → 0` → `safe_divide → 0.0`.
**Кто ещё это читает:** `_get_funding_rate` (`ml_strategy.py:1280-1282`)
берёт `funding_rate` **из вектора признаков**, т.е. то же устаревшее
значение уходит в `RiskEngine._check_funding_rate`. Фильтр экстремального
funding'а работает по чтению возрастом в недели.
**Как установлено:** замером — часы расчётов из parquet
(`/tmp/audit2_p6` шаг «funding settlement hours») и
`/tmp/audit2_p6/funding_stale.py`.
**Уверенность:** доказано.

### A2-061 Фаза амортизации Hurst зависит от длины кадра
**Севирити:** MEDIUM
**Тип:** математика
**Где:** `src/features/regime_detector.py:427-430`
**Что в коде:**
```
            if (i - min_bars) % _RECOMPUTE_EVERY == 0:
                start = max(0, i + 1 - self.hurst_window)
                last_hurst = calculate_hurst_exponent(close[start : i + 1])
            hursts[i] = last_hurst
```
**В чём дефект:** `i` — индекс внутри переданного кадра. Offline
`detect_all` получает всю историю (5778 баров), live — ровно 740.
Разность индексов последней строки по модулю 6 в общем случае не ноль,
поэтому Hurst последней строки пересчитан на баре, отстоящем на 0…5
баров от того, на котором его пересчитал бы offline-путь.
**Как проявляется:** это единственное расхождение при идеальных входах.
Замер по 20 барам: `hurst` — `max_rel = 0.01525`, `mean_rel = 0.004995`
(0.73250643 против 0.72952242 на последнем баре); через
`trend_strength = 0.4·|H−0.5|·2 + 0.6·min(adx/50,1)` расхождение
переходит на `trend_strength` и `regime_confidence`
(`max_rel = 0.01642`). Три признака из 46, расхождение ~1.5 %.
**Кто ещё это читает:** `regime_confidence` пишется в `ml_features` и
читается моделью; `hurst` — тоже признак. На классификацию режима Hurst
не влияет (`_classify` его игнорирует — `regime_detector.py:568`).
**Как установлено:** замером — `/tmp/audit2_p6/parity.py`, вариант L1.
**Уверенность:** доказано.

### A2-062 `PortfolioTracker` стирает флаг сработавшего circuit breaker'а из общего файла
**Севирити:** HIGH
**Тип:** логика / архитектура
**Где:** `src/risk/portfolio_tracker.py:431-441` против
`src/risk/circuit_breaker.py:258-284`
**Что в коде:**
```
    def _persist(self) -> None:
        if self._store is None:
            return
        try:
            self._store.save(self._snapshot())
```
против
```
            # Merge — preserve unrelated keys (the same file is shared with
            # PortfolioTracker in production).
            existing = self._store.load() if self._store is not None else {}
```
**В чём дефект:** оба класса получают один и тот же `state_path`
(`ml_strategy.py:288-307`). `CircuitBreaker` сливает; `PortfolioTracker`
перезаписывает файл своим `_snapshot()`, где ключей breaker'а нет.
Асимметрия односторонняя.
**Как проявляется:** замер (`/tmp`, живого файла не касались):
после `cb._persist()` файл содержит `breaker_daily_triggered = True`;
после любой мутации трекера — `<KEY GONE>`; новый `CircuitBreaker` на
том же файле поднимается с `_daily_triggered = False`. Halt по дневному
убытку не переживает рестарт — ровно тот сценарий, ради которого
`risk_state_store.py` и написан (его докстринг: «a bot that lost -2.9 %
over the morning and was restarted at noon could still lose another
-3 %»). В `on_bar` трекер персистится до проверки breaker'а
(`ml_strategy.py:692` против `:716`), т.е. флаг стирается на ближайшем
следующем событии.
**Кто ещё это читает:** `RiskStateStore.load` применяет дневной сброс и
сам выставляет `breaker_daily_triggered = False` при смене суток
(`risk_state_store.py:105-107`) — то есть отличить «сброшено по дате» от
«стёрто трекером» по файлу невозможно.
**Как установлено:** замером (скрипт в `/tmp`, вывод приведён в §4.3).
**Уверенность:** доказано.

### A2-063 4H-watchdog нацелен на testnet, тогда как бот работает на mainnet
**Севирити:** HIGH
**Тип:** архитектура / расхождение с практикой
**Где:** `deploy/atomicortex-bot.service` (ExecStart `--mode paper`) против
`deploy/atomicortex-watchdog.service:19` (`--trading-mode testnet`);
`src/execution/watchdog.py:179-181`; `scripts/run_watchdog.py:106-108, 192`
**Что в коде:**
```
        "--trading-mode",
        choices=["testnet", "live"],
        default="testnet",
```
```
        urls = _BINANCE_URLS.get(mode, _BINANCE_URLS["testnet"])
        self._base_url: str = urls["base"]
```
**В чём дефект:** словарь режимов у watchdog'а — `{testnet, live}`, у
`Settings` — `{testnet, paper, live}`. Значения `paper` для watchdog'а не
существует, поэтому юнит передаёт `testnet`. Но `--mode paper` в боте
резолвится через mainnet-ветку `live_trader.build_node` (это прямо
написано в комментарии юнита бота). Дополнительно
`run_watchdog.py:209-216` при `is_testnet = True` берёт
`settings.binance_testnet_api_key/secret`, про которые тот же юнит бота
пишет: «BINANCE_TESTNET_* are empty by design».
**Как проявляется:** `_run_emergency_close` обращается к
`https://testnet.binancefuture.com/fapi/v2/positionRisk` пустыми ключами.
Позиций бота там нет никогда. Watchdog отчитается «нечего закрывать» и
инцидент будет выглядеть обработанным. В `--dry-run` позиций нет вообще,
поэтому сегодня наблюдаемых последствий нет — но заявленная защита не
может сработать по построению.
**Кто ещё это читает:** `watchdog._scope_symbol` (`:186-189`) нормализует
`BTCUSDT-PERP.BINANCE` → `BTCUSDT` под формат `positionRisk` — эта часть
контракта как раз корректна, что делает несоответствие venue тем более
заметным.
**Как установлено:** чтением юнитов и `run_watchdog.py` / `watchdog.py`.
**Уверенность:** доказано (для конфигурации в `deploy/`; фактическая
конфигурация на VM не проверялась — на VM не ходили).

### A2-064 Метка режима и признак `atr_percentile` в одном баре считаются разными путями
**Севирити:** MEDIUM
**Тип:** математика / архитектура
**Где:** `src/execution/strategies/ml_strategy.py:1695-1700` (метка) против
`src/features/feature_pipeline.py:483` → `regime_detector.py:438-447` (признак);
две реализации перцентиля — `regime_detector.py:138-171` и `:438-447`
**Что в коде:**
```
            lookback = min(len(self._bars), 540)
            recent = self._bars[-lookback:]
            ...
            state = self._regime_detector.detect(df, idx=-1)
```
против
```
            lb_start = max(0, i + 1 - self.atr_lookback)
            history = atr_arr[lb_start : i + 1]
            history_valid = history[history > 0]
```
**В чём дефект:** метка режима, выбирающая модель, приходит из
`detect()` на 540 барах, где `calculate_atr_percentile` берёт
`valid[-540:]` из NaN-очищенного ряда (фактически 527 значений) и считает
ATR по 540-барному срезу. Признак `atr_percentile`, уходящий **в ту же
модель**, приходит из `detect_all` на 740-барном буфере с окном ровно
540 и ATR, посчитанным по всему кадру. Один бар — две величины под одним
именем.
**Как проявляется:** замер по 1200 барам (3 символа × 400):
`|atr_percentile_offline − atr_percentile_live|` среднее 0.020–0.022,
максимум 0.041, смещение **одностороннее** (live выше). Поскольку первое
правило `_classify` — `atr_percentile > atr_vol_threshold → HIGH_VOL`,
все 8 расхождений метки из 1200 идут в одну сторону: offline
`trend_up`/`trend_down`/`range` → live `high_vol`. `high_vol` маршрутизирует
на **другую модель**.
**Кто ещё это читает:** `_select_model` (метка), `signals_log.regime`
(метка), модель (признак). Три потребителя, две несогласованные величины.
**Как установлено:** замером — `/tmp/audit2_p6/regime_parity.py`,
вывод в `/tmp/audit2_p6/out_regime.txt`.
**Уверенность:** доказано.

### A2-065 28 % баров идут в модель, не обучавшуюся на этом режиме
**Севирити:** MEDIUM
**Тип:** логика / семантика
**Где:** `src/execution/strategies/ml_strategy.py:1759-1760` против
`src/models/lgbm_trainer.py:1310+` (`_filter_by_regime`)
**Что в коде:**
```
        if regime_label == "range":
            return self._trend_model, self._trend_features, max(base_threshold, 0.60)
```
против
```
        if regime == "trend":
            return df.filter(pl.col("regime").is_in(["trend_up", "trend_down"]))
```
**В чём дефект:** детектор выдаёт четыре метки, моделей две. Бары с
`regime == "range"` в live отправляются в trend-модель, а
`_filter_by_regime("trend")` при обучении отбирает **только**
`trend_up|trend_down` — ни одной строки `range` в обучающей выборке
trend-модели нет. `_load_models` (`:1720`) грузит ровно две модели:
`for regime in ["trend", "high_vol"]`; range-модели не существует.
**Как проявляется:** замер распределения меток по всей матрице:
`range` = 4258 из 15114 строк = **28.2 %**. Эти бары оцениваются моделью
вне её обучающего распределения. Компенсация — поднятый порог 0.60 —
не делает предсказание валидным, а только реже пропускает его.
**Кто ещё это читает:** `signals_log.regime` сохраняет `range` — в
снимке БД таких строк нет (все 8 — `trend_up`/`trend_down`/`high_vol`),
что согласуется с более высоким порогом.
**Как установлено:** замером (распределение меток) + чтением.
**Уверенность:** доказано.

### A2-066 `_preload_historical_bars` жёстко зашит на BTCUSDT
**Севирити:** MEDIUM
**Тип:** логика / недоделка
**Где:** `src/execution/strategies/ml_strategy.py:1798`
**Что в коде:**
```
        symbol_clean = "BTCUSDT"  # without -PERP
```
**В чём дефект:** метод не смотрит на `self._instrument_id`. Символ
используется и для `_preload_from_parquet(symbol_clean, n_bars)`, и для
`_preload_from_binance_api(symbol_clean, n_bars)`.
**Как проявляется:** при запуске с `--symbols BTCUSDT-PERP ETHUSDT-PERP`
(`live_trader.build_node:216` создаёт стратегию на каждый символ)
ETH-стратегия прогреет свои буферы **барами BTC**. Признаки будут
посчитаны по BTC, `symbol_encoded` — по ETH (=1), и модель получит
вектор из двух разных инструментов. В проде запускается один символ
(`deploy/atomicortex-bot.service`: `--symbols BTCUSDT-PERP`), поэтому
сегодня не проявляется.
**Кто ещё это читает:** `_compute_features_unified:1340-1343` и
`on_start:340-345` (`FeaturePipeline(symbol=sym_base)`) — оба корректно
выводят символ из `instrument_id`. Несогласован ровно один из трёх.
**Как установлено:** чтением.
**Уверенность:** доказано.

### A2-067 `atr_percentile` печатается под именем `atr_pct`
**Севирити:** LOW
**Тип:** семантика
**Где:** `src/execution/strategies/ml_strategy.py:761`
**Что в коде:**
```
                f"atr_pct={regime_state.atr_percentile:.2f} | "
```
**В чём дефект:** `RegimeState` несёт оба поля;
`atr_pct` — доля `ATR/close` (порядок 0.014), `atr_percentile` — ранг в
распределении (порядок 0.87). Печатается второе под именем первого.
Соседний журнал в том же файле (`:1704`) печатает `state.atr_pct` под
именем `ATR%` — то есть в одном модуле два несогласованных ярлыка.
**Как проявляется:** оператор, читающий `on_bar step 3`, видит
`atr_pct=0.87` и заключает, что волатильность 87 %. Числа в расчётах не
затронуты — строка только логирующая.
**Кто ещё это читает:** ничего; строка не парсится.
**Как установлено:** чтением + замером значений обеих величин
(0.014307574 против 0.87222222 на последнем баре BTCUSDT).
**Уверенность:** доказано.

### A2-068 Семь риск-критичных значений `.env` не читаются никем
**Севирити:** MEDIUM
**Тип:** архитектура / расхождение с практикой
**Где:** `src/config.py:87-94` (объявление) против фактических читателей
**Что в коде:** `Settings` объявляет `initial_capital`, `max_leverage`,
`risk_per_trade`, `max_open_positions`, `daily_loss_limit`,
`weekly_loss_limit`, `max_drawdown_kill` с алиасами на одноимённые
переменные окружения; все семь присутствуют и в `.env`, и в
`.env.example`. Замер читателей:
```
$ grep -roE "settings\.[a-z_]+" --include='*.py' src scripts | sort | uniq -c | sort -rn
      ...
      1 settings.confidence_threshold
```
— ни одного из семи в списке нет.
**В чём дефект:** фактические значения приходят из dataclass-дефолтов
(`RiskConfig`, `MLStrategyConfig`, `LiveTraderConfig`) и из констант
класса `CircuitBreaker` (`circuit_breaker.py:59-63`), которые нельзя
переопределить даже программно.
**Как проявляется:** правка `DAILY_LOSS_LIMIT`, `MAX_DRAWDOWN_KILL`,
`RISK_PER_TRADE` в `.env` не меняет ничего. Численно дефолты сегодня
совпадают с `.env`, поэтому расхождение поведения отсутствует —
расходится ожидание. Единственное значение `.env`, реально доходящее
до стратегии, — `CONFIDENCE_THRESHOLD` (через сентинел `None` в
`LiveTraderConfig`, паттерн H12).
**Кто ещё это читает:** `scripts/verify_setup.py` проверяет, что
`Settings` разбирается, но не что значения куда-то доходят.
**Как установлено:** замером (grep по читателям `settings.*`).
**Уверенность:** доказано.

### A2-069 Формат `SYMBOLS` в `.env` даёт `symbol_encoded = −1`
**Севирити:** MEDIUM
**Тип:** семантика
**Где:** `src/config.py:149-152` против
`src/execution/strategies/ml_strategy.py:1341-1344`
**Что в коде:**
```
    symbols_raw: str = Field(
        default="BTC-USDT-PERP,ETH-USDT-PERP,SOL-USDT-PERP",
        alias="SYMBOLS",
    )
```
против
```
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            base = sym_str.split("-")[0] if "-" in sym_str else sym_str.split(".")[0]
            rd["symbol_encoded"] = float(sym_map.get(base, -1))
```
**В чём дефект:** для `"BTC-USDT-PERP.BINANCE"` `split("-")[0]` даёт
`"BTC"`, чего в `sym_map` нет → дефолт `−1`. Модель обучена на
`symbol_encoded ∈ {0,1,2}` (manifest `symbols_in_train:
['BTCUSDT','ETHUSDT','SOLUSDT']`); −1 — код, которого она не видела.
Формат `BTC-USDT-PERP` — тот, что стоит и в `.env`, и в `.env.example`.
**Как проявляется:** сегодня не проявляется: `settings.symbols`
читается ровно в одном месте (`scripts/verify_setup.py:172`) и никогда
не попадает в `LiveTraderConfig.symbols`, который приходит из CLI
(`--symbols BTCUSDT-PERP` → `"BTCUSDT-PERP.BINANCE"` → `"BTCUSDT"` → 0).
Любая правка, начинающая читать `settings.symbols` для запуска, молча
подменит признак на −1 для всех трёх символов.
**Кто ещё это читает:** шесть копий отображения символ→код (A2-008); эта
находка объясняет, чем именно опасно их расхождение.
**Как установлено:** чтением + разбором фактического `instrument_id`
прод-юнита.
**Уверенность:** доказано (для латентного пути — вероятно, что он
когда-нибудь оживёт).

### A2-070 Дозапись funding при расчёте не дедуплицирована и сработала бы 60 раз
**Севирити:** LOW
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:626-635`
**Что в коде:**
```
                if dt.hour in (1, 9, 17) and dt.minute == 0:
                    self._live_state.funding_rate_history.append({
                        "fundingTime": ts_ms,
                        "fundingRate": rate,
                    })
```
**В чём дефект:** `BinanceFuturesMarkPriceUpdate` приходит раз в
секунду (докстринг `on_data:601-603` это признаёт). Условие
`dt.minute == 0` истинно все 60 секунд минуты, поэтому за один расчёт
в историю попало бы до 60 записей. `append` идёт напрямую в deque,
минуя `preload_funding`, где дедуп есть (`live_feature_state.py:158-168`).
При `maxlen=300` пять расчётов вытеснили бы весь preload.
**Как проявляется:** сегодня — никак: часы указаны неверно (A2-060), и
ветка не исполняется ни разу. Дефект проявится ровно в момент починки
A2-060, если чинить только часы.
**Кто ещё это читает:** `get_funding_df` → `add_funding_features` →
шесть признаков funding'а.
**Как установлено:** чтением (частота потока — из докстринга того же
метода).
**Уверенность:** вероятно (частота потока не измерена — бот не запускался).

### A2-071 `update_funding` не влияет на признак `funding_rate`; докстринг утверждает обратное
**Севирити:** LOW
**Тип:** семантика / мёртвый код
**Где:** `src/execution/strategies/ml_strategy.py:596-604, 619-622`
**Что в коде:**
```
        We update ``self._live_state.funding_rate`` on every tick (for the
        current ``funding_rate`` feature), but only append to
        ``funding_rate_history`` at settlement times so that rolling features
        (zscore, cum_24h) match the training distribution.
```
**В чём дефект:** `build_from_buffer` получает funding исключительно
через `get_funding_df()` (`ml_strategy.py:1325`), который читает
`funding_rate_history`, а не скаляр `self.funding_rate`. Скаляр
`LiveFeatureState.funding_rate` читается ровно в одном месте —
`on_bar:714` для `breaker.check(current_funding=...)`. То есть признак
`funding_rate` формируется **не** тем значением, которое обновляет
`update_funding`.
**Как проявляется:** предсказанная (predicted) ставка из потока
mark-price доходит только до circuit breaker'а; признак модели берётся
из истории расчётов. Само по себе это корректное разделение
predicted/settled — неверен докстринг, который направит следующего
читателя не туда.
**Кто ещё это читает:** `LiveFeatureState.funding_rate` — только
breaker; `funding_rate_history` — только `get_funding_df`.
**Как установлено:** чтением.
**Уверенность:** доказано.

### A2-072 `bars_seen` в heartbeat не читает никто
**Севирити:** LOW
**Тип:** мёртвый код
**Где:** `src/execution/heartbeat.py:160` (пишется), читателей нет
**Что в коде:**
```
                    "bars_seen": self._bars_seen,
```
**В чём дефект:** `watchdog._check_heartbeat_detailed` читает
`process_ts`, `started_ts`, `last_bar_ts`;
`check_signal_freshness` читает `last_signal_ts` и `started_ts`.
`bars_seen` не читается ни там, ни там. Замер:
```
$ grep -rn "bars_seen" --include='*.py' src scripts | grep -v heartbeat.py
src/execution/reconciler_signals.py:304:   "bars_seen": len(bars)}
```
— единственное совпадение относится к другой структуре.
**Как проявляется:** лишнее поле в payload; здоровье потока данных
измеряется по `last_bar_ts`, что достаточно.
**Кто ещё это читает:** никто.
**Как установлено:** замером (grep).
**Уверенность:** доказано.

### A2-073 `cost_model.calculate_funding_cost` принимает нотионал под именем `position_size`
**Севирити:** LOW
**Тип:** семантика
**Где:** `src/execution/cost_model.py:78-92`, вызов `:114-116`
**Что в коде:**
```
    def calculate_funding_cost(
        self,
        position_size: float,
        funding_rate: float,
```
и вызов:
```
        funding_cost = self.calculate_funding_cost(
            position_size=notional,
```
**В чём дефект:** во всей остальной системе `position_size` — контракты
базового актива (§1.9, подтверждено данными БД). Здесь под тем же именем
ожидается нотионал в USDT, что видно только из докстринга («Return
funding cost in USDT») и из строки вызова.
**Как проявляется:** сегодня оба вызывающих (`cost_model.py:115`,
`scripts/validate_cost_model.py:55`, `backtest_runner.py:373`) передают
нотионал — корректно. Вызывающий, взявший `decision.position_size` из
`RiskDecision`, получит funding, заниженный в `entry_price` раз
(для BTC ≈ 67 000×).
**Кто ещё это читает:** `estimate_total_cost` — единственный внутренний
потребитель, компенсирует.
**Как установлено:** чтением + сверкой с замером единиц из БД.
**Уверенность:** доказано.

### Сводка находок прохода 6

| ID | Имя | Севирити | Тип |
|---|---|---|---|
| A2-059 | `get_metrics_df` подаёт 100 записей и 2 колонки из 4 | HIGH | логика / семантика |
| A2-060 | История funding в live не пополняется (часы 01/09/17 против 00/08/16) | CRITICAL | логика |
| A2-061 | Фаза амортизации Hurst зависит от длины кадра | MEDIUM | математика |
| A2-062 | `PortfolioTracker` стирает флаг circuit breaker'а | HIGH | логика / архитектура |
| A2-063 | 4H-watchdog нацелен на testnet, бот — на mainnet | HIGH | архитектура / практика |
| A2-064 | Метка режима и `atr_percentile` считаются разными путями | MEDIUM | математика / архитектура |
| A2-065 | 28 % баров идут в модель, не обучавшуюся на этом режиме | MEDIUM | логика / семантика |
| A2-066 | `_preload_historical_bars` зашит на BTCUSDT | MEDIUM | логика / недоделка |
| A2-067 | `atr_percentile` печатается под именем `atr_pct` | LOW | семантика |
| A2-068 | Семь риск-критичных значений `.env` не читаются никем | MEDIUM | архитектура / практика |
| A2-069 | Формат `SYMBOLS` в `.env` даёт `symbol_encoded = −1` | MEDIUM | семантика |
| A2-070 | Дозапись funding не дедуплицирована, сработала бы 60 раз | LOW | логика |
| A2-071 | `update_funding` не влияет на признак; докстринг лжёт | LOW | семантика / мёртвый код |
| A2-072 | `bars_seen` в heartbeat без читателя | LOW | мёртвый код |
| A2-073 | `calculate_funding_cost` принимает нотионал под именем `position_size` | LOW | семантика |

CRITICAL 1, HIGH 3, MEDIUM 6, LOW 5. Всего 15.

### Статус находок, унаследованных этим проходом

| Находка | Статус после прохода 6 |
|---|---|
| A3 (35 % расхождения regime-лейбла) | **закрыта**: 0.2–1.0 % (§3.3). На её месте — A2-064 (другой механизм, меньший масштаб) |
| A2-008 (шесть копий символ→код) | жива; A2-069 показывает, чем именно она опасна |
| A2-009 (исключение `symbol_encoded` не действует) | подтверждена косвенно: признак присутствует в `feature_columns` бандла |
| A2-013 (в `--dry-run` у журнала сигналов нет производителя) | подтверждена: `_open_position` не вызывается (`ml_strategy.py:841-842`), значит `_emit_signal` недостижим; плюс §1.5 — весь риск-контур инертен |
| A2-021 (`p(UP) < 0.5` во всех режимах) | согласуется с БД: `regime='trend_down'` + `direction='long'` (id 7), `regime='trend_up'` + `direction='short'` (id 5) |
| A2-022 (две модели на разных событиях делят порог) | не закрыта; §1.2 добавляет: заявленный `signal_rate: 0.629` посчитан при пороге 0.55, а живёт бот с 0.65 |
| A2-003 (пять определений путей к БД) | подтверждена; §5.3 добавляет шестое обстоятельство: `ATOMICORTEX_DB_PATHS` нет ни в `Settings`, ни в `.env.example` |
| A2-010 (настройки API мимо `Settings`) | подтверждена; `ATOMICORTEX_API_KEY` отсутствует в `.env.example` |

---

## 8. ПРЯМОЙ ОТВЕТ

**Сколько признаков модель получает в live не такими, какими училась.**

Из 46 признаков прод-бандла `trend_model_v3.pkl`, при допуске
относительной ошибки 1e-6:

| Состояние процесса | Расходится | Доля |
|---|---|---|
| Нижняя граница — идеальные входные данные (только алгоритмическое расхождение) | **3** | 6.5 % |
| Первые часы после старта | **7** | 15.2 % |
| После ~8 ч работы | **8** | 17.4 % |
| **Рабочий режим (аптайм ≥ 7 суток, `taker_buy_volume` в порядке)** | **14** | **30.4 %** |
| Тот же режим при сбое REST-запроса `taker_buy_volume` | **21** | 45.7 % |

Рабочая цифра — **14 из 46**. Из них 11 расходятся порядково (значение
либо константный ноль вместо величины, либо кратно другое), 3 — на
1.5 %.

Поимённо (рабочий режим): `funding_zscore_30d`, `oi_zscore`,
`funding_zscore_7d`, `oi_delta_12h`, `ls_ratio`, `ls_ratio_zscore`,
`taker_vol_ratio`, `funding_abs`, `funding_rate`, `funding_cum_24h`,
`basis_approx`, `regime_confidence`, `trend_strength`, `hurst`.

Ни одно из этих значений не является NaN и ни одно не выходит за
диапазон, встречающийся в обучающей матрице: модель не может их
отличить от измерений.

---

## НЕ ИССЛЕДОВАНО (передаётся дальше)

1. **Округление `position_size` до шага инструмента.** `RiskEngine`
   отдаёт float полной точности; округление делает Nautilus при
   построении `Quantity`. Значит `signals_log.position_size` и
   количество в ордере могут расходиться до величины лота (для
   BTCUSDT-PERP шаг 0.001 → до 1.5 % на типичном размере 0.0645).
   Подтвердить можно только на живом ордере; в `--dry-run` ордера нет.
   Проход 7/8.
2. **Смешение форматов `created_at` в `signals_log`.** Сегодня в БД
   один формат (ISO-8601 с `T`), потому что пишет только
   `SignalBridge`. `telegram_bot/database.add_signal:360-364` не
   перечисляет `created_at` и при использовании даст формат
   `CURRENT_TIMESTAMP` (пробел). Тогда `MAX(created_at)` в
   `check_signal_freshness.py:384` — лексикографический — всегда вернёт
   строку с `T`, независимо от фактического времени. Кто и когда
   вызывает `telegram_bot/database.add_signal` — не прослежено.
3. **Расхождение схем `signals_log` между двумя `CREATE TABLE`.**
   Установлено, что схемы различаются шестью колонками и что побеждает
   тот, кто создал таблицу первым. Не установлено, какой порядок запуска
   гарантирован systemd-зависимостями: `atomicortex-telegram.service` не
   прочитан на предмет `After=`/`Requires=`. Проход 8.
4. **1H- и 15m-контуры.** Разобраны только там, где они пересекаются с
   4H (heartbeat-ключи, venue watchdog'ов, `bar_open_time_ms`). Паритет
   `build()` / `build_from_buffer()` для `interval='15m'` и `'1h'` не
   замерен: там участвуют `session_features`, `orb_features`,
   `mtf_context` и `alpha_v2` — 80+ дополнительных признаков, и
   `df_htf_4h` / `df_htf_1h` в live подаются частично (A3). Отдельный
   замер.
5. **`build_from_buffer` для 15m: `bar_duration_minutes`.**
   `add_funding_features` при вызове из скриптов сборки датасетов
   по умолчанию берёт 240 минут даже для 15m (докстринг
   `derivatives.py:84-88` сам это признаёт: «Offline build scripts that
   don't go through FeaturePipeline still default to 240 — known
   follow-up»). Разница окон offline/live для 15m не замерена.
6. **Частота `BinanceFuturesMarkPriceUpdate`.** A2-070 опирается на
   докстринг («streams … every second»), не на замер: бот не
   запускался. Подтвердить можно только на живом потоке.
7. **Фактическая конфигурация на VM.** A2-063 установлена по файлам
   `deploy/`. Что реально в `/etc/systemd/system` на VM и что в
   тамошнем `.env` — не проверялось (протокол: на VM не ходить).
8. **Признаки `live_enrichment`** (`feature_pipeline.py:112-136`:
   `liq_*`, `vpin`, `basis_bps`, `oi_velocity`, `fear_greed_*`,
   `sentiment_score` — 20 имён). В `FEATURE_GROUPS` объявлены, в
   `feature_columns` прод-бандла отсутствуют, в `ml_features`
   отсутствуют. Кто и куда их пишет — не прослежено. Пересекается с
   мёртвым кодом прохода 1.
