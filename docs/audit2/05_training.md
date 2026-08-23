# АУДИТ-2 / ПРОХОД 4 — Обучение

HEAD: `f4af5fd210b32db8af4478f0e2f440e2eb504ccc`. Протокол — [`00_method.md`](00_method.md).
Ни одной правки в `src/ scripts/ tests/ deploy/`. В `data/features/models/` ничего
не записано. Обучение — только на синтетике, артефакты в `/tmp/a2p4/`.

**Прямой ответ на вопрос прохода — §9.**

---

## 0. ЧТО ИЗМЕРЕНО

| № | Замер | Раздел |
|---|---|---|
| M1 | Кто пишет `{regime}_model.pkl`: grep по `model_suffix` | §1 |
| M2 | Манифесты четырёх бандлов: `best_iteration`, `git_commit`, `use_mtf_params`, whitelist | §1.2, §3.4, §7 |
| M3 | Сверка `created_at` (UTC) / mtime (IST) / даты коммитов — **исправляет таймлайн A2-023** | §1.2 |
| S1 | **Контроль стенда**: синтетика с известным сигналом → AUC 0.99 / 0.93 | §3.3 |
| S2 | Разделение гипотез R-1: сигнал / приор / веса | §3.3 |
| S3 | Слабый сигнал + сдвиг приора | §3.3 |
| M4 | Пересечение train/test в `feature_selection_v3._split` на реальных данных | §5.1 |

---

## 1. КАКОЙ КОД ОБУЧАЕТ ПРОД-МОДЕЛИ

### 1.1 Кто производит `{regime}_model.pkl` без суффикса

Имя строится в одном месте — `src/models/lgbm_trainer.py:850-853`:

```
        # model_suffix lets v3 retrains coexist with production weights
        # (empty string → legacy "{regime}_model.pkl"; "_v3" → "_v3.pkl").
```
```
            f"{self.config.regime}_model{self.config.model_suffix}.pkl"
```

Умолчание — `lgbm_trainer.py:243`:

```
    model_suffix: str = ""
```

Полный перечень мест, задающих суффикс:

```
$ grep -rn --include='*.py' 'model_suffix' src scripts
src/models/lgbm_trainer.py:243:    model_suffix: str = ""
src/models/lgbm_trainer.py:848-851:  (формирование имени)
scripts/retrain_v3.py:90:    model_suffix: str = "_v3",
scripts/retrain_v3.py:100:        model_suffix=model_suffix,
scripts/retrain_v3.py:111:    model_suffix: str,
scripts/retrain_v3.py:136:                model_suffix=model_suffix,
scripts/retrain_v3.py:401:            feature_whitelist=whitelist, model_suffix=args.model_suffix,
```

Скрипты 1H/15m строят имя мимо `save_bundle` (`train_1h_models.py:148`:
`filename=f"{regime}_model_1h.pkl"`). `feature_selection_v3.py:246` **читает**
`{regime}_model_v3.pkl`, не пишет.

**Единственные производители имени без суффикса:**

1. `scripts/train_models.py` → `TrainingPipeline.run` → `ModelConfig(...)`
   (`src/models/training_pipeline.py:66`) → `trainer.save_bundle(...)` (`:83`).
   `model_suffix` не задаётся → `""`.
2. `scripts/tune_models.py` — по докстрингу (`:425`):
   ```
        The final production model is saved to ``models_dir/{regime}_model.pkl``.
   ```

### 1.2 Кто писал текущие четыре бандла — замер

Ни один из них. Все четыре несут суффикс `_v3` и лежат в подкаталоге `v3/`:

```
bundle                             best_it  trees   mtf  uniq    wl nfeat        git      lgb       py  sv
v3/_grid/high_vol_model_v3.pkl           2      2  True  True  None    46    e67a451    4.3.0   3.11.9   1
v3/_grid/trend_model_v3.pkl              5      5  True  True  None    46    9d8bfbc    4.3.0   3.11.9   1
v3/high_vol_model_v3.pkl               145    145  True  True  None    46    e67a451    4.3.0   3.11.9   1
v3/trend_model_v3.pkl                  196    196  True  True  None    46    e67a451    4.3.0   3.11.9   1
```

`use_uniqueness_weights: True` + `_v3` + `use_mtf_params: True` — это подпись
`scripts/retrain_v3.py` (`:90`, `:97-100`, `:140`). **Производитель установлен:
`retrain_v3.py`.**

`train_models.py` и `tune_models.py` — единственные, кто мог бы создать имена,
которые ищет прод, — на диске следов не оставили: файлов `trend_model.pkl` /
`high_vol_model.pkl` в `data/features/models/` **нет** (проход 1, §4.1).

### M3. Исправление таймлайна из прохода 2

Проход 2 (A2-023) сопоставил `manifest.created_at` с датами коммитов и заключил,
что бандлы созданы до трёх исправлений. Сопоставление было неверным:
`created_at` записан в **UTC**, а `git log` печатал **IST (+05:30)**. Свежий
замер:

```
v3/_grid/high_vol_model_v3.pkl   created_at=2026-08-14T17:09:40  mtime=2026-08-14 22:39:40  git=e67a451
v3/_grid/trend_model_v3.pkl      created_at=2026-08-14T18:07:36  mtime=2026-08-14 23:37:36  git=9d8bfbc
v3/high_vol_model_v3.pkl         created_at=2026-08-14T17:09:41  mtime=2026-08-14 22:39:41  git=e67a451
v3/trend_model_v3.pkl            created_at=2026-08-14T17:09:29  mtime=2026-08-14 22:39:29  git=e67a451
```
```
e67a451: committed 2026-08-14 22:26:48 +0530 | fix(features): trim the offline warmup head by the detector's own window …
9d8bfbc: committed 2026-08-14 23:37:11 +0530 | fix(trainer): split train/test on one wall-clock boundary and embargo by time so test never overlaps train
```

**Корректный таймлайн (IST):**

```
19:24:45  b6cadf9  stamp a provenance manifest … move the disk write behind evaluate
20:48:09  2eafa92  refuse to write a model that fails the go-live thresholds
22:26:48  e67a451  trim the offline warmup head …
22:39:29  v3/trend_model_v3.pkl        (git=e67a451 ✓)
22:39:40  v3/_grid/high_vol_model_v3.pkl (git=e67a451 ✓)
22:39:41  v3/high_vol_model_v3.pkl     (git=e67a451 ✓)
23:37:11  9d8bfbc  … so test never overlaps train
23:37:36  v3/_grid/trend_model_v3.pkl  (git=9d8bfbc ✓, через 25 секунд после коммита)
```

Поле `git_commit` в каждом манифесте согласуется с его временем — провенанс
внутренне непротиворечив.

**Что из A2-023 подтверждается:** три бандла с пересечением train/test записаны
**до** коммита `9d8bfbc`, который это пересечение и устранял — разрыв
58 минут, а не часы. Единственный бандл без пересечения (`_grid/trend`) записан
через 25 секунд после того коммита. Причинная связь подтверждена точнее, чем в
проходе 2.

**Что из A2-023 не подтверждается:** утверждение, что бандлы созданы до
`b6cadf9` и `2eafa92`. Оба коммита предшествуют всем четырём файлам — поэтому
манифест вообще существует, а поле `written_despite_failing` присутствует и
осмысленно. Формулировка прохода 2 «записаны кодом, предшествующим двум
исправлениям того же дня» **неверна**; верно — «предшествующим одному».

### A2-031 Прод-путь и производитель бандлов не связаны ни по имени, ни по каталогу

**Севирити:** CRITICAL
**Тип:** архитектура

**Где:** `src/execution/strategies/ml_strategy.py:105`, `:1721`;
`scripts/retrain_v3.py:90`; `data/features/models/`

**Что в коде.** Прод ищет (`ml_strategy.py:105`, `:1719-1721`):

```
    models_dir: str = "./data/features/models"
```
```
        models_dir = Path(self._config.models_dir)
        for regime in ["trend", "high_vol"]:
            path = models_dir / f"{regime}_model.pkl"
```

Обучение v3 пишет (`retrain_v3.py:90`, `lgbm_trainer.py:851`):

```
    model_suffix: str = "_v3",
```
```
            f"{self.config.regime}_model{self.config.model_suffix}.pkl"
```

**В чём дефект:** производитель артефактов и потребитель используют разные
имена (`_v3` против пустого суффикса) и разные каталоги
(`data/features/models/v3/` против `data/features/models/`). Между ними нет ни
символьной ссылки, ни шага копирования, ни проверки. Единственные скрипты,
способные создать ожидаемое прод-именем, — `train_models.py` и
`tune_models.py`, — не оставили на диске ни одного файла и, по §6.2 ниже,
обучают на другой целевой переменной.

**Как проявляется:** `_load_models` при отсутствии файла только предупреждает
(`ml_strategy.py:1736`):

```
                self.log.warning(f"Model not found: {path}")
```

и оставляет `self._trend_model = None`. Стратегия стартует, подписывается на
бары и работает без модели — без ошибки, без остановки, без алерта.
Комментарий `lgbm_trainer.py:848-849` («so v3 retrains never overwrite
production weights») показывает, что разделение имён было намеренным; шаг
продвижения артефакта в прод при этом не написан.

**Кто ещё это читает:** `MetaSignalGate` (`meta_strategy.py:183`) ссылается на
третье имя — `./data/features/models/v3/meta_model_v3.pkl`, файла нет
(проход 1, A2-002). `validate_1h_models.py:777` строит четвёртое —
`{regime}_model_1h.pkl`. Итого четыре конвенции имён на один артефакт.

**Как установлено:** замером (M1, M2, листинг каталога из прохода 1) и чтением.
**Уверенность:** доказано.

### 1.3 Прямая формулировка

Производитель прод-имён установлен: `scripts/train_models.py` (через
`TrainingPipeline`) и `scripts/tune_models.py`. **Ни один из них не создавал
текущие бандлы**, и результата их работы на диске нет. Файлы, которые
существуют, созданы `scripts/retrain_v3.py` и лежат под именами, которых
прод-путь не ищет.

---

## 2. СПЛИТ

### 2.1 `prepare_data` — порядок операций

Дословно из докстринга (`src/models/lgbm_trainer.py:380-386`):

```
        3. Compute **one** OOS boundary on that combined frame, *before*
           the regime filter, via ``compute_default_oos_start_ms`` with
           ``oos_fraction = config.test_size_pct``.
        4. Apply the regime filter to the combined frame.
        5. Skip symbols with nothing on one side of the boundary (warning,
           not an exception); warn about a thin test side.
        6. Split with ``temporal_split_multi``, embargoing one label
           horizon off the tail of every symbol's train part.
```

Реализация — сначала разметка по каждому символу (`:424-437`), затем
конкатенация (`:443`), затем граница:

```
        oos_start_ms = compute_default_oos_start_ms(
            labelled,
            time_col="open_time",
            oos_fraction=self.config.test_size_pct,
        )
```

**По времени, не по строкам.** `compute_default_oos_start_ms`
(`temporal_split.py:48-52`):

```
    total_duration = int(t_max) - int(t_min)
    return int(t_min) + int(total_duration * (1.0 - oos_fraction))
```

### 2.2 Порядок «фильтр → сплит» — A13

**A13 закрыта.** Граница считается **до** режимного фильтра
(`lgbm_trainer.py:444-460`, фильтр — на `:462-465`). Докстринг прямо описывает
и старый дефект, и почему он был дефектом (`lgbm_trainer.py:388-397`):

```
        It used to be a per-symbol ``head(80%) / tail(20%)`` cut **by rows**,
        taken after the regime filter. Symbols survive the filter in
        different numbers, so each symbol's cut landed on a different date
        and the concatenated train frame overlapped test by weeks — WR / PF
        were scored on rows the booster had already fitted. Cutting once,
        on the combined frame, fixes that; cutting *before* the filter also
        keeps the OOS window identical across regimes, so their metrics stay
        comparable.
```

**Но исправление применено не везде.** Копия дефектного сплита живёт в
`scripts/feature_selection_v3.py` — см. A2-032 (§5.1), где расхождение измерено.

### 2.3 val-окно

`train()` (`lgbm_trainer.py:674-700`):

```
        val_frac = 0.85 if self.use_mtf_params else 0.90
        val_split = int(len(X_train) * val_frac)
```
```
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train_fit = X_train[:fit_end]
```

Хвост train-кадра по индексу строк. Это корректно **только потому**, что
`temporal_split_multi` возвращает глобально отсортированный по времени кадр
(`temporal_split.py:91-92`):

```
        train = pl.concat(train_parts, how="diagonal").sort(time_col)
        test = pl.concat(test_parts, how="diagonal").sort(time_col)
```

Без этой сортировки «последние 10% строк» кадра `[BTC][ETH][SOL]` были бы
хвостом одного лишь SOLUSDT. Сортировка есть; проверено чтением
`temporal_split.py:7-8` («The MULTI dataset is a per-symbol concatenation
`[all BTC][all ETH][all SOL]` — it is NOT globally time-sorted») — комментарий
описывает вход, а не выход `temporal_split_multi`.

**Перекрытие train↔val по меткам закрыто эмбарго** — `_embargo_fit_end`
(`lgbm_trainer.py:617-631`):

```
        open_times = train_df["open_time"].to_numpy()
        val_start_ms = int(open_times[val_split:].min())
        cutoff_ms = val_start_ms - horizon_bars * bar_ms
```
```
        violations = np.flatnonzero(open_times[:val_split] >= cutoff_ms)
        fit_end = int(violations[0]) if violations.size else val_split
        return max(fit_end, floor_fit_end)
```

Wall-clock, не строки — корректно для мультисимвольного кадра.

**Оговорка, не оформленная как находка:** `floor_fit_end = val_split -
max(0, val_split // 2)` (`:600`) молча ограничивает эмбарго половиной train.
Если требуемый разрыв превысит эту границу, эмбарго окажется недостаточным
**без единой строки лога** — `max(fit_end, floor_fit_end)` не предупреждает.
На 4H при `h=6` требуется ≈18 строк из ≈2900, порог не достигается. Дефект
латентен.

### 2.4 Фактические границы

Из манифестов (замер прохода 2, §8) — что реально попало в бандлы:

| Бандл | train | test | n_train | n_test | embargo_rows | after_embargo |
|---|---|---|---|---|---|---|
| `v3/trend_model_v3.pkl` | 2024-05-03 → 2025-10-02 | 2025-08-11 → 2025-12-23 | 3223 | 806 | 6 | 2733 |
| `v3/high_vol_model_v3.pkl` | 2024-05-23 → 2025-10-14 | 2025-08-17 → 2025-11-21 | 684 | 172 | 4 | 577 |
| `v3/_grid/high_vol_model_v3.pkl` | 2024-05-23 → 2025-10-12 | 2025-08-18 → 2025-11-21 | 1037 | 260 | 8 | 873 |
| `v3/_grid/trend_model_v3.pkl` | 2024-05-03 → 2025-08-28 | 2025-08-31 → 2025-12-23 | 2963 | 23 | 23 | — |

`v3/high_vol_model_v3.pkl` обучен на **684 строках** при 46 фичах — отношение
14.9 наблюдений на признак. Тест — 172 строки.

---

## 3. КРИТЕРИЙ ОСТАНОВКИ — R-1 и R-3

### 3.1 Где задаётся

`lgbm_trainer.py:745-751`:

```
        raw_params = MTF_LGBM_PARAMS if self.use_mtf_params else self.config.lgbm_params
        # n_estimators / early_stopping_rounds are not lgb.train params:
        # the former maps to num_boost_round, the latter to a callback.
        params = {
            k: v for k, v in raw_params.items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        num_rounds = raw_params.get("n_estimators", 200)
        stopping_rounds = raw_params.get("early_stopping_rounds", 50)
```

Валидационное множество — `val_data`, построенное из хвоста **train**
(`:781-786`):

```
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
```

MTF-профиль: `n_estimators: 2000` (`:179`), `learning_rate: 0.02` (`:180`).
`early_stopping_rounds` в `MTF_LGBM_PARAMS` не задан → фолбэк 50.

### 3.2 Используется ли тест для выбора числа итераций — ФАКТ

**Нет.** Полный перечень мест, где формируется набор валидации для остановки:

```
$ grep -rn 'valid_sets\|eval_set' src scripts
src/models/lgbm_trainer.py:784:            valid_sets=[train_data, val_data],
scripts/train_meta_model.py:185:        valid_sets=[train_data, val_data],
```

Оба раза — `train_data` и `val_data`, где `val_data` берётся из train-кадра
(`lgbm_trainer.py:693-694`; `train_meta_model.py:173-174`). Тестовый кадр в
`train()` не передаётся вовсе — сигнатура `train(self, train_df)`.

**R-3 не подтверждается.** Утечки через выбор числа итераций нет.

Побочно: включение `train_data` в `valid_sets` — не дефект (train-лосс
монотонно убывает и остановку не инициирует), но делает `best_iteration`
зависимым от того, какой набор сработал первым. На практике это всегда `val`.

### 3.3 R-1: почему val-loss минимален на итерации 1–5

**Факт из манифестов** (M2):

| Бандл | `best_iteration` | `num_trees` |
|---|---|---|
| `v3/_grid/high_vol_model_v3.pkl` | **2** | 2 |
| `v3/_grid/trend_model_v3.pkl` | **5** | 5 |
| `v3/high_vol_model_v3.pkl` | 145 | 145 |
| `v3/trend_model_v3.pkl` | 196 | 196 |

Симптом R-1 относится к двум `_grid`-бандлам, не ко всем четырём.

**Стенд.** Синтетический кадр: 3 символа × 1200 баров, отсортирован по времени,
45 признаков (1 «подсказка» + 44 шумовых), проход через **реальный**
`LGBMTrainer.train()` с `use_mtf_params=True` — тот же путь, что у v3.

**S1 — контроль стенда (обязателен по заданию):**

```
S1. КОНТРОЛЬ СТЕНДА: фича = метка + шум. Ожидание: AUC > 0.9, best_iter >> 5
сильный сигнал (signal=2.0), приор одинаковый 0.50/0.50 best_iter= 377  trees= 377  AUC(fit)=1.0000  AUC(val)=0.9948
сигнал средний (signal=1.0), приор одинаковый 0.50/0.50 best_iter= 277  trees= 277  AUC(fit)=0.9778  AUC(val)=0.9279
```

**Стенд исправен:** явный сигнал находится (AUC 0.99 и 0.93 > 0.9), остановка
происходит на сотнях итераций. Значит отрицательные результаты ниже относятся
к данным, а не к измерительному стенду.

**S2 — разделение гипотез, сигнал удалён полностью:**

```
S2. РАЗДЕЛЕНИЕ ГИПОТЕЗ R-1 (сигнала нет: signal=0.0)
нет сигнала, приор одинаковый     0.50 / 0.50        best_iter=   2  trees=   2  AUC(fit)=0.6705  AUC(val)=0.4945
нет сигнала, приор val сдвинут    0.50 / 0.42        best_iter=   6  trees=   6  AUC(fit)=0.7035  AUC(val)=0.5239
нет сигнала, приор как в проде    0.437 / 0.52       best_iter=   2  trees=   2  AUC(fit)=0.6536  AUC(val)=0.5016
нет сигнала, приор как в проде + uniq-веса           best_iter=  26  trees=  26  AUC(fit)=0.8387  AUC(val)=0.5126
```

**S3 — слабый, но ненулевой сигнал:**

```
S3. СЛАБЫЙ СИГНАЛ + СДВИГ ПРИОРА (ближе к реальности)
слабый сигнал 0.15, приор 0.437 / 0.437              best_iter=  65  trees=  65  AUC(fit)=0.8540  AUC(val)=0.5738
слабый сигнал 0.15, приор 0.437 / 0.52               best_iter=  70  trees=  70  AUC(fit)=0.8461  AUC(val)=0.5678
слабый сигнал 0.15, приор 0.437 / 0.36               best_iter=  98  trees=  98  AUC(fit)=0.8939  AUC(val)=0.5792
```

**Вывод — гипотезы разделены:**

| Гипотеза | Вердикт |
|---|---|
| Дефект разметки | **не подтверждается**: тот же путь разметки на синтетике с сигналом даёт `best_iter` 277–377 |
| Дефект весов (balanced) | **не подтверждается**: сдвиг приора 0.437/0.52 меняет `best_iter` с 2 на 2, сдвиг 0.50/0.42 — с 2 на 6 |
| Uniqueness-веса | **не причина коллапса**: они его смягчают (2 → 26), но val-AUC не растёт (0.5016 → 0.5126) |
| Хвост val / остаточная утечка | **не подтверждается**: утечка повышала бы val-AUC; она 0.49–0.52 |
| **Отсутствие обучаемого сигнала** | **подтверждается**: `best_iter` ∈ [2, 6] воспроизводится ровно при `signal = 0` |

`best_iteration` = 2 и 5 у `_grid`-бандлов численно совпадает с синтетикой при
нулевом сигнале (2 и 6). `best_iteration` = 145 и 196 у основных бандлов
соответствует режиму S3 («слабый сигнал»), где val-AUC 0.57 — того же порядка,
что заявленный walk-forward AUC 0.5375.

**R-1 объяснён и дефектом не является.** Ранняя остановка ведёт себя корректно:
она сообщает, что при этих барьерах и этом наборе признаков обучать нечему.
Диагноз «модель ничего не выучила» верен; поиск скрытого дефекта в разметке или
весах на этих данных исчерпан.

### A2-034 Модель из двух деревьев записана на диск как результат прогона

**Севирити:** MEDIUM
**Тип:** недоделка

**Где:** `data/features/models/v3/_grid/*.pkl` (`best_iteration` 2 и 5,
`written_despite_failing: true`)

**Что измерено:** M2 (таблица §1.2) и манифесты из прохода 2 (§8): оба
`_grid`-бандла несут `passes: false`, `written_despite_failing: true`,
`eval.win_rate: 0.0`, `eval.profit_factor: 0.0`.

**В чём дефект:** бустер из 2 деревьев при `learning_rate = 0.02` отличается от
константного априора на величину порядка 0.04 логита. Это не модель, это
артефакт неудавшегося прогона. Он записан в тот же каталог, тем же именем и в
том же формате, что и рабочие бандлы; отличается только вложенностью в `_grid/`.

**Как проявляется:** каталог `data/features/models/v3/` и его подкаталог
`_grid/` содержат по файлу с идентичными именами
(`trend_model_v3.pkl`, `high_vol_model_v3.pkl`). Различить рабочий бандл от
отброшенного можно только прочитав манифест. Ни один загрузчик манифест не
читает (`grep manifest src/execution/strategies/ml_strategy.py` — пусто,
проход 2, A2-022).

**Как установлено:** замером (M2).
**Уверенность:** доказано.

### 3.4 Фактическое число итераций в манифесте — D-1

**Записывается. D-1 закрыта.** `lgbm_trainer.py:1031-1032`:

```
            "best_iteration": int(booster.best_iteration),
            "num_trees": int(booster.num_trees()),
```

Замер подтверждает наличие поля во всех четырёх бандлах (M2).

---

## 4. ВЕСА И БАЛАНС

### 4.1 `scale_pos_weight` / `class_weight`

**Не задаются.** (Проход 3, §3.3; подтверждено повторно.) Единственный
механизм — `lgbm_trainer.py:701-703`:

```
        # Balanced sample weights — upweight minority classes (UP/DOWN)
        train_weights = compute_sample_weight("balanced", y_train_fit)
        val_weights = compute_sample_weight("balanced", y_val)
```

Считаются из фактического распределения классов **раздельно** для `y_train_fit`
и `y_val`.

### 4.2 Uniqueness-веса: путь в `lgb.Dataset`

`lgbm_trainer.py:710-719`:

```
        if self.config.use_uniqueness_weights:
            uniq_all = self._builder.compute_uniqueness_weights_by_symbol(
                train_df, max_holding=self.config.barrier_max_holding,
            )
            # Slice uniqueness weights to match the embargoed train_fit
            # length so train_weights * uniq_fit stays element-aligned.
            uniq_fit = uniq_all[:fit_end]
            uniq_val = uniq_all[val_split:]
            train_weights = train_weights * uniq_fit
            val_weights = val_weights * uniq_val
```

и далее `:734-741`:

```
        train_data = lgb.Dataset(
            X_train_fit, label=y_train_fit,
            weight=train_weights, feature_name=feature_names,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            weight=val_weights, feature_name=feature_names,
            reference=train_data,
        )
```

**Ответы на §4.2:**
- в `lgb.Dataset` попадают как произведение `balanced × uniqueness`;
- **применяются к валидации** (`val_weights`, строка 739);
- следовательно **участвуют в early stopping**, поскольку val-метрика взвешена.

Выравнивание срезов корректно: `_prepare_xy` не удаляет строк (проверено
чтением `lgbm_trainer.py:1326-1394` — ни одного `filter`/`drop_nulls`), поэтому
`uniq_all` той же длины, что `X_train`.

Нормировка весов на среднее 1 (A2-016, проход 2) действует и здесь: она
уничтожает поправку на ESS **и в обучении, и в валидации**, то есть early
stopping тоже видит завышенный эффективный объём.

### 4.3 R-4: нестационарный приор

**Пересчёта по окнам нет.** `compute_sample_weight("balanced", ...)`
вызывается один раз на прогон, для `y_train_fit` целиком и для `y_val` целиком.
Внутри train-части приор считается постоянным.

Замер прохода 2 (M2) показывает, насколько это неверно: `p(UP|бинарно)` по
режимам BTCUSDT при прод-конфиге варьирует 0.4151 … 0.4897 — то есть между
режимами разброс 7.5 п.п. Манифест `v3/trend_model_v3.pkl` фиксирует
`class_balance: 0.4372`, `v3/high_vol_model_v3.pkl` — `0.4678`.

Смягчающее обстоятельство: раздельный расчёт для `y_train_fit` и `y_val`
означает, что **между** fit и val приор всё-таки учитывается. Замер S2
показывает, что вклад этого фактора в поведение ранней остановки мал
(`best_iter` 2 против 6 при сдвиге приора на 8 п.п.).

**R-4 подтверждается как факт (пересчёта нет), но замером показано, что на
наблюдаемый симптом R-1 он почти не влияет.**

---

## 5. ОТБОР ФИЧ

### 5.1 `feature_selection_v3` — метод и срез

Метод — Clustered-MDA (докстринг `:5`: «Block 3 / Step 1 — Clustered-MDA
feature selection for v3 models»), перестановочная важность по кластерам
коррелированных признаков (`:143-155`).

Срез — `scripts/feature_selection_v3.py:175-198`:

```
    """Replicate LGBMTrainer.prepare_data: per-symbol triple-barrier
    target → regime filter → 80/20 temporal split → concat."""
    train_parts, test_parts = [], []
    for sym in symbols:
        df = builder.load_and_combine(features_dir, symbols=[sym])
```
```
        df = builder.create_target_triple_barrier(
            df, pt_multiplier=pt, sl_multiplier=sl, max_holding=hold,
        )
        if regime == "trend":
            df = df.filter(pl.col("regime").is_in(["trend_up", "trend_down"]))
```
```
        n = len(df)
        n_tr = int(n * (1.0 - test_pct))
        train_parts.append(df.head(n_tr))
        test_parts.append(df.tail(n - n_tr))
    return pl.concat(train_parts, how="diagonal"), pl.concat(test_parts, how="diagonal")
```

### A2-032 Отбор фич выполняется на сплите, дефект которого исправлен в тренере и скопирован сюда

**Севирити:** HIGH
**Тип:** математика

**Где:** `scripts/feature_selection_v3.py:175-198`

**Что в коде:** цитата выше. Докстринг заявляет «Replicate
`LGBMTrainer.prepare_data`», но воспроизводит его **прежнюю**, дефектную
версию: пер-символьный `head(80%)/tail(20%)` **по строкам**, взятый **после**
режимного фильтра.

Ровно эту конструкцию `lgbm_trainer.py:388-393` описывает как устранённый
дефект:

```
        It used to be a per-symbol ``head(80%) / tail(20%)`` cut **by rows**,
        taken after the regime filter. Symbols survive the filter in
        different numbers, so each symbol's cut landed on a different date
        and the concatenated train frame overlapped test by weeks — WR / PF
        were scored on rows the booster had already fitted.
```

**Замер на реальных данных** (`data/features/ml_features/*.parquet`,
барьеры прод-конфига pt=1.0/sl=0.8/h=6):

```
  trend     BTCUSDT: n= 1986 train_end=2026-03-31  test_start=2026-03-31
  trend     ETHUSDT: n= 1798 train_end=2026-03-18  test_start=2026-03-18
  trend     SOLUSDT: n= 2093 train_end=2026-03-17  test_start=2026-03-17
  >> trend: конкат train_end=2026-03-31, конкат test_start=2026-03-17 -> ПЕРЕСЕЧЕНИЕ 13.3 сут

  high_vol  BTCUSDT: n=  620 train_end=2025-11-15  test_start=2025-11-15
  high_vol  ETHUSDT: n=  537 train_end=2025-08-28  test_start=2025-08-28
  high_vol  SOLUSDT: n=  574 train_end=2025-09-26  test_start=2025-09-26
  >> high_vol: конкат train_end=2025-11-15, конкат test_start=2025-08-28 -> ПЕРЕСЕЧЕНИЕ 79.3 сут
```

Каждый символ теряет разное число строк на режимном фильтре (620 / 537 / 574
для `high_vol`), поэтому его 80%-я отсечка падает на свою дату. После
конкатенации train содержит бары до 2025-11-15, а test начинается с 2025-08-28
— **пересечение 79.3 суток** для `high_vol` и 13.3 для `trend`.

**Как проявляется:** перестановочная важность, определяющая, какие признаки
попадут в whitelist, считается на выборке, где train и test пересекаются на
2.6 месяца. Признак, полезный только на пересечении, получает завышенную
важность.

**Кто ещё это читает:** whitelist из этого скрипта передаётся в
`retrain_v3.py:401` (`feature_whitelist=whitelist`) → `ModelConfig` →
`_prepare_xy` (`lgbm_trainer.py:1358-1364`). То есть дефект переносится из
стадии отбора в обученную модель.

**Как установлено:** замером (M4) и чтением.
**Уверенность:** доказано.

### 5.2 `selected_features_v3.json` против бандлов

**Файл отсутствует.**

```
$ find . -name 'selected_features*' -not -path './.venv/*'
(пустой вывод)
```

И whitelist не применялся ни к одному бандлу — все четыре несут
`feature_whitelist_size: None` и `n_features: 46` (M2):

```
bundle                             …    wl nfeat
v3/_grid/high_vol_model_v3.pkl     …  None    46
v3/_grid/trend_model_v3.pkl        …  None    46
v3/high_vol_model_v3.pkl           …  None    46
v3/trend_model_v3.pkl              …  None    46
```

Все четыре имеют один и тот же `feature_columns_hash`
`3429c552be803bcd5c52ef747c01e6cf1dbe65285d55257c71586b1c6db9b036` — набор
признаков идентичен.

### A2-033 Отбор фич никогда не применялся к артефактам на диске

**Севирити:** HIGH
**Тип:** недоделка

**Где:** `data/features/models/v3/**/*.pkl` (`feature_whitelist_size: None`);
`scripts/feature_selection_v3.py`; `scripts/retrain_v3.py:401`

**В чём дефект:** заявленное «trend: 23 признака, high_vol: 45» не
соответствует ни одному артефакту. На диске — 46 признаков во всех четырёх
бандлах, с идентичным хэшем набора, и `feature_whitelist_size: None`,
означающий, что `ModelConfig.feature_whitelist` был `None`. Файл, из которого
whitelist читался бы, в дереве отсутствует.

**Как проявляется:** Block 3 / Step 1 («Clustered-MDA feature selection»)
существует как скрипт, не оставивший следа ни в артефакте, ни в манифесте.
Комментарий `retrain_v3.py:68-69` («`--best-only` refits at these so the only
delta vs the 45-feature baseline is the whitelist») описывает сравнение,
которого на диске нет.

**Как установлено:** замером (M2, `find`).
**Уверенность:** доказано.

### 5.3 `symbol_encoded` при неизвестном символе

`lgbm_trainer.py:1369-1372` (обучение):

```
            symbol_encoded = (
                df["symbol"]
                .replace(SYMBOL_ENCODING, default=-1)
```

`ml_strategy.py:1343-1345` (live):

```
            sym_map = {"BTCUSDT": 0, "ETHUSDT": 1, "SOLUSDT": 2}
            base = sym_str.split("-")[0] if "-" in sym_str else sym_str.split(".")[0]
            rd["symbol_encoded"] = float(sym_map.get(base, -1))
```

**Что получает модель при промахе: `-1.0`.** Не NaN, не исключение.

Разница существенна. `_safe_float` (`ml_strategy.py:140-148`) превращает
недоступные значения в NaN, а LightGBM маршрутизирует NaN в оптимальную ветвь —
механизм «данных нет» работает. Значение `-1` таким механизмом **не является**:
это конечное число вне обучающего диапазона `{0, 1, 2}`. Дерево, обученное на
трёх уровнях, отправит `-1` в ту же ветвь, что и `0` (BTCUSDT), поскольку все
пороги расщепления лежат правее.

**Практическое следствие:** при добавлении четвёртого символа или изменении
формата `instrument_id` инференс молча пойдёт «как для BTCUSDT». Проверка на
разбор `instrument_id` — `sym_str.split("-")[0]` для `BTCUSDT-PERP.BINANCE` —
даёт `BTCUSDT`, попадание есть; но для формата без дефиса
(`BTCUSDT.BINANCE`) ветка `split(".")[0]` тоже даёт `BTCUSDT`. Оба разбора
корректны для текущих трёх символов.

Дополняет A2-008 (шесть копий карты) фактом о поведении дефолта.

---

## 6. ГИПЕРПАРАМЕТРЫ

### 6.1 Откуда берутся

Захардкожены, два профиля. `ModelConfig.lgbm_params`
(`lgbm_trainer.py:257-269`) — профиль 4H:

```
    lgbm_params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_estimators": 200,
        "random_state": 42,
        "verbose": -1,
    })
```

`MTF_LGBM_PARAMS` (`lgbm_trainer.py:158-181`) — профиль 1H/15m:
`num_leaves: 25`, `max_depth: 5`, `feature_fraction: 0.75`,
`bagging_fraction: 0.8`, `bagging_freq: 1`, `lambda_l1: 0.05`,
`lambda_l2: 0.05`, `min_gain_to_split: 0.01`, `n_estimators: 2000`,
`learning_rate: 0.02`, плюс `feature_fraction_seed: 42`, `bagging_seed: 42`.

Тюнинг (`scripts/tune_models.py`) — Optuna по 9 параметрам
(`:124-134`), `n_trials: int = 100` (`:232`). Результат тюнинга в текущие
бандлы не попал: `lgbm_params` в манифесте `v3/trend_model_v3.pkl` (M2)
дословно совпадает с `MTF_LGBM_PARAMS` плюс `min_child_samples: 30` —
переопределение из `retrain_v3.py`.

### 6.2 `use_mtf_params` — D-3 и D-6

Профиль объявлен как принадлежащий 1H/15m. `lgbm_trainer.py:334-340`:

```
        # When True, train() uses the stricter MTF_LGBM_PARAMS profile
```
```
        # (1H=30, 15m=20). None → keep MTF_LGBM_PARAMS default. Ignored
        # when use_mtf_params=False (4H untouched).
```

и `:743-745`:

```
        # MTF profile (1H/15m) uses stricter regularization; 4H uses
        # config.lgbm_params defaults.
        raw_params = MTF_LGBM_PARAMS if self.use_mtf_params else self.config.lgbm_params
```

Все места, где флаг включается:

```
$ grep -rn 'use_mtf_params' scripts/ src/ | grep -v lgbm_trainer.py
scripts/train_1h_models.py:116, :245        use_mtf_params=True   # 1H
scripts/train_15m_models.py:115, :242       use_mtf_params=True   # 15m
scripts/validate_1h_models.py:243, :316     use_mtf_params=True
scripts/validate_15m_models.py:244, :316    use_mtf_params=True
scripts/retrain_v3.py:140, :256, :316       use_mtf_params=True
```

### A2-035 4H-модели обучены профилем, объявленным как «1H/15m»

**Севирити:** MEDIUM
**Тип:** семантика / расхождение с практикой

**Где:** `scripts/retrain_v3.py:140`, `:256`, `:316`;
`src/models/lgbm_trainer.py:158-181`, `:743-745`

**Что измерено:** все четыре бандла несут `interval: "4h"` (проход 2, §8) и
одновременно `use_mtf_params: True` (M2). Их `lgbm_params` в манифесте —
`MTF_LGBM_PARAMS`, а не `ModelConfig.lgbm_params`.

**В чём дефект:** комментарий на строке 744 утверждает «4H uses
config.lgbm_params defaults». Для артефактов, которые существуют, это неверно.
Расхождение не косметическое:

| | 4H-профиль (`config.lgbm_params`) | применённый MTF-профиль |
|---|---|---|
| `n_estimators` | 200 | **2000** |
| `learning_rate` | 0.05 | **0.02** |
| `num_leaves` | 31 | 25 |
| `max_depth` | — (без ограничения) | **5** |
| `lambda_l1` / `lambda_l2` | — | 0.05 / 0.05 |
| `val_frac` (`:675`) | 0.90 | **0.85** |
| `min_child_samples` | 20 | 30 (переопределён) |

Меняется не только регуляризация, но и **доля валидации** — 15% вместо 10%,
что напрямую влияет на раннюю остановку, то есть на R-1.

**Как проявляется:** ни одна из существующих 4H-моделей не обучена профилем,
который код и комментарии называют 4H-профилем. Воспроизвести «4H по
умолчанию» нельзя ни одним из скриптов: `retrain_v3.py` жёстко ставит `True`
во всех трёх точках, а `train_models.py` (единственный путь с `False`) на
диске следов не оставил.

**Как установлено:** замером (M2) и чтением.
**Уверенность:** доказано.

### A2-036 Единственный производитель прод-имён обучает на другой целевой переменной

**Севирити:** HIGH
**Тип:** логика / семантика

**Где:** `src/models/lgbm_trainer.py:236`; `src/models/training_pipeline.py:66`;
`scripts/train_models.py:75-81`

**Что в коде.** Умолчание (`lgbm_trainer.py:232-240`):

```
    # v3: triple-barrier target + AFML uniqueness weights.
    # Enable for retraining; defaults preserve legacy sign(return) target
    # so existing callers (production trend/high_vol/range models) are
    # untouched.
    use_triple_barrier: bool = False
    use_uniqueness_weights: bool = False
```

`TrainingPipeline` (`training_pipeline.py:66`) конструирует `ModelConfig(...)`,
не задавая ни одного из этих флагов:

```
$ grep -n 'use_triple_barrier\|use_uniqueness\|model_suffix\|feature_whitelist\|barrier' src/models/training_pipeline.py
(нет совпадений)
```

**В чём дефект:** `train_models.py` — единственный скрипт, чей выход носит имя,
которое ищет прод (§1.1), — обучает на `sign(return)` с горизонтом
`forward_bars = 1` (один 4H-бар), а не на тройном барьере. Это **третье**
определение события в системе, в дополнение к двум из прохода 2 (A2-022):

| Путь | Целевая переменная |
|---|---|
| `train_models.py` → `{regime}_model.pkl` (прод-имя) | `sign(return)` за 1 бар |
| `retrain_v3.py` → `trend_model_v3.pkl` | triple barrier pt=1.0/sl=0.8/h=6 |
| `retrain_v3.py` → `high_vol_model_v3.pkl` | triple barrier pt=1.25/sl=1.0/h=4 |

**Как проявляется:** если оператор запустит `train_models.py`, чтобы получить
файлы под именами, которые загружает прод, он получит модель другого события,
без uniqueness-весов, с профилем гиперпараметров 4H — и она бесшумно встанет на
место v3-моделей. Ни `_load_models`, ни манифест этого не проверяют.

**Как установлено:** замером (grep по `training_pipeline.py`) и чтением.
**Уверенность:** доказано.

### 6.3 Тюнинг и его срез

`scripts/tune_models.py` в текущие бандлы вклада не внёс (§6.1). Его собственный
срез не разбирался — см. «НЕ ИССЛЕДОВАНО» п.4. Из `data/mlflow.db` (проход 3):
123 прогона с именем `optuna_best_all` из 2827.

---

## 7. ЗАПИСЬ АРТЕФАКТА

### 7.1 Что попадает в бандл

Ключи верхнего уровня (замер):

```
ключи бандла: ['booster', 'feature_columns', 'manifest', 'regime', 'symbols']
```

Манифест (`lgbm_trainer.py:971-1035`) содержит 24 поля, включая
`feature_columns_hash`, `barriers`, `data_range`, `oos_start_ms`,
`lgbm_params`, `best_iteration`, `num_trees`, `git_commit`,
`lightgbm_version`, `python_version`, `class_balance`, `embargo_rows`.

**Чего не хватает для воспроизводимости:**

| Отсутствует | Почему нужно |
|---|---|
| хэш/версия входных данных | `ml_features/*.parquet` не версионированы, `data/` в `.gitignore`; какой снимок использовался — не записано |
| **список `feature_columns` записан, но хэш взят от `sorted()`** (`:980-982`) | порядок признаков — часть контракта (`symbol_encoded` последний); хэш от отсортированного списка не фиксирует порядок |
| `random_state` фактического прогона отдельно | есть внутри `lgbm_params`, но `bagging_seed`/`feature_fraction_seed` присутствуют только в MTF-профиле; 4H-профиль их не задаёт |
| `polars` / `numpy` / `scikit-learn` версии | записаны только `lightgbm` и `python` |
| путь `features_dir` | не записан вовсе |
| команда запуска / CLI-аргументы | не записаны |
| ссылка на whitelist-файл | записан только размер (`None`) |

### 7.2 Можно ли по бандлу восстановить, на каких данных он обучен

**Частично.** Замер на четырёх реальных файлах:

Восстанавливается: символы (`symbols_in_train`), интервал (`4h`), режим,
барьеры, границы train/test в миллисекундах (`data_range`, `oos_start_ms`),
число строк, число признаков и их имена, коммит кода.

**Не восстанавливается:** содержимое `ml_features/*.parquet` на момент
прогона. Между 2026-08-14 (создание бандлов) и сегодня файлы перезаписывались —
mtime `BTCUSDT_4h_features.parquet` = 2026-08-22 17:34 (проход 1, §4.1), то
есть **на 8 дней позже** бандла. Восстановить входные данные того прогона
нечем: ни хэша, ни снимка, ни версии.

### 7.3 Мета-модель мимо `LGBMTrainer` — D-7

`scripts/train_meta_model.py:181-188` вызывает `lgb.train` напрямую и
`pickle.dump` на `:258`. Что теряется по сравнению с путём `LGBMTrainer`:

| Компонент | `LGBMTrainer` | `train_meta_model.py` |
|---|---|---|
| Манифест провенанса (24 поля) | есть (`:971-1035`) | **нет** |
| Гейт перед записью (`passes_minimum_thresholds`) | есть (`save_bundle`) | **нет** |
| Эмбарго train→val по wall-clock | есть (`_embargo_fit_end`) | **нет** — `val_split` по строкам (`:173-174`) |
| Uniqueness-веса | опционально | **нет** |
| Логирование в MLflow | есть (`:788`) | **нет** |
| Веса классов | `compute_sample_weight("balanced")` | своя формула (`:169`): `np.where(y_tr == 0, base_rate_tr / (1 - base_rate_tr), 1.0)` |

**D-7 подтверждён.** Смягчающее: подсистема мертва (проход 1, A2-002) —
`meta_model_v3.pkl` на файловой системе отсутствует.

---

## 8. ЗАМЕР ВОСПРОИЗВОДИМОСТИ

Взят `data/features/models/v3/trend_model_v3.pkl` — бандл с `passes: true`,
наиболее близкий к «прод-модели».

**Что даёт манифест:**

```
git_commit        = e67a451
lightgbm_version  = 4.3.0
python_version    = 3.11.9
use_mtf_params    = True
use_uniqueness_weights = True
feature_whitelist_size = None
n_features        = 46
feature_columns_hash = 3429c552be803bcd5c52ef747c01e6cf1dbe65285d55257c71586b1c6db9b036
class_balance     = 0.43717033819422896
best_iteration    = 196
barriers          = pt 1.0, sl 0.8, max_holding 6
oos_start_ms      = (отсутствует — None)
data_range        = train 2024-05-03 → 2025-10-02, test 2025-08-11 → 2025-12-23
lgbm_params       = {objective binary, metric binary_logloss, verbose -1, num_leaves 25,
                     max_depth 5, feature_fraction 0.75, feature_fraction_seed 42,
                     bagging_fraction 0.8, bagging_freq 1, bagging_seed 42,
                     lambda_l1 0.05, lambda_l2 0.05, min_gain_to_split 0.01,
                     learning_rate 0.02, min_child_samples 30, random_state 42}
```

**Чего не хватает, чтобы повторить прогон:**

1. **Входные данные.** `ml_features/*.parquet` перезаписаны 2026-08-22, на
   8 дней позже бандла. Хэша данных в манифесте нет. Восстановить снимок
   нечем — `data/` в `.gitignore`, бэкапа нет.
2. **Код.** `git_commit = e67a451` известен, но HEAD сейчас `f4af5fd`, и между
   ними лежит `9d8bfbc` — тот самый коммит, который изменил логику сплита.
   Откат к `e67a451` возможен (дерево чисто), но тогда воспроизводится
   **дефектный** сплит: сегодняшний прогон на сегодняшнем коде даст **другие**
   границы train/test.
3. **`oos_start_ms = None`** — граница OOS не записана. Восстановить её из
   `data_range` нельзя: `test_start_ms` — это первый бар, переживший режимный
   фильтр, а не сама отсечка (это прямо сказано в комментарии
   `lgbm_trainer.py:1013-1016`).
4. **Команда запуска.** Какая ячейка `BARRIER_GRID` и какие CLI-флаги
   `retrain_v3.py` — не записано. Барьеры `pt=1.0/sl=0.8/h=6` в манифесте не
   совпадают ни с одной ячейкой `BEST_CELLS` (`retrain_v3.py:70-71`:
   `trend` → pt=1.25/sl=1.0/hold=4), то есть это был не `--best-only`-прогон.
5. **`feature_whitelist`.** `None` в манифесте, файла `selected_features_v3.json`
   нет — но набор из 46 признаков зависит от того, какие колонки были в
   parquet на тот момент. Сегодня в `ml_features` 64 колонки (проход 2), из
   которых `get_feature_columns` отберёт какое-то число; совпадёт ли оно с 46 —
   не проверено и не проверяемо без данных того дня.
6. **Версии зависимостей помимо LightGBM.** `polars 0.20.31` определяет порядок
   и содержимое `get_feature_columns` (сортировка `sorted(feature_cols)` —
   `dataset_builder.py:465`); версия не записана.

**Вердикт:** прогон **невоспроизводим**. Из шести необходимых компонентов
отсутствуют пять; шестой (код) восстановим, но приведёт к другому результату,
так как логика сплита с тех пор изменена.

Отдельно — даже при полном совпадении входов детерминизм не гарантирован:
4H-профиль `ModelConfig.lgbm_params` не задаёт `bagging_seed` и
`feature_fraction_seed` (только `random_state`), тогда как MTF-профиль их
задаёт. Для бандлов на диске (MTF-профиль) сиды присутствуют — этот конкретный
риск не реализуется, но он реализуется для любого прогона `train_models.py`.

---

## 9. ПРЯМОЙ ОТВЕТ

> **Воспроизводим ли хоть один результат обучения?**
>
> **Нет — ни один из четырёх.**

Для каждого из четырёх бандлов на диске отсутствуют одни и те же пять
компонентов: снимок входных данных (перезаписаны на 8 дней позже, хэша нет),
граница OOS (`oos_start_ms: None` у трёх из четырёх), команда запуска, версии
зависимостей кроме LightGBM, и файл whitelist (которого не существует).
Шестой компонент — код — восстановим по `git_commit`, но откат к нему
воспроизведёт логику сплита, исправленную коммитом `9d8bfbc`, то есть даст
заведомо иные границы train/test.

Формально повторить можно только один аспект: гиперпараметры (записаны
полностью, с сидами) и число итераций (`best_iteration` в манифесте).

**Существенная оговорка в пользу системы.** Манифест провенанса — сильная
часть этого кода. 24 поля, включая `git_commit`, `feature_columns_hash`,
`best_iteration`, `data_range`, `written_despite_failing`, — это больше, чем
в типичном исследовательском репозитории. Именно благодаря манифесту стало
возможно установить и производителя бандлов (§1.2), и корректный таймлайн
(M3), и природу R-1 (§3.3). Недостающее — хэш входных данных и команда
запуска — это два поля, а не архитектурная перестройка.

**Отдельный результат прохода, важнее находок.** Гипотеза R-1 («val-loss
минимален на итерации 1–5 — значит где-то дефект») **опровергнута замером**.
Стенд с известным сигналом даёт AUC 0.99 и остановку на 377-й итерации;
тот же стенд при нулевом сигнале даёт остановку на 2-й и 6-й — то есть ровно
те значения, что записаны в `_grid`-бандлах (2 и 5). Ни разметка, ни веса, ни
хвост val, ни утечка симптом не производят. Ранняя остановка работает
правильно и сообщает, что учить нечему. Искать дефект в этом месте больше не
нужно.

R-3 («критерий остановки может выбираться по тесту») **также не
подтверждается**: `valid_sets` во всех двух местах кодовой базы содержит
только `train_data` и `val_data`, тест в `train()` не передаётся.

---

## РЕЕСТР НАХОДОК ПРОХОДА 4

| ID | Имя | Севирити | Тип |
|---|---|---|---|
| A2-031 | Прод-путь и производитель бандлов не связаны ни по имени, ни по каталогу | CRITICAL | архитектура |
| A2-032 | Отбор фич на сплите с пересечением 13.3 / 79.3 суток | HIGH | математика |
| A2-033 | Отбор фич никогда не применялся к артефактам на диске | HIGH | недоделка |
| A2-036 | Единственный производитель прод-имён обучает на другой целевой переменной | HIGH | логика / семантика |
| A2-034 | Модель из двух деревьев записана на диск как результат прогона | MEDIUM | недоделка |
| A2-035 | 4H-модели обучены профилем, объявленным как «1H/15m» | MEDIUM | семантика / практика |

**Сводка прохода 4:** CRITICAL 1, HIGH 3, MEDIUM 2. Всего 6.
**Нарастающим итогом (проходы 1–4):** CRITICAL 6, HIGH 17, MEDIUM 10, LOW 3 — 36.

### Проверено и дефекта не найдено

| Предмет | Результат |
|---|---|
| A13 — порядок «режимный фильтр → сплит» | **закрыта** в `LGBMTrainer.prepare_data` (граница до фильтра, по времени). Копия дефекта осталась в `feature_selection_v3` — A2-032 |
| D-1 — `best_iteration` в манифесте | **закрыта**, поле есть во всех четырёх бандлах |
| R-1 — коллапс ранней остановки | **опровергнута**: воспроизводится ровно при нулевом сигнале (S1/S2/S3) |
| R-3 — остановка по тесту | **не подтверждается**: тест в `valid_sets` не попадает |
| R-4 — нестационарный приор | факт подтверждён (пересчёта нет), но замером показано, что вклад в R-1 мал (`best_iter` 2 → 6) |
| Выравнивание uniqueness-весов со срезами | корректно: `_prepare_xy` строк не удаляет |
| Сортировка кадра по времени перед выделением val | корректна (`temporal_split.py:91-92`) |
| **Таймлайн A2-023** | **исправлен** (M3): `created_at` в UTC против `git log` в IST. Суть находки подтверждена точнее, формулировка про «два исправления» неверна — исправление было одно |
| D-7 — мета-модель мимо `LGBMTrainer` | подтверждён, перечень потерь в §7.3; подсистема мертва |

---

## НЕ ИССЛЕДОВАНО

1. **`TrainingPipeline.run` целиком** — прочитаны только строки с `ModelConfig`
   и `save_bundle`. Как он выбирает символы и обрабатывает отказы, не разобрано.
2. **`_filter_by_regime`** (`lgbm_trainer.py`) не читался: не проверено, как
   режим `range` соотносится с `trend`/`high_vol` и что происходит при
   неизвестном лейбле.
3. **`compute_uniqueness_weights_by_symbol` на реальном train-кадре** — веса
   считались в проходе 2 отдельно от тренера; их фактическое влияние на
   `best_iteration` реальных бандлов не измерялось (только на синтетике).
4. **`scripts/tune_models.py`**: срез, на котором работает Optuna, не проверен.
   §6.3 задания («тюнинг на тесте — утечка») остался открытым. 123 прогона
   `optuna_best_all` в MLflow.
5. **`floor_fit_end`** — латентное молчаливое ограничение эмбарго половиной
   train. На 4H не срабатывает; на 15m/1H с большим `h` может. Не измерялось.
6. **Почему `v3/high_vol_model_v3.pkl` обучен на 684 строках** при 46 признаках
   (14.9 наблюдений на признак) — не проверено, ожидаемо ли это при
   `high_vol` = 947 баров на символ.
7. **Соответствие 46 признаков сегодняшним 64 колонкам `ml_features`**: не
   проверено, даёт ли `get_feature_columns` сегодня те же 46.
8. **Тесты обучения** не читались — почему 1923 теста не поймали A2-031
   (несовпадение имён прод-пути и производителя). Проход 8.
