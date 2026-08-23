# АУДИТ-2 / ПРОХОД 3 — Математика и валидация

HEAD: `f4af5fd210b32db8af4478f0e2f440e2eb504ccc`. Протокол — [`00_method.md`](00_method.md).
Ни одной правки в `src/ scripts/ tests/ deploy/`. Все замеры — `/tmp/a2p3/`.
`data/mlflow.db` открывалась `mode=ro&immutable=1` (сайдкары не создавались).

**Прямой ответ на вопрос прохода — §8.**

---

## 0. ЧТО ИЗМЕРЕНО

| № | Замер | Раздел |
|---|---|---|
| N1 | DSR: код против канона Bailey & LdP на реалистичном наборе | §1.1 |
| N2 | Вклад каждого отклонения по отдельности | §1.1 |
| N3 | **Проверка на известном ответе** (SR\* ≡ SR0 → канон даёт ровно 0.5) | §1.5 |
| N4 | Чувствительность DSR к N: код против канона | §2.3 |
| N5 | DSR как функция произвольной константы ×10 в прокси | §1.5 |
| N6 | PBO ≡ 0: 36 000 случайных наборов + 5 крайних случаев | §3.1 |
| N7 | Слиппедж: годовая σ вместо дневной, ×19.10 | §6.2 |
| N8 | Доверительные интервалы win rate | §5.3 |
| N9 | Инфляция Sharpe/t от перекрытия меток | §4.4 |
| N10 | t-stat: веса не входят в знаменатель | §5.4 |
| N11 | Честный N по `data/mlflow.db` | §2.2 |

---

## 1. DSR — ФОРМУЛА ПРОТИВ КАНОНА

### 1.1 Построчное сопоставление

Канон (Bailey & López de Prado 2014, «The Deflated Sharpe Ratio»):

```
SR0 = √V[SR_trials] · ( (1−γ)·Z⁻¹(1 − 1/N) + γ·Z⁻¹(1 − 1/(N·e)) )
DSR = Φ( (SR* − SR0)·√(T−1) / √(1 − γ3·SR* + ((γ4−1)/4)·SR*²) )
```
где γ — постоянная Эйлера–Маскерони, γ4 — **сырой** эксцесс (3 для нормального).

Код, `src/models/statistical_tests.py:82-123`:

```
    sr_array = np.array(sharpe_ratios, dtype=np.float64)
    best_sr = float(np.max(sr_array))
    std_sr = float(np.std(sr_array, ddof=1))
```
```
    sqrt_2logn = math.sqrt(2 * log_n)
    log_logn = math.log(log_n) if log_n > 0 else 0.0

    expected_max_sr = sqrt_2logn - (log_logn + math.log(4 * math.pi)) / (
        2 * sqrt_2logn
    )
```
```
    t_obs = n_obs if n_obs is not None else len(sharpe_ratios)
    if t_obs < 2:
        return 0.0
    se_sr = math.sqrt(
        (1 - skewness * best_sr + ((kurtosis - 3) / 4) * best_sr ** 2)
        / (t_obs - 1)
    )
```
```
    z = (best_sr - expected_max_sr) / se_sr
    dsr = float(sp_stats.norm.cdf(z))
```

**Расхождения:**

| № | Канон | Код | Куда смещает |
|---|---|---|---|
| D1 | `SR0 = √V[SR] · (…)` | `expected_max_sr` **не умножен** на `std_sr` (строки 97-99) | зависит от масштаба: при `std(SR) < 1` порог завышен → DSR занижен; при `std(SR) > 1` — наоборот |
| D2 | `(γ4 − 1)/4 · SR*²` | `((kurtosis - 3) / 4) * best_sr ** 2` (строка 114) | член обнуляется при γ4=3 вместо `+0.5·SR*²` → SE занижен → **DSR завышен** |
| D3 | `T` = число наблюдений доходности | `t_obs = len(sharpe_ratios)` при `n_obs=None` (строка 110) | T ≈ 10 вместо сотен → SE завышен → DSR занижен |
| D4 | аппроксимация E[max] через два квантиля | аппроксимация Эйлера–Маскерони `√(2 ln N) − (ln ln N + ln 4π)/(2√(2 ln N))` | само по себе допустимо: расхождение с каноном < 0.5% |

**D2 требует отдельного пояснения, потому что комментарий в коде утверждает
обратное.** `statistical_tests.py:103-109`:

```
    # Two bugs were here pre-fix:
    #   * the denominator used the number of FOLDS (5-15) instead of the
    #     number of return observations T (hundreds-thousands), inflating
    #     SE by ~sqrt(T/N_folds) and collapsing DSR to ≈0.5;
    #   * the kurtosis term was ``(γ4 - 3)/4`` so for a normal distribution
    #     (γ4 = 3) it added ``0.5·SR²`` of spurious variance instead of
    #     vanishing.
```

(в файле — `(γ4 - 1)/4`; цитата выше приведена по строке 107 дословно:
«the kurtosis term was ``(γ4 - 1)/4``»).

Дисперсия `0.5·SR²` **не является spurious**. Это ровно дисперсия оценки
Шарпа по Mertens (2002):
`Var(SR) ≈ (1 + 0.5·SR² − γ3·SR + (γ4−3)/4·SR²)/T`. Канон Bailey–LdP
сворачивает `1 + 0.5·SR²` и `(γ4−3)/4·SR²` в `1 + ((γ4−1)/4)·SR²`, что при
γ4=3 даёт ровно `1 + 0.5·SR²`. Убрав этот член, код удалил основную часть
дисперсии оценки Шарпа. **«Исправление» устранило не дефект, а корректную
формулу.**

### 1.2 Размерность: per-period против аннуализированного

`√365` внутри `calculate_dsr` **отсутствует**. Аннуализация выполняется
снаружи, в ветке реальных доходностей `run_all_tests:400-404`:

```
            sr = (
                float(np.mean(daily_rets))
                / (float(np.std(daily_rets, ddof=1)) + 1e-10)
                * math.sqrt(annualization_factor)
            )
```

Эта ветка **никогда не исполняется** (проход 2, A2-007: ни один валидатор не
передаёт `per_fold_daily_returns`). Исполняется прокси-ветка,
`run_all_tests:428-441`, где «Sharpe» — это `(wr_frac - 0.5) * pf * 10`,
величина без периода и без аннуализации вообще.

**Вывод по §1.2:** размерности нет. В формулу подаётся безразмерная
конструкция, а порог `expected_max_sr` — квантиль стандартной нормали. Обе
величины сравниваются как если бы они были в одних единицах, но единиц нет ни
у одной.

### 1.3 `std_sr` — вычисляется, не используется

`std_sr` вычисляется на строке 84 и встречается ровно дважды:

```
$ grep -n 'std_sr' src/models/statistical_tests.py
84:    std_sr = float(np.std(sr_array, ddof=1))
86:    if std_sr < 1e-12:
127:        f"std_sr={std_sr:.4f}, se_sr={se_sr:.4f}, z={z:.4f}, DSR={dsr:.4f}"
```

Строка 86 — guard, строка 127 — лог. **В расчёт `z` (строка 122) `std_sr` не
входит.** `expected_max_sr` на `√V[SR]` не умножен.

### 1.4 Возврат `0.0` — «плохо» или «не смогли»

`calculate_dsr` возвращает `0.0` в **пяти** различных ситуациях:

| Строка | Условие | Смысл |
|---|---|---|
| 79-80 | `len(sharpe_ratios) < 2 or n_trials < 2` | недостаточно данных |
| 86-87 | `std_sr < 1e-12` | все трейлы одинаковы |
| 91-92 | `log_n <= 0` | N ≤ 1 |
| 111-112 | `t_obs < 2` | недостаточно наблюдений |
| 118-119 | `se_sr < 1e-12` | вырожденная SE |

Плюс шестая, у вызывающего — `scripts/validate_1h_models.py:568-573`:

```
    else:
        dsr = 0.0
        pbo = 1.0
        t_stat = 0.0
        dsr_by_n = {}
        _log.warning("  Insufficient data for statistical tests")
```

Потребитель — `StatTestResult.passes_all_thresholds` (`statistical_tests.py:331`):

```
            self.dsr >= 0.95
```

и рендер `summary()` (`:342`):

```
            f"  DSR:         {self.dsr:.4f}  {'✅' if self.dsr >= 0.95 else '❌'}  ← goal ≥ 0.95",
```

**Потребитель не различает.** `0.0` рендерится как `❌` — «модель не проходит
порог» — во всех шести случаях, включая «расчёт невозможен». Отдельного
состояния «не определено» в типе нет: `dsr: float`, не `float | None`.

Это тот же класс, что A2-012 из прохода 1: тихая деградация к значению, которое
потребитель принимает за данные. Здесь безопасно по направлению (0.0 не даст
ложного «прошло»), но диагностически лживо.

### 1.5 ЧИСЛЕННАЯ ПРОВЕРКА НА ИЗВЕСТНОМ ОТВЕТЕ

Сконструирован набор из 10 трейлов со `std(SR) = 1.0` и `max(SR)`, равным
**в точности** каноническому порогу `SR0` при N=100. По определению
PSR/DSR, когда `SR* = SR0`, ответ обязан быть **ровно 0.5**.

```
N3. ЧИСЛЕННАЯ ПРОВЕРКА НА ИЗВЕСТНОМ ОТВЕТЕ
  сконструировано: std(SR)=1.000000, max(SR)=2.530603, SR0_канон=2.530603
  ОЖИДАЕМЫЙ КАНОНИЧЕСКИЙ ОТВЕТ (SR*==SR0)          : DSR = 0.500000
  канон, посчитанный независимо                     : DSR = 0.500000
  КОД calculate_dsr(srs, n_trials=100)              : DSR = 0.689009
  (код: E[SR_max]=2.3663 против SR0=2.5306; SE=0.33333)
```

**Код даёт 0.689 там, где правильный ответ 0.500.** Абсолютная ошибка 0.189.
Независимая реализация канона в том же скрипте воспроизводит 0.500000 точно —
значит расхождение в коде, а не в проверке.

**Вклад каждого отклонения по отдельности** (N=100, реалистичный
прокси-набор `best_sr=1.12`, `std(SR)=0.2961`):

```
N2. ВКЛАД КАЖДОГО ОТКЛОНЕНИЯ ПО ОТДЕЛЬНОСТИ (N=100, тот же набор)
вариант                       порог SR0         SE         z        DSR
код как есть                     2.3663    0.33333    -3.739   0.000092
+ умножить E[max] на std         0.6591    0.33333     1.383   0.916634
+ канон (g4-1)/4                 0.6591    0.42521     1.084   0.860820
канон целиком                    0.7048    0.42521     0.976   0.835556
```

Одна пропущенная строка кода — умножение на `std_sr` — меняет DSR с
**0.000092 на 0.916634**, то есть на четыре порядка.

**Сторона смещения не фиксирована.** Она определяется `std(SR)`, а `std(SR)` —
масштабом прокси. Прокси содержит произвольный множитель `× 10`
(`statistical_tests.py:142`, `:432`, `:438`):

```
        proxies.append((wr_frac - 0.5) * pf * 10)
```

```
N5. DSR — ФУНКЦИЯ ПРОИЗВОЛЬНОЙ КОНСТАНТЫ ×10 В ПРОКСИ
 масштаб k   best_sr   std(SR)  DSR код (N=100)
         1    0.1120    0.0296         0.000000
         5    0.5600    0.1481         0.000000
        10    1.1200    0.2961         0.000092
        20    2.2400    0.5923         0.352431
       100   11.2000    2.9613         1.000000
```

**Замените `× 10` на `× 100` — и DSR станет 1.0000 при тех же самых win rate и
profit factor.** Отчётная величина определяется константой, у которой нет ни
вывода, ни источника.

### A2-024 DSR не реализует формулу Bailey & López de Prado; результат определяется произвольной константой

**Севирити:** CRITICAL
**Тип:** математика
**Где:** `src/models/statistical_tests.py:84`, `:97-99`, `:110`, `:114`, `:122`, `:142`

**Что в коде:** цитаты выше (§1.1).

**В чём дефект:** четыре отклонения от канона (D1–D4 в §1.1), из которых D1
(отсутствие умножения на `√V[SR]`) — разрыв размерности, а D2 (`(γ4−3)/4`
вместо `(γ4−1)/4`) — удаление основного члена дисперсии оценки Шарпа.

**Как проявляется:** на проверке с известным ответом код даёт 0.689 вместо
0.500 (N3). На реалистичном наборе — 0.000092 вместо 0.836 (N1, N=100).
Величина является функцией произвольного множителя `×10` в прокси: при `×100`
DSR = 1.0000 (N5).

**Кто ещё это читает:** `StatTestResult.passes_all_thresholds`
(`statistical_tests.py:331`) — порог `dsr >= 0.95`;
`dsr_sensitivity` (`:150-170`) — строит таблицу чувствительности той же
сломанной функцией; `scripts/validate_1h_models.py:563`,
`validate_15m_models.py:569`, `validate_ml_models.py:189` — печатают в отчёт.
Ни один потребитель не проверяет размерность.

**Отношение к Аудиту-1:** A4. **Не закрыта.** Из четырёх заявленных дефектов
A4 закрыт один (D3, `n_obs`, — механизм добавлен), но он не активен, так как
`n_obs` никто не передаёт. Дефект D2 **введён исправлением** — комментарий
`statistical_tests.py:107-109` называет корректный член дисперсии «spurious».

**Как установлено:** замером (N1, N2, N3, N5) и чтением.
**Уверенность:** доказано.

---

## 2. N ТРЕЙЛОВ

### 2.1 Что подставляется в N

```
$ grep -rn 'n_experiments\|n_trials=' scripts/validate*.py src/models/statistical_tests.py
scripts/validate_ml_models.py:186:            n_experiments=10,
scripts/validate_1h_models.py:562:            n_experiments=_N_EXPERIMENTS_DEFAULT,
scripts/validate_15m_models.py:568:            n_experiments=_N_EXPERIMENTS_DEFAULT,
src/models/statistical_tests.py:168:        n: round(calculate_dsr(proxies, n_trials=n), 4)
src/models/statistical_tests.py:360:    n_trials: int = 10,
src/models/statistical_tests.py:441:        dsr = calculate_dsr(sharpe_proxies, n_trials=n_experiments)
```

`_N_EXPERIMENTS_DEFAULT = 100` (`validate_1h_models.py:104`,
`validate_15m_models.py:105`). Дефолт функции — `10`. CLI-аргумент
`--n-experiments` в расчёт не попадает (A2-006, проход 2).

Итого три значения: **10** (`validate_ml_models`), **100** (1H/15m), и
`[20, 50, 100, 200, 500]` в таблице чувствительности (`:165`).

### 2.2 Честный N — замер

`data/mlflow.db` — журнал прогонов, который система вела сама:

```
$ sqlite3 (mode=ro&immutable=1) data/mlflow.db
runs 2827
experiments 3
params 47815
--- runs по experiment_id ---
   (1, 2704)
   (2, 123)
distinct run_uuid: 2827
--- топ имён параметров ---
   ('forward_bars', 2704)
   ('lgbm_bagging_fraction', 2704)
   ('lgbm_num_leaves', 2704)
   ('lgbm_learning_rate', 2704)
   ...
--- распределение по имени run ---
   ('lgbm_all', 2656)
   ('optuna_best_all', 123)
   ('lgbm_trend', 41)
   ('lgbm_high_vol', 7)
```

Каталог `mlruns/` независимо подтверждает порядок:

```
$ find mlruns -maxdepth 2 -mindepth 2 -type d | wc -l
1913
mlruns/523265834685266483/ : 71 runs
mlruns/839555096382368734/ : 1842 runs
```

Все 2704 прогона эксперимента 1 несут полный набор гиперпараметров LightGBM
(`lgbm_num_leaves`, `lgbm_learning_rate`, `lgbm_bagging_fraction`, …), то есть
это перебор конфигураций, а не повторы одной.

К этому добавляются источники, не попавшие в MLflow:

| Источник | Замер |
|---|---|
| Сетка барьеров | `scripts/retrain_v3.py:52-58` — 6 ячеек `BARRIER_GRID`, × 2 режима |
| Optuna | `scripts/tune_models.py:232` — `n_trials: int = 100` по умолчанию, 9 гиперпараметров (`:124-134`) |
| Отбор фич | `scripts/feature_selection_v3.py` — Clustered-MDA, `n_repeats` перестановок |
| Линии моделей | v1 / v2 / v3 (по именам в `data/models/` и суффиксам `_v3`) |
| Пороги confidence | `--confidence-threshold` в `validate_1h_models.py`, `_CONF_OVERRIDE` |

**Нижняя граница честного N — 2827**, документированная самой системой.

### 2.3 Насколько занижение N завышает DSR

Замер по канону (единственная работающая реализация), тот же прокси-набор:

```
  N=   10: SR0=0.4386  DSR_канон=0.9455
  N=  100: SR0=0.7048  DSR_канон=0.8356
  N= 2827: SR0=0.9859  DSR_канон=0.6237
```

Переход от заявленного N=10 к честному N=2827 снижает DSR с **0.9455 до
0.6237** — то есть с «проходит порог 0.95» (с точностью до округления) до
«не проходит с большим запасом». При N=100 — 0.8356, тоже ниже порога.

Полная таблица код против канона (N4):

```
     N    DSR код  DSR канон
    10   0.233990   0.945487
    20   0.039217   0.917578
    50   0.001627   0.873467
   100   0.000092   0.835556
   200   0.000004   0.794595
   500   0.000000   0.737113
  1000   0.000000   0.692082
  5000   0.000000   0.586441
```

### A2-025 N трейлов занижен минимум в 28 раз относительно журнала самой системы

**Севирити:** HIGH
**Тип:** математика
**Где:** `scripts/validate_ml_models.py:186`, `scripts/validate_1h_models.py:104`,
`scripts/validate_15m_models.py:105`

**Что в коде:**

```
scripts/validate_ml_models.py:186:            n_experiments=10,
```
```
scripts/validate_1h_models.py:104:_N_EXPERIMENTS_DEFAULT = 100
```
с пояснением в справке CLI (`validate_1h_models.py:719`):

```
            f"Default={_N_EXPERIMENTS_DEFAULT} (honest project estimate). "
```

**В чём дефект:** «honest project estimate» = 100 при 2827 прогонах,
зарегистрированных в `data/mlflow.db` тем же проектом. DSR — поправка именно на
множественное тестирование; занижение N есть отказ от поправки, ради которой
метрика вводится.

**Как проявляется:** DSR по канону при N=2827 равен 0.6237 против 0.9455 при
N=10 — разница между «почти проходит» и «не проходит». Порог `dsr >= 0.95`
недостижим при честном N на этих данных.

**Кто ещё это читает:** `dsr_sensitivity` строит таблицу до N=500
(`statistical_tests.py:165`) — верхняя граница диапазона в 5.7 раза меньше
фактического числа прогонов, то есть даже анализ чувствительности не покрывает
реальный случай.

**Как установлено:** замером (N11, N4) и чтением.
**Уверенность:** доказано.

---

## 3. PBO

### 3.1 Это не CSCV. И это тождественный ноль

Код целиком — `src/models/statistical_tests.py:212-255`:

```
    n = len(cv_results)
    if n < 4:
        _log.warning("PBO needs ≥ 4 folds for meaningful estimate; got %d", n)
        return 0.5  # uninformative prior
```
```
    for oos_idx in range(n):
        # IS = all folds except the current OOS fold
        is_indices = [j for j in range(n) if j != oos_idx]
        is_metrics = metrics_arr[is_indices]

        # Best IS fold (mapped back to original index)
        best_is_pos = int(np.argmax(is_metrics))
        best_is_original = is_indices[best_is_pos]
```
```
        other_vals = np.delete(metrics_arr, best_is_original)
        other_median = float(np.median(other_vals))

        if metrics_arr[best_is_original] < other_median:
            overfit_count += 1

    pbo = overfit_count / n
```

**Опровержение того, что это CSCV.** CSCV по Bailey et al. (2014) требует:
(1) матрицы `T × S` доходностей **множества стратегий-кандидатов**;
(2) разбиения на `S` подматриц и перебора всех `C(S, S/2)` комбинаций IS/OOS;
(3) для каждой комбинации — выбора лучшей стратегии по IS и вычисления её
**относительного ранга** среди всех стратегий на OOS;
(4) PBO = доля комбинаций, где логит относительного ранга отрицателен.

В коде: кандидатов нет — есть фолды **одной** модели; комбинаций нет — есть
`n` итераций leave-one-out; ранга среди стратегий нет — есть сравнение с
медианой других фолдов. Докстринг это признаёт (`:183`):

```
    Leave-one-out cross-validation approach:
```

**A5 подтверждена. Но найдено большее.**

**Утверждение: при `n ≥ 4` функция возвращает ровно 0.0 всегда.**

Доказательство. `best_is_original` выбран как argmax по `is_metrics` — это
максимум над `n−1` значениями (все фолды кроме `oos_idx`). Массив `other_vals`
содержит `n−1` значений: все фолды кроме `best_is_original`. Из них `n−2`
принадлежат `is_metrics` и по определению максимума **не превосходят**
`metrics_arr[best_is_original]`. Единственное значение, которое может его
превосходить, — `metrics_arr[oos_idx]`. Чтобы медиана `n−1` элементов
превысила `best`, требуется, чтобы более половины из них превышали `best`,
то есть `> (n−1)/2` элементов. Но таких элементов не более одного. Отсюда
`(n−1)/2 < 1`, то есть `n < 3`. При `n ≥ 4` условие
`metrics_arr[best_is_original] < other_median` невыполнимо. ∎

**Замер, подтверждающий:**

```
N6. PBO ≡ 0 — ПРОВЕРКА ПЕРЕБОРОМ
  случайных наборов прогнано : 36000  (n фолдов 4..15)
  из них PBO > 0             : 0
  множество полученных PBO   : [0.0]
  монотонно возр.      → PBO = 0.0000
  монотонно убыв.      → PBO = 0.0000
  один выброс вверх    → PBO = 0.0000
  один провал вниз     → PBO = 0.0000
  все равны            → PBO = 0.0000
```

### 3.2 Что означает возвращаемое число

Ничего. Оно константа. Докстринг обещает шкалу (`:194-197`):

```
    Interpretation:
        PBO = 0.0 → no overfitting
        PBO = 0.5 → random selection
        PBO > 0.5 → overfitting
```

Достижимы ровно два значения: `0.5` при `n < 4` (ранний возврат, строка 215)
и `0.0` при `n ≥ 4`. Значения `> 0.5`, для которых написана интерпретация
«overfitting», недостижимы ни при каких данных.

Как вероятность переобучения по Bailey et al. величина **не интерпретируема**:
она не зависит от данных.

### 3.3 Используется ли в гейте

Да. `StatTestResult.passes_all_thresholds` (`statistical_tests.py:328-335`):

```
    def passes_all_thresholds(self) -> bool:
        """Check against master-document go-live criteria."""
        return (
            self.dsr >= 0.95
            and self.pbo <= 0.30
            and self.t_stat >= 3.0
            and self.n_oos_signals >= 300
        )
```

Условие `self.pbo <= 0.30` при `pbo ≡ 0.0` выполняется **всегда**. Один из
четырёх критериев go-live тождественно истинен.

Смягчающее обстоятельство: сам `passes_all_thresholds` вызывается только из
тестов (проход 1, §3.2), то есть в реальном гейте не участвует — см. §7.2.

### A2-026 `calculate_pbo` тождественно равна нулю при любых входных данных

**Севирити:** CRITICAL
**Тип:** математика / логика
**Где:** `src/models/statistical_tests.py:236-247`

**Что в коде:**

```
        best_is_pos = int(np.argmax(is_metrics))
        best_is_original = is_indices[best_is_pos]
```
```
        other_vals = np.delete(metrics_arr, best_is_original)
        other_median = float(np.median(other_vals))

        if metrics_arr[best_is_original] < other_median:
            overfit_count += 1
```

**В чём дефект:** сравнивается максимум подмножества с медианой того же
множества за вычетом этого максимума. Условие невыполнимо при `n ≥ 4`
(доказательство в §3.1). Сверх того, конструкция не является CSCV: она
оценивает не выбор между стратегиями, а разброс фолдов одной модели.

**Как проявляется:** отчёт печатает `PBO: 0.0000 ✅ ← goal ≤ 0.30`
(`statistical_tests.py:343`) вне зависимости от того, переобучена модель или
нет. Оператор видит зелёную галочку на метрике, которая ничего не измеряет.
Критерий `pbo <= 0.30` в `passes_all_thresholds` тождественно истинен.

**Кто ещё это читает:** `run_all_tests:444` (`pbo = calculate_pbo(cv_results,
metric="win_rate")`), `validate_1h_models.py:565`,
`validate_15m_models.py:571`, `StatTestResult.summary()`.
Единственное место, где PBO может быть не нулём, — фолбэк
`validate_1h_models.py:570` (`pbo = 1.0` при нехватке данных), то есть
единственное информативное значение PBO в системе возникает при **отказе**
расчёта.

**Отношение к Аудиту-1:** A5 («PBO — LOO-эвристика, не CSCV»). **Не закрыта,
и недооценена**: это не приблизительная эвристика, а константа.

**Как установлено:** замером (N6: 36 000 случайных наборов + 5 крайних случаев,
ни одного ненулевого) и доказательством.
**Уверенность:** доказано.

---

## 4. WALK-FORWARD

### 4.1 Схема, эмбарго, purging

Нарезка — `src/execution/walk_forward.py:349-363`:

```
        cursor = start
        while True:
            train_start = cursor
            train_end = _add_months(cursor, self.train_months)
            # Embargo shifts the test window forward so triple-barrier
            # labels generated in the last bars of train cannot reach
            # into the test window. With embargo=timedelta(0) this is
            # a no-op (legacy behaviour).
            test_start = train_end + self.embargo
            test_end = _add_months(test_start, self.test_months)
```

Умолчание — `src/execution/walk_forward.py:328`:

```
        embargo: timedelta = timedelta(0),
```

**Замер: ни одна из шести точек конструирования не передаёт `embargo`.**

```
$ grep -rn 'WalkForwardValidator(' src scripts
src/models/ml_validator.py:298      → train_months, test_months, step_months
scripts/train_1h_models.py:215      → train_months, test_months, step_months
scripts/train_15m_models.py:213     → train_months, test_months, step_months
scripts/run_walk_forward.py:121     → train_months, test_months, step_months
scripts/validate_1h_models.py:287   → train_months, test_months, step_months
scripts/validate_15m_models.py:288  → train_months, test_months, step_months
```

Следовательно `test_start = train_end + timedelta(0) = train_end` везде.
Комментарий кода сам называет это состояние: «With embargo=timedelta(0) this is
a no-op (legacy behaviour)».

Purging (удаление обучающих меток, чьё окно пересекает тест) — **отсутствует**:
`walk_forward_ml` (`ml_validator.py:315-323`) режет фрейм только по времени:

```
            train_df = full_df.filter(
                (pl.col(ts_col) >= train_start)
                & (pl.col(ts_col) < train_end)
            )
            test_df = full_df.filter(
                (pl.col(ts_col) >= test_start)
                & (pl.col(ts_col) < test_end)
            )
```

`PurgedKFoldCV` эмбарго имеет (`walk_forward.py:124`, `embargo_pct=0.01`), но
это **другой** класс, используемый в K-fold-ветке, не в walk-forward.

### A2-027 В walk-forward эмбарго объявлено, задокументировано и нигде не включено

**Севирити:** HIGH
**Тип:** математика / недоделка
**Где:** `src/execution/walk_forward.py:328`, `:356`; шесть точек вызова

**Что в коде:** цитаты выше.

**В чём дефект:** параметр существует, снабжён комментарием со ссылкой на
AFML Ch.7 (`walk_forward.py:334`) и во всех шести местах остаётся нулевым.
Обучающая метка на последнем баре train с горизонтом `h` заглядывает на `h`
баров внутрь теста.

**Как проявляется:** каждое из окон walk-forward содержит утечку глубиной `h`
баров на стыке. При `h=6` и 4H-барах это 24 часа теста, «увиденные» обучением.
Именно `walk_forward_ml` производит `WalkForwardMLResult`, из которого строятся
и прокси-Sharpe для DSR (`run_all_tests:435-439`), и t-stat
(`run_all_tests:447-449`), и счётчик OOS-сигналов (`:452`).

**Контраст.** В `LGBMTrainer` тот же самый эмбарго **включён** и измеряется в
wall-clock (`lgbm_trainer.py:522-525`, `:552-554`; проход 2, §4.3 — проверено,
дефекта нет). То есть одна и та же защита реализована в тренере и не
подключена в валидаторе — разные части системы дают разные гарантии на один и
тот же вопрос.

**Кто ещё это читает:** `MLValidator.walk_forward_ml` → `run_all_tests` →
`StatTestResult`; `scripts/validate_1h_models.py`, `validate_15m_models.py`,
`train_1h_models.py`, `train_15m_models.py`, `run_walk_forward.py`.

**Отношение к Аудиту-1:** смежно A17, но по существу это отдельный дефект —
не аннуализация, а отсутствие разрыва.

**Как установлено:** замером (grep по шести точкам вызова) и чтением.
**Уверенность:** доказано.

### 4.2 Как считается P&L фолда

**Реальных сделок нет. Есть прокси, и он двухступенчатый.**

Ступень 1 — метрики окна, `src/models/lgbm_trainer.py:1445-1478`:

```
        # A "win" = prediction direction matches actual return direction
```
```
        correct = (dir_preds * dir_returns) > 0
        win_rate = float(correct.sum()) / len(dir_preds) * 100

        # Profit factor = sum of |returns| on wins / sum of |returns| on losses
        wins_abs = np.abs(dir_returns[correct]).sum()
        losses_abs = np.abs(dir_returns[~correct]).sum()
```

где `dir_returns` — это `future_return` из тройного барьера
(`lgbm_trainer.py:1108`: `future_returns = test_df["future_return"].to_numpy()`).
Ни комиссий, ни слиппеджа, ни фандинга, ни размера позиции.

Ступень 2 — «Sharpe» для DSR, `src/models/statistical_tests.py:435-439`:

```
        for w in wf_result.windows:
            wr_frac = w.win_rate / 100.0
            pf = w.profit_factor if w.profit_factor < 100 else 1.0
            sr_proxy = (wr_frac - 0.5) * pf * 10
            sharpe_proxies.append(sr_proxy)
```

**Дословная формула прокси: `sr_proxy = (win_rate/100 − 0.5) × profit_factor × 10`.**

В `WindowMLResult` поля Sharpe нет вовсе:

```
$ grep -n 'sharpe' src/models/ml_validator.py
(пусто)
```

`WindowMLResult` (`ml_validator.py:41-55`) содержит `win_rate`,
`profit_factor`, `signal_rate`, `n_signals`, `n_test_bars` — и только.

Отдельно: строка `pf = w.profit_factor if w.profit_factor < 100 else 1.0`
превращает **лучший** результат (PF ≥ 100, в пределе `inf` при нуле убытков —
`lgbm_trainer.py:1475`) в **нейтральный** (PF = 1.0), то есть окно без единого
убытка вносит в DSR тот же вклад, что окно с PF = 1.0.

### 4.3 `bar_duration_minutes` в аннуализации Sharpe (A17)

```
$ grep -rn 'bar_duration' src scripts
src/execution/walk_forward.py:336:        # as a duration (caller computes max_holding_bars × bar_duration).
```

Единственное вхождение — комментарий. Идентификатора `bar_duration_minutes` в
кодовой базе **нет**. Аннуализация Sharpe в walk-forward отсутствует, потому
что Sharpe в walk-forward не вычисляется вовсе (§4.2).

Там, где Sharpe всё-таки считается — `scripts/validate_1h_models.py:484`,
`validate_15m_models.py:485` — используется `np.sqrt(252)` на дневных
агрегатах, а `run_all_tests` по умолчанию берёт `annualization_factor = 365.0`
(проход 2, A2-007). Длительность бара ни там, ни там не участвует.

**A17 в исходной формулировке неприменима: аннуализировать нечего.** Но
породивший её дефект жив в другом виде — две конвенции (252 и 365) в одном
отчёте.

### 4.4 Sharpe на перекрывающихся метках

Поправки нет. `future_return` берётся с бара выхода тройного барьера
(проход 2, §1.1), окна соседних баров перекрываются, и ни `_compute_trading_metrics`,
ни `_sharpe_proxies`, ни `calculate_t_stat` не получают весов уникальности.
Веса применяются только внутри обучения (`lgbm_trainer.py:710-719`) и там же
нормируются до среднего 1 (A2-016).

```
N9. ПЕРЕКРЫТИЕ МЕТОК И ИНФЛЯЦИЯ SHARPE / t
  ESS=1053, номинал=3178: √(n/ESS) = 1.7373 → наивные Sharpe/t/z завышены в 1.74 раза
  ESS=1048.8, номинал=3482: √(n/ESS) = 1.8221 → наивные Sharpe/t/z завышены в 1.82 раза
  пример: z-тест WR 59.01% на n=610 даёт z=4.451; при поправке на ESS/n=0.302 → z=2.446, p=0.0145
```

Замер `ESS = 1048.8` при номинале 3482 получен независимо в проходе 2 (M5) и
согласуется с входным фактом 1053/3178. Коэффициент инфляции **1.74–1.82**.

Пример последствия: заявленная WR 59.01% на 610 наблюдениях даёт наивный
`z = 4.451` (`p < 0.0001`); с поправкой на перекрытие `z = 2.446`
(`p = 0.0145`) — разница между «неоспоримо» и «на грани».

---

## 5. МЕТРИКИ КАЧЕСТВА

### 5.1 Точные определения

Из `src/models/lgbm_trainer.py`:

**`signal_rate`** (`:1114-1116`):

```
        max_proba = np.maximum(proba_up, 1.0 - proba_up)
        signal_mask = max_proba >= confidence_threshold
        signal_rate = float(signal_mask.sum()) / len(signal_mask) if len(signal_mask) > 0 else 0.0
```

Доля баров тестового среза, где `max(p, 1−p) ≥ порог`. Знаменатель — **все**
бары теста (после `drop_timeout`, см. A2-017).

**`win_rate`** (`:1467-1468`):

```
        correct = (dir_preds * dir_returns) > 0
        win_rate = float(correct.sum()) / len(dir_preds) * 100
```

Доля сигнальных баров, где знак предсказания совпал со знаком `future_return`.
**Не** доля прибыльных сделок: издержек нет, размера позиции нет, R:R нет.

**`profit_factor`** (`:1471-1477`):

```
        wins_abs = np.abs(dir_returns[correct]).sum()
        losses_abs = np.abs(dir_returns[~correct]).sum()

        if losses_abs == 0:
            profit_factor = float("inf") if wins_abs > 0 else 0.0
```

Сумма модулей доходностей на «выигрышах», делённая на сумму на «проигрышах».
Валовая величина.

Мёртвая ветка: `directional = predictions != 0` (`:1459`). В бинарной схеме
`CLASS_TO_LABEL = {0: -1, 1: +1}` — нуля не бывает, маска всегда полностью
истинна.

### 5.2 PF при малом n и при нуле убытков (A16)

`float("inf")` возвращается при `losses_abs == 0` (`:1475`). Дальше:

- `EvaluationResult.profit_factor = round(profit_factor, 4)` (`:1138`) —
  `round(inf, 4)` даёт `inf`, не ошибку;
- гейт `passes_minimum_thresholds` (`:295`): `self.profit_factor >= 1.3` →
  `inf >= 1.3` = `True`. **Окно без единого убытка проходит гейт.**
- прокси-Sharpe (`statistical_tests.py:431`, `:437`):
  `pf if pf < 100 else 1.0` → `inf` схлопывается в **1.0**, нейтральное
  значение.

То есть одна и та же величина `inf` в гейте означает «отлично», а в DSR —
«никак». Минимального `n` для PF нет нигде: единственная проверка —
`if len(predictions) == 0` (`:1453`).

**A16 жива.**

### 5.3 Доверительные интервалы

**Отсутствуют полностью.**

```
$ grep -rn 'confidence_interval\|conf_int\|\bCI\b\|binom\|proportion_confint' src/models src/execution/walk_forward.py
(пусто)
```

`StatTestResult.summary()` (`:337-350`) печатает четыре числа с галочками и без
единого интервала. `ValidationResult` в `validate_1h_models.py` — то же.

Замер, показывающий масштаб пропущенного:

```
N8. ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ WIN RATE
  WR=59.01%  n= 610  95% CI = [55.11%, 62.91%]  ±3.90 п.п.  z-тест против 50%: z=4.451, p=0.0000
  WR=54.83%  n= 806  95% CI = [51.39%, 58.27%]  ±3.44 п.п.  z-тест против 50%: z=2.742, p=0.0061
  WR=60.15%  n= 172  95% CI = [52.83%, 67.47%]  ±7.32 п.п.  z-тест против 50%: z=2.662, p=0.0078
  WR=50.00%  n=   8  95% CI = [15.35%, 84.65%]  ±34.65 п.п.  z-тест против 50%: z=0.000, p=1.0000
```

WR 59.01% на 610 наблюдениях имеет 95% CI ±3.90 п.п. — величина сообщается как
`59.01`, с двумя знаками после запятой, при неопределённости в четвёртом
знаке слева от них. С поправкой на перекрытие (§4.4) интервал шире ещё в
1.82 раза: примерно ±7.1 п.п.

WR 60.15% на 172 наблюдениях (`high_vol_model_v3.pkl`, манифест из прохода 2)
имеет CI [52.83%, 67.47%] — нижняя граница едва выше порога гейта 52.0%.

### 5.4 t-stat против z-теста пропорции (A15)

Код — `src/models/statistical_tests.py:299-306`:

```
    weighted_mean = float(np.average(wr_arr, weights=n_arr))
    std_wr = float(np.std(wr_arr, ddof=1))
```
```
    n_windows = len(win_rates)
    t = (weighted_mean - 50.0) / (std_wr / math.sqrt(n_windows))
```

Три дефекта в трёх строках:

1. **Числитель взвешен по числу сделок, знаменатель — нет.** `std_wr` —
   невзвешенное стандартное отклонение по окнам. Смешивание взвешенной оценки
   центра с невзвешенной оценкой разброса не даёт корректной t-статистики.
2. **Число сделок не входит в знаменатель вообще.** Окно с 5 сделками и окно
   с 1000 вносят в `std_wr` одинаково.
3. **Это не тест пропорции.** H0 сформулирована как `win_rate = 50%`
   (`:268`), но проверяется гипотеза о среднем `n_windows` наблюдений, а не о
   доле `Σ n_trades` испытаний Бернулли. Правильная статистика —
   `z = (p̂ − 0.5)/√(0.25/N)` с `N = Σ n_trades`.

Замер, показывающий (2):

```
N10. t-stat: ВЕС ЧИСЛА СДЕЛОК НЕ ВЛИЯЕТ НА ЗНАМЕНАТЕЛЬ
  win_rates=[52.0, 58.0, 54.0, 56.0, 55.0]  n_trades=[100, 100, 100, 100, 100] → t = 5.0000
  win_rates=[52.0, 58.0, 54.0, 56.0, 55.0]  n_trades=[5, 5, 5, 5, 1000] → t = 5.0000
  win_rates=[52.0, 58.0, 54.0, 56.0, 55.0]  n_trades=[1000, 5, 5, 5, 5] → t = 2.0735
```

Первые две строки: 500 сделок и 1020 сделок дают **тождественно одинаковую**
t-статистику. Третья: те же win rate, то же общее число сделок, что и во
второй, но иное их распределение — и `t` падает с 5.00 до 2.07. Статистика
реагирует на то, в каком окне лежит вес, и не реагирует на объём выборки.

**A15 жива.**

### 5.5 AUC

```
$ grep -rn 'roc_auc\|auc' src/models src/execution/walk_forward.py
(пусто)
```

**AUC в кодовой базе не вычисляется вообще.** `evaluate` возвращает
`accuracy`, `precision`, `recall`, `f1` (`lgbm_trainer.py:1104-1107`), причём
все три последних — с `average="weighted"`.

Входной факт «walk-forward AUC 0.5375 (p=0.163)» относится к величине, которую
текущий код не производит. Откуда она — настоящим проходом не установлено
(см. «НЕ ИССЛЕДОВАНО» п.2).

Про перекрытие меток: поскольку AUC не считается, вопрос о поправке
неприменим. Для тех метрик, что считаются (`accuracy` и др.), поправка на
перекрытие также отсутствует (§4.4).

### A2-028 Ни одна метрика качества не сопровождается мерой неопределённости

**Севирити:** HIGH
**Тип:** математика / расхождение с практикой
**Где:** `src/models/lgbm_trainer.py:1131-1143`; `src/models/statistical_tests.py:337-350`;
`scripts/validate_1h_models.py:610-620`

**Что в коде:** `EvaluationResult` содержит восемь скалярных полей и ни одного
интервала; `StatTestResult.summary()` печатает четыре числа с галочками:

```
            f"  DSR:         {self.dsr:.4f}  {'✅' if self.dsr >= 0.95 else '❌'}  ← goal ≥ 0.95",
            f"  PBO:         {self.pbo:.4f}  {'✅' if self.pbo <= 0.30 else '❌'}  ← goal ≤ 0.30",
            f"  t-stat:      {self.t_stat:.4f}  {'✅' if self.t_stat >= 3.0 else '❌'}  ← goal ≥ 3.0",
            f"  OOS signals: {self.n_oos_signals}     {'✅' if self.n_oos_signals >= 300 else '❌'}  ← goal ≥ 300",
```

**В чём дефект:** пороговые сравнения выполняются над точечными оценками, чья
стандартная ошибка сопоставима с расстоянием до порога. WR 60.15% при n=172
проходит порог 52.0%, но 95% CI = [52.83%, 67.47%] — нижняя граница в 0.83 п.п.
от порога; с поправкой на перекрытие (×1.82) интервал накрывает порог.

**Как проявляется:** решение «модель проходит / не проходит» принимается без
информации о том, отличается ли наблюдённое значение от порога значимо.
Округление до `.2f` и `.4f` создаёт видимость точности на 2–4 знака при
неопределённости в первом.

**Кто ещё это читает:** `passes_minimum_thresholds` (`lgbm_trainer.py:291`) —
гейт записи модели на диск; манифест бандла (`lgbm_trainer.py:983-991`, поле
`passes`); отчёты валидаторов.

**Как установлено:** замером (N8) и чтением.
**Уверенность:** доказано.

---

## 6. ИЗДЕРЖКИ

### 6.1 `cost_model.py` — формулы

`src/execution/cost_model.py:14-24`, комиссии:

```
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    use_bnb_discount: bool = True
```
```
    def effective_maker(self) -> float:
        return self.maker_fee * (0.9 if self.use_bnb_discount else 1.0)
```
```
    def effective_taker(self) -> float:
        return self.taker_fee * (0.9 if self.use_bnb_discount else 1.0)
```

`cost_model.py:61-76`, слиппедж:

```
    def calculate_slippage(
        self,
        notional: float,
        daily_volume: float,
        volatility: float,
    ) -> float:
        """
        Return one-way slippage in USDT using the square-root market impact model.

        slippage = notional × 0.5 × σ_annual × √(Q / V)
        where Q = notional, V = daily_volume_usdt, σ = annualised fractional vol.
        """
        if daily_volume <= 0:
            return 0.0
        slippage_fraction = 0.5 * volatility * math.sqrt(notional / daily_volume)
        return notional * slippage_fraction
```

`cost_model.py:78-93`, фандинг:

```
        num_payments = hours_held / 8.0
        gross = position_size * funding_rate * num_payments
        return gross if is_long else -gross
```

Откуда `σ`: `src/risk/risk_engine.py:56-58`:

```
    default_daily_volume: float = 1_000_000_000  # $1B
    default_volatility: float = 0.60              # 60% annualised
    default_hours_held: float = 8.0
```

Единицы: `volatility` — **годовая** доля (0.60 = 60% годовых). Именно она
подставляется в формулу рыночного воздействия.

### 6.2 A9 — численная проверка

Модель квадратного корня (Almgren et al., Kyle) определена на **дневной**
волатильности: `impact ≈ σ_daily · √(Q/V)`. Подстановка годовой σ завышает в
`√365 = 19.105` раза.

```
N7. СЛИППЕДЖ: ГОДОВАЯ σ ВМЕСТО ДНЕВНОЙ (A9)
  notional=$  1,000  σ_ann=0.60 →   0.0548 USDT ( 0.548 bps) | σ_day=0.03141 → 0.002867 USDT (0.0287 bps) | завышение ×19.10
  notional=$  7,174  σ_ann=0.60 →   1.0525 USDT ( 1.467 bps) | σ_day=0.03141 → 0.055088 USDT (0.0768 bps) | завышение ×19.10
  notional=$ 10,000  σ_ann=0.60 →   1.7321 USDT ( 1.732 bps) | σ_day=0.03141 → 0.090660 USDT (0.0907 bps) | завышение ×19.10
  notional=$ 50,000  σ_ann=0.60 →  19.3649 USDT ( 3.873 bps) | σ_day=0.03141 → 1.013606 USDT (0.2027 bps) | завышение ×19.10

  Полный круг на notional 7174 USDT (медиана 8 реальных сделок):
    σ_annual : fees=6.4566 slip=2.1049 fund=2.1522 → 14.934 bps, min_required=44.802 bps
    σ_daily  : fees=6.4566 slip=0.1102 fund=2.1522 → 12.154 bps, min_required=36.461 bps
    константа валидаторов _ROUND_TRIP_FEES_BPS = 7 bps
```

**Коэффициент ровно 19.10 = √365** на всех размерах — подтверждение точное.

**A9 жива.** Но важно: на реальных размерах позиции проекта (7 тыс. USDT при
объёме 30 млрд) завышение слиппеджа даёт всего +2.78 bps к полному кругу —
14.93 против 12.15 bps. Эффект есть, но он **много меньше** другого расхождения:
валидаторы вычитают 7 bps там, где сама модель издержек даёт 12.2–14.9 bps.

### 6.3 Где издержки применяются, а где нет

| Место | Формула | Gross / Net |
|---|---|---|
| `LGBMTrainer.evaluate` → `win_rate`, `profit_factor` (`lgbm_trainer.py:1467-1477`) | по `future_return`, издержки не вычитаются | **GROSS** |
| `passes_minimum_thresholds` (`lgbm_trainer.py:291-297`) | те же WR/PF | **GROSS** |
| `MLValidator.walk_forward_ml` → `WindowMLResult` (`ml_validator.py:345-359`) | `trainer.evaluate(...)` | **GROSS** |
| Прокси-Sharpe для DSR (`statistical_tests.py:432`, `:438`) | из тех же WR/PF | **GROSS** |
| `calculate_t_stat` (`statistical_tests.py:299-306`) | из тех же WR | **GROSS** |
| `scripts/validate_1h_models.py:461`, `:468` | `signed_pnl = signal_preds * signal_returns - _ROUND_TRIP_COST` | NET, но 7 bps |
| `scripts/validate_15m_models.py:462`, `:469` | то же | NET, 7 bps |
| `RiskEngine` pre-trade (`risk_engine.py:189`, `:344`) | полный `CostModel` | NET, 14.9 bps |
| `backtest_runner.py:357` | `cm.calculate_slippage(...) * 2` | частично |

**Гейт, определяющий запись модели на диск, — валовый.** `passes` в манифесте
бандла (проход 2, §8) вычислен без единого базисного пункта издержек.

Три разные величины круговых издержек сосуществуют: **0 bps** (гейт и DSR),
**7 bps** (валидаторы), **14.93 bps** (`CostModel` в проде). Отношение
крайних — бесконечность; отношение двух ненулевых — 2.13.

### 6.4 Комиссии Binance UM

| Компонент | В коде | Замечание |
|---|---|---|
| taker | `0.0005` × 0.9 (BNB) = `0.00045` | соответствует базовой ставке VIP 0 |
| maker | `0.0002` × 0.9 = `0.00018` | соответствует |
| `is_maker` по умолчанию | `False` (`cost_model.py:104`) | консервативно, верно: вход market IOC (`ml_strategy.py:1107-1113`) |
| funding | `position_size × rate × hours/8` | знак зависит от `is_long`, корректно |
| **фактическая ставка фандинга** | `default` не задан; `RiskEngine` берёт `signal.funding_rate` | в 8 реальных сделках `funding_rate` был null в 6 из 8 (проход 2, §6.3) |
| liquidation fee | **отсутствует** | |
| проскальзывание на стопе (gap) | **отсутствует** | стоп предполагается исполненным ровно по цене — замер прохода 2: `close_price == stop_loss` 8/8 |

### A2-029 Гейт записи модели вычисляется на валовых метриках

**Севирити:** HIGH
**Тип:** математика
**Где:** `src/models/lgbm_trainer.py:291-297`, `:1467-1477`

**Что в коде:**

```
    def passes_minimum_thresholds(self) -> bool:
        """Check against master-document go-live criteria."""
        return (
            self.win_rate >= 52.0
            and self.profit_factor >= 1.3
            and self.signal_rate >= 0.10  # at least 10% signals
        )
```

`win_rate` и `profit_factor` приходят из `_compute_trading_metrics`, где
`dir_returns` — сырой `future_return` без вычета издержек.

**В чём дефект:** порог PF ≥ 1.3 применяется к валовой величине. Круговые
издержки по собственной модели проекта — 14.93 bps при типичном размере
позиции; средняя валовая доходность на сигнальный бар в замерах прохода 2
имела порядок 1–3%, то есть издержки съедают 5–15% от неё, что смещает PF вниз
примерно на 0.05–0.15. Порог 1.3 при этом не скорректирован.

**Как проявляется:** `passes: true` в манифестах `trend_model_v3.pkl`
(`profit_factor 1.3249`) и `high_vol_model_v3.pkl` (`profit_factor 1.404`) —
проход 2, §8. Первое значение отстоит от порога на 0.0249 при валовом
измерении. Нетто-величина порога не достигает.

**Кто ещё это читает:** `manifest.passes` (`lgbm_trainer.py:983-991`), решение
о записи бандла на диск (коммит `2eafa92`, «refuse to write a model that fails
the go-live thresholds»), отчёты валидаторов.

**Отношение к Аудиту-1:** A8. **Не закрыта.**

**Как установлено:** замером (N7 — величина издержек; манифесты из прохода 2)
и чтением.
**Уверенность:** доказано.

---

## 7. ГЕЙТ GO-LIVE

### 7.1 `passes_minimum_thresholds`

Полный текст — `src/models/lgbm_trainer.py:291-297` (процитирован в A2-029).
Три порога: `win_rate >= 52.0`, `profit_factor >= 1.3`, `signal_rate >= 0.10`.

**Происхождение:** докстринг ссылается на «master-document go-live criteria».
Ни одного вывода, расчёта или ссылки на источник в коде нет. Порог 52.0 не
сопровождается ни требуемым числом наблюдений, ни уровнем значимости — при
n=172 нижняя граница 95% CI для WR 60.15% равна 52.83% (N8), то есть
буквально на границе.

### 7.2 Собственный критерий проекта: DSR ≥ 0.95, PBO ≤ 0.30, ≥300 OOS, net-of-costs

Реализован — `StatTestResult.passes_all_thresholds`
(`src/models/statistical_tests.py:328-335`):

```
        return (
            self.dsr >= 0.95
            and self.pbo <= 0.30
            and self.t_stat >= 3.0
            and self.n_oos_signals >= 300
        )
```

**Но не подключён.** Замер (проход 1, §3.2, метод — 0 упоминаний в `src/` и
`scripts/` вне собственного файла):

```
src/models/statistical_tests.py:328  method:StatTestResult passes_all_thresholds   tests_refs=6 in 1 files
```

Подтверждение:

```
$ grep -rn 'passes_all_thresholds' src scripts
src/models/statistical_tests.py:328:    def passes_all_thresholds(self) -> bool:
src/models/statistical_tests.py:348:        verdict = "✅ PASSES" if self.passes_all_thresholds() else "❌ Does not pass (yet)"
```

Единственный вызов — внутри `summary()`, то есть печать строки. **Ни одно
решение в системе на нём не основано.** Решение о записи модели принимает
`passes_minimum_thresholds` — WR/PF/signal_rate, валовые, без DSR, без PBO, без
требования 300 OOS-сигналов.

Требование «net-of-costs» **не реализовано нигде** (§6.3).
CSCV-PBO не реализован (§3.1).

### A2-030 Заявленный критерий go-live существует только как печатная строка

**Севирити:** HIGH
**Тип:** архитектура / недоделка

**Где:** `src/models/statistical_tests.py:328-335`, `:348`;
`src/models/lgbm_trainer.py:291-297`

**В чём дефект:** в системе два разных «критерия go-live».

| | Строгий (`passes_all_thresholds`) | Действующий (`passes_minimum_thresholds`) |
|---|---|---|
| Что проверяет | DSR ≥ 0.95, PBO ≤ 0.30, t ≥ 3.0, OOS ≥ 300 | WR ≥ 52, PF ≥ 1.3, signal_rate ≥ 0.10 |
| Издержки | — | валовые |
| Где вызывается | только `summary()` (печать) и тесты | `evaluate()`, запись бандла, `manifest.passes` |

Строгий критерий, к тому же, не работал бы и будучи подключённым: DSR сломан
(A2-024), PBO тождественно ноль (A2-026), t-stat не является тестом пропорции
(A15/§5.4). Из четырёх его условий **одно тождественно истинно**, одно
считается по неверной формуле, одно — по формуле, чувствительной к
произвольной константе, и лишь `n_oos_signals >= 300` — простой счётчик.

**Как проявляется:** документы проекта описывают строгий критерий; артефакты на
диске несут `passes: true`, полученное по мягкому. Разрыв между заявленной и
фактической планкой не виден ни из отчёта, ни из манифеста — оба используют
слово `passes`.

**Как установлено:** замером (grep по вызовам) и чтением.
**Уверенность:** доказано.

### 7.3 `signal_rate >= 0.10`

Единственное пояснение в коде — комментарий на той же строке
(`lgbm_trainer.py:296`):

```
            and self.signal_rate >= 0.10  # at least 10% signals
```

Комментарий пересказывает выражение и ничего не обосновывает. Ни расчёта
мощности, ни требуемого числа сделок, ни связи с `n_oos_signals >= 300` из
строгого критерия.

**Что это требование делает статистически.** `signal_rate` — доля баров, где
`max(p, 1−p) ≥ порог`. Понижая порог, эту долю можно поднять произвольно;
повышая — обнулить. То есть требование ограничивает не качество модели, а
**согласованность порога с распределением её выходов**. Как критерий качества
оно не работает: модель, выдающая `p = 0.5 ± ε` на всех барах, провалит его
при пороге 0.55 и пройдёт при пороге 0.50.

Обратный эффект тоже реален: требование `signal_rate ≥ 0.10` создаёт давление
**понижать** порог, что прямо противоположно назначению порога.

Связь с живым состоянием системы: входной факт «live confidence 0.50–0.54
против порога 0.65» означает `signal_rate = 0` в проде при пороге 0.65, тогда
как манифесты бандлов записаны с `confidence_threshold: 0.55` (проход 2, §8) и
`signal_rate` 0.629 и 0.7733.

**Вывод:** порог взят произвольно; статистического обоснования в коде нет.

---

## 8. ПРЯМОЙ ОТВЕТ

> **Можно ли доверять хотя бы одному числу, которое система о себе сообщает?**
>
> **Из чисел статистического слоя — ни одному.**

Построчно:

| Число | Состояние |
|---|---|
| **DSR** | Не реализует формулу. На проверке с известным ответом даёт 0.689 вместо 0.500 (N3). Зависит от произвольного множителя `×10`: при `×100` даёт 1.0000 (N5). N занижен в ≥28 раз (N11). |
| **PBO** | Тождественно 0.0 при любых данных — 36 000 случайных наборов, ни одного ненулевого (N6). Не CSCV. Гейт `pbo ≤ 0.30` истинен всегда. |
| **t-stat** | Не тест пропорции. Знаменатель не зависит от числа сделок: 500 и 1020 сделок дают одинаковый `t = 5.0000` (N10). |
| **Sharpe** | В walk-forward не вычисляется. В DSR подставляется `(WR/100 − 0.5) × PF × 10`. Там, где вычисляется (валидаторы), — `√252` против `365` в `run_all_tests`. |
| **AUC** | Кодом **не вычисляется вообще** (`grep roc_auc` — пусто). |
| **win_rate** | Определён корректно как доля совпадений знака, но: валовый, без CI (±3.90 п.п. при n=610), без поправки на перекрытие (×1.82). |
| **profit_factor** | Валовый; `inf` при нуле убытков проходит гейт как «отлично» и схлопывается в 1.0 в DSR. |
| **signal_rate** | Величина корректна, но порог 0.10 произвольный и измеряет согласованность порога, а не качество. |
| **`passes`** | Вычисляется мягким критерием (WR/PF/signal_rate, gross), тогда как заявленный строгий критерий существует только как печатная строка. |

**Что всё-таки можно считать надёжным.** Не «числа о качестве», а
**инвентарные факты**: число строк в данных, диапазоны дат, число сигналов,
содержимое манифестов бандлов, длительности сделок. Они получены прямым
подсчётом и в проходах 2–3 воспроизводились независимо и сходились
(ESS 1048.8 против заявленных 1053; автокорреляция 0.635 против 0.64; SL/ATR
1.5 и TP/ATR 2.25 в 8 сделках из 8). То есть **учётный слой системе доверять
можно, измерительный — нет.**

Отдельно стоит отметить направление ошибок. Они **не** все в сторону
оптимизма: DSR при текущем масштабе прокси занижен (0.000092 против канонических
0.836), и это создаёт впечатление строгости там, где её нет. Опасен не знак, а
то, что знак **не определён**: он переворачивается при изменении произвольной
константы в прокси. Величина, чей знак смещения зависит от множителя, не
является измерением.

---

## РЕЕСТР НАХОДОК ПРОХОДА 3

| ID | Имя | Севирити | Тип |
|---|---|---|---|
| A2-024 | DSR не реализует формулу Bailey & LdP; результат зависит от произвольной константы | CRITICAL | математика |
| A2-026 | `calculate_pbo` тождественно равна нулю при любых данных | CRITICAL | математика / логика |
| A2-025 | N трейлов занижен минимум в 28 раз относительно `mlflow.db` | HIGH | математика |
| A2-027 | В walk-forward эмбарго объявлено и нигде не включено | HIGH | математика / недоделка |
| A2-028 | Ни одна метрика не сопровождается мерой неопределённости | HIGH | математика / расхождение с практикой |
| A2-029 | Гейт записи модели вычисляется на валовых метриках (A8) | HIGH | математика |
| A2-030 | Заявленный критерий go-live существует только как печатная строка | HIGH | архитектура / недоделка |

**Сводка прохода 3:** CRITICAL 2, HIGH 5. Всего 7.
**Нарастающим итогом (проходы 1–3):** CRITICAL 5, HIGH 14, MEDIUM 8, LOW 3 — 30.

### Статус находок Аудита-1 по этому предмету

| A-номер | Предмет | Статус |
|---|---|---|
| A4 | 4 дефекта DSR | **жива**; один механизм добавлен (`n_obs`) но не активен; дефект `(γ4−3)/4` **введён** «исправлением» (A2-024) |
| A4d | proxy Sharpe вместо per-fold P&L | **жива**; ветка реальных доходностей есть, не вызывается (A2-007, проход 2) |
| A5 | PBO — LOO, не CSCV | **жива и недооценена**: величина не эвристика, а константа 0.0 (A2-026) |
| A8 | eval gross без издержек | **жива** (A2-029) |
| A9 | слиппедж завышен ~19× | **жива**, коэффициент подтверждён точно: ×19.10 = √365 (N7) |
| A15 | t-stat против z-теста пропорции | **жива** (§5.4, N10) |
| A16 | PF при нуле убытков → inf | **жива** (§5.2) |
| A17 | `bar_duration_minutes` в аннуализации | **неприменима**: Sharpe в walk-forward не считается; породивший дефект жив как 252 против 365 (A2-007) |

---

## НЕ ИССЛЕДОВАНО

1. **`PurgedKFoldCV`** прочитан только в части эмбарго (`walk_forward.py:124-140`).
   Корректность purging по `t1` (а не только по времени) не проверена. Класс
   используется в `validate_*` через `MLValidator.purged_kfold_cv` — путь не
   прослежен до конца.
2. **Источник AUC 0.5375 из входных фактов не установлен.** Текущий код AUC не
   вычисляет. Возможные источники: `docs/retrain_v3_results.txt`, ноутбуки,
   удалённый код. Не искалось.
3. **`backtest_runner.py:352-357`** — третий путь применения издержек, не
   разобран (проход 7).
4. **`_compute_per_symbol`** (`lgbm_trainer.py:1129`) не читался; per-symbol
   метрики в манифестах не сверялись с агрегатом.
5. **Ранняя остановка LightGBM** и её взаимодействие с val-срезом не
   проверялись — влияет на то, сколько раз данные фактически «просмотрены»
   (то есть на честный N).
6. **2827 прогонов MLflow** приняты как нижняя граница N. Не проверено, сколько
   из них — независимые конфигурации, а сколько окна walk-forward одной
   конфигурации. Это может как увеличить (окна × конфигурации), так и
   уменьшить (повторы) оценку.
7. **`dsr_sensitivity`** строит таблицу той же сломанной функцией; отдельно не
   разбиралась.
8. **Тесты** (`tests/test_*statistical*`) не читались: не установлено, почему
   1923 теста не поймали PBO ≡ 0 и DSR-размерность. Проход 8.
