# AtomiCortex v3.0 — Research & Roadmap
## Дата: 2026-05-19
## Статус: COMPLETE

Глубокий исследовательский анализ production-системы AtomiCortex
(Nautilus Trader / Python / LightGBM, BTCUSDT/ETHUSDT/SOLUSDT
Binance Futures Perpetual, TF 4H/1H/15m).

Источники: arXiv, SSRN, GitHub, mlfinlab/FreqAI docs,
QuantConnect/Quant StackExchange. Актуальность: 2025–2026.

---

## ВОПРОС 1: Почему 1H модель неуверена?

### A) Near-50% confidence в LightGBM для финансов

**Это нормально и ожидаемо для intraday crypto.** Не баг,
а отражение низкого signal-to-noise ratio на 1H BTC.

- Atsalakis-style и обзорные работы по daily BTC дают R²≈0.43
  (LightGBM vs RF, *Comparative Analysis of LightGBM and Random
  Forest for Daily Bitcoin*, JAIC 2025) — это для **регрессии цены**;
  для направленной классификации это транслируется в accuracy
  ~52–55%, ровно ваш диапазон.
- *Bitcoin at High Frequency* (Scaillet et al., JRFM 2019) и
  *Stylized Facts of High-Frequency Bitcoin Time Series*
  (arXiv:2402.11930, 2024): предсказуемость BTC значима только
  до ~6h, на ≥1d исчезает; 1H автокорреляция **слабо
  отрицательная** и **time-varying** (после 2015 предсказуемость
  на 30m/1h резко сжалась и часто статистически незначима).
- Вывод: модель с avg proba 0.48–0.52 на 1H — это **корректная
  неуверенность**, НЕ underfitting, ЕСЛИ кривые train/val loss
  сошлись. Underfitting проверяется отдельно: train accuracy
  тоже ~52%? → данные/фичи слабые. Train 65%+, val 52% →
  overfit. У вас почти наверняка первое (информационный предел TF).

**Как с этим работают production-системы:**
- **FreqAI** (freqtrade docs): не торгует по argmax. Порог
  определяется не фиксированным 0.55, а **Youden-index точкой
  ROC-кривой** на валидации, ИЛИ `&-target` регрессией +
  z-score фильтр (`DI_threshold`, dissimilarity index) для
  отбраковки точек вне обучающего распределения. То есть берут
  не «уверенность», а **edge при данном пороге**.
- Confidence-threshold framework (*ML Analytics for
  Blockchain-Based Financial Markets*, Applied Sciences
  15(20):11145, 2025): отделять directional prediction от
  execution; торговать только верхний/нижний дециль proba.
  Типичная картина profitable intraday crypto: **общая accuracy
  53–56%, но в топ-дециле proba accuracy 60–68%** — edge живёт
  в хвостах распределения, не в среднем.

**Практический вывод для 1H:** не повышать порог до 0.55 на
сырой proba (даёт 0 сигналов), а:
1. калибровать (isotonic/Platt) и резать по квантилю обучающей
   proba (top/bottom 15–20%), не по абсолюту;
2. оценивать модель по precision@top-decile, не по global acc;
3. ввести meta-labeling слой (см. Вопрос 4B).

### B) Качество triple-barrier меток

- TBM (López de Prado, *Advances in Financial ML*, 2018):
  ваш class balance 28% UP / 72% DOWN — **симптом
  несимметричных барьеров** pt=1.5/sl=1.0. При нейтральном
  дрейфе SL ближе → срабатывает чаще → перекос вниз. Это
  напрямую тянет proba к одному классу и снижает «уверенность»
  на UP.
- *Algorithmic crypto trading using information-driven bars,
  triple barrier labeling and deep learning* (Financial
  Innovation, Springer, 2025): лучшие результаты дают
  **volatility-scaled барьеры** (pt/sl = k·σ_t, σ_t —
  rolling/EWMA волатильность на горизонте бара), а НЕ
  фиксированные множители ATR. Информационные бары
  (volume/dollar bars) дополнительно повышают качество меток
  vs time bars.
- *Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling*
  (Mathematics MDPI 12(5):780, 2024): pt/sl подбирали GA per
  режим — оптимум для crypto pair-trading вышел в зоне
  **pt≈1.0–1.3, sl≈0.8–1.0 (близко к симметрии), horizon
  короткий**. Сильная асимметрия pt>>sl ухудшала Sharpe из-за
  class imbalance.
- Рекомендация для 1H crypto futures 2025–26:
  **симметричные или near-symmetric vol-scaled барьеры**
  (pt≈sl≈1.0–1.5·EWMA-σ), отдельная калибровка по режиму;
  при сохранении дисбаланса — `scale_pos_weight` /
  sample weights, НЕ ресэмплинг (ломает temporal structure).
- Альтернативы: trend-scanning labels (López de Prado 2019,
  t-value наклона) — устойчивее к выбору барьеров; «zigzag/
  perfect-profit» меток избегать (lookahead).

### C) Prediction horizon

- Из autocorrelation-работ выше: значимая предсказуемость BTC
  ограничена ~6h. Ваш 1H max_holding=6 баров (=6h) **на самом
  пределе** информационного горизонта — это сознательный
  выбор, близкий к оптимуму, но барьерное время стоит
  тестировать в сетке **4/6/8 баров**.
- Autocorr decay BTC 1H: power-law, период 2021–22 — более
  быстрый decay (рынок эффективнее). Практически: edge на 1H
  концентрируется в **2–4h forward**, не 8h+.
- Связь horizon↔predictability: чем длиннее horizon, тем выше
  SNR на тренде, но тем реже чистые сигналы; для 1H sweet
  spot — 3–5h forward с vol-scaled барьером.

**Источники Q1:**
- López de Prado, *Advances in Financial Machine Learning*,
  Wiley 2018 (TBM, meta-labeling, uniqueness weights).
- Scaillet et al., «Bitcoin at High Frequency», *J. Risk
  Financial Manag.* 12(1):36, 2019.
- «Stylized Facts of High-Frequency Bitcoin Time Series»,
  arXiv:2402.11930v2, 2024.
- «Algorithmic crypto trading using information-driven bars,
  triple barrier labeling and deep learning», *Financial
  Innovation*, Springer 2025.
- «Enhanced GA-Driven Triple Barrier Labeling», *Mathematics*
  12(5):780, MDPI 2024.
- «ML Analytics for Blockchain-Based Financial Markets:
  Confidence-Threshold Framework», *Applied Sciences*
  15(20):11145, 2025.
- «Comparative Analysis of LightGBM and Random Forest for
  Daily Bitcoin», *JAIC* 2025.
- FreqAI documentation (freqtrade.io), 2025.

---

## ВОПРОС 2: Какие фичи дают наибольший edge на 1H?

### A) Order flow фичи которых нет

- **Multi-level OFI (MLOFI)** — самый сильный недостающий
  класс. *Forecasting High Frequency Order Flow Imbalance*
  (arXiv:2408.03594, 2024) + Cont/Kukanov/Stoikov-линия:
  включение imbalance на глубоких уровнях стакана даёт до
  **75% улучшения RMSE** прогноза mid-price vs single-level
  OFI на large-tick инструментах (BTCUSDT — large-tick).
- *Order Flow and Cryptocurrency Returns* (Anastasopoulos &
  Gradojevic, EFMA 2025): модели с order flow дают Sharpe
  **3.0–3.6 против 1.1–2.7** без него — крупнейший
  задокументированный economic gain среди фич-классов.
  OFI horizon- и **regime-dependent** (это плюс к вашему
  regime_label).
- Что доступно бесплатно с **Binance Data Portal /
  data.binance.vision**: aggTrades (для tick-rule CVD/
  taker-flow по сделкам), bookTicker (L1 best bid/ask —
  L1-OFI считается), metrics (5m sum OI), depth snapshots
  ограниченно. Полный L20 book — НЕ бесплатно исторически
  (нужен сбор live через WS `depth20@100ms` или
  Tardis.dev / CryptoLake платно).
- **Реалистично для 1H без покупки данных:**
  - L1-OFI из bookTicker (есть исторически бесплатно) —
    агрегировать в 1H: sum/EWMA, sign-persistence;
  - trade-flow OFI из aggTrades (signed volume, tick rule);
  - **VPIN** на volume-buckets из aggTrades — работает без
    L2 book; для 1H брать VPIN с 50–100 buckets, как фичу
    toxicity (Easley/López de Prado/O'Hara 2012, актуален).
- **L/S ratio топ-трейдеров (Binance API
  `topLongShortPositionRatio`, period=1h)**: топ-20% по
  марже считаются информированными; как 1H фича — уровень
  + дельта + z-score. Это sentiment/positioning фактор,
  работает как contrarian на экстремумах и trend-confirm
  в середине. Доступно через REST бесплатно, исторически
  тоже (futures data portal: `metrics`).

### B) Funding rate паттерны

- *Predictability of Funding Rates* (Inan, SSRN 5576424,
  2024): funding **сам по себе предсказуем** DAR-моделью
  (бьёт no-change по directional accuracy) — значит фичи
  типа funding_forecast / funding_momentum несут сигнал
  сверх текущего значения.
- Presto Labs, *Can Funding Rate Predict Price Change?*
  (2024): значимые **изменения** funding скоррелированы с
  последующим движением цены; уровень менее информативен
  чем скорость/ускорение → ваши funding_change_1/3,
  funding_zscore — правильное направление, добавить
  **funding_momentum знак-persistence** и
  **funding × OI interaction** (растущий OI + растущий
  funding = переполненная сторона).
- *Two-Tiered Structure of Crypto Funding Rate Markets*
  (Mathematics MDPI 14(2):346, 2026; 35.7M 1-мин
  наблюдений, 26 бирж): Granger-causality между биржами →
  **cross-exchange funding spread** (Binance vs Bybit/OKX)
  — отдельный leading сигнал; стоит добавить как фичу,
  данные дешёвы (REST по каждой бирже).
- **Pre-funding window effect**: академически
  подтверждён слабо для крупных коинов на больших биржах
  (арбитражеры выравнивают за минуты до сеттлмента);
  ваш `pre_funding_window` лучше работает на 15m, на 1H
  эффект почти полностью съедается размером бара —
  держать, но не ждать большого edge на 1H.

### C) Межрыночные зависимости

- **BTC→ETH lead-lag на 1H**: Sifat, Mohamad & Shariff,
  *Lead-Lag between Bitcoin and Ethereum* (Research in
  Int. Business & Finance 50:306-321, 2019, hourly):
  causality в основном **bi-directional**, чистый
  arbitrageable lead на hourly **минимален** (рынок
  эффективен). Практический вывод: BTC-as-feature для
  ETH/SOL модели работает не как «BTC лидирует N баров»,
  а как **contemporaneous beta / dominance regime** и
  divergence (ETH/SOL vs implied-by-BTC).
- **TradFi correlation на 1H в 2026**: корреляция
  BTC–S&P 500 крайне **нестабильна** (30d corr скакала
  0.18 → 0.74 → 0.88 за 2025–нач.2026; Nasdaq100 ~0.35–
  0.52). Вывод: статичная фича «corr с TradFi» бесполезна
  на 1H; полезен только **rolling regime-flag**
  (risk-on/off), и то RTH-сессии США — у вас уже есть
  session-фичи, этого достаточно. Прямой TradFi-feed на
  1H crypto в 2026 — низкий приоритет.
- **Dominance ratio (BTC.D)**: предсказательная сила для
  альтов средняя; полезнее как **ETH/BTC и SOL/BTC
  relative-strength** фича для multi-symbol модели, чем
  абсолютный BTC.D.

### D) Чем profitable 1H отличается от unprofitable

Из SHAP-обзоров (*Explainable AI to forecast Bitcoin
prices*, White Rose 2023; *Explainable Patterns in
Crypto Microstructure*, arXiv 2602.00776 2026; Coinbase
Institutional ML Outlook 2024; ACM Ethereum multi-factor
2025) топ-фичи в опубликованных моделях:

1. **Order-flow imbalance / signed volume** (микроструктура)
   — есть только частично (cvd), нет MLOFI/VPIN.
2. **EMA-cross / MA-distance normalized** — есть (alpha-v2).
3. **Volatility/ATR regime** — есть.
4. **RSI / mean-reversion oscillators** — **НЕТ** (нет
   ни одного нормированного осциллятора перекупленности).
5. **On-chain / flow** для daily (не для 1H).

**Чего принципиально нет в вашем наборе (по приоритету):**
- MLOFI / L1-OFI / VPIN (order-flow toxicity) — **#1 gap**;
- RSI/Stoch-style mean-reversion осциллятор (нормированный);
- cross-exchange funding spread;
- relative-strength ETH/BTC, SOL/BTC (для MULTI);
- funding×OI и flow×regime **interaction**-фичи
  (LightGBM ловит взаимодействия, но явные ускоряют).

**Источники Q2:** arXiv:2408.03594 (2024);
Anastasopoulos & Gradojevic EFMA 2025; Inan SSRN
5576424 (2024); Presto Labs (2024); *Mathematics*
14(2):346 MDPI 2026; Sifat et al. RIBAF 50:306 (2019);
ainvest/Phemex BTC–SPX correlation trackers (2025–26);
White Rose XAI Bitcoin (2023); arXiv:2602.00776 (2026);
Coinbase Institutional ML Outlook (July 2024);
Easley, López de Prado, O'Hara «VPIN» RFS 25 (2012).

---

## ВОПРОС 3: MTF как улучшение моделей

### A) LTF сигналы → улучшение HTF моделей

- Работает, но как **агрегаты, не raw**. Подтверждённый
  паттерн (*Neural Network-Based Algo Trading: Multi-TF
  Analysis and HF Execution in Crypto*, arXiv:2508.02356,
  2025): LTF используется для микроструктуры/тайминга,
  HTF — для контекста; LTF microstructure, агрегированный
  в HTF-фичу, повышает direction accuracy.
- Конкретно для вашей 4H-модели: добавить агрегаты 15m
  внутри 4H-бара:
  - **aggregated 15m CVD** (sum signed volume 16×15m) и
    его наклон — раньше отражает накопление/распределение,
    чем 4H-CVD;
  - **15m realized vol / vol-of-vol** внутри 4H-бара
    (intrabar volatility signature) — сильный режимный
    предиктор;
  - доля 15m-баров с ORB_BREAKOUT-режимом внутри 4H —
    связывает вашу рабочую 15m-модель с 4H.
  Числа: в arXiv:2508.02356 multi-TF фичи давали прирост
  направленной точности порядка единиц % vs single-TF;
  реалистично ждать **+1–3% accuracy / +0.1–0.3 Sharpe**,
  не больше.
- ⚠️ Skew-риск: агрегаты 15m в live должны считаться
  ровно тем же кодом, что в train (см. Вопрос 4A — единый
  feature store обязателен, иначе повторите 4H funding/OI
  проблему).

### B) Cross-timeframe ensemble & meta-labeling

- **Stacking 1H+4H+15m**: правильная схема не «усреднить
  proba», а **meta-labeling по López de Prado**:
  1. базовые модели каждого TF дают сторону (primary
     signal);
  2. **meta-модель** (отдельный LightGBM) на фичах
     {proba каждого TF, mtf_alignment, regime, volatility,
     согласованность сторон} предсказывает **бинарно:
     брать сделку или нет** (precision-ориентир);
  3. размер позиции ∝ proba meta-модели.
  Это именно то, что чинит «1H неуверена»: 1H перестаёт
  быть solo-сигналом, становится фичой meta-слоя; 0
  сигналов при 0.55 → meta-слой сам учит порог.
- Hudson & Thames (*Does Meta Labeling Add to Signal
  Efficacy?*, 2025): meta-labeling стабильно повышает
  precision и Sharpe при сохранении recall на разумном
  уровне — **главный single-наибольший рычаг** для вашей
  ситуации.
- Взвешивание TF: не равное. Вес ∝ (OOS-DSR данного TF
  на текущем режиме). У вас 15m-ORB DSR=1.0 на
  ORB_BREAKOUT, 4H PF~2 → меню весов **режим-зависимое**,
  что естественно ложится на meta-фичу regime_label.

### C) 5m и 1m специфика

- **5m рентабельность**: round-trip Binance Futures
  реально **0.25–0.30%** ($1k BTC, нормальные часы),
  не 0.2% из бэктестов; реальный WR на ~6 п.п. ниже
  бэктеста (*StratProof*, калибровочное исследование
  апр. 2026; 16 из 22 стратегий убыточны на реальных
  комиссиях). Минимальный edge для 5m: **gross EV/trade
  > ~2× round-trip ≈ 0.5–0.6%** на сделку, иначе coin
  flip. С maker-only (limit, rebate/0 fee) порог падает,
  но добавляется fill-risk.
  → 5m имеет смысл только: (а) maker-execution, либо
  (б) очень селективный режим (ORB-подобный, как ваш
  15m), таргет ≥ 0.6% net, низкая частота.
- **5m vs 15m microstructure**: на 5m доминирует
  order-flow toxicity/HFT-шум; ваш 15m-ORB-эдж на 5m
  частично съедается комиссией и конкуренцией. Optimal
  holding 5m: короткий, 3–8 баров (15–40 мин), строгий
  vol-scaled barrier.
- **1m без co-location**: нереально для систематического
  edge — конкуренция с HFT/market-makers, латентность
  retail-API (десятки–сотни мс) убивает любой
  microstructure-сигнал. Вывод: **1m не делать**
  (обоснование в roadmap «Не делать»).

**Источники Q3:** arXiv:2508.02356 (2025); López de
Prado *Advances in Financial ML* гл.3 (meta-labeling);
Hudson & Thames «Does Meta Labeling Add to Signal
Efficacy?» (2025); StratProof «22 strategies on real
fees» / calibration study (2026); Binance Futures fee
schedule (2025–26).

---

## ВОПРОС 4: Архитектурные улучшения

### A) Инфраструктура (ПРИОРИТЕТ #1) — live funding/OI

**Nautilus имеет это нативно — не надо городить
cryptofeed.** (nautilustrader.io/docs binance integration,
GitHub Discussion #1625):
- Rust-адаптер Binance эмитит **`FundingRateUpdate`** как
  first-class data type через
  `subscribe_funding_rates(instrument_id)`; обрабатывается
  в `on_data`.
- `BinanceFuturesMarkPriceUpdate` стрим содержит funding
  info (mark price + funding rate) — подписка в
  `on_start`, приём в `on_data`.
- Open Interest: REST `futures/data/openInterestHist`
  (5m granularity) + `/fapi/v1/openInterest` (текущий) —
  poll'ить в actor по таймеру; нет нативного WS-стрима
  OI у Binance, поэтому **periodic REST poll** (каждые
  1–5 мин) — корректная архитектура.
- cryptofeed (PyPI) поддерживает Funding + OpenInterest
  channels — fallback, если нужен cross-exchange, но для
  Binance-only это лишний слой.

**Правильная архитектура feature store (фикс train/serve
skew — корень и 4H, и 1H проблем):**
1. **Единая функция фичей** `compute_features(df)` —
   ОДИН модуль, вызывается и в offline-обучении, и в
   live `on_bar`. Сейчас `_compute_features()` ручной и
   расходится с offline → плейсхолдеры funding/OI=0.
   Это **критический баг**, не «известная проблема».
2. **Online feature buffer**: rolling-окно баров +
   последние FundingRateUpdate / OI в state стратегии;
   фичи считаются из буфера тем же кодом.
3. **Контракт-тест skew**: на каждом релизе прогонять
   offline `compute_features` и live-replay на одном
   историческом срезе, assert max abs diff < ε по каждой
   колонке. Без этого теста любая фича-правка
   реинтродуцирует skew.
4. **Point-in-time дисциплина**: funding известен только
   на/после settlement; OI с лагом publish — лагировать
   на 1 бар, не использовать значение «текущего
   незакрытого» бара.

### B) Sample weights

López de Prado *AFML* 2018, гл.4 (mlfinlab
`get_av_uniqueness_from_triple_barrier`):
- **Concurrency**: для каждого бара c_t = число активных
  (открытых) меток. **Uniqueness** метки i:
  ū_i = mean_t( 1/c_t ) по времени жизни метки.
- **Sample weight** ∝ ū_i (либо return-attribution:
  w_i ∝ |Σ_t (r_t / c_t)| — взвешивает по абсолютной
  доходности на бар, нормированной на конкуренцию).
- Когда нужны: **всегда при overlapping labels** —
  ваш 1H max_holding=6 и 15m=8 → метки сильно
  перекрываются, обучение без weights = переоценка
  уверенности и leakage концентрации → **прямой вклад
  в «модель неуверена / переобучена на кластерах»**.
- Влияние в published results: López de Prado / Hudson
  & Thames sequential bootstrap + uniqueness weights
  обычно дают **+3–8% OOS accuracy и заметный рост DSR**
  на overlapping-label задачах; снижают вариативность
  feature importance. Return-attribution weights нужны,
  когда хотите, чтобы модель приоритизировала
  крупно-движущие события (полезно для PF-ориентира).
- Реализация: LightGBM `sample_weight=` в `fit`;
  для бэггинга — **sequential bootstrap** вместо
  стандартного (mlfinlab `seq_bootstrap`).

### C) Online learning

- **Concept drift в BTC 2024–26**: режимная нестабильность
  высокая (см. Q2C — corr с TradFi прыгает 0.18↔0.88 за
  месяцы; funding-структура меняется). Drift скорее
  **gradual + периодические sudden** (ETF-флоу, делеверидж-
  волны).
- **Walk-forward частота**: для 1H/15m моделей —
  **еженедельный** rolling refit (expanding или sliding
  ~12–18 мес окно) оптимален: дневной даёт переобучение
  на шуме и операционный риск, месячный отстаёт от
  drift. Drift-aware ретрейн (триггер по PSI/KS на
  распределении фич или просадке live precision) «в
  пределах 1% от периодического, но эффективнее»
  (drift-retraining literature 2025) — практичнее:
  **еженедельный refit + drift-триггер на досрочный**.
- **Incremental LightGBM**: поддерживается частично —
  `init_model=` + `keep_training_booster=True`
  продолжает обучение/добавляет деревья, НО это не
  истинный online (старые деревья замораживаются, дрейф
  плохо забывается). Для финансов **rolling full refit
  предпочтительнее** инкрементального; инкремент только
  для тёплого старта между еженедельными рефитами.

### D) Feature selection (84 фичи → отбор)

- MDI/MDA/SHAP все страдают **substitution effect** на
  коррелированных фичах (López de Prado; Man & Chan,
  *Cluster-based Feature Selection*, SSRN 3880641, 2021):
  при дублирующих фичах важность размазывается, обе
  выглядят нерелевантными.
- **Решение — Clustered MDA (cMDA)**, López de Prado
  2020 / *Machine Learning for Asset Managers*:
  1. корреляционная матрица фич → иерархическая
     кластеризация (1−|ρ| или информационная метрика);
  2. перестановка **всего кластера** разом → важность
     на уровне кластеров (устойчива к substitution);
  3. из каждого значимого кластера оставить 1–2
     представителя.
  Man & Chan: cMDA повышает predictive performance и
  **стабильность** отбора vs plain MDA/SHAP/LIME.
- **Что у вас почти точно избыточно / под нож:**
  - `returns_1/3/6/12/24` — высокая взаимная корреляция;
    оставить 2–3 (напр. 1, 6, 24) или заменить на
    EWMA-momentum;
  - `cvd_slope_3/6/12` + `funding_change_1/3` +
    `funding_zscore` — кластер деривативов, проредить;
  - `session_hour_sin/cos` + `trading_session` +
    `session_*` — сессионный кластер, многие дублируют
    друг друга;
  - `ema9/21_slope_normalized` vs `ema9_cross_ema21` vs
    `efficiency_ratio_10/20` — трендовый кластер;
  - `basis_approx`, `cvd_divergence`,
    `price_oi_divergence`, `oi_price_div_vec` — проверить
    на near-zero cMDA и низкую дисперсию (часто шум на 1H).
  Цель: с ~84 до **~25–35 некоррелированных** фич —
  меньше шума → выше калибровка proba (прямо помогает
  1H уверенности).

**Источники Q4:** NautilusTrader docs (Binance
integration, 2025–26) + GitHub Discussion #1625;
cryptofeed PyPI (2025); López de Prado *AFML* (2018)
гл.4 + *ML for Asset Managers* (2020); Man & Chan SSRN
3880641 (2021); mlfinlab (Hudson & Thames) docs;
drift-aware retraining literature (fxis/arXiv 2025);
LightGBM docs (`init_model`/continued training).

---

## ВОПРОС 5: Специфика каждого TF

**1M.** HFT-доминируемый. Edge на 1m — это
order-flow/queue-position, требует co-location и
tick/L2-данных; retail-API латентность убивает сигнал.
Минимальный edge на сделку должен покрывать ≥0.25–0.30%
round-trip при крайне частой торговле → нереально без
maker-инфраструктуры. **Вердикт: не торговать.** Tick-
данные сами по себе не спасают без латентности.

**5M.** ORB применим, но эдж тоньше 15m и почти весь
съедается комиссией (Q3C): нужен net ≥0.5–0.6%/trade,
maker-execution или жёсткая селективность. Microstructure
vs 15m: больше toxicity/шум, короче память. Optimal
holding 5m: 3–8 баров (15–40 мин), vol-scaled barrier.
Условно-жизнеспособен только как узко-режимная maker-
стратегия — низкий приоритет.

**1H.** Принципиальное отличие от 4H: на 1H **SNR
заметно ниже**, автокорреляция слабо-отрицательная и
time-varying, предсказуемость на грани горизонта (≤6h);
4H ближе к funding-периоду и агрегирует шум. Поэтому 1H
требует: meta-labeling (не solo-сигнал), sample weights,
vol-scaled симметричные барьеры, order-flow фичи.
Наиболее предсказуемый режим на 1H — **трендовый/breakout
с подтверждением order-flow и MTF-alignment** (а не
range/mean-reversion, где 1H-шум максимален).

**4H.** Работает лучше всех, потому что: (1) период
кратен funding (8h = 2 бара) → funding/OI-сигналы чище;
(2) агрегирование подавляет microstructure-шум, растёт
SNR; (3) меньше сделок → комиссии незначимы (PF~2
устойчив). Optimal features 4H: funding/OI-режим,
agg-15m CVD/vol (Q3A), regime/volatility,
MTF-context. Главный риск 4H — train/serve skew
(funding/OI=0 live) — критический фикс.

**Daily.** Минимальный датасет: 2022–2025 ≈ ~1200–1400
дневных баров — мало для сложной модели, но достаточно
для простой режимной/трендовой (или как HTF-контекст).
On-chain реально и полезно именно на daily (Glassnode/
free CryptoQuant-подобные: active addresses, exchange
netflow, SOPR) — это лучший TF для on-chain;
предсказуемость на daily структурно ниже intraday
(autocorr-работы: ≥1d предсказуемость ≈ 0), поэтому
daily — для контекста/режима, не для solo-альфы.

---

## AtomiCortex v3.0 Roadmap

Принцип: **сначала инфраструктура (train/serve skew),
потом метки и веса, потом фичи, потом ансамбль**.
Никакой новой фичи нет смысла добавлять, пока live ≠ train.

### Критические (делать первыми)

#### 1. Единый feature store + фикс train/serve skew
**Таймфреймы:** 4H, 1H, 15m (все)
**Проблема:** `_compute_features()` ручной в live,
funding/OI=0 плейсхолдеры → модель в live видит другие
данные, чем в train. Корень слабых live-результатов 4H
и недоверия к 1H.
**Решение:**
1. вынести ОДНУ `compute_features(df, state)` в общий
   модуль, удалить ручной live-путь;
2. в стратегии вести rolling-буфер баров + последний
   funding/OI в state;
3. контракт-тест: offline vs live-replay на одном
   срезе, assert max|Δ| < 1e-6 по каждой колонке в CI.
**Критерии приёмки:** skew-тест зелёный; live feature
snapshot == offline на том же баре; funding/OI ≠ 0 в
live логах.
**Ожидаемый impact:** 4H live приближается к бэктесту
(сейчас разрыв из-за skew); снимает ложный «1H NO-GO».
**Приоритет:** КРИТИЧЕСКИЙ

#### 2. Live funding/OI feed через нативный Nautilus
**Таймфреймы:** 4H (сильнее всего), 1H
**Проблема:** нет реального funding/OI в live.
**Решение:** `subscribe_funding_rates` +
`BinanceFuturesMarkPriceUpdate` в `on_start`/`on_data`;
periodic REST-poll OI (`openInterestHist`, 1–5 мин) в
actor; point-in-time лаг funding на settlement, OI на
1 бар.
**Критерии приёмки:** в live state ненулевые
актуальные funding/OI; значения совпадают с REST
ground-truth ±1 тик.
**Ожидаемый impact:** восстанавливает derivatives-блок
фич 4H/1H (это топ-importance класс).
**Приоритет:** КРИТИЧЕСКИЙ

#### 3. Sample weights (uniqueness) + sequential bootstrap
**Таймфреймы:** 1H, 15m (overlapping labels), 4H
**Проблема:** перекрывающиеся triple-barrier метки →
переоценка уверенности, leakage кластеров, нестабильный
importance.
**Решение:** `get_av_uniqueness_from_triple_barrier`
→ `sample_weight` в LightGBM.fit; sequential bootstrap
вместо bagging-random.
**Критерии приёмки:** OOS DSR не падает, feature
importance стабильнее между фолдами; калибровка proba
(reliability curve) ближе к диагонали.
**Ожидаемый impact:** +3–8% OOS accuracy на 1H,
рост DSR; реалистичнее доверять proba.
**Приоритет:** КРИТИЧЕСКИЙ

### Высокие (после критических)

#### 4. Vol-scaled симметричные triple-barrier + сетка
**Таймфреймы:** 1H (приоритет), 15m
**Проблема:** pt=1.5/sl=1.0 фикс → дисбаланс 28/72,
смещённая proba.
**Решение:** барьеры = k·EWMA-σ_t, near-symmetric;
сетка pt∈{1.0,1.25,1.5}, sl∈{0.8,1.0}, horizon∈{4,6,8}
по DSR на walk-forward; при остаточном дисбалансе —
`scale_pos_weight`, не ресэмплинг.
**Критерии приёмки:** class balance → 40–60%; OOS DSR
не ниже текущего; снижение разрыва UP/DOWN precision.
**Ожидаемый impact:** 1H WR 52→55–58%, рост числа
валидных сигналов.
**Приоритет:** ВЫСОКИЙ

#### 5. Meta-labeling слой (MTF-ансамбль)
**Таймфреймы:** ансамбль 1H+4H+15m
**Проблема:** 1H solo неуверена (0 сигналов @0.55).
**Решение:** базовые модели дают сторону; отдельный
LightGBM-meta на {proba каждого TF, mtf_alignment,
regime, volatility, согласие сторон} → бинарно
take/skip; size ∝ meta-proba; порог по precision-target,
не по абсолюту.
**Критерии приёмки:** meta повышает precision при
recall≥ разумного; OOS Sharpe ансамбля > max(одиночных);
1H перестаёт быть «0 сигналов».
**Ожидаемый impact:** ансамбль Sharpe +0.2–0.5 vs
лучший одиночный; 1H вносит вклад как фича.
**Приоритет:** ВЫСОКИЙ

#### 6. Order-flow фичи: L1-OFI + trade-flow OFI + VPIN
**Таймфреймы:** 1H (приоритет), 15m, 4H
**Проблема:** нет самого сильного класса фич
(documented Sharpe 3.0+ vs 1–2 без него).
**Решение:** L1-OFI из bookTicker, signed trade-flow
OFI и VPIN из aggTrades (бесплатно с data.binance.vision),
агрегаты в TF; добавить в общий feature store (п.1).
**Критерии приёмки:** новые фичи в топ-кластерах cMDA;
OOS DSR растёт vs baseline.
**Ожидаемый impact:** наибольший среди фич-улучшений;
реалистично +0.2–0.5 Sharpe на 1H.
**Приоритет:** ВЫСОКИЙ

#### 7. Clustered-MDA feature selection (84 → ~30)
**Таймфреймы:** все
**Проблема:** избыточные коррелированные фичи → шум,
плохая калибровка, нестабильный importance.
**Решение:** иерархическая кластеризация по |ρ| →
cMDA → 1–2 представителя на кластер (кандидаты на
удаление перечислены в Q4D).
**Критерии приёмки:** ≤35 фич, OOS DSR не ниже; рост
стабильности importance между фолдами; лучше reliability
curve.
**Ожидаемый impact:** косвенный — улучшает калибровку
1H, ускоряет refit, снижает overfit.
**Приоритет:** ВЫСОКИЙ

### Средние (после получения результатов)

#### 8. Доп. фичи: RSI/Stoch-нормированный, cross-exchange
funding spread, relative-strength ETH/BTC, SOL/BTC,
funding×OI и flow×regime interaction
**Таймфреймы:** 1H, MULTI
**Проблема:** нет mean-reversion-осциллятора и
cross-asset/cross-exchange сигналов.
**Решение:** добавить через feature store, прогнать
через cMDA, оставить только значимые.
**Критерии приёмки:** попадают в значимые кластеры;
DSR не падает.
**Ожидаемый impact:** скромный, +0.05–0.2 Sharpe.
**Приоритет:** СРЕДНИЙ

#### 9. Еженедельный walk-forward refit + drift-триггер
**Таймфреймы:** 1H, 15m
**Проблема:** concept drift BTC 2024–26.
**Решение:** rolling refit (12–18 мес окно) еженедельно;
PSI/KS на фичах + live-precision-просадка → досрочный
refit; warm-start через LightGBM `init_model`.
**Критерии приёмки:** live precision стабильна между
рефитами; drift-триггер срабатывает на known regime
shifts в бэктесте.
**Ожидаемый impact:** удержание edge во времени, без
прироста пикового WR.
**Приоритет:** СРЕДНИЙ

#### 10. Agg-15m фичи в 4H-модель
**Таймфреймы:** 4H
**Решение:** agg-15m CVD/наклон, intrabar realized
vol, доля ORB_BREAKOUT 15m внутри 4H — через feature
store + skew-тест.
**Критерии приёмки:** в топ-кластерах cMDA; OOS DSR↑.
**Ожидаемый impact:** +1–3% accuracy / +0.1–0.3 Sharpe
на 4H.
**Приоритет:** СРЕДНИЙ

### Не делать (с обоснованием)

- **1m TF** — HFT-доминирован, нужен co-location и
  tick/L2 + субмиллисекундная латентность; retail-API
  не конкурентен, edge < комиссий. (Q5/Q3C)
- **Прямой TradFi-feed (S&P/Nasdaq) как 1H-фича** —
  корреляция нестабильна (0.18↔0.88 за месяцы 2025–26),
  статичная фича бесполезна; risk-on/off уже покрыт
  session-фичами. (Q2C)
- **Истинный online/incremental LightGBM в проде** —
  замороженные деревья плохо забывают drift; rolling
  full refit надёжнее. (Q4C)
- **Ресэмплинг для class imbalance** — ломает temporal
  structure; использовать sample/scale_pos_weight. (Q1B)
- **Покупка платных L20-book данных сейчас** — L1-OFI +
  trade-flow OFI + VPIN с бесплатного Binance portal
  дают 80% эффекта; платный book — только если п.6
  докажет ценность order-flow. (Q2A)
- **Повышение сырого proba-порога до 0.55 на 1H** —
  даёт 0 сигналов; вместо этого калибровка + квантильный
  порог + meta-labeling. (Q1A)

---

## Ожидаемые результаты v3.0 (реалистично)

| TF  | Сейчас | Цель v3.0 (реалистично) |
|-----|--------|--------------------------|
| 4H  | WR 62–69%, PF 1.8–2.5, мало сигналов; live≠backtest | live ≈ backtest после skew-фикса; PF 1.8–2.4 стабильно; +20–40% валидных сигналов за счёт agg-15m + order-flow |
| 15m ORB | WR 76.6%, Sharpe 2.18, DSR 1.0, 338 OOS | удержать WR ~70–76% на бóльшей выборке (ожидать регрессию к среднему ~−3–6 п.п.); расширить за счёт sample weights |
| 1H  | WR 52%, 0 сигналов @0.55, NO-GO | WR 55–58%, precision@top-decile 60–66%; вносит вклад через meta-слой; **самостоятельно — осторожный GO, в ансамбле — да** |
| Ансамбль (meta) | — | Sharpe лучшего одиночного +0.2–0.5; основной production-режим |

**Честная оговорка:** 1H вряд ли станет «сильной»
solo-моделью — информационный предел TF (предсказуемость
BTC ≤6h, autocorr ≈0). Реалистичная роль 1H в v3.0 —
**компонент meta-ансамбля**, а не отдельная стратегия.
Никаких 80–90% WR: документированные profitable intraday
crypto-модели живут в зоне accuracy 53–58% global /
60–68% в топ-дециле. Все приросты — аддитивны и скромны;
наибольший единичный рычаг — meta-labeling (#5) и
order-flow фичи (#6) поверх честной инфраструктуры
(#1–#3).

---

## Статус: COMPLETE (2026-05-19)
