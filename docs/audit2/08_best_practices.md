# АУДИТ-2 · ПРОХОД 7 — СВЕРКА С ПРАКТИКОЙ ИНДУСТРИИ

**Файл:** `docs/audit2/08_best_practices.md`
**Дата прохода:** 2026-08-23
**HEAD:** `f4af5fd210b32db8af4478f0e2f440e2eb504ccc` (`2026-08-22 16:18:08 +0530`, `chore: ignore local snapshot directory`)
**Рабочее дерево:** `git status --short` → `?? docs/audit2/` (только каталог отчётов; `src/` не тронут)

---

## 0. Метод этого прохода

Предмет — расхождение реализации с тем, как устроены референсные реализации,
официальные контракты бирж и первоисточники метода. Стиль, мода и вкусовщина
не рассматриваются: в отчёт попадает только то, что меняет число или поведение.

### 0.1 Правило достоверности

Использованы основания только четырёх допустимых видов. Каждое утверждение
о внешнем мире снабжено ссылкой и датой обращения. Там, где основания нет,
стоит «не проверено» — без формулировок вида «принято считать».

| # | Вид основания | Использовано в проходе |
|---|---|---|
| 1 | Исходный код референсной реализации | `.venv/.../nautilus_trader/**` (1.221.0), `.venv/.../lightgbm/**` (4.3.0), `raw.githubusercontent.com/microsoft/LightGBM/v4.3.0/**` |
| 2 | Официальная документация библиотеки/биржи | `lightgbm.readthedocs.io/en/v4.3.0`, `scikit-learn.org`, `binance.com/en/support/faq`, `developers.binance.com` |
| 3 | Рецензируемая статья / книга | Bailey & López de Prado (2014) DSR; Bailey, Borwein, López de Prado, Zhu (PBO/CSCV); López de Prado, *Advances in Financial Machine Learning* (AFML) |
| 4 | Открытый код признанного проекта | `github.com/rubenbriones/Probabilistic-Sharpe-Ratio` (оценка зрелости — в §4.5) |

### 0.2 Измерения, выполненные в этом проходе

Все числа ниже получены свежим прогоном на машине аудита. Обучение, бот и
`pytest` не запускались. Единственный сетевой вызов к бирже — публичный
`GET /fapi/v1/exchangeInfo` (чтение, без ключей).

| Что | Команда | Результат |
|---|---|---|
| Часы расчёта funding в данных проекта | polars по `data/features/exchange=BINANCE_UM/symbol=*/funding_rate/**` | BTC/ETH/SOL: по 2889 записей, часы **строго {0: 963, 8: 963, 16: 963}** |
| `close_time − open_time` для 4H | polars по `klines_4h/**` (963 файла) | единственное значение **14399999** мс на 5778 строк |
| Часы открытия 4H-баров | там же | {0,4,8,12,16,20} — по 963 каждый |
| Фильтры инструментов | `GET https://fapi.binance.com/fapi/v1/exchangeInfo` | см. §1.4 |
| Сдвиг funding-окна | прогон `src.features.session_features` на сетках 4H/1H | см. §1.1 |
| E[SR_max] и SE(SR) | numpy/scipy по формулам кода и статьи | см. §4.5 |

---

## 1. БИРЖЕВОЙ КОНТРАКТ BINANCE UM

### 1.1 Время расчёта funding

**Что говорит биржа.**
Официальная FAQ Binance «Introduction to Binance Futures Funding Rates»
(https://www.binance.com/en/support/faq/360033525031, обращение 2026-08-23):

> «00:00 (UTC), 08:00 (UTC), and 16:00 (UTC)»
> «The default funding interval is every 8 hours.»

Там же зафиксировано, что частота расчёта не константа:

> «Effective from 2025-05-02 08:00 (UTC), Binance Futures will adjust the
> settlement frequency from every eight hours or every four hours to every
> one hour when the previous funding rate settlement of USDⓈ-M perpetual
> contracts reaches the funding rate cap or floor.»

> «Effective from 2026-01-02 12:00 (UTC), if the funding rate of USDⓈ-M
> perpetual contracts with funding rate settlement frequency of every one
> hour is less than or equal to the absolute value of 0.025% for 16
> consecutive cycles, Binance Futures will revert the settlement frequency
> from every one hour to every four hours.»

Эндпойнт `GET /fapi/v1/fundingInfo` отдаёт `fundingIntervalHours` — то есть
интервал биржа считает атрибутом символа, а не глобальной константой
(https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-Info,
обращение 2026-08-23).

**Что в данных проекта.** Замер по собственным parquet-файлам проекта:

```
BTCUSDT rows 2889 {0: 963, 8: 963, 16: 963}
ETHUSDT rows 2889 {0: 963, 8: 963, 16: 963}
SOLUSDT rows 2889 {0: 963, 8: 963, 16: 963}
min 2024-01-01 00:00:00  max 2026-08-20 16:00:00.005000
```

Ни одной записи вне {00, 08, 16} за 963 дня по трём символам. Для этих трёх
символов период 8 ч держался весь диапазон данных — динамического перехода
на 1 ч/4 ч в выборке нет.

**Что в коде.** Две независимые константы:

`src/features/session_features.py:60`
```python
_FUNDING_MARKS = [1, 9, 17]
```

`src/execution/strategies/ml_strategy.py:626`
```python
                if dt.hour in (1, 9, 17) and dt.minute == 0:
```

**Вердикт.** Верно 00/08/16; 01/09/17 неверно. Менялась не сетка часов,
а *частота* (8 ч → 4 ч → 1 ч и обратно) — сдвига сетки на +1 ч в истории
Binance по документации не обнаружено; см. §9, п. 1.

Последствие в рантайме — уже зафиксированная **A2-060** (история funding
в live не пополняется). Здесь добавляется второй, независимый потребитель
той же ошибочной константы — признаки `session_features`, и он ведёт себя
иначе. Замер (прогон `_FUNDING_MARKS` на обеих сетках):

```
--- 4H grid | code marks=[1, 9, 17] ---
 hours_to_funding_mark  code: [1.0, 5.0]  true: [0.0, 4.0]
 pre_funding_window(<=2) code: [0, 8, 16]  true: [0, 8, 16]
 post_funding_window(<=1) code: []  true: [0, 8, 16]
--- 1H grid | code marks=[1, 9, 17] ---
 pre_funding_window(<=2) code: [0, 1, 7, 8, 9, 15, 16, 17, 23]  true: [0, 6, 7, 8, 14, 15, 16, 22, 23]
 post_funding_window(<=1) code: [1, 2, 9, 10, 17, 18]  true: [0, 1, 8, 9, 16, 17]
```

`pre_funding_window` на 4H совпадает случайно (дистанция 1 ч и 0 ч обе ≤ 2 ч),
`post_funding_window` на 4H — **тождественно False**, тогда как правильная
сетка помечала бы 1/3 баров. На 1H пересечение правильного и фактического
множеств — 3 часа из 6. Признак называется «после расчёта funding», а
описывает окно 1–2 ч после момента, в который ничего не происходит.
→ **A2-082**.

**Как это делают те, кто делает это правильно.** Не хардкодят сетку вообще.
Nautilus 1.221.0 доставляет момент следующего расчёта в каждом тике —
см. §2.4 и **A2-081**.

### 1.2 Формат kline

**Официальная схема.** Ответ `/fapi/v1/klines` — массив массивов; индекс 0 —
open time, индекс 6 — close time
(https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Kline-Candlestick-Data,
обращение 2026-08-23). Референсная реализация той же схемы —
`nautilus_trader/adapters/binance/common/schemas/market.py:236-252`:

```python
class BinanceKline(msgspec.Struct, array_like=True):
    open_time: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time: int
    asset_volume: str
    trades_count: int
    taker_base_volume: str
    taker_quote_volume: str
    ignore: str
```

**`close_time = open_time + duration − 1 мс` — подтверждено замером** по
данным самого проекта (963 файла `klines_4h`, 5778 строк):

```
shape: (1, 2)
│ dur      ┆ count │
│ 14399999 ┆ 5778  │
```

`4 ч = 14 400 000` мс; единственное наблюдаемое значение — `14399999`.
Свеча закрывается за миллисекунду до открытия следующей.

**Возврат незакрытой свечи.** REST-эндпойнт возвращает формирующуюся свечу
последним элементом. WS-поток помечает закрытие полем `k.x`; референсная
реализация отбрасывает незакрытые бары —
`nautilus_trader/adapters/binance/data.py:1000-1002`:

```python
    def _handle_kline(self, raw: bytes) -> None:
        msg = self._decoder_candlestick_msg.decode(raw)
        if not msg.data.k.x:
            return  # Not closed yet
```

**Что в проекте.** `src/execution/strategies/ml_strategy_15m.py:586-593`
делает то же самое для REST-предзагрузки, но по времени, а не по флагу
(флага в REST-ответе нет — это корректный выбор):

```python
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            unclosed = 0
            for k in klines:
                close_ms = int(k[6])
                if close_ms >= now_ms:
                    unclosed += 1
                    continue
```

**Вердикт по 1.2: расхождений нет.** Индекс 6, отбрасывание незакрытой
свечи и вывод `open_time` из `close_time`
(`src/features/live_feature_state.py:39`, формула
`open = ((ts_event + 1) // duration - 1) * duration`) согласованы с
контрактом биржи и с референсной реализацией. Это записано в §6.

Остаточное расхождение — не в формате, а в жёстко зашитой точности цены:
`src/execution/strategies/ml_strategy_15m.py:598-601`

```python
                    open=Price(float(k[1]), precision=1),
```

`precision=1` соответствует `tickSize=0.10` только у BTCUSDT (см. §1.4);
у ETHUSDT `tickSize=0.01`. Это подмножество уже зафиксированной **A2-066**
(предзагрузка прибита к BTCUSDT), новой находкой не оформляется.

### 1.3 Ордера: SL + TP

**Что делает код.** В 4H-стратегии на биржу уходит ровно один защитный
ордер — `STOP_MARKET` с `reduce_only=True`
(`src/execution/strategies/ml_strategy.py:1240-1251`):

```python
                stop_order = self.order_factory.stop_market(
                    instrument_id=self._instrument_id,
                    order_side=exit_side,
                    quantity=qty,
                    trigger_price=sl_price,
                    trigger_type=TriggerType.LAST_PRICE,
                    time_in_force=TimeInForce.GTC,
                    reduce_only=True,
                    tags=[f"SL-attempt-{attempt}"],
                )
```

Take-profit **не отправляется никуда**. Замер: grep по `src/execution/`
и `src/risk/` даёт `take_profit` только в `RiskDecision` (расчёт R:R 1.5),
в журнале сигналов `signal_bridge` и в тексте лога. Ни одного вызова
`order_factory.limit(...)`, `take_profit_market`, `bracket` в проекте нет.
15m-стратегия не отправляет ордеров вообще (grep по `submit_order`
в `ml_strategy_15m.py` — пусто).

**Как правильно ставить пару SL+TP.** Три варианта, по убыванию
предпочтительности, с источниками:

1. **Nautilus `OrderFactory.bracket(...)` с `contingency_type=OUO`** —
   есть в 1.221.0, проектом не используется. Сигнатура (прочитана из
   установленного пакета):

   ```
   OrderFactory.bracket(..., contingency_type=ContingencyType.OUO,
       entry_order_type=OrderType.MARKET, tp_order_type=OrderType.LIMIT,
       tp_price=None, sl_order_type=OrderType.STOP_MARKET,
       sl_trigger_price=None, ...) -> OrderList
   ```
   Докстринг: «Create a bracket order with optional entry of take-profit
   order types. The stop-loss order will always be ``STOP_MARKET``.»

   Что произойдёт со второй заявкой, когда сработает первая, определено
   в `nautilus_trader/execution/manager.pyx:499-509`:

   ```python
            elif order.contingency_type == ContingencyType.OUO:
                ...
                if leaves_qty._mem.raw == 0 and order.exec_spawn_id is not None:
                    self.cancel_order(contingent_order)
                elif order.is_closed_c() and (order.exec_spawn_id is None or not is_spawn_active):
                    self.cancel_order(contingent_order)
                elif leaves_qty._mem.raw != contingent_order.leaves_qty._mem.raw:
                    self.modify_order_quantity(contingent_order, leaves_qty)
   ```

   То есть: исполнение одной ноги закрывает вторую, частичное исполнение
   уменьшает объём второй. Механизм включается флагом
   `manage_contingent_orders` в `StrategyConfig`
   (`nautilus_trader/trading/config.py:51-53`):

   > «manage_contingent_orders : bool, default False — If OUO and OCO
   > **open** contingent orders should be managed automatically by the
   > strategy.»

   Замер: grep `manage_contingent_orders` по `src/` — **ни одного
   вхождения**. Значение остаётся `False`.

2. **Две отдельные заявки на бирже + `reduceOnly`.** Работает, но Binance
   **не связывает** их: при срабатывании одной вторая остаётся висеть.
   На UM-фьючерсах `reduceOnly` не даёт перевернуть позицию, но оставляет
   «сироту» в книге, которая сработает на следующей позиции по тому же
   символу. Именно поэтому в Nautilus и существует OUO-менеджер выше.

3. **`closePosition=true`** — заявка закрывает всю позицию, объём не
   указывается; Binance сам снимает такие заявки при обнулении позиции.
   Точную формулировку правил снятия в официальной документации в рамках
   этого прохода подтвердить не удалось — см. §9, п. 2.

**Расхождение и его цена.** Разметка (`apply_triple_barrier`) определяет
событие как «первым тронут верхний барьер `1.5 × ATR` или нижний `1.0 × ATR`»,
исполнение реализует только нижний. Это ядро уже зафиксированной **A2-015**
(разметка и исполнение описывают разные события). Новая часть, которая
относится к практике: готовый механизм, закрывающий именно этот разрыв,
лежит в установленной версии библиотеки и не включён → **A2-079**.

*Замечание по нумерации.* Задание прохода ссылается на «A2-059 из прохода 5»
как на находку про пару SL+TP. В реестре §5 файла `00_method.md` номер
A2-059 занят находкой прохода 6 «`get_metrics_df` подаёт модели 100 записей
и две колонки из четырёх» — другой предмет. Ссылку на A2-059 не использую;
см. §9, п. 6.

### 1.4 Округление: tickSize, stepSize, minNotional

**Откуда берутся.** Из `GET /fapi/v1/exchangeInfo`, поле `filters`
каждого символа. Замер, выполненный в этом проходе (обращение 2026-08-23):

```
BTCUSDT pricePrecision 2 quantityPrecision 3
  tickSize 0.10 stepSize 0.001 marketStep 0.001 minNotional 50
ETHUSDT pricePrecision 2 quantityPrecision 3
  tickSize 0.01 stepSize 0.001 marketStep 0.001 minNotional 20
SOLUSDT pricePrecision 4 quantityPrecision 2
  tickSize 0.0100 stepSize 0.01 marketStep 0.01 minNotional 5
rateLimits: [
 {"rateLimitType": "REQUEST_WEIGHT", "interval": "MINUTE", "intervalNum": 1, "limit": 2400},
 {"rateLimitType": "ORDERS", "interval": "MINUTE", "intervalNum": 1, "limit": 1200},
 {"rateLimitType": "ORDERS", "interval": "SECOND", "intervalNum": 10, "limit": 300}
]
```

**Обязательны ли.** Да — это фильтры матчинг-движка: заявка, нарушившая
`PRICE_FILTER`, `LOT_SIZE` или `MIN_NOTIONAL`, отклоняется биржей.

**Что в проекте.** Замер: grep `min_notional|minNotional|tickSize|stepSize|exchangeInfo`
по всему `src/` даёт вхождения **только** в `src/execution/data_catalog.py`,
и там это не запрос к бирже, а зашитая таблица:

```python
    "SOLUSDT": {
        "base": SOL,
        "price_precision": 3,
        "size_precision": 0,
        "price_increment": 0.001,
        "size_increment": 1.0,
    },
```
(`src/execution/data_catalog.py:38-44`)

Сверка с биржей: у SOLUSDT `tickSize = 0.0100` (не 0.001) и
`stepSize = 0.01` (не 1.0). То есть в бэктесте объём позиции по SOL
округляется до **целых монет**, тогда как биржа принимает сотые. При цене
SOL ~$150 и риске $100 на сделку истинный размер 0.66 SOL превращается
в 0 или 1 — ошибка квантования от −100 % до +50 % на каждой SOL-сделке.
`AtomiCortexCatalog` читается только из `src/execution/backtest_runner.py:82`,
так что радиус — бэктест и бумажный прогон, не live (в live инструмент
приходит из адаптера Binance). → **A2-075**.

`minNotional` не читает никто. Для BTCUSDT это 50 USDT: если риск-движок
выдаст меньший нотионал, биржа отклонит входной ордер, а сигнал к этому
моменту уже отправлен в Telegram (`_emit_signal` вызывается до
`submit_order`, `src/execution/strategies/ml_strategy.py:1132`).
→ **A2-076**.

**Отдельный путь без округления вообще — watchdog.** Аварийное закрытие
шлёт REST-заявку напрямую (`src/execution/watchdog.py:778-789`):

```python
                    "type": "LIMIT",
                    "timeInForce": "IOC",
                    "quantity": qty,
                    "price": f"{limit_price:.2f}",
                    "reduceOnly": "true",
```

`:.2f` даёт два знака после запятой. Для BTCUSDT `tickSize = 0.10` —
цена вида `112345.67` не кратна тику, и `PRICE_FILTER` её отклонит.
То есть защита H16 от проскальзывания (лимит-IOC перед MARKET) для
BTCUSDT не работает **никогда**: код каждый раз падает в MARKET-ветку.
Для ETHUSDT и SOLUSDT (`tickSize = 0.01`) формат совпадает с тиком
случайно. → **A2-077**.

Как это делают правильно: цена приводится к `tickSize` округлением вниз/вверх
по стороне, объём — к `stepSize`; в Nautilus это `instrument.make_price()` /
`instrument.make_qty()`, и 4H-стратегия в основном пути их использует
(`ml_strategy.py:1101`, `1237-1238`) — то есть в проекте есть правильный
образец, просто watchdog идёт мимо него.

### 1.5 Rate limits и веса запросов

**Лимиты (замер выше):** REQUEST_WEIGHT 2400/мин на IP; ORDERS 1200/мин
и 300/10 с. Заголовок `X-MBX-USED-WEIGHT-1M` в каждом ответе —
авторитетный счётчик.

**Что в проекте.** `src/execution/binance_rate_limiter.py:52`:

```python
    MAX_WEIGHT_PER_MINUTE: int = 1200   # 50 % of the 2400 hard cap
```

Число 2400 совпадает с замеренным. Подход правильный: скользящее окно 60 с,
бюджет вдвое ниже лимита, заголовок с провода как источник истины,
fail-soft. Это записано в §6.

Два расхождения:

1. Лимитер считает **только вес запросов**. Лимиты ORDERS (1200/мин,
   300/10 с) не отслеживаются ничем. При ретраях SL
   (`_submit_stop_loss_with_retry`, до 3 попыток) и аварийном закрытии
   watchdog по всем символам сразу счётчик заявок никто не сторожит.

2. Докстринг `src/execution/binance_rate_limiter.py:25-29` перечисляет
   «Wired into … `LiveFeatureState.fetch_taker_buy_volume` — H1c». Замер:
   grep `BinanceRateLimiter|limiter.acquire` по `src/` даёт вхождения
   только в `src/execution/watchdog.py:857,891,924` и
   `src/execution/reconciler.py:240-242`. В `src/features/live_feature_state.py`
   лимитера нет. `ml_strategy.py` (klines-предзагрузка с `limit=1500`,
   `openInterestHist`, funding-предзагрузка, пер-баровый taker-volume)
   не подключён — что честно указано в том же докстринге как «Still
   pending», но первый список от этого не перестаёт быть неверным.

   Структурная причина: лимитер асинхронный, а `ml_strategy` ходит в сеть
   синхронным `requests` прямо в `on_bar` — подключить его без переделки
   нельзя.

   → **A2-087**.

---

## 2. NAUTILUS TRADER 1.221.0

Версия подтверждена прогоном:
`.venv/bin/python -c "import nautilus_trader; print(nautilus_trader.__version__)"` → `1.221.0`.
Всё ниже — чтение установленного пакета в `.venv/lib/python3.11/site-packages/nautilus_trader/`.

### 2.1 Конвенция `ts_event` для баров — подтверждена по исходникам

Базовый докстринг `Bar` (`model/data.pyx:1424-1425`) говорит нейтрально:

```
    ts_event : uint64_t
        UNIX timestamp (nanoseconds) when the data event occurred.
```

Конкретику задаёт адаптер Binance — и он одинаков для REST и WS:

`adapters/binance/common/schemas/market.py:262` (REST klines):
```python
        ts_event = millis_to_nanos(self.close_time)
```

`adapters/binance/common/schemas/market.py:710` (WS kline, поле `T` = «Kline close time»):
```python
        ts_event = millis_to_nanos(self.T)
```

**Вывод:** для баров Binance `ts_event` = close_time = `open_time + duration − 1 мс`.
Проект использует ровно эту конвенцию: `ml_strategy_15m.py:595`
(`ts_ns = close_ms * 1_000_000`) и `bar_open_time_ms()` в
`src/features/live_feature_state.py:39`, которая снимает обе конвенции
(`open + d` и `open + d − 1`) одной формулой. **Расхождения нет.**

### 2.2 Bracket-ордер в Nautilus

Готовый механизм есть и не используется — разобрано в §1.3. Кратко:
`OrderFactory.bracket(...)` (список методов фабрики получен интроспекцией:
`['bracket', 'create_list', ..., 'stop_market', ...]`), контингенция OUO,
исполнитель — `execution/manager.pyx:450,502-509`, выключатель —
`StrategyConfig.manage_contingent_orders` (`trading/config.py:51`, default `False`).

Проект вместо этого держит собственную машину: словарь `_pending_sl_params`,
зеркало на диск `PendingOrdersStore`, отложенная подача SL в `on_order_filled`,
ретраи. Каждая деталь по отдельности осмысленна (окно «вход без стопа»
действительно закрыто), но заменяет она только половину bracket'а — TP
в ней нет, и связи между ногами нет, потому что второй ноги нет.
→ **A2-079**.

### 2.3 `on_position_changed`, `on_order_filled` — что не переопределено

Полный список хуков базового класса (`trading/strategy.pyx`, `common/actor.pyx`)
и пересечение с проектом:

| Хук Nautilus | 4H `ml_strategy.py` | 15m `ml_strategy_15m.py` |
|---|---|---|
| `on_start` / `on_stop` | ✔ 267 / 481 | ✔ 129 / 192 |
| `on_bar` | ✔ 641 | ✔ 380 |
| `on_data` | ✔ 593 | — |
| `on_order_filled` | ✔ 884 | — |
| `on_order_rejected` / `on_order_denied` | ✔ 1157 / 1166 | — |
| `on_position_opened` / `on_position_closed` | ✔ 943 / 950 | — |
| **`on_position_changed`** | **—** | — |
| **`on_order_canceled`** | **—** | — |
| **`on_order_expired`** | **—** | — |
| **`on_order_triggered`** | **—** | — |
| **`on_order_updated`** | **—** | — |
| `on_order_modify_rejected` / `on_order_cancel_rejected` | — | — |
| `on_event` | — | — |

Два пропуска меняют поведение, остальные — нет.

**(а) Частичные исполнения входа.** `on_order_filled` различает вход и выход
наличием ключа в `_pending_sl_params` (`ml_strategy.py:907`):

```python
        is_entry_fill = client_oid in self._pending_sl_params
```

и **снимает** ключ на первом же филле (`ml_strategy.py:922`):

```python
            sl_params = self._pending_sl_params.pop(client_oid)
```

Референсная реализация показывает, что филлов на один `client_order_id`
может быть несколько. `adapters/binance/futures/schemas/user.py:511-547`:

```python
            last_qty = Quantity(float(self.l), size_precision)
            last_px = Price(float(self.L), price_precision)
            ...
            exec_client.generate_order_filled(
                ...
                trade_id=TradeId(str(self.t)),  # Trade ID
                ...
                last_qty=last_qty,
                last_px=last_px,
```

`self.l` — объём **последней сделки** (не накопленный `z`), `self.t` — ID
сделки. То есть Binance шлёт по событию `ORDER_TRADE_UPDATE` на каждую
сделку, а Nautilus генерирует по `OrderFilled` на каждое такое событие.
Рыночная IOC-заявка, прошедшая по книге на нескольких уровнях, даёт
несколько филлов.

Следствия для кода:
* второй и последующие филлы входа попадают в `else`-ветку и логируются
  как «Exit fill (SL/close)» — `PortfolioTracker.update_fill` для них не
  вызывается, позиция в трекере занижена;
* `_submit_stop_loss_with_retry(fill_qty=fill_qty)` получает объём
  **первого** филла, а не итоговый — стоп покрывает часть позиции,
  остаток остаётся без защиты;
* при этом `on_position_changed`, который Nautilus эмитит именно на
  доборы/частичные изменения позиции, не переопределён, так что и
  компенсировать это негде.

→ **A2-078**.

*Замечание по нумерации.* Задание ссылается на «A2-079 прохода 5» как на
находку про непереопределённые события. В реестре §5 файла `00_method.md`
последний занятый номер — **A2-073**; A2-079 в нём отсутствует. Нумерация
этого прохода продолжается с A2-074, номер A2-079 выдан заново (см. §7).
См. §9, п. 6.

**(б) Снятая или истёкшая защита.** `on_order_canceled` / `on_order_expired`
не переопределены. Стоп, снятый на бирже — вручную, watchdog'ом
(`watchdog.py:308` явно снимает висящие заявки перед reduceOnly-закрытием)
или движком при переоткрытии — исчезает молча. `on_order_rejected` умеет
писать `POSITION UNPROTECTED!` (`ml_strategy.py:1204`), но для отмены
такого пути нет. → **A2-080**.

Остальные пропуски (`on_order_updated`, `on_order_triggered`,
`on_order_modify_rejected`, `on_order_cancel_rejected`) при текущем наборе
типов заявок ни на какое число не влияют — находкой не оформляются.

### 2.4 Что проект реализует вручную, имея это в 1.221.0

| Реализовано вручную | Что есть в 1.221.0 | Оценка |
|---|---|---|
| `_pending_sl_params` + отложенный SL | `OrderFactory.bracket` + `manage_contingent_orders` | **Расхождение**, TP теряется → A2-079 |
| `dt.hour in (1, 9, 17)` как детектор расчёта funding | `BinanceFuturesMarkPriceUpdate.next_funding_ns` | **Расхождение** → A2-081 |
| Аналитическая оценка проскальзывания после прогона | `FillModel` в `add_venue(...)` | **Расхождение** → A2-083 |
| Отсутствие модели задержки | `LatencyModel` в `add_venue(...)` | **Расхождение** → A2-083 |
| `PortfolioTracker` поверх позиций | `Portfolio` / `Cache.positions_open()` | Дублирование; ломается отдельно (A2-062). Здесь не переоткрывается |
| Ручной `on_data` c `isinstance(BinanceFuturesMarkPriceUpdate)` | `Actor.on_funding_rate(FundingRateUpdate)` | **Не расхождение**, см. ниже |

**Про `on_funding_rate`.** В ядре 1.221.0 есть и тип `FundingRateUpdate`
(`model/data.pyx:5908`), и хук `Actor.on_funding_rate`
(`common/actor.pyx:513`), и подписка (`common/actor.pyx:1786`). Но замер
`grep -rn FundingRateUpdate adapters/binance/` даёт **ноль вхождений** —
адаптер Binance этот тип не производит. Ручной разбор
`BinanceFuturesMarkPriceUpdate` в `on_data` — единственный доступный путь.
**Проект здесь прав; находки нет.**

**Про `next_funding_ns`.** Тот же объект, который проект уже получает и уже
разбирает, несёт момент следующего расчёта. `adapters/binance/futures/types.py:43-44`:

```
    next_funding_ns : uint64_t
        UNIX timestamp (nanoseconds) when next funding will occur.
```

Заполняется из поля `T` WS-потока markPrice
(`adapters/binance/futures/schemas/market.py:227,238`):

```python
    T: int  # Next funding time
        ...
            next_funding_ns=millis_to_nanos(self.T),
```

То есть биржа сообщает момент расчёта в каждом тике (раз в секунду),
объект уже в руках у `on_data`, и вместо чтения этого поля код сверяет
`dt.hour` с зашитой тройкой часов. Это не только источник **A2-060**, но и
причина, по которой переход Binance на 1 ч/4 ч (§1.1) сломал бы код
второй раз. → **A2-081**.

---

## 3. LIGHTGBM 4.3.0

Версия подтверждена: `import lightgbm; lightgbm.__version__` → `4.3.0`.

### 3.1 Ранняя остановка, `valid_sets`, `callbacks` — расхождений нет

Проект передаёт обучающий набор в `valid_sets`
(`src/models/lgbm_trainer.py:783-787`):

```python
        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )
```

Документация `lightgbm.early_stopping` (`.venv/.../lightgbm/callback.py:436-445`):

> «Requires at least one validation data and one metric. If there's more
> than one, will check all of them. But the training data is ignored anyway.»

Проверка, что «ignored anyway» действительно срабатывает при
`valid_names=["train", ...]`, а не только при имени по умолчанию
`"training"` — цепочка из двух мест установленного пакета:

`lightgbm/engine.py:205-221`
```python
    is_valid_contain_train = False
    train_data_name = "training"
    ...
        for i, valid_data in enumerate(valid_sets):
            # reduce cost for prediction training data
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
```
`lightgbm/engine.py:255-257`
```python
        booster = Booster(params=params, train_set=train_set)
        if is_valid_contain_train:
            booster.set_train_data_name(train_data_name)
```
`lightgbm/callback.py:310-312`
```python
        # for lgb.train(), it's possible to pass the training data via valid_sets with any eval_name
        if isinstance(env.model, Booster) and ds_name == env.model._train_data_name:
            return True
```

Так как объект `train_data` физически тот же, LightGBM подменяет
`_train_data_name` на `"train"`, и `_is_train_set` его распознаёт →
`continue` в `callback.py:420-425`, ранняя остановка считается только
по `val`. **Реализация корректна.** Записано в §6.

Остальное в этой части тоже сходится: `n_estimators` и
`early_stopping_rounds` — не параметры `lgb.train`, и код их отфильтровывает
явно (`lgbm_trainer.py:749-753`), маппя в `num_boost_round` и в callback.

### 3.2 `scale_pos_weight` против `sample_weight`

Проект использует `sample_weight` (`lgbm_trainer.py:702-703`):

```python
        train_weights = compute_sample_weight("balanced", y_train_fit)
        val_weights = compute_sample_weight("balanced", y_val)
```

и домножает на веса уникальности AFML (`lgbm_trainer.py:718-719`).

Официальная документация LightGBM 4.3.0 о `scale_pos_weight`
(https://lightgbm.readthedocs.io/en/v4.3.0/Parameters.html, обращение 2026-08-23):

> «weight of labels with positive class»
> «**Note**: while enabling this should increase the overall performance
> metric of your model, it will also result in poor estimates of the
> individual class probabilities.»

Вопрос — распространяется ли это предупреждение на `sample_weight`.
Ответ по исходнику референсной реализации
(`github.com/microsoft/LightGBM`, тег `v4.3.0`,
`src/objective/binary_objective.hpp`, обращение 2026-08-23):

```
label_weights_[1] *= scale_pos_weight_;
...
gradients[i] = static_cast<score_t>(response * label_weight);
hessians[i]  = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight);
...
gradients[i] = static_cast<score_t>(response * label_weight * weights_[i]);
hessians[i]  = static_cast<score_t>(abs_response * (sigmoid_ - abs_response) * label_weight * weights_[i]);
```

`scale_pos_weight` входит как `label_weight`, пользовательские веса — как
`weights_[i]`, **в то же самое произведение**. `compute_sample_weight("balanced")`
из sklearn задаёт `weights_[i]`, постоянный внутри класса, то есть с точностью
до общего масштаба совпадает с `scale_pos_weight`. Предупреждение документации
применимо дословно.

**Что это значит здесь.** Априор нестационарен (по A2-021 доля `p(UP)` уходит
ниже 0.5 во всех режимах, включая `trend_up`). «Balanced»-веса выравнивают
классы **обучающей** выборки — то есть подменяют априор на 50/50, привязанный
к конкретному train-срезу. Выход бустера после этого — не оценка вероятности
события, а оценка вероятности в перевзвешенной популяции, которой не
существует. Следствие — §3.3.

**Про выбор `sample_weight` вместо `scale_pos_weight`.** Сам выбор
**правильный**: только он позволяет домножить веса уникальности AFML
поэлементно. Расхождения нет; расхождение — в отсутствии следующего шага.

### 3.3 Калибровка вероятностей

**Как выход используется.** `src/models/lgbm_trainer.py:1288-1296`:

```python
        p_up = float(model.predict(features)[0])
        direction = 1 if p_up > 0.5 else -1
        confidence = p_up if direction == 1 else 1.0 - p_up
        ...
        if confidence < confidence_threshold:
            return 0, confidence
```

Порог: `src/config.py:90` — `confidence_threshold: float = Field(default=0.65, ...)`.

**Что говорят источники.**

1. Определение калиброванности — документация scikit-learn
   (https://scikit-learn.org/stable/modules/calibration.html, обращение 2026-08-23):

   > «Well calibrated classifiers are probabilistic classifiers for which
   > the output of the predict_proba method can be directly interpreted as
   > a confidence level. For instance, a well calibrated (binary) classifier
   > should classify the samples such that among the samples to which it
   > gave a predict_proba value close to, say, 0.8, approximately 80%
   > actually belong to the positive class.»

2. Взвешивание классов портит именно эти оценки — документация LightGBM,
   цитата в §3.2 («poor estimates of the individual class probabilities»),
   применимость к `sample_weight` доказана исходником там же.

3. Об искажении вероятностей ансамблями деревьев та же страница sklearn
   цитирует Niculescu-Mizil & Caruana (2005), «Predicting Good Probabilities
   with Supervised Learning», ICML 2005 — но конкретно для **bagging /
   random forests**. Утверждение «бустинг растягивает вероятности к краям»
   на этой странице **отсутствует**; см. §9, п. 3.

**Что в проекте.** Замер: grep `calibrat|isotonic|platt|brier|reliability`
по `src/` и `scripts/` даёт только комментарии — «Threshold 0.55 is
calibrated for binary» (`lgbm_trainer.py:1113`), «recalibrate after retrain»
(`configs/strategy_1h.py:31`, `configs/strategy_15m.py:41`), «Keeps proba
calibration sane» (`scripts/train_meta_model.py:168`). **Ни одной строки
кода калибровки.** Ни `CalibratedClassifierCV`, ни изотонической регрессии,
ни Платта, ни кривой надёжности, ни Brier score в отчётах.

**Что делают вместо.** Два раздельных шага, оба отсутствуют:
* **калибровка** на отдельном срезе (изотоническая или сигмоидная) —
  после неё `p̂` можно сравнивать с числом;
* **выбор порога** по целевой функции (ожидаемая прибыль с учётом комиссий
  и funding), а не подбором «красивой» константы.

Сейчас 0.65 сравнивается с числом, у которого нет единиц измерения:
после «balanced»-взвешивания сдвиг выхода — функция дисбаланса конкретного
train-среза, и он меняется при каждом переобучении. Это ровно то, почему
живой confidence 0.50–0.54 (вход, §3 файла `00_method.md`) невозможно
интерпретировать: неизвестно, модель молчит потому, что не уверена, или
потому, что шкала съехала. → **A2-086**.

### 3.4 Воспроизводимость: seed, deterministic, num_threads

Документация LightGBM 4.3.0 (обращение 2026-08-23):

* `deterministic` (default `false`): «setting this to `true` should ensure
  the stable results when using the same data and the same parameters
  (and different `num_threads`)»; «when you use the different seeds,
  different LightGBM versions, the binaries compiled by different compilers,
  or in different systems, the results are expected to be different»;
  рекомендуется вместе с `force_col_wise=true` или `force_row_wise=true`.
* `num_threads` (default `0`): «`0` means default number of threads in
  OpenMP»; «set this to the number of **real CPU cores**».
* `seed`: «this seed is used to generate other seeds, e.g. `data_random_seed`,
  `feature_fraction_seed`, etc.»; «has lower priority in comparison with
  other seeds».

**Что в проекте** (`src/models/lgbm_trainer.py:158-186`, `MTF_LGBM_PARAMS`):
заданы `feature_fraction_seed: 42`, `bagging_seed: 42`, `random_state: 42`.
Замер grep `deterministic|num_threads|force_col_wise|force_row_wise` по
`src/` — вхождений нет.

**Оценка.** Посевы покрывают обе стохастические компоненты (подвыборка
признаков и бэггинг), это правильно. `deterministic=false` при
неопределённом `num_threads` означает, что число потоков задаёт OpenMP по
числу логических ядер — и результат обучения на машине с другим числом
ядер будет **другим** при том же посеве. Для проекта, где артефакт
собирается на одной машине, а метрики считаются в другом прогоне
(A2-031, A2-023), это не гипотетическая проблема.

Тем не менее оформлять отдельной находкой не буду: расхождение с практикой
есть, но **прямого доказательства расхождения чисел** в этом проходе не
получено (обучение запускать запрещено). Записано в §9, п. 4 как
неподтверждённое.

Что зафиксировано в манифесте и работает на воспроизводимость —
`lightgbm_version`, `python_version`, `git_commit`
(`lgbm_trainer.py:1033-1035`) — записано в §6.

---

## 4. ФИНАНСОВЫЙ ML — КАНОН

Первоисточник — López de Prado, *Advances in Financial Machine Learning*
(Wiley, 2018), далее AFML. Структура глав сверена с оглавлением издателя
(https://www.wiley.com/en-ae/Advances+in+Financial+Machine+Learning-p-9781119482086
и агрегаторы, обращение 2026-08-23):

* Гл. 3 «Labeling»: 3.4 The Triple-Barrier Method, 3.5 Learning Side and
  Size, 3.6 Meta-Labeling, 3.7 How to Use Meta-Labeling, 3.9 Dropping
  Unnecessary Labels.
* Гл. 4 «Sample Weights»: 4.3 Number of Concurrent Labels, 4.4 Average
  Uniqueness of a Label, 4.5 Bagging Classifiers and Uniqueness,
  4.6 Return Attribution, 4.7 Time Decay, 4.8 Class Weights.
* Гл. 7 «Cross-Validation in Finance»: 7.3 Why K-Fold CV Fails in Finance,
  7.4 A Solution: Purged K-Fold CV, 7.4.1 Purging the Training Set,
  7.4.2 Embargo, 7.4.3 The Purged K-Fold Class.

Номера страниц не приводятся: физического экземпляра у аудита нет,
восстанавливать их по памяти запрещено протоколом. См. §9, п. 5.

### 4.1 Triple barrier: книга против кода

| Аспект | AFML гл. 3.4 | `src/features/triple_barrier.py` | Расхождение |
|---|---|---|---|
| Ширина барьеров | `trgt × ptSl[0]` / `trgt × ptSl[1]`, где `trgt` — оценка волатильности (в книге — EWMA дневной волатильности) | `atr_pct × 1.5` / `atr_pct × 1.0` (строки 55-57) | Нет. Другая оценка волатильности — допустимая замена |
| По какой цене проверяется касание | путь `close[t0:t1]`, нормированный на `close[t0]` | `close[k : k+valid_n]` (строка 126) | **Нет** — книга тоже использует close |
| Вертикальный барьер | момент времени (`numDays`) | число баров `max_holding_bars` | Нет |
| Метка | `sign(ret)` в момент касания | `+1 / −1 / 0` (строки 130-131, 148) | Нет |
| Что записывается как `ret` | доходность до фактического `t1` | `(fut − entry) / entry` на баре касания (строки 139-140) | Нет; в коде это сделано осознанно, см. комментарий строк 132-138 |
| `t1` (момент выхода) | обязателен, нужен для весов гл. 4 | `t1_bar` (строки 160-163) | Нет |

**Важная поправка к предыдущим проходам.** Находка **A2-018** («барьеры
проверяются только по `close`») — расхождение **не с книгой**. Референсный
снипет AFML `applyPtSlOnT1` работает по колонке close, а не по high/low.
Расхождение здесь другое и оно реально: **между разметкой и исполнением**.
Биржевой `STOP_MARKET` с `trigger_type=TriggerType.LAST_PRICE`
(`ml_strategy.py:1246`) срабатывает внутри бара по любой сделке, а метка
считает, что бар закрылся выше стопа и сделка жива. То есть A2-018 остаётся
дефектом, но по причине §1.3 (A2-015), а не по причине «отступили от книги».
Это уточнение основания, не новая находка.

**Что действительно расходится с книгой — 3.9 «Dropping Unnecessary Labels».**
Книга снимает **редкие** классы (те, у которых слишком мало примеров для
обучения). Код снимает класс `0` (таймаут) — то есть класс, принадлежность
к которому определяется **исходом в будущем**. Это уже зафиксировано как
**A2-017** (смещение отбора); здесь только фиксируется, что первоисточник
такой операции не предписывает.

### 4.2 Sample uniqueness и average uniqueness

**Формула книги** (гл. 4.3–4.4). Для каждого бара `t` число одновременных
меток `c_t`; уникальность метки `i` со сроком `[t_{i,0}, t_{i,1}]`:

```
    ū_i = (1 / |[t_{i,0}, t_{i,1}]|) · Σ_{t ∈ [t_{i,0}, t_{i,1}]} 1 / c_t
```

**Что в коде** (`src/models/dataset_builder.py:373-388`) — та же формула,
посчитанная разностным массивом и кумулятивной суммой:

```python
            t_max = int(t1.max()) + 1
            diff = np.zeros(t_max + 1, dtype=np.int64)
            np.add.at(diff, idx, 1)
            np.add.at(diff, np.minimum(t1 + 1, t_max), -1)
            concur = np.cumsum(diff)[:t_max].astype(np.float64)
            concur = np.maximum(concur, 1.0)
            inv_c = 1.0 / concur
            cum = np.concatenate(([0.0], np.cumsum(inv_c)))
            end = t1 + 1
            span = np.maximum(end - idx, 1)
            u = (cum[end] - cum[idx]) / span
```

**Расхождение — одна строка** (`dataset_builder.py:392-394`):

```python
        u_mean = u.mean()
        if u_mean > 0:
            u = u / u_mean
```

Разбор последствий — **R-2 / A2-016**, проход 2; здесь по заданию только
сверка с первоисточником. Она даёт следующее:

* Книга **не подаёт** `ū_i` напрямую как `sample_weight` в обучение.
  В гл. 4.5 средняя уникальность идёт в `max_samples` бэггинг-классификатора
  и в последовательный бутстрап; веса обучения в гл. 4.6–4.7 строятся из
  **атрибуции доходности** (`|Σ r_t / c_t|`) и **временного затухания**.
* Нормировка на среднее в книге встречается — но именно для весов
  атрибуции доходности, где она приводит сумму весов к числу наблюдений.
  Перенести её на `ū_i` — значит стереть абсолютный уровень уникальности,
  который и есть поправка на ESS.
* Разбиение по символам (`compute_uniqueness_weights_by_symbol`) —
  правильно и книге не противоречит: перекрытие меток определено внутри
  одного ряда.

**Итог сверки:** формула concurrency воспроизведена верно; расходятся
(а) роль величины (веса обучения вместо бутстрапа/`max_samples`) и
(б) нормировка. Новой находки не открываю — это A2-016.

### 4.3 Purged K-Fold с эмбарго: определение против реализации

**Определение (AFML гл. 7.4).**
* *Purging* (7.4.1): из обучающей выборки удаляются наблюдения, чьи
  **метки перекрываются по времени** с метками тестовой выборки. Критерий —
  интервал `[t_{i,0}, t_{i,1}]` конкретной метки, а не фиксированный отступ.
* *Embargo* (7.4.2): дополнительно исключаются обучающие наблюдения,
  попадающие в окно **после** тестового блока, размером `h` (доля выборки).
* *PurgedKFold* (7.4.3): K смежных тестовых блоков; обучающая выборка —
  **всё остальное, включая данные после теста**; отсюда и нужен эмбарго.

**Что в коде** (`src/execution/walk_forward.py:42-53`):

```python
class PurgedKFoldCV:
    """Time-series cross-validation with an embargo gap between train and test.

    Fold layout (expanding train, fixed-size test block):

        Fold 1: [==TRAIN==][GAP][TEST]
        Fold 2: [=====TRAIN=====][GAP][TEST]
        Fold 3: [========TRAIN========][GAP][TEST]
```

и (`walk_forward.py:124-126`):

```python
        block = total / (self.n_splits + 1)
        embargo = self.embargo_pct * total
```

**Три расхождения, каждое меняет содержимое обучающей выборки.**

1. Это не purged K-fold, а **растущий walk-forward с зазором**. Обучение
   никогда не содержит данных после теста. Следовательно эмбарго в смысле
   7.4.2 (защита от утечки *назад* из пост-тестовых обучающих меток)
   защищать нечего — то, что код называет «embargo», по функции является
   purging.

2. Purging сделан **отступом**, а не по перекрытию меток. Отступ равен
   `embargo_pct × (t_max − t_min)` — доле **всего диапазона данных**, а не
   горизонту метки. При `embargo_pct = 0.02` (значение по умолчанию в
   `scripts/validate_1h_models.py:216` и `scripts/validate_15m_models.py:217`)
   и диапазоне 2024-01-01 … 2026-08-20 (963 дня) зазор ≈ 19.3 суток,
   тогда как горизонт метки 4H при `max_holding = 6` равен 1 суткам.
   Зазор задан величиной, которая не имеет отношения к причине зазора: он
   меняется при изменении длины датасета и не меняется при изменении
   горизонта метки. При меньшем `embargo_pct` или более длинном горизонте
   знак ошибки переворачивается на «недопуржено».

3. Название класса утверждает соответствие AFML, докстринг
   `walk_forward.py:334-337` ссылается на «AFML Ch.7 embargo». Читатель,
   опирающийся на имя, получит не то, что в книге.

→ **A2-088**.

Отдельно: `WalkForwardValidator` в ML-пути инстанцируется **без** `embargo`
(`src/models/ml_validator.py:298-302`):

```python
        wf_validator = WalkForwardValidator(
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
        )
```
при `embargo: timedelta = timedelta(0)` по умолчанию
(`walk_forward.py:328`). Это подтверждение **A2-027** (эмбарго объявлено и
нигде не включено), не новая находка.

### 4.4 CSCV / PBO: оригинальная процедура против кода

**Оригинал.** Bailey, Borwein, López de Prado, Zhu, «The Probability of
Backtest Overfitting» (https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf,
обращение 2026-08-23). Алгоритм 2.3 (CSCV), цитаты дословно из PDF:

> «First, we form a matrix M by collecting the performance series from the
> N trials. In particular, each column n = 1, . . . , N represents a vector
> of profits and losses over t = 1, . . . , T observations associated with
> a particular model configuration tried by the researcher.»

> «Second, we partition M across rows, into an even number S of disjoint
> submatrices of equal dimensions.»

> «Third, we form all combinations CS of Ms, taken in groups of size S/2.»
> «For instance, if S = 16, we will form 12, 780 combinations.»

> «e) Determine the element n∗ such that r^c_{n∗} ∈ Ω∗_{n∗}. In other
> words, n∗ is the best performing strategy IS.»

> «f) Define the relative rank of r̄^c_{n∗} by ω̄c := r̄^c_{n∗}/(N + 1) ∈ (0, 1).»

> «g) We define the logit λc = ln(ω̄c/(1−ω̄c)). High logit values imply a
> consistency between IS and OOS performances, which indicates a low level
> of backtest overfitting.»

PBO — доля распределения логитов ниже нуля.

**Что в коде** (`src/models/statistical_tests.py:177-249`). Существенное:

```python
    for oos_idx in range(n):
        is_indices = [j for j in range(n) if j != oos_idx]
        is_metrics = metrics_arr[is_indices]
        best_is_pos = int(np.argmax(is_metrics))
        best_is_original = is_indices[best_is_pos]
        other_vals = np.delete(metrics_arr, best_is_original)
        other_median = float(np.median(other_vals))
        if metrics_arr[best_is_original] < other_median:
            overfit_count += 1
```

**Пять отличий, каждое существенное:**

1. **Нет измерения N.** В CSCV столбцы `M` — это *разные конфигурации*.
   Здесь `cv_results` — фолды **одной** конфигурации. Размерность, по
   которой определяется «лучшая стратегия IS», отсутствует физически.
2. **Нет разбиения IS/OOS.** У каждого фолда одно число; IS-оценка и
   OOS-оценка одной и той же стратегии не вычисляются.
3. **Нет комбинаторики.** Вместо C(S, S/2) (для S = 16 — 12 780
   комбинаций) — `n` итераций leave-one-out.
4. **Нет логита и нет распределения.** PBO определён как площадь под
   плотностью λ левее нуля; здесь это доля счётчика.
5. **Результат тождественно ноль.** `best_is_original` — максимум по
   `n − 1` значениям (все, кроме `oos_idx`). `other_vals` — те же `n − 1`
   значений, но с заменой `best` на `metrics[oos_idx]`. Значит из `n − 1`
   элементов `other_vals` максимум **один** может превосходить `best`,
   а `n − 2` заведомо ≤ `best`. При `n ≥ 4` медиана такого набора всегда
   ≤ `best`, условие `<` не выполняется ни разу. Это и есть **A2-026**,
   доказанное здесь алгебраически, а не только замером.

**Как выглядит корректная процедура (по алгоритму 2.3):**

```
1. M ← (T × N): T синхронных наблюдений P&L, N конфигураций,
   реально перебранных исследователем (по A2-025 — ≥ 28× больше,
   чем сейчас записывается).
2. Разбить строки M на чётное S непересекающихся блоков.
3. Для каждой из C(S, S/2) комбинаций c:
     J   ← блоки комбинации, в исходном порядке   (IS,  T/2 × N)
     J̄   ← дополнение J в M, в исходном порядке    (OOS, T/2 × N)
     R^c ← вектор метрики по столбцам J;   r^c  ← ранги
     R̄^c ← вектор метрики по столбцам J̄;  r̄^c  ← ранги
     n*  ← argmax r^c                      (лучшая IS)
     ω̄c  ← r̄^c[n*] / (N + 1)
     λc  ← ln(ω̄c / (1 − ω̄c))
4. PBO ← доля λc ≤ 0 по всем комбинациям.
```

Требование, которого проект сегодня не удовлетворяет ни в одной точке:
нужны сохранённые **временные ряды P&L по каждой пробе**, синхронные по
строкам. Сейчас хранятся только агрегаты (`win_rate`, `profit_factor`)
по фолдам.

Отдельно — предупреждение самих авторов, которое стоит держать рядом
с любой реализацией:

> «We emphasize that the CSCV implementation is only one illustrative
> [method] … task-specific methods for estimating the PBO.»

Новой находки не открываю — это A2-026 и A2-025.

### 4.5 DSR: формула из статьи с указанием обозначений

**Первоисточник.** Bailey & López de Prado, «The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality»
(https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf, обращение
2026-08-23; журнальная публикация — *Journal of Portfolio Management*,
2014). Текст извлечён `pdftotext`; формулы в PDF набраны шрифтом, который
не извлекается символ-в-символ, поэтому ниже — авторский **исходный код**
из статьи (Snippet 1), извлечённый дословно, и словесные определения,
извлечённые дословно.

**Уравнение (1), E[max SR] — словесное определение автора:**

> «Appendix 1 proves that, under these assumptions, the expected maximum
> of {ŜR_n} after N independent trials can be approximated as […] where
> γ (approx. 0.5772) is the Euler-Mascheroni constant, Z is the cumulative
> function of the standard Normal distribution, and e is Euler's number.»

**Snippet 1 из статьи (дословно):**

```python
def getExpMaxSR(mu,sigma,numTrials):
    # Compute the expected maximum Sharpe ratio (Analytically)
    emc=0.5772156649 # Euler-Mascheroni constant
    maxZ=(1-emc)*ss.norm.ppf(1-1./numTrials)+emc*ss.norm.ppf(1-1./(numTrials*np.e))
    return mu+sigma*maxZ
```

**Уравнение (2), DSR — словесное определение автора:**

> «where ŜR_0 = √V[{ŜR_n}] ((1−γ) Z^{-1}[1 − 1/N] + γ Z^{-1}[1 − 1/(N e)])
> […] V[{ŜR_n}] is the variance across the trials' estimated SR and N is
> the number of independent trials. We also use information concerning the
> selected strategy: ŜR is its estimated SR, T is the sample length, γ̂_3 is
> the skewness of the returns distribution and γ̂_4 is the kurtosis of the
> returns distribution for the selected strategy. Z is the cumulative
> function of the standard Normal distribution.»

**Обозначения:** `ŜR` — оценка Шарпа выбранной стратегии; `ŜR_0` — порог
отсечения (= E[max SR] при нулевом истинном Шарпе); `N` — число независимых
проб; `V[{ŜR_n}]` — дисперсия оценок Шарпа **по пробам**; `T` — длина ряда
доходностей; `γ̂_3` — асимметрия; `γ̂_4` — куртозис (сырой, для нормального
распределения = 3); `γ` ≈ 0.5772 — постоянная Эйлера–Маскерони;
`Z` — функция стандартного нормального распределения.

**Оценка зрелости открытой реализации, использованной для сверки.**
`github.com/rubenbriones/Probabilistic-Sharpe-Ratio` — небольшой
однофайловый репозиторий, не пакет и не индустриальный стандарт. Взят
не как авторитет, а как **независимая вторая реализация**: она совпадает
со Snippet 1 статьи строка в строку, что и делает её пригодной для
перекрёстной проверки той части, которая в PDF не извлеклась. Дословно
(`src/sharpe_ratio_stats.py`, обращение 2026-08-23):

```python
    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) +
             (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))
...
    maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1./independent_trials) + \
           emc * scipy_stats.norm.ppf(1 - 1./(independent_trials * np.e))
    expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
...
    dsr = probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)
```

#### 4.5.1 Расхождение первое — E[SR_max] (подтверждение A2-024)

Код (`src/models/statistical_tests.py:96-99`):

```python
    expected_max_sr = sqrt_2logn - (log_logn + math.log(4 * math.pi)) / (
        2 * sqrt_2logn
    )
```

Это классическая асимптотика максимума `N` **стандартных** нормальных
величин. В ней нет ни `E[{ŜR_n}]`, ни `√V[{ŜR_n}]` — то есть неявно
принято `μ = 0`, `σ = 1`. При этом `std_sr` в коде **вычисляется**
(`statistical_tests.py:85`) и **логируется** (строка 125), но в формулу
не входит.

Замер расхождения (свежий прогон numpy/scipy):

```
     N  code E[SRmax]   paper maxZ
     2         0.2582       0.5198
     5         0.9561       1.1926
    10         1.3619       1.5746
    50         2.1009       2.2763
   100         2.3663       2.5306
  1000         3.1165       3.2551
 10000         3.7384       3.8607
 86000         4.2466       4.3579

mu=0.5 sigma=0.3 N=100:  paper E[SRmax]=1.2592   code=2.3663
mu=1.0 sigma=0.5 N=1000: paper E[SRmax]=2.6276   code=3.1165
mu=0.0 sigma=1.0 N=100:  paper E[SRmax]=2.5306   code=2.3663
```

Направление ошибки **не фиксировано**: при `σ < 1` код завышает порог
(вдвое в первой строке), при `σ = 1` — занижает. Это ровно то, что
означает **A2-024** («зависит от произвольной константы»); здесь оно
подтверждено первоисточником и числами. Не новая находка.

#### 4.5.2 Расхождение второе — стандартная ошибка Шарпа (НОВОЕ)

Код (`src/models/statistical_tests.py:113-116`):

```python
    se_sr = math.sqrt(
        (1 - skewness * best_sr + ((kurtosis - 3) / 4) * best_sr ** 2)
        / (t_obs - 1)
    )
```

Статья и обе независимые реализации требуют дополнительного слагаемого
`+ 0.5 · SR²`:

```
    SE(ŜR) = √( (1 + ŜR²/2 − γ̂_3·ŜR + (γ̂_4 − 3)/4 · ŜR²) / (T − 1) )
```

Тождество, объясняющее обе формы записи (Мертенс/Ло):
`1 + SR²/2 + (γ_4 − 3)·SR²/4  ≡  1 + (γ_4 − 1)·SR²/4`.
То есть форма с `(γ_4 − 1)/4` **без** `1 + SR²/2` — тоже верна, а форма
с `(γ_4 − 3)/4` **без** `1 + SR²/2` — не верна ни при каком соглашении
о куртозисе.

Комментарий в коде (`statistical_tests.py:104-110`) утверждает обратное:

```python
    #   * the kurtosis term was ``(γ4 - 1)/4`` so for a normal distribution
    #     (γ4 = 3) it added ``0.5·SR²`` of spurious variance instead of
    #     vanishing.
```

Слагаемое `0.5·SR²` не «spurious» — это дисперсия оценки Шарпа при
нормальных доходностях (член Ло 2002). Прежняя формула была верна;
«исправление» её сломало.

Замер, `T = 1000`, `γ_3 = 0`, `γ_4 = 3`:

```
SR=0.5: correct SE=0.033558 code SE=0.031639 ratio=1.0607
SR=1.0: correct SE=0.038749 code SE=0.031639 ratio=1.2247
SR=2.0: correct SE=0.054800 code SE=0.031639 ratio=1.7321
```

SE занижена в √(1 + SR²/2) раз. Занижение знаменателя завышает `z` и,
следовательно, **завышает DSR** — систематически, в оптимистичную сторону,
тем сильнее, чем выше заявленный Шарп. При SR = 2 ошибка 73 %.
→ **A2-074**.

#### 4.5.3 Что сделано верно

Гейт `if len(sharpe_ratios) < 2 or n_trials < 2: return 0.0`, отдельный
аргумент `n_obs` с явным предупреждением в докстринге («If omitted, falls
back to `len(sharpe_ratios)` — a legacy approximation that massively
understates DSR») и полный лог входов/выходов
(`best_sr`, `E[SR_max]`, `std_sr`, `se_sr`, `z`, `DSR`) — правильная
практика: величина, которую нельзя проверить снаружи, печатается по частям.
Записано в §6.

### 4.6 Meta-labeling: как в книге, что реализовано

**Книга, гл. 3.6–3.7.** Первичная модель даёт **сторону** (side).
Вторичная модель решает **брать или пропускать** (`{0, 1}`), обучаясь на
метках «была бы сделка первичной модели прибыльной». Цель — поднять
precision, не трогая recall первичной модели.

**Что в коде** (`scripts/build_meta_dataset.py:149-150`):

```python
        net_pnl = future_ret * direction.astype(np.float64) - cost
        meta_target = (net_pnl > 0).astype(np.int32)
```

и `scripts/train_meta_model.py:90`:

```python
    y = df["meta_target"].to_numpy().astype(np.int32)
```

**Оценка: реализовано верно и по книге.** Сторона берётся из первичной
модели (`base_direction`), цель бинарная «take/skip», и — сверх книги —
из P&L вычитается стоимость (`- cost`, по умолчанию 6 б.п.), то есть
вторичная модель учится отличать сделку, прибыльную **после издержек**.
Это строже, чем в первоисточнике. Разбиение 70/30 хронологическое,
тестовый срез лежит внутри OOS-области базовых моделей — заявлено в
докстринге `train_meta_model.py:10-14`.

Единственная проблема — не математическая: подсистема недостижима из
рантайма и артефакт отсутствует (**A2-002**, проход 1). Здесь фиксируется
только то, что реализованный метод соответствует первоисточнику.
Записано в §6.

---

## 5. ЧЕГО НЕТ ВООБЩЕ

Каждый пункт — с источником, подтверждающим, что это стандарт, а не
пожелание. Кандидаты, которые проверка **не** подтвердила как отсутствующие,
перенесены в §6.

### 5.1 Покрытие сделок в бэктесте реальными комиссиями и funding

**Комиссии — есть, но посчитаны дважды.** Инструмент создаётся с
`maker_fee` / `taker_fee` (`backtest_runner.py:113-116`), и симулятор
Nautilus их списывает. После прогона тот же расход считается **второй раз**
аналитически (`backtest_runner.py:352-360`):

```python
    cm = CostModel()
    fee_per_rt = (
        cm.calculate_fee(avg_notional, is_maker=False, fee_config=cfg.fee_config) * 2
    )
```
и печатается как «Est. Total Cost» (`backtest_runner.py:233`) рядом с
equity, которая эти же комиссии уже включает. Два числа с одним смыслом
и без указания, что их нельзя складывать.

**Funding — отсутствует в P&L.** В `add_venue` не передан ни один модуль,
списывающий funding; список `modules` не передан вовсе
(`backtest_runner.py:101-109`). Единственный `SimulationModule` в
1.221.0 — `FXRolloverInterestModule` (`backtest/modules.pyx:92`), к
перпетуалам не относится. Вместо этого funding оценивается **после**
прогона одним умножением (`backtest_runner.py:363-377`):

```python
    funding_rate = _load_actual_funding_rate(cfg)
    ...
    position_hours = num_rt * cfg.avg_holding_hours
    total_funding = cm.calculate_funding_cost(
        position_size=avg_notional,
        funding_rate=funding_rate,
        hours_held=position_hours,
        is_long=True,
    )
```

Здесь: средняя **по модулю** ставка за весь период вместо ставки на момент
расчёта; предположение `avg_holding_hours` вместо фактического времени в
позиции; `is_long=True` жёстко — то есть шорт всегда платит, хотя при
положительном funding он получает.

**Почему это стандарт, а не пожелание.** Funding — часть контракта
инструмента, а не фрикция: Binance списывает/начисляет его каждые 8 часов
(§1.1, официальная FAQ). Бэктест перпетуала, не начисляющий funding на
кривую капитала, измеряет другой инструмент. Знак ошибки не случайный:
по замеру §1.1 у BTCUSDT за 963 дня — 2889 расчётов, и историческое
среднее funding на BTC положительно, то есть лонги платят. Стратегия с
преобладанием лонгов (по A2-021 модель систематически даёт `p(UP) < 0.5`,
то есть шорты — но это про другой артефакт) получает смещённую кривую.

→ **A2-084**.

### 5.2 Учёт задержки исполнения

**Что предоставляет референсная реализация.** Сигнатура
`BacktestEngine.add_venue` в 1.221.0 (получена интроспекцией установленного
пакета) содержит:

```
FillModel fill_model: FillModel | None = None,
FeeModel fee_model: FeeModel | None = None,
LatencyModel latency_model: LatencyModel | None = None,
...
bar_adaptive_high_low_ordering: bool = False,
```

**Что передаёт проект** (`backtest_runner.py:101-109`): `venue`, `oms_type`,
`account_type`, `base_currency`, `starting_balances`, `default_leverage`,
`bar_execution=True`. Ни `fill_model`, ни `latency_model`, ни `fee_model`,
ни `bar_adaptive_high_low_ordering`.

Последствие `bar_adaptive_high_low_ordering=False` — из докстринга
`backtest/engine.pyx:560-566`:

> «Determines whether the processing order of bar prices is adaptive based
> on a heuristic. This setting is only relevant when `bar_execution` is True.
> If False, bar prices are always processed in the fixed order:
> Open, High, Low, Close.»

Фиксированный порядок O→H→L→C означает, что на баре, где задеты оба
барьера, движок всегда сначала видит **максимум**. Для брекета это
систематическое смещение в пользу тейк-профита. Сегодня тейк-профита нет
(§1.3), поэтому эффект приглушён — но это ровно тот флаг, который зрелая
система включает перед тем, как добавить TP.

Отсутствие `LatencyModel` означает нулевую задержку между решением и
исполнением. Для 4H-стратегии, торгующей на закрытии бара, это менее
критично, чем для внутридневной, — но здесь важнее, что параметр
существует в API и просто не заполнен, а не что его нет.

→ **A2-083**.

### 5.3 Walk-forward с переобучением

**Есть.** `src/models/ml_validator.py:340-341` в каждом окне обучает
новую модель:

```python
                trainer._feature_columns = []
                model = trainer.train(train_df)
                result = trainer.evaluate(model, test_df)
```

Кандидат **не подтверждён** как отсутствующий. Дефект в этом пути другой —
отсутствие эмбарго (**A2-027**, подтверждено в §4.3). Записано в §6.

### 5.4 Контроль дрейфа признаков

**Отсутствует.** Замер: grep `drift|psi|ks_2samp|population_stability`
по `src/` и `scripts/` даёт только `check_clock_drift` в
`src/ingestion/data_quality.py:367` — это монотонность таймстемпов
`agg_trades`, не дрейф распределения признаков.

**Почему это стандарт.** Breck, Cai, Nielsen, Salib, Sculley,
«The ML Test Score: A Rubric for ML Production Readiness and Technical Debt
Reduction», Proceedings of IEEE Big Data 2017
(https://research.google.com/pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf,
обращение 2026-08-23). Дословно:

> «**Data 1: Feature expectations are captured in a schema**: It is useful
> to encode intuitions about the data in a schema so they can be
> automatically checked. […] Such expectations can be used for tests on
> input data during training and serving (see test Monitor 2).»

> «**Monitor 2: Data invariants hold in training and serving inputs**:
> […] analyzing and comparing data sets is the first line of defense for
> detecting problems where the world is changing in ways that can confuse
> an ML system. **How?** Using the schema constructed in test Data 1,
> measure whether data matches the schema and alert when they diverge
> significantly.»

Ни схемы признаков, ни сравнения распределений train/serving в проекте нет.
Для системы, у которой живой confidence 0.50–0.54 при пороге 0.65 (вход
§3 файла `00_method.md`), это и есть недостающий инструмент: без сравнения
распределений невозможно отличить «рынок изменился» от «признак в live
считается иначе».

→ **A2-089**.

**Что при этом сделано правильно** — `Monitor 3` того же источника:

> «**Monitor 3: Training and serving features compute the same values**:
> The codepaths that actually generate input features may differ at
> training and inference time. […] This is sometimes called
> "training/serving skew".»

Проект закрывает это архитектурно: и обучение, и live идут через один
`FeaturePipeline`, live-путь — `build_from_buffer()`
(`src/features/feature_pipeline.py:472`), и в `ml_strategy.py:2010` это
названо прямо («Phase 6 — eliminates train/serve skew by using the same
transforms»). Это правильнее, чем мониторить расхождение двух кодовых
путей. Записано в §6.

### 5.5 Версионирование датасета

**Отсутствует — при том что манифест есть.** Манифест
(`src/models/lgbm_trainer.py:971-1036`) содержит 30 полей, включая:

```python
            "feature_columns_hash": hashlib.sha256(
                json.dumps(sorted(self._feature_columns)).encode()
            ).hexdigest(),
```

Хэшируется **список имён колонок**, не данные. Два датасета, пересобранные
разными версиями кода признаков, с исправленным look-ahead или на другом
срезе истории, дадут **один и тот же** `feature_columns_hash`, если набор
имён не изменился. Манифест при этом фиксирует `data_range`, `n_train_rows`,
`oos_start_ms` — то есть границы, но не содержимое.

Тот же источник (ML Test Score, `Infra 1`):

> «**Infra 1: Training is reproducible**: Ideally, training twice on the
> same data should produce two identical models. […] This sort of
> diff-testing relies entirely on deterministic training.»

Без хэша содержимого «the same data» непроверяемо. Конкретно для этого
проекта проблема не гипотетическая: **A2-023** установила, что бандлы
записаны до фиксов и train/test пересекаются на 51–57 суток, а **A2-031** —
что прод-путь и производитель бандлов не связаны. Хэш содержимого датасета
позволил бы обнаружить оба факта автоматически, из самого бандла.

→ **A2-085**.

### 5.6 Отдельный OOS-период, не тронутый ни разу

**Формально есть, фактически — нет.** Механизм:
`src/models/temporal_split.py:25-42` — `compute_default_oos_start_ms`,
последние 20 % **временного** диапазона; граница записывается в манифест
(`oos_start_ms`, `oos_start_iso`).

Но тот же диапазон многократно используется как test:
* в `purged_kfold_cv` последний фолд лежит в этой зоне;
* в walk-forward последнее окно — тоже;
* результаты этих прогонов участвуют в отборе (A2-029 — гейт записи модели);
* `scripts/train_meta_model.py:10-14` прямо строит мета-тест внутри
  base-OOS.

«Не тронутый ни разу» означает срез, по которому не принималось **ни одного**
решения — ни выбора гиперпараметров, ни выбора порога, ни решения записать
модель. Такого среза в проекте нет: последние 20 % видели все процедуры.

Отдельной находкой не оформляю: это следствие уже открытых **A2-029** и
**A2-030** (гейт на валовых метриках; критерий go-live существует только
как печатная строка). Фиксирую как подтверждение.

---

## 6. ЧТО СДЕЛАНО ХОРОШО

Не «неплохо для одиночки», а «правильно по источнику». Каждый пункт —
с тем же стандартом доказательства, что и находки.

### 6.1 Конвенция времени бара выдержана сквозь весь проект

`ts_event` = close time — ровно то, что делает адаптер Binance в Nautilus
(`adapters/binance/common/schemas/market.py:262` и `:710`). Проект не просто
совпал, а **явно снял неоднозначность** обеих конвенций одной формулой
(`src/features/live_feature_state.py:39-58`):

```python
        open = ((ts_event + 1) // duration - 1) * duration
```

с разбором в докстринге, почему наивное `ts_event - duration` промахивается
на миллисекунду и почему эта миллисекунда важна для `startTime=` и для
asof-join. Это выше среднего уровня: большинство систем на этом месте
хранят молчаливое допущение.

### 6.2 Отбрасывание незакрытой свечи с объяснением причины

`ml_strategy_15m.py:586-593` — то же поведение, что в
`adapters/binance/data.py:1001-1002` («Not closed yet»), плюс защита от
усечённых строк и **отказ от частичной истории** вместо тихой склейки:

```python
            short_rows = sum(1 for k in klines if len(k) < 7)
            if short_rows:
                self._warmup_complete = False
```

с обоснованием («a hole silently shifts them all… a short row is a format
change, not a network glitch»). Отказ вместо деградации — правильный выбор
для позиционных скользящих окон.

### 6.3 Ранняя остановка LightGBM — корректна

Разобрано в §3.1 по цепочке `engine.py:205-221` → `engine.py:255-257` →
`callback.py:310-312`. Передача `train_data` в `valid_sets` при
`valid_names=["train","val"]` **не** ломает раннюю остановку. Явная
фильтрация `n_estimators` / `early_stopping_rounds` из параметров
(`lgbm_trainer.py:749-753`) с комментарием, почему они там не место, —
тоже правильно.

### 6.4 Meta-labeling соответствует первоисточнику и строже него

§4.6. Сторона от первичной модели, бинарная цель take/skip, и — сверх
AFML — вычет издержек из P&L перед разметкой
(`scripts/build_meta_dataset.py:149-150`).

### 6.5 Формула concurrency AFML воспроизведена точно

§4.2. Разностный массив + кумулятивная сумма дают в точности
`ū_i = (1/|span|) Σ 1/c_t`, при этом посчитано **по фактическому** `t1_bar`,
а не по фиксированному горизонту — то есть исправлено то самое
переоценивание concurrency, о котором предупреждает гл. 4.4. Разбиение по
символам корректно. Испорчена ровно одна строка (нормировка, A2-016).

### 6.6 Рейт-лимитер построен правильно

`src/execution/binance_rate_limiter.py`. Совпадает с замеренным лимитом
2400/мин; бюджет 50 %; скользящее окно 60 с; **заголовок с провода как
источник истины** (`X-MBX-USED-WEIGHT-1M`) — то есть локальный счётчик не
может недооценить; async-safe singleton; fail-soft. Пункт «источник истины
— провод» встречается редко и здесь сделан.

### 6.7 Fail-soft отделён от fail-safe осознанно

Проект последовательно различает два режима, и в каждом месте выбор
объяснён:
* **fail-soft** там, где отказ не влияет на риск: предзагрузка баров
  (`ml_strategy_15m.py:522-523`, «Never raises»), `on_data`
  (`ml_strategy.py:636`), рейт-лимитер, `_git_commit` («no git, no repo,
  a timeout or any other failure must never take a training run down»);
* **fail-safe** там, где влияет: `_get_funding_rate` возвращает `None`,
  а не `0.0`, с прямым разбором почему
  (`ml_strategy.py:1259-1267`): «Returning `0.0` here would be
  indistinguishable from a legitimate neutral-market funding and would
  silently let trades through». Апстрим-фильтр трактует `None` как жёсткий
  блок (`src/risk/risk_engine.py:372`).

Это именно та граница, на которой ошибаются чаще всего.

### 6.8 Устранение train/serve skew архитектурой, а не мониторингом

§5.4. Один `FeaturePipeline` на оба пути, live через `build_from_buffer()`.
По классификации ML Test Score (Monitor 3) это сильнее, чем мониторить
расхождение двух реализаций: расхождения не может быть по построению.

### 6.9 Провенанс-манифест

`lgbm_trainer.py:971-1036`, 30 полей. Существенно то, что в нём есть
поля, которые обычно забывают:
* `passes` — вердикт гейта, и `written_despite_failing` рядом, «так что
  исследовательский артефакт нельзя принять за проверенный»;
* `oos_start_ms` — сама граница отсечения, а не первый выживший бар
  (комментарий PR-K объясняет, почему это разные вещи);
* `embargo_rows` и `train_rows_after_embargo` — «n_train_rows counts the
  frame, these two count what LightGBM actually fitted on»;
* `symbols_in_train` отдельно от `symbols` — «a symbol whose feature file
  is missing is skipped with a warning upstream, so config.symbols can
  overstate the training set»;
* `lgbm_params` снимаются **после** подмены профиля и переопределений
  (`self._effective_lgbm_params`, строка 767), то есть записан реальный
  вход `lgb.train`, а не дефолты конфига;
* `git_commit`, `lightgbm_version`, `python_version`.

Недостающее звено — хэш содержимого данных (§5.5, A2-085), но набор
записанного заметно полнее типового.

### 6.10 Логирование величин, которые нельзя проверить снаружи

`statistical_tests.py:123-126` печатает все компоненты DSR по отдельности
(`best_sr`, `E[SR_max]`, `std_sr`, `se_sr`, `z`, `DSR`). Именно благодаря
этому обе ошибки §4.5 удалось локализовать чтением, без запуска обучения:
`std_sr` виден в логе и виден его неучёт.

### 6.11 Разделение known-bad / unknown в watchdog

`src/execution/watchdog.py:38-62`: вердикт heartbeat — три состояния,
`UNKNOWN` выделен отдельно, с комментарием, почему:

```python
# to return "alive" and the watchdog stood down. They now return UNKNOWN,
```

плюс бюджет последовательных «слепых» проверок и проверка согласованности
этого бюджета с интервалом (`_UNKNOWN_BUDGET_TOLERANCE`, строки 157,
222-226). Различение «знаю, что плохо» и «не знаю» — редкое и правильное
решение: без него слепая проверка читается как доказательство жизни.

### 6.12 Walk-forward действительно переобучает

§5.3. `trainer.train(train_df)` в каждом окне, с явным сбросом
`_feature_columns`.

---

## 7. НАХОДКИ ПРОХОДА 7 (A2-074 … A2-089)

Нумерация продолжает реестр §5 файла `00_method.md` (последний занятый
номер там — A2-073).

### A2-074 Стандартная ошибка Шарпа в DSR потеряла член `1 + SR²/2`
**Севирити:** HIGH
**Тип:** математика / расхождение с практикой
**Где:** `src/models/statistical_tests.py:104-116`
**Что в коде:**
```python
    #   * the kurtosis term was ``(γ4 - 1)/4`` so for a normal distribution
    #     (γ4 = 3) it added ``0.5·SR²`` of spurious variance instead of
    #     vanishing.
    t_obs = n_obs if n_obs is not None else len(sharpe_ratios)
    if t_obs < 2:
        return 0.0
    se_sr = math.sqrt(
        (1 - skewness * best_sr + ((kurtosis - 3) / 4) * best_sr ** 2)
        / (t_obs - 1)
    )
```
**В чём дефект:** Bailey & López de Prado (2014), уравнение (2), задаёт
знаменатель `√((1 − γ̂₃·ŜR + (γ̂₄ − 1)/4 · ŜR²)/(T−1))`. Тождественная
форма записи — `1 + ŜR²/2 − γ̂₃·ŜR + (γ̂₄ − 3)/4 · ŜR²`. Код взял вторую
форму куртозисного члена, но выбросил `1 + ŜR²/2`. Комментарий называет
`0.5·SR²` «spurious variance» — это дисперсия оценки Шарпа при нормальных
доходностях (Ло 2002 / Мертенс 2002), она не исчезает при нормальности.
Независимая реализация той же формулы:
`github.com/rubenbriones/Probabilistic-Sharpe-Ratio`, `src/sharpe_ratio_stats.py`
(обращение 2026-08-23):
`sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))`
**Как проявляется:** SE занижена в `√(1 + SR²/2)` раз. Замер при `T=1000`,
`γ₃=0`, `γ₄=3`:
```
SR=0.5: correct SE=0.033558 code SE=0.031639 ratio=1.0607
SR=1.0: correct SE=0.038749 code SE=0.031639 ratio=1.2247
SR=2.0: correct SE=0.054800 code SE=0.031639 ratio=1.7321
```
`z = (SR − E[SR_max]) / SE` завышен на те же 6–73 %, DSR завышен
систематически и тем сильнее, чем выше заявленный Шарп.
**Кто ещё это читает:** `calculate_dsr` вызывается из `run_all_tests`
(`statistical_tests.py`) и из трёх валидаторов (A2-007). Контракт «DSR ∈ [0,1],
цель ≥ 0.95» держится формально, но число смещено вверх у всех трёх.
**Как установлено:** чтением статьи (дословные цитаты §4.5) + замером
(численное сравнение формул).
**Уверенность:** доказано.

### A2-075 Спецификация SOLUSDT в каталоге бэктеста противоречит бирже
**Севирити:** HIGH
**Тип:** семантика / расхождение с практикой
**Где:** `src/execution/data_catalog.py:38-44`
**Что в коде:**
```python
    "SOLUSDT": {
        "base": SOL,
        "price_precision": 3,
        "size_precision": 0,
        "price_increment": 0.001,
        "size_increment": 1.0,
    },
```
**В чём дефект:** замер `GET /fapi/v1/exchangeInfo` (2026-08-23) даёт для
SOLUSDT `tickSize 0.0100`, `stepSize 0.01`, `pricePrecision 4`,
`quantityPrecision 2`. Код задаёт шаг цены 0.001 (в 10 раз мельче тика) и
шаг объёма 1.0 (в 100 раз крупнее лота).
**Как проявляется:** в бэктесте `instrument.make_qty()` округляет объём SOL
до целых монет. При SOL ≈ $150 и риске $100 истинный размер 0.66 SOL
превращается в 0 (сделки нет) или 1 (риск +50 %). Все SOL-метрики бэктеста
считаются на квантованных объёмах.
**Кто ещё это читает:** `AtomiCortexCatalog` читается только из
`src/execution/backtest_runner.py:82` (замер grep). Live-инструмент приходит
из адаптера Binance, там значения настоящие — то есть бэктест и live
работают с **разными** инструментами под одним именем.
**Как установлено:** замером (`exchangeInfo` + grep по потребителям).
**Уверенность:** доказано.

### A2-076 Фильтры биржи (minNotional / tickSize / stepSize) не читает никто
**Севирити:** MEDIUM
**Тип:** архитектура / расхождение с практикой
**Где:** весь `src/` — отсутствие
**Что в коде:** замер
`grep -rn "min_notional|minNotional|tickSize|stepSize|exchangeInfo" src/`
даёт вхождения только в `src/execution/data_catalog.py` (зашитая таблица,
не запрос).
**В чём дефект:** `MIN_NOTIONAL` — фильтр матчинг-движка. Замер:
BTCUSDT `minNotional 50`, ETHUSDT `20`, SOLUSDT `5`. Риск-движок
(`src/risk/risk_engine.py`) считает размер позиции от риска в процентах,
не сверяясь с этим порогом.
**Как проявляется:** при малом счёте или узком стопе нотионал уходит ниже
порога, биржа отклоняет входной ордер. К этому моменту `_emit_signal`
уже выполнен (`ml_strategy.py:1132`, до `submit_order`), то есть сигнал
ушёл в Telegram, а позиции нет. Обработчик `on_order_rejected` пометит
сигнал rejected — то есть система деградирует не молча, но причина
предотвратима на входе.
**Кто ещё это читает:** `RiskEngine` → `RiskDecision.position_size` →
`instrument.make_qty()` → `submit_order`. Ни одно звено порога не знает.
**Как установлено:** замером (exchangeInfo + grep).
**Уверенность:** доказано (отсутствие проверки); последствие — вероятно
(зависит от размера счёта).

### A2-077 Аварийный Limit-IOC watchdog нарушает tickSize BTCUSDT
**Севирити:** HIGH
**Тип:** логика / расхождение с практикой
**Где:** `src/execution/watchdog.py:785`
**Что в коде:**
```python
                    "price": f"{limit_price:.2f}",
```
**В чём дефект:** формат даёт два знака после запятой. Замер:
BTCUSDT `tickSize = 0.10`. Цена, не кратная тику, не проходит
`PRICE_FILTER`.
**Как проявляется:** ветка H16 («Limit-IOC при markPrice ± 0.3 %, чтобы
закрытие не съело 1–5 % проскальзывания на тонкой книге») для BTCUSDT
не срабатывает никогда — `_try_limit_ioc_close` возвращает `False` и код
уходит в безусловный MARKET. Fail-soft отработает, позиция закроется, но
именно та защита, ради которой ветка написана, отсутствует. Для ETHUSDT
и SOLUSDT (`tickSize = 0.01`) формат совпал случайно.
**Кто ещё это читает:** `_emergency_close_all` (`watchdog.py:381`) —
единственный вызывающий; ветка MARKET исполняется всегда.
**Как установлено:** чтением кода + замером `exchangeInfo`.
**Уверенность:** доказано (нарушение фильтра); конкретный код ошибки
Binance не проверен — см. §9, п. 7.

### A2-078 Частичный филл входа классифицируется как выходной
**Севирити:** HIGH
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:907`, `922`, `934-937`
**Что в коде:**
```python
        is_entry_fill = client_oid in self._pending_sl_params
...
            sl_params = self._pending_sl_params.pop(client_oid)
...
            self._submit_stop_loss_with_retry(
                decision=sl_params["decision"],
                signal=sl_params["signal"],
                fill_qty=fill_qty,
            )
```
**В чём дефект:** ключ снимается на первом филле, а филлов на один
`client_order_id` может быть несколько. Референсная реализация:
`nautilus_trader/adapters/binance/futures/schemas/user.py:511,533-547` —
`last_qty = Quantity(float(self.l), ...)` (объём **последней сделки**),
`trade_id = TradeId(str(self.t))`, и `generate_order_filled` вызывается
на каждое событие `ORDER_TRADE_UPDATE`. Рыночная IOC-заявка, прошедшая по
нескольким уровням книги, даёт несколько `OrderFilled`.
**Как проявляется:** (1) второй и последующие филлы уходят в `else` и
логируются как «Exit fill (SL/close)» — `PortfolioTracker.update_fill`
для них не вызывается, размер позиции в трекере занижен; (2) стоп
выставляется на объём первого филла — остаток позиции без защиты.
**Кто ещё это читает:** `PortfolioTracker` (размер позиции, просадка,
`close_position`), `RiskEngine` (лимит на позицию), биржевой SL.
Контракт «`fill_qty` = размер позиции» не держится.
**Как установлено:** чтением исходников Nautilus и адаптера Binance.
**Уверенность:** доказано (механизм); частота — зависит от ликвидности,
не измерена (обучение/бот запускать запрещено).

### A2-079 `OrderFactory.bracket` и `manage_contingent_orders` не используются
**Севирити:** MEDIUM
**Тип:** архитектура / расхождение с практикой
**Где:** `src/execution/strategies/ml_strategy.py:1240-1251`;
отсутствие `manage_contingent_orders` во всём `src/`
**Что в коде:** отправляется только `order_factory.stop_market(...)`;
`take_profit` из `RiskDecision` никуда не уходит (замер grep по
`src/execution/`, `src/risk/`).
**В чём дефект:** в 1.221.0 есть `OrderFactory.bracket(...)` —
вход + TP + SL одним `OrderList` с `contingency_type=ContingencyType.OUO`
(докстринг: «Create a bracket order with optional entry of take-profit
order types. The stop-loss order will always be ``STOP_MARKET``»), и
исполнитель контингенции `execution/manager.pyx:502-509`, который при
исполнении одной ноги снимает вторую, а при частичном — уменьшает её
объём. Включается флагом `StrategyConfig.manage_contingent_orders`
(`trading/config.py:51`, default `False`; замер grep по `src/` — ноль
вхождений).
**Как проявляется:** тейк-профит, на котором построена разметка
(верхний барьер `1.5 × ATR`), не существует в исполнении; сделка живёт
до стопа или до ручного закрытия. Это исполнительная сторона **A2-015**.
**Кто ещё это читает:** `signal_bridge` (пишет `take_profit` в журнал
сигналов), `reconciler_signals.py:292` (читает `tp` оттуда), Telegram-текст
(`ml_strategy.py:848`). Все три показывают число, которое биржа не знает.
**Как установлено:** чтением (интроспекция `OrderFactory`, исходники
`manager.pyx`, `config.py`) + замером grep.
**Уверенность:** доказано.

### A2-080 Отмена или истечение стопа не обрабатывается
**Севирити:** MEDIUM
**Тип:** логика / недоделка
**Где:** `src/execution/strategies/ml_strategy.py` — отсутствие
`on_order_canceled` / `on_order_expired`
**Что в коде:** переопределены `on_order_filled`, `on_order_rejected`,
`on_order_denied`, `on_position_opened`, `on_position_closed`
(замер `grep -n "def on_"`). `on_order_canceled`, `on_order_expired`,
`on_order_triggered`, `on_order_updated`, `on_position_changed`,
`on_event` — нет.
**В чём дефект:** база Nautilus объявляет все эти хуки
(`trading/strategy.pyx`). Стоп, снятый на бирже — вручную, watchdog'ом
(`watchdog.py:308` снимает висящие заявки перед reduceOnly-закрытием)
или движком, — исчезает без единой записи. Для отклонения стопа путь есть:
`ml_strategy.py:1204` пишет `POSITION UNPROTECTED!`; для отмены —
симметричного пути нет.
**Как проявляется:** позиция без стопа при живом процессе и молчащем
логе. Watchdog это не поймает: он смотрит heartbeat и позиции, а не
наличие защитной заявки.
**Кто ещё это читает:** никто — событие не доходит ни до трекера, ни до
Telegram.
**Как установлено:** чтением (список хуков базы против списка
переопределений).
**Уверенность:** доказано (отсутствие обработчика).

### A2-081 `next_funding_ns` приходит в каждом тике и игнорируется
**Севирити:** MEDIUM
**Тип:** расхождение с практикой
**Где:** `src/execution/strategies/ml_strategy.py:626`
**Что в коде:**
```python
                if dt.hour in (1, 9, 17) and dt.minute == 0:
```
**В чём дефект:** объект, который код уже получил и уже разобрал, несёт
момент расчёта. `nautilus_trader/adapters/binance/futures/types.py:43-44`:
«`next_funding_ns : uint64_t` — UNIX timestamp (nanoseconds) when next
funding will occur». Заполняется из поля `T` WS-потока markPrice
(`adapters/binance/futures/schemas/market.py:227` — `T: int  # Next funding time`;
строка 238 — `next_funding_ns=millis_to_nanos(self.T)`).
**Как проявляется:** это исполнительная причина **A2-060** (история funding
в live не пополняется: расчёты 00/08/16, код ждёт 01/09/17). Сверх того —
код не переживёт смену частоты расчёта, которую Binance выполняет
автоматически при достижении cap/floor (официальная FAQ, §1.1), тогда как
`next_funding_ns` отражает её сразу.
**Кто ещё это читает:** `LiveFeatureState.funding_rate_history` →
`add_funding_features` → `funding_zscore_7d/30d`, `funding_cum_24h`.
**Как установлено:** чтением исходников адаптера + документацией биржи.
**Уверенность:** доказано.

### A2-082 Признаки окна funding сдвинуты на час; `post_funding_window` мёртв на 4H
**Севирити:** MEDIUM
**Тип:** семантика / математика
**Где:** `src/features/session_features.py:60`, `:341-343`
**Что в коде:**
```python
_FUNDING_MARKS = [1, 9, 17]
...
            hours_to_mark.alias("hours_to_funding_mark"),
            (hours_to_mark <= _PRE_FUNDING_WINDOW_H).alias("pre_funding_window"),
            (min_hours_since <= 1.0).alias("post_funding_window"),
```
**В чём дефект:** расчёт идёт в 00/08/16 UTC (официальная FAQ Binance;
замер по данным проекта — 2889 записей на символ, часы строго {0,8,16}).
**Как проявляется:** замер прогоном модуля:
```
--- 4H grid | code marks=[1, 9, 17] ---
 hours_to_funding_mark  code: [1.0, 5.0]  true: [0.0, 4.0]
 pre_funding_window(<=2) code: [0, 8, 16]  true: [0, 8, 16]
 post_funding_window(<=1) code: []  true: [0, 8, 16]
--- 1H grid | code marks=[1, 9, 17] ---
 pre_funding_window(<=2) code: [0, 1, 7, 8, 9, 15, 16, 17, 23]  true: [0, 6, 7, 8, 14, 15, 16, 22, 23]
 post_funding_window(<=1) code: [1, 2, 9, 10, 17, 18]  true: [0, 1, 8, 9, 16, 17]
```
На 4H `post_funding_window` тождественно `False` (правильная сетка дала бы
`True` на 1/3 баров), `hours_to_funding_mark` принимает {1,5} вместо {0,4}.
На 1H множества совпадают на 3 часа из 6.
**Кто ещё это читает:** `FEATURE_GROUPS_MTF["session"]`
(`src/features/feature_pipeline.py:149-150`) — то есть 1h/15m-профили.
В 4H-датасет эти колонки не входят (замер: в
`data/features/ml_features/BTCUSDT_4h_features.parquet` 64 колонки,
funding-колонок 7, ни одной из четырёх названных). Радиус — MTF-модели.
**Как установлено:** замером (прогон модуля на обеих сетках + проверка
состава колонок датасета).
**Уверенность:** доказано.

### A2-083 Бэктест-площадка без FillModel, LatencyModel и с фиксированным порядком O→H→L→C
**Севирити:** MEDIUM
**Тип:** расхождение с практикой
**Где:** `src/execution/backtest_runner.py:101-109`
**Что в коде:**
```python
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USDT,
            starting_balances=[Money(cfg.initial_capital, USDT)],
            default_leverage=Decimal(str(cfg.leverage)),
            bar_execution=True,
        )
```
**В чём дефект:** сигнатура 1.221.0 принимает `fill_model`, `fee_model`,
`latency_model`, `modules`, `bar_adaptive_high_low_ordering` — ни один не
передан. Докстринг `backtest/engine.pyx:560-566`: «If False, bar prices are
always processed in the fixed order: Open, High, Low, Close.»
**Как проявляется:** нулевая задержка исполнения; проскальзывание не
моделируется в движке (считается отдельно и постфактум, §5.1); на баре,
где задеты оба барьера, движок всегда видит максимум первым —
систематическое смещение в пользу тейк-профита, как только TP появится.
**Кто ещё это читает:** `BacktestResult.sharpe / profit_factor / win_rate`
и всё, что строится на них (гейт A2-029, DSR-прокси).
**Как установлено:** чтением сигнатуры установленного пакета и кода
вызова.
**Уверенность:** доказано (параметры не переданы); величина смещения не
измерена.

### A2-084 Funding не начисляется на кривую капитала бэктеста; комиссии посчитаны дважды
**Севирити:** HIGH
**Тип:** математика / расхождение с практикой
**Где:** `src/execution/backtest_runner.py:363-377`, `:352-360`, `:179`
**Что в коде:**
```python
    funding_rate = _load_actual_funding_rate(cfg)
    if funding_rate is None:
        funding_rate = cfg.typical_funding_rate
    position_hours = num_rt * cfg.avg_holding_hours
    total_funding = cm.calculate_funding_cost(
        position_size=avg_notional,
        funding_rate=funding_rate,
        hours_held=position_hours,
        is_long=True,
    )
```
**В чём дефект:** funding — часть контракта перпетуала, а не фрикция:
Binance списывает/начисляет его каждые 8 часов (официальная FAQ, §1.1;
замер — 2889 расчётов за 963 дня на символ). В `add_venue` не передан ни
один модуль, который бы это делал, и `SimulationModule` в 1.221.0 только
один — `FXRolloverInterestModule` (`backtest/modules.pyx:92`), к
перпетуалам не относящийся. Вместо начисления — одно умножение после
прогона, на трёх допущениях: средняя **по модулю** ставка за весь период;
`avg_holding_hours` вместо фактического времени в позиции; `is_long=True`
жёстко (шорт всегда платит, хотя при положительном funding он получает).
Отдельно: комиссии уже списаны движком через `maker_fee`/`taker_fee`
инструмента (`backtest_runner.py:113-116`), и те же комиссии считаются
второй раз в `_estimate_costs` и печатаются рядом с equity как
«Est. Total Cost» — два числа с одним смыслом, без пометки, что их нельзя
складывать.
**Как проявляется:** кривая капитала и все производные от неё метрики
(Sharpe, PF, max DD) описывают инструмент без funding. Знак смещения
зависит от направления сделок и знака ставки, но не случаен.
**Кто ещё это читает:** `BacktestResult` → отчёт → гейт записи модели
(A2-029) → DSR-прокси (A2-024, A2-074).
**Как установлено:** чтением кода + замером (число расчётов funding
в данных проекта) + чтением сигнатуры `add_venue` и списка модулей.
**Уверенность:** доказано.

### A2-085 Манифест хэширует имена признаков, а не данные
**Севирити:** MEDIUM
**Тип:** архитектура / расхождение с практикой
**Где:** `src/models/lgbm_trainer.py:979-981`
**Что в коде:**
```python
            "feature_columns_hash": hashlib.sha256(
                json.dumps(sorted(self._feature_columns)).encode()
            ).hexdigest(),
```
**В чём дефект:** хэш берётся от **списка имён колонок**. Два датасета,
пересобранные разными версиями кода признаков, с исправленным look-ahead
или на другом срезе истории, дадут одинаковый хэш при том же наборе имён.
Источник: Breck et al., ML Test Score, IEEE Big Data 2017, `Infra 1`:
«Ideally, training twice on the same data should produce two identical
models. […] This sort of diff-testing relies entirely on deterministic
training.» Без хэша содержимого «the same data» непроверяемо.
**Как проявляется:** бандл на диске не позволяет установить, на каких
именно данных он обучен. Ровно эту проверку не удалось выполнить в
проходах 2 и 4 — **A2-023** (бандлы записаны до фиксов, train/test
пересекаются на 51–57 суток) и **A2-031** (прод-путь и производитель
бандлов не связаны) пришлось устанавливать косвенно.
**Кто ещё это читает:** `load_bundle` (`lgbm_trainer.py:1173-1191`),
стратегии при старте, скрипты валидации.
**Как установлено:** чтением + сопоставлением с уже открытыми находками.
**Уверенность:** доказано.

### A2-086 Порог 0.65 применяется к некалиброванному выходу перевзвешенного бустера
**Севирити:** HIGH
**Тип:** математика / расхождение с практикой
**Где:** `src/models/lgbm_trainer.py:702-703`, `:1288-1296`;
`src/config.py:90`
**Что в коде:**
```python
        train_weights = compute_sample_weight("balanced", y_train_fit)
...
        p_up = float(model.predict(features)[0])
        direction = 1 if p_up > 0.5 else -1
        confidence = p_up if direction == 1 else 1.0 - p_up
...
        if confidence < confidence_threshold:
            return 0, confidence
```
```python
    confidence_threshold: float = Field(default=0.65, alias="CONFIDENCE_THRESHOLD")
```
**В чём дефект:** документация LightGBM 4.3.0 о `scale_pos_weight`:
«while enabling this should increase the overall performance metric of your
model, it will also result in **poor estimates of the individual class
probabilities**». Что предупреждение относится и к `sample_weight`,
показывает исходник референсной реализации
(`github.com/microsoft/LightGBM`, `v4.3.0`,
`src/objective/binary_objective.hpp`): `label_weights_[1] *= scale_pos_weight_;`
и далее `gradients[i] = response * label_weight * weights_[i];` — оба
множителя входят в одно и то же произведение градиента и гессиана.
`compute_sample_weight("balanced")` задаёт `weights_[i]`, постоянный внутри
класса, то есть с точностью до общего масштаба тождествен
`scale_pos_weight`. Определение калиброванности — документация sklearn:
«a well calibrated (binary) classifier should classify the samples such
that among the samples to which it gave a predict_proba value close to,
say, 0.8, approximately 80% actually belong to the positive class».
Замер: `grep -rni "calibrat|isotonic|platt|brier|reliability"` по `src/`
и `scripts/` даёт только комментарии, **ни одной строки кода калибровки**.
**Как проявляется:** число, с которым сравнивается 0.65, — вероятность в
перевзвешенной популяции, которой не существует; сдвиг шкалы зависит от
дисбаланса конкретного train-среза и меняется при каждом переобучении.
Живой confidence 0.50–0.54 при пороге 0.65 (вход §3 файла `00_method.md`)
поэтому нельзя интерпретировать: неизвестно, модель не уверена или шкала
съехала. Комментарии в коде («Threshold 0.55 is calibrated for binary»,
`lgbm_trainer.py:1113`; «recalibrate after retrain»,
`configs/strategy_1h.py:31`) обещают калибровку, которой нет.
**Кто ещё это читает:** `MLStrategyConfig.confidence_threshold` (4H),
`MLStrategyConfig1H` (0.55), `MLStrategyConfig15M` (0.58) — три разных
числа против одной и той же несопоставимой шкалы; `signal_bridge` пишет
`confidence` в журнал, Telegram показывает его пользователю как «уверенность».
**Как установлено:** чтением документации LightGBM и sklearn + исходника
LightGBM v4.3.0 + замером grep.
**Уверенность:** доказано (отсутствие калибровки и применимость
предупреждения); величина смещения не измерена (обучение запрещено).

### A2-087 Рейт-лимитер не считает заявки; список «Wired into» неверен
**Севирити:** LOW
**Тип:** недоделка / семантика
**Где:** `src/execution/binance_rate_limiter.py:25-29`, `:52`
**Что в коде:**
```
Wired into
----------
* ``LiveFeatureState.fetch_taker_buy_volume`` — H1c
* ``Watchdog._signed_*``                       — H15/H16
* ``reconciler.fetch_position_risk``           — H10/this step
```
```python
    MAX_WEIGHT_PER_MINUTE: int = 1200   # 50 % of the 2400 hard cap
```
**В чём дефект:** (1) замер `exchangeInfo` даёт три лимита, а не один:
`REQUEST_WEIGHT 2400/мин`, `ORDERS 1200/мин`, `ORDERS 300/10 с`. Лимитер
считает только вес; заявки не считает никто, при том что
`_submit_stop_loss_with_retry` делает до 3 попыток, а
`_emergency_close_all` шлёт заявки по всем символам сразу. (2) Замер
`grep -rn "BinanceRateLimiter|limiter.acquire" src/` даёт вхождения только
в `src/execution/watchdog.py:857,891,924` и
`src/execution/reconciler.py:240-242`. В `src/features/live_feature_state.py`
лимитера нет — первый пункт списка неверен.
**Как проявляется:** превышение лимита ORDERS даёт HTTP 429 и, при
повторе, бан IP — то есть отказ ровно в момент аварийного закрытия
позиций. Ошибочная строка докстринга создаёт ложную уверенность, что
пер-баровые сетевые вызовы уже под контролем.
**Кто ещё это читает:** watchdog и reconciler (реальные потребители);
`ml_strategy.py` ходит в сеть синхронным `requests` в `on_bar` и
структурно не может использовать async-лимитер.
**Как установлено:** замером (`exchangeInfo` + grep).
**Уверенность:** доказано.

### A2-088 `PurgedKFoldCV` не реализует purged K-fold из AFML
**Севирити:** HIGH
**Тип:** математика / семантика
**Где:** `src/execution/walk_forward.py:42-53`, `:124-126`, `:334-337`
**Что в коде:**
```python
class PurgedKFoldCV:
    """Time-series cross-validation with an embargo gap between train and test.

    Fold layout (expanding train, fixed-size test block):

        Fold 1: [==TRAIN==][GAP][TEST]
```
```python
        block = total / (self.n_splits + 1)
        embargo = self.embargo_pct * total
```
**В чём дефект:** AFML гл. 7.4 определяет: K смежных тестовых блоков;
обучение — **всё остальное, включая данные после теста**; purging (7.4.1) —
удаление обучающих наблюдений, **метки которых перекрываются по времени**
с метками теста; embargo (7.4.2) — дополнительное окно **после** теста.
В коде: растущее обучение только слева, зазор один и справа его нет;
purging не по перекрытию меток, а фиксированной долей **всего диапазона
данных**. При `embargo_pct = 0.02` (умолчание в
`scripts/validate_1h_models.py:216` и `scripts/validate_15m_models.py:217`)
и диапазоне 963 дня зазор ≈ 19.3 суток, тогда как горизонт метки 4H при
`max_holding = 6` — 1 сутки. Величина зазора зависит от длины датасета и
не зависит от горизонта метки — то есть от единственной причины, по
которой зазор нужен.
**Как проявляется:** имя класса и докстринг («AFML Ch.7 embargo»,
`walk_forward.py:334-337`) утверждают соответствие первоисточнику; читатель
результатов CV получает не то, что в книге. Знак ошибки переворачивается
при изменении длины датасета или горизонта метки.
**Кто ещё это читает:** `MLValidator.purged_kfold_cv`
(`src/models/ml_validator.py:149`), `scripts/validate_1h_models.py:230`,
`scripts/validate_15m_models.py:231`; результаты идут в `calculate_pbo`
(A2-026) и в DSR-прокси (A2-024, A2-074).
**Как установлено:** чтением первоисточника (структура гл. 7.4 сверена
с оглавлением издателя) и кода.
**Уверенность:** доказано.

### A2-089 Контроля дрейфа признаков нет
**Севирити:** MEDIUM
**Тип:** недоделка / расхождение с практикой
**Где:** весь `src/` — отсутствие
**Что в коде:** замер
`grep -rni "drift|psi|ks_2samp|population_stability" src/ scripts/` даёт
только `check_clock_drift` (`src/ingestion/data_quality.py:367`) —
монотонность таймстемпов `agg_trades`, не распределение признаков.
**В чём дефект:** Breck et al., ML Test Score, IEEE Big Data 2017,
`Data 1` («Feature expectations are captured in a schema») и `Monitor 2`
(«Data invariants hold in training and serving inputs: […] analyzing and
comparing data sets is the first line of defense for detecting problems
where the world is changing»). Ни схемы признаков, ни сравнения
распределений train/serving в проекте нет.
**Как проявляется:** живой confidence 0.50–0.54 против порога 0.65
(вход §3 файла `00_method.md`) не поддаётся диагностике: без сравнения
распределений неотличимо «рынок изменился» от «признак в live считается
иначе». При этом сама причина отсутствует по построению (§6.8, один
`FeaturePipeline`), но проверить это утверждение нечем.
**Кто ещё это читает:** никто — величины не вычисляются.
**Как установлено:** замером grep + рецензируемым источником.
**Уверенность:** доказано (отсутствие).

### Сводка прохода 7

| Севирити | Кол-во | Номера |
|---|---|---|
| CRITICAL | 0 | — |
| HIGH | 7 | A2-074, A2-075, A2-077, A2-078, A2-084, A2-086, A2-088 |
| MEDIUM | 8 | A2-076, A2-079, A2-080, A2-081, A2-082, A2-083, A2-085, A2-089 |
| LOW | 1 | A2-087 |
| **Всего** | **16** | A2-074 … A2-089 |

### Подтверждённые, но не переоткрытые находки прошлых проходов

| ID | Чем подтверждена в этом проходе |
|---|---|
| A2-015 | §1.3 — TP не отправляется на биржу ни в одной стратегии (замер grep) |
| A2-016 | §4.2 — сверка формулы с AFML гл. 4.3–4.4; расходится только нормировка |
| A2-017 | §4.1 — AFML 3.9 снимает **редкие** классы, не класс таймаута |
| A2-018 | §4.1 — **уточнение основания**: снипет AFML тоже работает по close; расхождение не с книгой, а с исполнением (`TriggerType.LAST_PRICE`) |
| A2-024 | §4.5.1 — E[SR_max] без `E[{ŜR_n}]` и `√V[{ŜR_n}]`; численное сравнение со Snippet 1 статьи |
| A2-025 | §4.4 — CSCV требует N конфигураций; сохранённых рядов P&L по пробам нет |
| A2-026 | §4.4 — PBO ≡ 0 доказано алгебраически, не только замером |
| A2-027 | §4.3 — `WalkForwardValidator` в ML-пути создаётся без `embargo` |
| A2-029 / A2-030 | §5.6 — «нетронутого» OOS нет: последние 20 % видели все процедуры |
| A2-060 | §1.1 — 00/08/16 подтверждено документацией и замером 2889 записей ×3 символа |
| A2-062 | §2.4 — `PortfolioTracker` дублирует `Portfolio`/`Cache` из Nautilus |
| A2-066 | §1.2 — `Price(..., precision=1)` в предзагрузке верен только для BTCUSDT |

---

## 8. НЕ ИССЛЕДОВАНО

Перечислено то, что входило в предмет прохода, но не было закрыто
доказательством. Каждый пункт — с указанием, что именно осталось.

1. **Binance: точный код ошибки при нарушении `PRICE_FILTER`.**
   A2-077 доказывает нарушение фильтра (`:.2f` против `tickSize=0.10`),
   но конкретный код (`-1111` «Precision is over the maximum defined for
   this asset» против `-4014` «Price not increased by tick size»)
   не подтверждён ни документацией, ни ответом биржи. Отправлять заявку
   для проверки запрещено протоколом.

2. **Поведение второй заявки пары при `closePosition=true`.**
   §1.3, вариант 3. Официальную формулировку правил автоматического
   снятия таких заявок в документации Binance в рамках прохода найти не
   удалось.

3. **Частота частичных филлов рыночной IOC-заявки на реальном объёме.**
   A2-078 доказывает механизм по исходникам адаптера, но частота зависит
   от ликвидности и размера заявки. Измерить нельзя — бот запускать
   запрещено, а исторических журналов филлов в репозитории нет.

4. **Величина смещения DSR на реальных данных проекта.**
   A2-074 и A2-024 дают формулы и коэффициенты искажения, но пересчитать
   фактические DSR из `mlruns` / `data/mlflow.db` в этом проходе не
   пробовал — это работа прохода 3 (`04_math_validation.md`), повтор был
   бы дублированием.

5. **Влияние `deterministic=false` + неопределённого `num_threads` на
   воспроизводимость обучения.** §3.4. Документация LightGBM 4.3.0
   говорит, что `deterministic=true` «should ensure the stable results …
   (and different `num_threads`)», из чего следует, что при `false`
   стабильность не гарантирована. Прямого доказательства расхождения
   чисел нет: обучение запускать запрещено. Вынесено в §9, п. 4, а не в
   находки.

6. **Величина смещения бэктеста от отсутствия funding в P&L.**
   A2-084 доказывает отсутствие начисления, но не считает, сколько это
   в процентах доходности: для этого нужен прогон бэктеста.

7. **`FeeModel` в Nautilus 1.221.0.** Параметр в `add_venue` есть,
   но какие реализации поставляются в комплекте и покрывают ли они
   ступенчатые VIP-комиссии Binance — не проверял.

8. **Остальные адаптеры и лимиты Binance.** Проверены только
   REQUEST_WEIGHT и ORDERS для UM-фьючерсов. Лимиты WebSocket-подписок,
   `RAW_REQUESTS` и лимиты по ключу (а не по IP) не рассматривались.

9. **Веса запросов по конкретным эндпойнтам.** §1.5 фиксирует, что
   ml_strategy не подключён к лимитеру, но соответствие весов, которыми
   оперирует `_binance_weight_for` в watchdog, документированным весам
   Binance не сверялось.

10. **Гл. 12 AFML (CPCV).** Задание требовало разбор CSCV/PBO —
    он сделан по оригинальной статье (§4.4). Комбинаторно-очищенная
    кросс-валидация из книги как альтернатива не разбиралась.

---

## 9. УТВЕРЖДЕНИЯ, КОТОРЫЕ НЕ УДАЛОСЬ ПОДТВЕРДИТЬ ИСТОЧНИКОМ

Каждое из этого списка **не** использовано как основание находки.

1. **«Сетка часов расчёта funding у Binance никогда не была 01/09/17».**
   Найдено подтверждение, что сейчас 00/08/16 и что менялась *частота*
   (8 ч → 4 ч → 1 ч и обратно). Официального документа, перечисляющего
   всю историю изменений с 2019 года, найти не удалось. Утверждение
   §1.1 «сдвига сетки на +1 ч в истории по документации не обнаружено»
   сформулировано именно как отсутствие находки, а не как доказанное
   отрицание.

2. **Правила автоматического снятия заявок с `closePosition=true`.**
   См. §8, п. 2. В §1.3 помечено «не проверено».

3. **«Бустинг систематически растягивает вероятности к 0 и 1».**
   Страница sklearn о калибровке цитирует Niculescu-Mizil & Caruana (2005)
   применительно к **bagging / random forests**, а не к бустингу.
   Оригинальную статью ICML 2005 в этом проходе не читал. Поэтому A2-086
   опирается **только** на документированный эффект взвешивания классов
   (LightGBM) и на определение калиброванности (sklearn), но не на
   утверждение о форме искажения у бустинга.

4. **«`deterministic=false` даёт разные модели на машинах с разным числом
   ядер».** Следует из документации LightGBM, но прямым измерением в этом
   проходе не подтверждено. Находкой не оформлено.

5. **Номера страниц AFML.** Главы и разделы сверены с оглавлением
   издателя; страницы не приводятся — физического экземпляра нет,
   реконструкция по памяти запрещена протоколом (§2 файла `00_method.md`).

6. **Ссылки задания на «A2-059 из прохода 5» и «A2-079 прохода 5».**
   В реестре §5 файла `00_method.md` A2-059 — находка прохода 6 про
   `get_metrics_df` (другой предмет), а A2-079 отсутствует вовсе
   (последний занятый номер — A2-073). Соответствие номеров восстановить
   не удалось; нумерация этого прохода продолжена с A2-074, номера A2-078
   и A2-079 выданы заново.

7. **Код ошибки Binance при нарушении `PRICE_FILTER`.** См. §8, п. 1.
   A2-077 утверждает только факт нарушения фильтра, не конкретный ответ.

8. **Уравнения (1) и (2) статьи о DSR не извлеклись из PDF символ-в-символ.**
   Математический шрифт исходника не поддаётся `pdftotext`. Использованы:
   (а) дословные словесные определения автора, извлёкшиеся полностью;
   (б) авторский Snippet 1, извлёкшийся дословно; (в) независимая
   реализация `github.com/rubenbriones/Probabilistic-Sharpe-Ratio`,
   совпадающая со Snippet 1 строка в строку. Тождество
   `1 + SR²/2 + (γ₄−3)SR²/4 ≡ 1 + (γ₄−1)SR²/4` проверено алгебраически
   здесь, а не процитировано.

9. **Оценка зрелости `rubenbriones/Probabilistic-Sharpe-Ratio`.**
   Однофайловый репозиторий, не пакет, не индустриальный стандарт.
   Использован исключительно как **вторая независимая реализация** для
   перекрёстной проверки нечитаемой части PDF, а не как авторитет.

---

## 10. Итог прохода

**Открыто:** 16 находок (A2-074 … A2-089): HIGH 7, MEDIUM 8, LOW 1,
CRITICAL 0.

**Подтверждено первоисточниками, без переоткрытия:** 12 находок прошлых
проходов (таблица в конце §7). Одна из них — **A2-018** — получила
уточнение основания: расхождение не с книгой, а с исполнением.

**Записано как сделанное правильно:** 12 позиций (§6), каждая с
проверкой по источнику, а не по впечатлению.

**Два самых дорогих расхождения этого прохода:**

1. **A2-086 + A2-074** — обе про одно и то же свойство системы: число,
   с которым сравнивается порог, не является тем, чем его считают.
   В инференсе это `p_up` после взвешивания классов; в валидации это
   `DSR` с заниженной стандартной ошибкой. В обоих случаях расхождение
   с первоисточником — одна строка, а последствие — что решение
   принимается по величине без известной шкалы.

2. **A2-084 + A2-083** — бэктест измеряет инструмент, которого не
   существует: перпетуал без funding, с нулевой задержкой, с
   проскальзыванием, посчитанным после прогона, и с комиссиями,
   учтёнными дважды в двух соседних строках отчёта.

**Что в проекте уже есть и просто не включено** (самая дешёвая часть
списка): `OrderFactory.bracket` + `manage_contingent_orders` (A2-079),
`next_funding_ns` (A2-081), `fill_model` / `latency_model` /
`bar_adaptive_high_low_ordering` (A2-083), `embargo=` у
`WalkForwardValidator` (A2-027). Четыре расхождения из шестнадцати
закрываются параметрами, которые уже лежат в установленной версии
библиотеки.
