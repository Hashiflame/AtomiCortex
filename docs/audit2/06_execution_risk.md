# Аудит-2, проход 5 — ИСПОЛНЕНИЕ И РИСК

Файл: `docs/audit2/06_execution_risk.md`
Дата прохода: 2026-08-23
HEAD на момент прохода: `f4af5fd210b32db8af4478f0e2f440e2eb504ccc`
(`chore: ignore local snapshot directory`, 2026-08-22 16:18:08 +0530)
Рабочее дерево: `?? docs/audit2/` — единственное изменение, правок в `src/` нет.

Предмет: путь от предсказания модели до ордера и обратно — `risk_engine`,
сайзинг, SL/TP, circuit breaker, `portfolio_tracker`, обработка исполнений,
реконсиляция. Ссылки Аудита-1: A6, A10, A19.

## Как получены числа

| Инструмент | Команда |
|---|---|
| Копия снимка БД | `cp docs/audit/atomicortex_20260702.db.bak /tmp/ac_snap.db` (оригинал не открывался) |
| Интерпретатор | `/home/asus/Desktop/AtomiCortex/.venv/bin/python` (системный `python3` не имеет `loguru`) |
| Фичи | `data/features/ml_features/{BTCUSDT,ETHUSDT,SOLUSDT}_4h_features.parquet` |

`pytest` не запускался. Бот и watchdog не запускались.

`nautilus_trader` доступен в venv проекта (системный `python3` его не видит):
`nautilus_trader.__version__` → `1.221.0`. Хуки и типы верифицированы
замером через интроспекцию — см. §11.

---

## 1. ПУТЬ ОРДЕРА

### 1.1 От `_open_position` до `submit_order` — целиком

Порядок вызовов в `on_bar` (`src/execution/strategies/ml_strategy.py`):

```
641  def on_bar(self, bar: Bar) -> None:
665      self._live_state.add_bar(...)
686      for _sym in list(self._tracker._positions.keys()):
687          self._tracker.update_price(_sym, close_px)      # mark-to-market
692      self._record_equity(bar.ts_event)                    # sync/seed equity
695      if not self._warmup_complete: ... return
716      if self._breaker is not None and self._tracker is not None:
724          breaker_state = self._breaker.check(...)
730          if breaker_state.is_triggered: ... return
743      regime_state = self._detect_regime()
766      model, features_list, conf_threshold = self._select_model(regime_label)
778      feature_vector = self._compute_features_unified(features_list)
786      direction, confidence = LGBMTrainer.get_signal(...)
811      signal = TradeSignal(...)
826      decision = self._risk_engine.evaluate(signal, portfolio_state)
842      if not self._config.dry_run:
843          self._open_position(decision, signal)
```

`TradeSignal` строится так (строки 804–821):

```python
            current_price = bar.close.as_double()
            atr_dollar = regime_state.atr_pct * current_price
            now_utc = datetime.fromtimestamp(bar.ts_event / 1e9, tz=timezone.utc)

            # Read funding rate from feature data (PROD-003 fix)
            funding_rate = self._get_funding_rate(feature_vector, features_list)

            signal = TradeSignal(
                symbol=str(self._instrument_id),
                direction=direction,
                confidence=confidence,
                regime=regime_label,
                entry_price=current_price,
                atr=atr_dollar,
                atr_pct=regime_state.atr_pct,
                funding_rate=funding_rate,
                timestamp=now_utc,
            )
```

`_open_position` (1079–1155) — полный контур отправки:

```python
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            self.log.error(f"Instrument {self._instrument_id} not found in cache")
            return

        # Direction → OrderSide
        entry_side = OrderSide.BUY if signal.direction == 1 else OrderSide.SELL

        # Quantity (round to instrument precision)
        qty = instrument.make_qty(decision.position_size)

        # Client order ID (idempotent)
        ts_ms = int(time.time() * 1000)
        dir_str = "L" if signal.direction == 1 else "S"
        entry_tag = f"AC-{dir_str}-{ts_ms}"

        # Market entry
        entry_order = self.order_factory.market(
            instrument_id=self._instrument_id,
            order_side=entry_side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
            tags=[entry_tag],
        )

        # Store SL params for deferred submission (on_order_filled will use these)
        client_oid = str(entry_order.client_order_id)
        self._pending_sl_params[client_oid] = {
            "decision": decision,
            "signal": signal,
        }
```

затем зеркалирование на диск (1123–1129), `self._emit_signal(decision, signal)`
(1133) и, наконец, `self.submit_order(entry_order)` (1136).

**Формирование SL** отложено в `on_order_filled` (884–941):

```python
        is_entry_fill = client_oid in self._pending_sl_params

        if is_entry_fill and self._tracker:
            # Entry fill → update tracker + place SL
            direction = 1 if is_buy else -1
            self._tracker.update_fill(...)

            # Submit deferred stop-loss (PROD-005 fix: SL after confirmed entry)
            sl_params = self._pending_sl_params.pop(client_oid)
            ...
            self._submit_stop_loss_with_retry(
                decision=sl_params["decision"],
                signal=sl_params["signal"],
                fill_qty=fill_qty,
            )
        else:
            # Exit fill (SL or manual close) → fees tracked via close_position
            self.log.debug(...)
```

Сам стоп (1238–1255):

```python
        for attempt in range(1, max_retries + 1):
            try:
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
                self.submit_order(stop_order)
```

**Порядок отправки:** market IOC → (ждём `OrderFilled`) → stop-market GTC
reduce-only. Между ними нет ничего.

**Take-profit ордер не отправляется никогда.** Полный список употреблений
`take_profit` в файле (`grep -n "take_profit" src/execution/strategies/ml_strategy.py`):

```
848:                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
1023:                take_profit=decision.take_profit,
```

Строка 848 — печать в ветке `[DRY RUN]`. Строка 1023 — запись в журнал
`signals_log` через `SignalBridge.log_signal`. Ни одного `limit(...)`,
`stop_limit(...)` или иного ордера в файле нет:
`grep -n "limit(\|LIMIT" ml_strategy.py` даёт только `_BINANCE_KLINES_MAX_LIMIT`
и `"limit": min(n_bars, ...)` в HTTP-запросе к klines.

Единственные пути выхода из позиции в живом коде:
1. срабатывание биржевого `STOP_MARKET`;
2. `on_stop` (495–497): `cancel_all_orders` + `close_all_positions` при
   остановке процесса.

Нет выхода по времени, нет выхода по развороту сигнала, нет TP.

### 1.2 Частичное исполнение, отклонение entry, отклонение SL

**Частичное исполнение.** `is_entry_fill` определяется наличием
`client_oid` в `_pending_sl_params` (строка 907), а первая же обработка
делает `self._pending_sl_params.pop(client_oid)` (922). Nautilus порождает
отдельное `OrderFilled` на каждый частичный филл одного ордера с тем же
`client_order_id`. Следовательно второй и последующие частичные филлы того
же entry попадают в ветку `else` (937–941): в трекер не записываются, стоп
на их количество не выставляется. Стоп ставится на `fill_qty` **первого**
филла (935), а не на итоговый размер позиции. Ордер отправлен с
`TimeInForce.IOC` (1111) — режим, в котором частичное исполнение является
штатным, а не исключительным исходом.

**Отклонение entry.** Обрабатывается (1157–1198). Ключ
`_pending_signal_ids` — `signal.symbol`, а `signal.symbol =
str(self._instrument_id)` (812); `_handle_order_reject` снимает по
`str(event.instrument_id)` (1193). Ключи совпадают. Сигнал переводится в
`result = 'rejected'` (`signal_bridge.py:280-287`).

**A10 закрыта частично.** Отклонённый entry больше не остаётся
`result='open'` и не выглядит исполненным. Но:
* причина отклонения (`reason`) только логируется — в схеме `signals_log`
  колонки под неё нет (`PRAGMA table_info` даёт 24 колонки, ни одной
  для причины);
* `_emit_signal` вызывается **до** `submit_order` (1133 против 1136), то
  есть в интервале между записью и филлом строка в журнале утверждает
  открытую позицию, которой ещё нет.

**Отклонение SL** (1199–1212):

```python
            try:
                order = self.cache.order(client_oid)
            except Exception:
                order = None

            if order and any(t.startswith("SL-") for t in (order.tags or [])):
                # TODO(PR-0.2): Route this to Telegram via a watchdog-channel (no asyncio.run here)
                self.log.error(
                    f"POSITION UNPROTECTED! Stop-loss order rejected: {reason} | oid={client_oid}"
                )
            else:
                self.log.warning(f"Unknown order rejected | oid={client_oid} | reason={reason}")
```

`client_oid` здесь — `str`, полученный на 1161 как
`str(event.client_order_id)`. `Cache.order` в Nautilus принимает
`ClientOrderId`, а не строку. Никакого повторного выставления стопа нет —
только запись в журнал; маршрут в Telegram помечен `TODO(PR-0.2)` и не
реализован.

### 1.3 Гарантии, что позиция не остаётся без стопа — нет

Пути, ведущие к позиции без защиты:

| # | Путь | Строки |
|---|---|---|
| 1 | Частичное исполнение: стоп на количество первого филла, остаток без стопа | 907, 922, 935 |
| 2 | `self._tracker` is None → условие `if is_entry_fill and self._tracker:` ложно → entry-филл уходит в ветку `else`, стоп не ставится вообще | 909 |
| 3 | Все три попытки `submit_order(stop_order)` бросили исключение → `log.error("CRITICAL: ... UNPROTECTED")` и выход, ретраев больше нет | 1256–1265 |
| 4 | `instrument` отсутствует в кэше на момент филла → `return` без стопа | 1226–1232 |
| 5 | Биржа отклонила стоп после успешной отправки → только `log.error`, повторной отправки нет | 1206–1210 |
| 6 | Падение процесса в окне между `submit_order(entry)` и `OrderFilled`, при этом филл случился во время окна: Binance событие не переигрывает | док-строка `pending_orders_store.py:20-23` |
| 7 | Второй сигнал по тому же символу при открытой позиции: `update_fill` усредняет в существующую позицию, стоп ставится на `fill_qty` добавки — часть позиции покрыта старым стопом на старую цену, часть новым | `portfolio_tracker.py:124-133` |

Ни один из путей 1–5 и 7 не приводит ни к закрытию позиции, ни к
блокировке новых входов, ни к внешнему оповещению. Все они завершаются
записью в журнал процесса.

Путь 6 документирован как покрытый:

```
unprotected position on the exchange; that scenario is covered by the
``PositionReconciler`` which surfaces it as ORPHAN.
```
(`src/execution/pending_orders_store.py:22-23`)

`PositionReconciler` мёртв — установлено в проходе 1 (**A2-001**). То есть
единственный документированный механизм обнаружения незащищённой позиции
не имеет исполнителя.

### 1.4 Идемпотентность

**Дубль сигнала на одном баре.** Защиты нет. В `on_bar` не хранится
`ts_event` последнего обработанного бара, нет множества обработанных
меток. `grep -n "_last_bar\|dedup\|duplicate\|seen"` по файлу даёт только
комментарий на строке 211. Поле `self._pending_stops: dict[str, str]  #
instrument_id → client_order_id` (212), заявленное комментарием «Track
pending SL/TP per instrument to avoid duplicate orders», не читается и не
пишется нигде: `grep -rn "_pending_stops" src/` возвращает ровно одну
строку — само объявление.

`entry_tag = f"AC-{dir_str}-{ts_ms}"` (1104) строится из `time.time()`, а
не из времени бара, и является только тегом — на дедупликацию он не влияет,
`client_order_id` генерирует `order_factory`.

При повторной доставке того же бара будет отправлен второй market-ордер,
трекер усреднит две позиции в одну (`update_fill`, 124–133), а на бирже
образуется удвоенная позиция с двумя `reduce_only` стопами по разным ценам
на разные количества.

**Сигнал в противоположную сторону при открытой позиции.** `update_fill`
не сверяет `direction` с существующей записью:

```python
        if symbol in self._positions:
            pos = self._positions[symbol]
            # Average in
            total_qty = pos.quantity + quantity
```
(`portfolio_tracker.py:124-127`)

`pos.direction` сохраняется от первого филла. SHORT поверх LONG даст в
трекере LONG удвоенного размера, тогда как на бирже неттинг сведёт позицию
почти к нулю. Дальнейшие `unrealized_pnl`, `equity`, `drawdown` — фикция.

**Рестарт между submit и fill.** Покрыт частично: `_init_pending_store`
(548–587) восстанавливает `_pending_sl_params` из `pending_sl_4h.json`,
чтобы следующий `OrderFilled` был опознан как entry. Если филл произошёл
внутри окна падения — см. путь 6 выше.

---

## 2. САЙЗИНГ

### 2.1 `calculate_position_size` целиком

`src/risk/risk_engine.py:233-262`:

```python
        dollar_risk = equity * self._config.risk_per_trade
        stop_distance = signal.atr * self._config.atr_stop_multiplier

        if stop_distance <= 0:
            return 0.0, 0.0, 0.0

        contracts = dollar_risk / stop_distance
        notional = contracts * signal.entry_price
        leverage = notional / equity if equity > 0 else 0.0

        # Cap leverage
        max_notional = equity * self._config.max_leverage
        if notional > max_notional:
            notional = max_notional
            contracts = notional / signal.entry_price if signal.entry_price > 0 else 0.0
            leverage = float(self._config.max_leverage)

        return contracts, notional, leverage
```

### 2.2 Разбор по единицам — размерность верна

| Величина | Единица | Источник |
|---|---|---|
| `equity` | USDT | `PortfolioTracker._get_equity()` |
| `risk_per_trade` | доля (0.01) | `RiskConfig:35` |
| `dollar_risk` | USDT | произведение |
| `signal.atr` | **USDT** | `ml_strategy.py:805`: `atr_dollar = regime_state.atr_pct * current_price` |
| `atr_stop_multiplier` | безразмерный (1.5) | `RiskConfig:37` |
| `stop_distance` | USDT | произведение |
| `contracts` | базовая монета (BTC) | USDT / USDT-за-монету |
| `notional` | USDT | contracts × цена |

`regime_state.atr_pct` — доля: `regime_detector.py:329` `atr_pct =
(current_atr / price) if price > 0 else 0.0`. В `TradeSignal` уходят
**оба**: `atr=atr_dollar` (доллары) и `atr_pct=regime_state.atr_pct` (доля).
Размерности согласованы; ошибки «на порядки» здесь нет.

**Замер на 8 реальных сделках снимка** (`/tmp/ac_snap.db`, `signals_log`):

```
id   dir   SL/atr   TP/atr  atr_pct   eq_impl     lev exp_ret_bps  volflt   conf
 1 short 1.500000 2.250000  0.00929  10000.00  0.7174        92.9    PASS 0.7366
 2 short 1.500000 2.250000  0.01034  10000.00  0.6445       103.4    PASS 0.6526
 3 short 1.500000 2.250000  0.00875  10000.00  0.7618        87.5    PASS 0.6542
 4 short 1.500000 2.250000  0.01048  10000.00  0.6361       104.8    PASS 0.6672
 5 short 1.500000 2.250000  0.01028  10000.00  0.6486       102.8    PASS 0.6714
 6 short 1.500000 2.250000  0.00938  10000.00  0.7105        93.8    PASS 0.6608
 7  long 1.500000 2.250000  0.00961  10000.00  0.6940        96.1    PASS 0.6655
 8  long 1.500000 2.250000  0.01535  10000.00  0.4344       153.5    PASS 0.6557
```

`eq_impl` восстановлен двумя независимыми способами и совпал:
`notional / leverage` и `position_size × atr × 1.5 / 0.01`. Оба дают
ровно `10000.0` во всех восьми строках. Также во всех восьми
`position_size × entry_price == notional` с точностью до 1e-9.

Из этого следует: формула сайзинга воспроизводится **в точности**, а
`equity` за все восемь сделок не сдвинулась ни на цент — при том, что
между ними 4 выигрыша и 4 проигрыша по 1.4–2.3 % каждый. Трекер
эквити не двигал (см. §6.4 и §7.3).

### 2.3 Кэп по плечу недостижим на 4H

Кэп срабатывает при `notional > equity × max_leverage`, то есть при

```
leverage = risk_per_trade / (atr_stop_multiplier × atr_pct) > max_leverage
        ⇔ atr_pct < 0.01 / (1.5 × 10) = 0.000667  (0.0667 % от цены)
```

Измеренные `leverage` на 8 сделках: 0.434 … 0.762, то есть в 13–23 раза
ниже кэпа. Медиана `atr_pct` по `data/features/ml_features/BTCUSDT_4h_features.parquet`
— 0.01288, минимум 5-го перцентиля — 0.00793. Кэп на 4H-баре BTC не
достигается.

Когда он всё же срабатывает, `contracts` **уменьшается**, а значит
фактический риск на сделку становится **меньше** объявленного 1 %:
`contracts_capped × stop_distance < equity × risk_per_trade`. Это
изменение нигде не логируется и не возвращается вызывающему — `RiskDecision`
не несёт поля «фактический риск». `leverage` при этом присваивается
константа `float(self._config.max_leverage)` (260), а не пересчитывается.

### 2.4 Биржевые фильтры не учитываются

`grep -rn "min_notional\|min_quantity\|minNotional\|minQty\|step_size" src/execution/ src/risk/`
даёт только `src/execution/data_catalog.py` — синтетические инструменты
для бэктеста (`price_increment`, `size_increment`), к живому пути отношения
не имеющие.

Единственное округление — `instrument.make_qty(decision.position_size)`
(1099) и `instrument.make_price(decision.stop_loss)` (1235), то есть
приведение к точности инструмента из кэша Nautilus. Явных проверок
`minQty` / `stepSize` / `minNotional` нет; отказ биржи по этим фильтрам
попадёт в `on_order_rejected` уже после отправки. Плечо на самой бирже
(`POST /fapi/v1/leverage`) не выставляется нигде — `max_leverage`
существует только как локальный расчёт.

---

## 3. SL / TP

### 3.1 Обе функции целиком

`src/risk/risk_engine.py:264-300`:

```python
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float,
    ) -> float:
        """
        Stop-loss = entry ± (ATR × multiplier).

        LONG:  stop = entry - (ATR × 1.5)
        SHORT: stop = entry + (ATR × 1.5)
        """
        offset = atr * self._config.atr_stop_multiplier
        if direction == 1:  # LONG
            return entry_price - offset
        return entry_price + offset  # SHORT

    def calculate_take_profit(
        self,
        entry_price: float,
        direction: int,
        stop_loss: float,
        rr_ratio: float = 1.5,
    ) -> float:
        """
        Take-profit achieving the target risk:reward ratio.

        risk   = |entry - stop_loss|
        reward = risk × rr_ratio
        LONG:  TP = entry + reward
        SHORT: TP = entry - reward
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * rr_ratio
        if direction == 1:  # LONG
            return entry_price + reward
        return entry_price - reward  # SHORT
```

`evaluate` вызывает `calculate_take_profit` без `rr_ratio` (184–186), то
есть всегда с дефолтом 1.5. Отсюда TP/ATR = 1.5 × 1.5 = **2.25**.

### 3.2 Откуда 1.5 и 2.25 — обоснования нет

* `atr_stop_multiplier: float = 1.5    # stop = 1.5 × ATR` — `RiskConfig:37`,
  комментарий повторяет значение, не обосновывает его.
* `rr_ratio: float = 1.5` — дефолт параметра `calculate_take_profit:286`.
* `2.25` нигде не записано как константа; это произведение двух дефолтов.
* `MLStrategyConfig.rr_ratio: float = 1.5` (`ml_strategy.py:119`) —
  **мёртвое поле**. `grep -rn "rr_ratio" src/` показывает, что оно нигде
  не читается: единственные потребители имени `rr_ratio` в риске — это
  локальная переменная в `evaluate:201` и параметр функции `:286`. Кнопка
  в конфиге стратегии не соединена ни с чем.
* Ни `1.5`, ни `2.25` не выводятся из горизонта метки, из барьеров
  triple-barrier и из измеренной статистики. В проходе 2 (**A2-015**)
  установлено, что геометрия сделки и геометрия метки описывают разные
  события; настоящий проход подтверждает это со стороны исполнения:
  метка ставится по одной паре барьеров, а на биржу уходит один стоп на
  1.5·ATR и ничего больше.

### 3.3 Округление к тик-сайзу

Для стопа есть: `sl_price = instrument.make_price(decision.stop_loss)`
(1235). Для входа не требуется — ордер market. Для TP вопрос не стоит,
поскольку TP-ордера нет.

Но в журнал `signals_log` и в Telegram уходит **неокруглённое**
`decision.stop_loss` (`_emit_signal:1022`) — то есть отображаемая
пользователю цена стопа отличается от той, что реально ушла на биржу.
В снимке это видно напрямую: `stop_loss = 82349.22576545` при
`price_increment` BTCUSDT-PERP = 0.1.

### 3.4 Проверка на 8 реальных сделках — формула совпадает

Из таблицы §2.2: `|entry − stop_loss| / atr = 1.500000` и
`|take_profit − entry| / atr = 2.250000` во всех восьми строках, до
шестого знака. Записанные значения воспроизводятся сегодняшним кодом
без расхождений.

Однако `close_price` в этих же строках равен **ровно** `stop_loss` или
`take_profit` (сравнение по всем 8 строкам: 4 закрытия по TP, 4 по SL,
все точные). Реальные филлы такого не дают. Источник этих закрытий —
`reconciler_signals`, см. §7.1.

---

## 4. ФИЛЬТРЫ RISK ENGINE

### 4.1 Девять фильтров по порядку

Цепочка задана в `evaluate` (`risk_engine.py:145-155`):

```python
        filters = [
            (self._check_max_drawdown, (portfolio_state,)),
            (self._check_weekly_loss, (portfolio_state,)),
            (self._check_daily_loss, (portfolio_state,)),
            (self._check_consecutive_losses, (portfolio_state, signal.timestamp)),
            (self._check_max_positions, (portfolio_state,)),
            (self._check_confidence, (signal,)),
            (self._check_funding_rate, (signal,)),
            (self._check_volatility, (signal,)),
            (self._check_expected_return, (signal, portfolio_state.equity)),
        ]
```

| # | Фильтр | Что проверяет | На каких данных | Отказ означает |
|---|---|---|---|---|
| 1 | `_check_max_drawdown` | `current_drawdown_pct > 0.15` | `PortfolioTracker.get_drawdown()` = `(peak − equity)/peak` | kill switch, ручной сброс |
| 2 | `_check_weekly_loss` | `weekly_pnl_pct <= −0.08` | `_weekly_realized_pnl + Σ unrealized`, делитель `_initial_equity` | стоп на неделю |
| 3 | `_check_daily_loss` | `daily_pnl_pct <= −0.03` | `_daily_realized_pnl + Σ unrealized`, делитель `_day_start_equity` | стоп на день |
| 4 | `_check_consecutive_losses` | `consecutive_losses >= 5` и с последнего убытка прошло < 4 ч | счётчик трекера + `signal.timestamp` | пауза 4 ч |
| 5 | `_check_max_positions` | `open_positions >= 3` | `len(self._positions)` **одного** трекера | лимит позиций |
| 6 | `_check_confidence` | `confidence < threshold` | выход LightGBM | слабый сигнал |
| 7 | `_check_funding_rate` | `funding is None` **или** `abs(funding) > 0.001` | фича `funding_rate` из вектора | fail-safe блок (H4) |
| 8 | `_check_volatility` | `atr_pct > 2.0 × 0.01` | `regime_state.atr_pct` | «всплеск волатильности» |
| 9 | `_check_expected_return` | `atr_pct × 10000 < max(15, 3 × cost_bps)` | ATR% и `CostModel` | издержки не окупаются |

Замечание к #5: `_positions` ключуется символом, а каждый экземпляр
стратегии обслуживает ровно один инструмент (`live_trader.py:216-251`) и
создаёт **собственный** `PortfolioTracker` (`ml_strategy.py:295-297`).
Значит `open_positions ∈ {0, 1}` и порог 3 недостижим по построению.
Фильтр #5 никогда не отказывает.

### 4.2 `_check_volatility` — отдельный разбор единиц

Код целиком (`risk_engine.py:387-401`):

```python
    def _check_volatility(self, signal: TradeSignal) -> tuple[bool, str]:
        """Block if ATR > vol_spike_multiplier × average (circuit breaker)."""
        # atr_pct is ATR/price; spike detection uses the raw ratio.
        # Average ATR is approximated as atr_pct / vol_spike_multiplier
        # threshold, i.e. the user passes actual ATR and we compare to the
        # multiplier against itself.  A true vol-spike means atr_pct is
        # unexpectedly large; we approximate "average" as 1% and flag when
        # atr_pct ≥ 2× that, but the actual comparison is done via the
        # circuit breaker.  Here we simply cap at an absolute level.
        if signal.atr_pct > self._config.vol_spike_multiplier * 0.01:
```

**Кто пишет `atr_pct`.** `regime_detector.py:329`:
`atr_pct = (current_atr / price) if price > 0 else 0.0`, затем
`atr_pct=round(atr_pct, 6)` (353). Единица — **доля**.

**Кто читает.**
* `ml_strategy.py:805` — умножает на цену, получая ATR в долларах. Значит
  читатель трактует поле как долю. Согласовано.
* `ml_strategy.py:818` — передаёт как есть в `TradeSignal.atr_pct`.
* `risk_engine.py:396` — сравнивает с `2.0 × 0.01 = 0.02`, то есть с
  долей. Согласовано.
* `risk_engine.py:354` — умножает на 10 000, получая bps. Согласовано.

**Единицы в коде совпадают.** Ошибки «доли против процентов» нет.

**Откуда 0.07–0.69 в логах прода.** Это не `atr_pct`. Строка
`ml_strategy.py:761`:

```python
                f"atr_pct={regime_state.atr_percentile:.2f} | "
```

Под именем `atr_pct` печатается **`atr_percentile`** — ранг ATR в
исторической выборке, по определению ∈ [0, 1] (`regime_detector.py:203`:
`atr_percentile: float     # ATR in historical context (0–1)`; клип
на 448: `np.clip(pct, 0.0, 1.0)`). Диапазон 0.07–0.69 для перцентиля
совершенно нормален. То есть расхождение существует только в **строке
лога**, а не в вычислениях. Настоящий `atr_pct` в проде — порядка 0.008–0.015,
что видно из 8 записей снимка (0.00875 … 0.01535).

**Чем плох хардкод 1 %.** Комментарий сам признаёт, что «average»
аппроксимирован константой, а «actual comparison is done via the circuit
breaker» — но в circuit breaker ветка всплеска получает нули (§5.1), то
есть не работает. В результате фильтр #8 — это фиксированный абсолютный
потолок ATR% = 2 %, одинаковый для BTC, ETH и SOL.

**Замер по всей истории фич** (`ml_features/*_4h_features.parquet`,
колонка `atr_pct`, n = 5038 на символ):

```
BTCUSDT n=5038 median=0.01288 p05=0.00793 p95=0.02333 | below=5.0% inside=84.6% above=10.4%
ETHUSDT n=5038 median=0.01919 p05=0.01203 p95=0.03422 | below=0.5% inside=55.0% above=44.5%
SOLUSDT n=5038 median=0.02327 p05=0.01359 p95=0.04072 | below=0.1% inside=28.5% above=71.4%
```

где `above` — доля баров, отвергаемых фильтром #8 как «всплеск
волатильности». Для SOL это **71.4 % всех баров истории**, для ETH —
44.5 %. Константа 1 % откалибрована (неявно) под BTC и делает
два из трёх заявленных инструментов почти неторгуемыми — не по решению
модели, а по хардкоду.

### 4.3 `_check_expected_return` — «ожидаемая доходность» не связана с моделью

```python
        expected_return_bps = signal.atr_pct * 10_000  # ATR% → bps
        threshold = max(
            self._config.min_expected_return_bps,
            rt_cost.total_cost_bps * 3,  # rule of 3×
        )
```
(`risk_engine.py:354-358`)

`signal.confidence` в этой функции не участвует. Ожидаемая доходность
приравнена к **одному ATR**, безусловно и для любой уверенности модели.
При этом сделка геометрически устроена иначе: стоп на 1.5·ATR, TP (если бы
он выставлялся) на 2.25·ATR. Ни 1.0, ни 1.5, ни 2.25 не выводится из
`confidence`, из вероятностей LightGBM или из статистики исходов.

`confidence` вообще влияет ровно на одно: проходит/не проходит фильтр #6.
Ни на размер позиции, ни на геометрию, ни на порог издержек он не влияет.
Это прямое расхождение с практикой — при бинарной классификации размер
обычно масштабируется краем (`2p − 1`, Kelly-подобно, либо ступенчато).

**Издержки, замер на 8 реальных сделках** (`CostModel`, реальные
`notional` и `funding_rate` из снимка, дефолты `RiskConfig`:
`daily_volume=1e9`, `volatility=0.60`, `hours_held=8.0`):

```
id  notional cost_bps  thresh  exp_bps verdict  margin
 1   7173.86    25.07   75.21    92.93    PASS   17.72
 2   6444.96    24.23   72.70   103.44    PASS   30.74
 3   7618.18    25.56   76.68    87.51    PASS   10.83
 4   6360.72    24.13   72.40   104.81    PASS   32.41
 5   6485.72    24.28   72.84   102.79    PASS   29.95
 6   7105.05    24.99   74.98    93.83    PASS   18.85
 7   6940.11    24.72   74.17    96.06    PASS   21.89
 8   4343.95    22.13   66.40   153.47    PASS   87.07
```

Порог `min_expected_return_bps = 15` не связывает никогда: `3 × cost_bps`
всегда больше (66–77 bps). Запас у сделки №3 — 10.8 bps, то есть 12 % от
порога.

**Совместное действие фильтров #8 и #9** задаёт жёсткий коридор ATR%.
Численно (BTC, цена 80 000, equity 10 000, шаг 5e-5):

```
admissible atr_pct band at price=80000, equity=10000: 0.00795 .. 0.01995
  in %: 0.795 % .. 1.995 %
```

Ниже 0.795 % сделка отвергается как «издержки не окупаются», выше 1.995 % —
как «всплеск волатильности». Обе границы — следствие констант
(`0.01` в `_check_volatility`, `default_daily_volume = 1e9`,
`default_volatility = 0.60`, множитель 3), ни одна не измерена.

Отдельно: `default_hours_held = 8.0` даёт ровно один фандинговый платёж
(`calculate_funding_cost:91`: `num_payments = hours_held / 8.0`). Измеренные
удержания в снимке — 239 … 5039 минут, то есть от 4 до **84 часов**
(10.5 фандинговых платежей). Издержки удержания занижены до порядка.

### 4.4 Regime-multiplier (A19) — не подключён, находка жива

Два независимых множителя существуют и оба не вызываются из прода:

1. `RegimeState.position_size_multiplier()` (`regime_detector.py:211-219`,
   HIGH_VOL → 0.5, RANGE → 0.7, TREND → 1.0, UNKNOWN → 0.0).
2. `CircuitBreaker.get_position_size_multiplier()`
   (`circuit_breaker.py:215-245`, 1.0 / 0.5 / 0.0).

`grep -rn "get_position_size_multiplier\|position_size_multiplier" src/ scripts/ tests/`:

```
src/features/regime_detector.py:211:    def position_size_multiplier(self) -> float:
scripts/analyze_regimes.py:158:    print(f"  Pos mult     : {last_state.position_size_multiplier()}")
tests/test_risk_engine.py:274:        mult = cb.get_position_size_multiplier(state)
tests/test_risk_engine.py:290:        mult = cb.get_position_size_multiplier(state)
tests/test_regime_detector.py:289:    def test_position_size_multiplier(...)
```

Единственный не-тестовый вызов — печать в диагностическом скрипте.
`decision.position_size` нигде не умножается ни на что: в
`_open_position` он идёт прямо в `instrument.make_qty(...)` (1099).
**A19 не закрыта.**

---

## 5. CIRCUIT BREAKER И ПРОСАДКА

### 5.1 Пороги

```python
    DAILY_LOSS_SOFT: float = -0.02        # -2%: reduce positions 50%
    DAILY_LOSS_HARD: float = -0.03        # -3%: stop trading today
    WEEKLY_LOSS: float = -0.08            # -8%: stop for the week
    MAX_DRAWDOWN_WARNING: float = -0.10   # -10%: alert
    MAX_DRAWDOWN_KILL: float = -0.15      # -15%: full stop
    VOL_SPIKE: float = 2.0               # ATR > 2× average
    FUNDING_EXTREME: float = 0.001        # |funding| > 0.1%
    CONSECUTIVE_LOSSES: int = 5           # 5 consecutive losses
```
(`circuit_breaker.py:59-66`)

Происхождение — «from master document» (док-строка модуля, строки 7–16).
Ни один порог не выведен из измеренного распределения PnL, из размера
выборки, из ожидаемой частоты ложных срабатываний. Дублируются в
`RiskConfig:40-53` независимыми литералами: `daily_loss_limit = -0.03`,
`weekly_loss_limit = -0.08`, `max_drawdown_kill = -0.15`,
`consecutive_losses_limit = 5`. Два источника одних и тех же чисел, без
общей константы.

Границы срабатывания при этом **не совпадают**:
`RiskEngine._check_max_drawdown` использует строгое `>` (`443`),
`CircuitBreaker.check` — нестрогое `<=` (`112`). При просадке ровно 15.00 %
breaker сработает, risk engine — нет.

Ветка всплеска волатильности получает нули от единственного вызывающего:

```python
                    breaker_state = self._breaker.check(
                        portfolio_state=self._tracker.get_state(),
                        current_atr=0.0,
                        avg_atr=0.0,
                        current_funding=funding,
                    )
```
(`ml_strategy.py:724-729`)

`if avg_atr > 0 and ...` (187) — условие ложно всегда. Комментарий в
`_check_volatility` ссылается на эту ветку как на «actual comparison».
Её нет.

`self._daily_triggered` **записывается** (153), персистится (158),
восстанавливается (299) и логируется как «restored as TRIGGERED» (302–305),
но `check()` его **никогда не читает**. Липкий дневной стоп не существует
как поведение; при рестарте breaker пересчитывает всё заново из
`portfolio_state`. `reset_daily()` (247–252) не имеет ни одного вызова вне
тестов.

Сверх того, `PortfolioTracker._persist` пишет полный снапшот через
`RiskStateStore.save`, который перезаписывает файл целиком
(`risk_state_store.py:116-141`, `json.dump(state, f)`), а `_snapshot()`
(`portfolio_tracker.py:409-429`) не содержит ключей breaker'а. То есть
любой филл, закрытие или откат периода стирает
`breaker_daily_triggered` из общего файла. `CircuitBreaker._persist`
делает load-merge (262–267) — слияние одностороннее.

### 5.2 Что breaker видит в проде сейчас

`_record_equity` (`ml_strategy.py:2196-2217`):

```python
        # --- Simulated mode: the exchange is not consulted at all. ---
        if self._config.dry_run:
            ...
            return
```

Возврат происходит **до** `seed_from_authoritative_equity` (2247) и
`sync_equity` (2260). Значит в `--dry-run`:

* база просадки не засевается с биржи — ветка S0-2 недостижима
  (подтверждает **A2-012** со стороны исполнения);
* `_cash` остаётся `initial_equity = 10 000`;
* `_positions` пуст всегда, потому что `_open_position` не вызывается
  (`ml_strategy.py:842`), значит `update_fill` не вызывается, значит
  `unrealized_pnl` = 0;
* следовательно `equity ≡ 10000`, `peak_equity ≡ 10000`,
  `drawdown ≡ 0`, `daily_pnl_pct ≡ 0`, `weekly_pnl_pct ≡ 0`,
  `consecutive_losses ≡ 0`.

Все четыре срабатывающих ветви `check()` питаются исключительно этими
полями. **Circuit breaker в текущем проде — тождественный no-op.** Он не
проверялся ни разу за всё время работы в dry_run, потому что его входы
константны по построению.

Развёрнутая конфигурация подтверждает это: `deploy/atomicortex-bot.service:26-32`

```
ExecStart=/home/hashiflame/AtomiCortex/.venv/bin/python \
    scripts/run_live.py \
    --mode paper \
    --dry-run \
    --symbols BTCUSDT-PERP \
    --capital 10000 \
    --log-level INFO
```

### 5.3 `_roll_periods` и `day_start_equity` — тупик по дневному лимиту

```python
    def _roll_periods(self, now: datetime) -> None:
        """Reset daily/weekly accumulators if boundaries crossed."""
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
```
(`portfolio_tracker.py:381-403`)

Часовой пояс: `now` приходит из `update_fill` / `close_position`, которые
получают его от `ml_strategy` как
`datetime.fromtimestamp(event.ts_event / 1e9, tz=timezone.utc)` (896, 957).
`__init__` берёт `datetime.now(timezone.utc)` (73). `RiskStateStore.load`
считает границы по стенным часам (`_today_start_utc`, 50–52). Всё в UTC,
рассогласования зон нет. Смешаны, однако, два источника времени: во время
работы дни катятся по времени события, при рестарте — по стенным часам.

**Главное:** `_roll_periods` вызывается ровно из двух мест —
`update_fill:121` и `close_position:272`. `grep -rn "_roll_periods" src/`
подтверждает: других вызовов нет. `get_state()` его не вызывает.

Отсюда — тупик. Замер:

```
day1 daily_pnl_pct = -0.045
3 days later, get_state().daily_pnl_pct = -0.045
day_start held at 2026-08-22T00:00:00+00:00
_check_daily_loss -> (False, 'Daily loss -4.50% <= limit -3.00%')
```

(скрипт: `PortfolioTracker(10000)` → `update_fill(0.09 BTC @ 80000)` →
`close_position(@75000)` в день 1, затем `get_state()` без единого филла
спустя трое суток, результат подан в `RiskEngine._check_daily_loss`.)

Механика: дневной лимит пробит → `evaluate` отказывает → ордер не
отправляется → филла нет → `_roll_periods` не вызывается → дневной
счётчик не обнуляется → лимит пробит навсегда. Единственный выход —
перезапуск процесса, при котором `RiskStateStore.load` применит дневной
сброс (`risk_state_store.py:100-106`).

То же самое, ещё жёстче, для последовательных убытков.
`CircuitBreaker.check` (171–184) блокирует при `consecutive_losses >= 5`
**без окна паузы**, и вызывается в `on_bar` **раньше** risk engine
(716–735, `return` на 735). Значит четырёхчасовая пауза
`_check_consecutive_losses` (403–424) недостижима: до неё управление не
доходит. `_consecutive_losses` обнуляется только в `close_position` при
`realized_pnl >= 0` (296–297), то есть требует прибыльного закрытия,
которое требует новой сделки, которая заблокирована. Пять убытков подряд
= бессрочная остановка без автоматического восстановления.

---

## 6. УЧЁТ ПОЗИЦИЙ

### 6.1 Инвариант `equity = cash + unrealized` — держится

Замер последовательности fill → mark → mark → close:

```
start      equity 10000.0 cash 10000.0
after fill equity 9996.76 cash 9996.76 dd 0.00032399999999997817
mark 81k   equity 10086.76 peak 10086.76 daily 0.009
mark 79k   equity 9906.76 peak 10086.76 dd 0.01784517525944902
closed     realized -96.44 equity 9903.56 cash 9903.56
fees paid expected 6.44 | gross pnl -90.0
10000 + gross - fees = 9903.56
```

Итог сходится точно: `10000 − 90 − 6.44 = 9903.56`. Внутри
`PortfolioTracker` двойного счёта комиссий нет: открывающая списывается в
`update_fill:122` (`self._cash -= fee`), закрывающая — в
`close_position:288` (`self._cash += gross_pnl - fee`), а `pos.total_fees`
участвует только в возвращаемом `realized_pnl`, не в `_cash`.

Побочно видно: `peak_equity` поднимается нереализованной переоценкой
(10086.76 после mark 81k), и на возврате к 79k фиксируется просадка 1.78 %
по позиции, которая ещё открыта. Это заявленное поведение
(`update_price` док-строка, 156–161), но означает, что kill switch −15 %
считается от внутрибарного максимума нереализованного PnL.

Также: `daily 0.009` после mark 81k. Комиссия −3.24 списана из `_cash`, но
в `_daily_realized_pnl` не попала. `get_daily_pnl` (331–343) считает
`_daily_realized_pnl + Σ unrealized` — то есть дневной PnL систематически
занижает убыток на величину уплаченных комиссий.

Делители разные: дневной — `_day_start_equity` (340), недельный —
`_initial_equity` (349–351). Один и тот же лимит «процент от капитала»
измеряется от двух разных баз.

### 6.2 Комиссии — закрывающая теряется на уровне стратегии

Внутри трекера всё корректно (§6.1). Но связка в `ml_strategy` теряет
комиссию выхода:

* `on_order_filled` списывает комиссию только в ветке entry
  (`fee=commission`, 917). Ветка `else` для exit-филлов (937–941) не
  делает ничего, кроме `log.debug`.
* `on_position_closed` вызывает `close_position(..., fee=0.0, ...)` с
  комментарием `# fees already accounted in on_order_filled` (970).

Утверждение комментария неверно: `on_order_filled` учитывает комиссию
**только entry**. Комиссия закрывающего ордера (по taker-ставке
~4.5 bps от нотионала, то есть ~3.2 USDT на нотионал 7000) не списывается
нигде. При round-trip издержках ~25 bps это ~18 % от полной стоимости
сделки, теряемых из учёта в одну сторону — в пользу завышения PnL.

### 6.3 Funding в P&L удерживаемой позиции — не учитывается

В `PortfolioTracker` слова `funding` нет вовсе. `_Position` (27–38) не
имеет поля накопленного фандинга; `update_price` (155–173) считает
`unrealized_pnl = direction × (price − entry) × qty` без него.

Единственный путь, которым фандинг может попасть в эквити, —
`sync_equity` (175–202), которая просто присваивает `_cash = target −
unrealised` от биржевого баланса. В `--dry-run` она недостижима (§5.2).
В `--mode live` она сработает, но фандинг тогда войдёт в эквити «оптом»,
без атрибуции к позиции: `_daily_realized_pnl` и `_weekly_realized_pnl`
его не увидят (док-строка 184–186 объявляет это намеренным). То есть
дневной и недельный лимиты потерь фандинг не учитывают вообще.

`CostModel.calculate_funding_cost` существует и используется **только**
для предторговой оценки в `_check_expected_return`, с константой
`hours_held = 8.0` (§4.3). При измеренных удержаниях до 84 часов и
горизонте метки 6 баров (24 ч) это занижение в 3–10 раз.

### 6.4 Расхождение с биржей

Обнаружителя нет. `sync_equity` не сравнивает — она присваивает:

```python
        unrealised = sum(p.unrealized_pnl for p in self._positions.values())
        self._cash = target - unrealised
```
(`portfolio_tracker.py:195-196`)

Расхождение любого размера молча поглощается корректировкой `_cash`. Ни
порога, ни счётчика, ни лога при большом сдвиге.

Хуже: `_read_nautilus_equity` (2163–2173) складывает **общий** баланс
счёта с нереализованным PnL **только своего** инструмента:

```python
            balance = account.balance_total(USDT)
            upnl = self.portfolio.unrealized_pnl(self._instrument_id)
            return balance.as_double() + (
                upnl.as_double() if upnl is not None else 0.0
            )
```

При нескольких символах каждый экземпляр стратегии подаёт своему трекеру
полный баланс счёта плюс свой собственный uPnL, не видя uPnL остальных.
Три трекера, три независимых представления об одном счёте.

Позиций трекер вообще не сверяет: `RiskStateStore` док-строка прямо
говорит `no positions; those are reconciled separately`
(`risk_state_store.py:12-13`), а «separately» — это `PositionReconciler`,
мёртвый (**A2-001**).

---

## 7. РЕКОНСИЛЯЦИЯ

### 7.1 `reconciler_signals` — что сверяет

Не сверяет ничего с биржей. Читает открытые строки журнала и **доигрывает
цену по историческим свечам**:

```python
            rows = conn.execute(
                "SELECT * FROM signals_log "
                "WHERE result = 'open' AND closed_at IS NULL "
                "ORDER BY id ASC"
            ).fetchall()
```
(`reconciler_signals.py:263-267`)

```python
            if is_long:
                worst = min(worst, low)
                best = max(best, high)
                sl_hit = low <= sl
                tp_hit = high >= tp
            else:
                worst = max(worst, high)
                best = min(best, low)
                sl_hit = high >= sl
                tp_hit = low <= tp

            if sl_hit or tp_hit:
                # Tie on same bar → SL (loss), conservative.
                if sl_hit:
                    close_px, result = sl, "loss"
                else:
                    close_px, result = tp, "win"
```
(`reconciler_signals.py:220-236`)

Это **симуляция**: закрытие книжится ровно по `sl` или `tp`, без
проскальзывания, без комиссий, по `high`/`low` бара.

Отсюда происхождение восьми «сделок» снимка. Проверка: сделка №1, short,
entry 81217.1, `close_price = take_profit = 79518.91135182502`,
`pnl_pct = 2.090924999999978`. Формула: `2.25 × atr_pct = 2.25 × 0.00929
= 2.0909 %`. Совпадение до 4 знаков, потому что цена закрытия — это ровно
TP, а не филл.

**И этот TP на биржу никогда не отправлялся** (§1.1). Реконсилятор
закрывает сделки по ордеру, которого нет. Четыре из восьми «побед» —
исходы стратегии, не реализованной в коде исполнения.

Периодичность объявлена в `deploy/atomicortex-reconciler.timer`:
`OnBootSec=2min`, `OnUnitActiveSec=15min`, `Persistent=true`.

Но `deploy/units.enabled` перечисляет реально развёрнутое:

```
atomicortex-bot.service
atomicortex-telegram.service
atomicortex-signal-check.service
atomicortex-signal-check.timer
```

с прямой оговоркой в шапке файла: «the 15m bot, the watchdogs, the REST
API and **the reconciler** are kept for reference and for the unit linters,
and are not part of the current paper --dry-run deployment».

**По декларации самого репозитория реконсиляция в проде не развёрнута.**
Ни в каком виде. Фактическое состояние VM в этом проходе не проверялось
(протокол запрещает выход на VM), поэтому источник — `deploy/units.enabled`
и его шапка, а не наблюдение живой машины.

### 7.2 Позиция, открытая не этим процессом

Не обнаруживается. `reconciler_signals` не обращается к бирже за
позициями — его источники цен `DataStorePriceSource` (локальный parquet)
и REST klines (`reconciler_signals.py:69-154`). Ни `GET /fapi/v2/positionRisk`,
ни `account.positions()` в модуле нет.

`PortfolioTracker` знает только о том, что ему сообщили через
`update_fill`. `sync_equity` поглотит чужой PnL как дрейф `_cash` (§6.4),
не подняв флага.

Единственное место, где позиция биржи могла бы сравниваться с локальной, —
`src/execution/reconciler.py`, мёртвый.

### 7.3 Что потеряно вместе с мёртвым `reconciler.py`

`reconciler.py` (275 строк) — единственный носитель понятия ORPHAN, то
есть позиции на бирже без соответствующего локального состояния. С ним
потеряны:

1. обнаружение позиции, открытой не этим процессом (§7.2);
2. обнаружение **незащищённой** позиции после падения в окне
   submit → fill — сценарий, который `pending_orders_store` док-строкой
   (20–23) прямо переадресует `PositionReconciler`;
3. сверка количества/направления локальной позиции с биржевой;
4. обнаружение «висящих» стоп-ордеров после закрытия позиции.

Живой `reconciler_signals` не покрывает ни один из этих четырёх пунктов —
он работает исключительно с таблицей `signals_log` и историческими
свечами. Замена не эквивалентна; имя вводит в заблуждение.

---

## 8. ЧТО БУДЕТ ПРИ СНЯТИИ `--dry-run`

### 8.0 Как именно снимается

`--mode paper` без `--dry-run` **отвергается** (`scripts/run_live.py:136-155`,
выход 78, `RestartPreventExitStatus=78`). Единственный путь к реальным
ордерам — `--mode live` с TTY и вводом `YES`
(`scripts/run_live.py:159-165`). Комментарий в юните
(`deploy/atomicortex-bot.service:20-25`) фиксирует, что `paper` уже
резолвит **mainnet**-ключи и mainnet-эндпоинты: `--dry-run` — это
единственное, что отделяет текущий процесс от боевых ордеров.

### 8.1 Ветки, которые оживут

| Ветка | Где | Исполнялась ли | Тесты |
|---|---|---|---|
| `_open_position` целиком | `ml_strategy.py:1079-1155` | нет с 14.08 | только на моках |
| `submit_order(entry)` | `1136` | нет | мок |
| `_emit_signal` | `992-1054` | **нет вообще**: вызывается только из `_open_position` (**A2-013**) | мок |
| `on_order_filled` (ветка entry) | `909-936` | нет | мок |
| `_submit_stop_loss_with_retry` | `1214-1265` | нет | мок |
| `on_order_rejected` / `on_order_denied` | `1157-1173` | нет | мок |
| `_handle_order_reject`, ветка SL | `1199-1212` | нет | не покрыта (см. §1.2) |
| `on_position_opened` / `on_position_closed` | `943-986` | нет | мок |
| `PortfolioTracker.update_fill` / `close_position` | `111-153`, `262-315` | нет в проде | юнит-тесты |
| `_roll_periods` | `381-403` | нет в проде (нет филлов) | юнит |
| `seed_from_authoritative_equity` (S0-2) | `204-260` | **нет** — недостижима в dry_run (**A2-012**) | юнит |
| `sync_equity` | `175-202` | нет | юнит |
| `CircuitBreaker.check` со значащими входами | `86-213` | нет — входы константны (§5.2) | юнит |
| Все девять фильтров `RiskEngine` со значащей `PortfolioState` | `137-176` | нет | юнит |
| `on_stop` → `cancel_all_orders` + `close_all_positions` | `495-497` | нет | мок |
| `PendingOrdersStore.put` / `pop` | `94-124` | нет | юнит |

То есть **весь** путь исполнения — от отправки ордера до закрытия позиции
и обновления риск-состояния — ни разу не работал против биржи. Всё, что
существует, — юнит- и мок-тесты.

### 8.2 Ветки, которые НЕ оживут

* Take-profit — его нет вообще, ни в каком режиме (§1.1).
* `CircuitBreaker.get_position_size_multiplier` и
  `RegimeState.position_size_multiplier` — вызывающих нет (§4.4).
* `CircuitBreaker.reset_daily` — вызывающих нет.
* `CircuitBreaker._daily_triggered` — не читается в `check()`.
* Ветка всплеска волатильности в `check()` — получает нули (§5.1).
* `_check_max_positions` — недостижим (§4.1).
* `MLStrategyConfig.rr_ratio` — не читается.
* `_pending_stops` — не читается.
* `reconciler_signals` — не развёрнут (§7.1).
* `PositionReconciler` — мёртв (**A2-001**).
* Меж-символьные эффекты — развёрнут один символ (`--symbols BTCUSDT-PERP`).

---

## 9. РЕЕСТР НАХОДОК ПРОХОДА 5

### A2-037 Take-profit вычисляется, пишется в журнал и не отправляется на биржу
**Севирити:** CRITICAL
**Тип:** логика / архитектура / семантика
**Где:** `src/execution/strategies/ml_strategy.py:1023`, `848`; отсутствие ордера — по всему файлу
**Что в коде:**
```
848:                    f"SL=${decision.stop_loss:.2f} TP=${decision.take_profit:.2f}"
1023:                take_profit=decision.take_profit,
```
**В чём дефект:** `decision.take_profit` употребляется ровно дважды — в печати сухого прогона и в записи в `signals_log`. Ордера на фиксацию прибыли не существует: в файле нет ни одного `order_factory.limit(...)`, ни `stop_limit`, ни OCO. Единственный биржевой выходной ордер — `stop_market` на 1.5·ATR.
**Как проявляется:** позиция может выйти только по стопу или при остановке процесса. Заявленное R:R 1:1.5 недостижимо: убыточная сторона биржевая, прибыльная — нет. Журнал, Telegram, `daily_stats` и `performance_cache` описывают стратегию, отличную от исполняемой. Из 8 записей снимка 4 закрыты «по TP» — это симуляция (§7.1), а не исполнение.
**Кто ещё это читает:** `SignalBridge.log_signal` → `signals_log.take_profit` → `reconciler_signals._evaluate` (закрывает по TP) → `_update_daily_stats` → `performance_cache` → `src/api/main.py`, `src/telegram_bot/database.py`. Контракт «TP — это ордер» не держится ни на одном звене.
**Как установлено:** чтением + замером (`grep -n "take_profit\|limit(" ml_strategy.py`; сверка `close_price` с `take_profit` в снимке — совпадение до последнего знака в 4 строках из 8).
**Уверенность:** доказано

### A2-038 Дневной лимит потерь становится бессрочным: счётчики катятся только при филле
**Севирити:** CRITICAL
**Тип:** логика
**Где:** `src/risk/portfolio_tracker.py:121`, `272`, `381-403`
**Что в коде:**
```
121:        self._roll_periods(timestamp)      # update_fill
272:        self._roll_periods(timestamp)      # close_position
```
**В чём дефект:** `_roll_periods` — единственный сброс дневных/недельных аккумуляторов, и вызывается он только из `update_fill` и `close_position`. `get_state()` его не вызывает. Пробитый дневной лимит блокирует вход → филла нет → сброса нет → лимит пробит навсегда.
**Как проявляется:** замер:
```
day1 daily_pnl_pct = -0.045
3 days later, get_state().daily_pnl_pct = -0.045
_check_daily_loss -> (False, 'Daily loss -4.50% <= limit -3.00%')
```
Бот перестаёт торговать до перезапуска процесса. Восстановление только через `RiskStateStore.load` при рестарте.
**Кто ещё это читает:** `RiskEngine._check_daily_loss` (`risk_engine.py:315-322`), `CircuitBreaker.check` (`circuit_breaker.py:152-168`), `get_position_size_multiplier` (мёртв), `SignalBridge.update_metrics` (`ml_strategy.py:866-871`) — все получают одно и то же застрявшее значение.
**Как установлено:** замером (скрипт на `PortfolioTracker` + `RiskEngine`, вывод выше).
**Уверенность:** доказано

### A2-039 Пять убытков подряд — бессрочная остановка; заявленная пауза 4 ч недостижима
**Севирити:** CRITICAL
**Тип:** логика
**Где:** `src/risk/circuit_breaker.py:171-184`; `src/execution/strategies/ml_strategy.py:716-735`; `src/risk/risk_engine.py:403-424`
**Что в коде:**
```
171:        if portfolio_state.consecutive_losses >= self.CONSECUTIVE_LOSSES:
...
176:            return CircuitBreakerState(
177:                is_triggered=True,
```
**В чём дефект:** у breaker'а нет окна паузы. В `on_bar` он вызывается раньше risk engine и делает `return` при срабатывании (735), поэтому `_check_consecutive_losses` с его четырёхчасовой паузой (`consecutive_losses_pause_hours: int = 4`) до исполнения не доходит. Счётчик обнуляется только в `close_position` при `realized_pnl >= 0` (`portfolio_tracker.py:296-297`) — то есть требует прибыльной сделки, которая заблокирована.
**Как проявляется:** после пятого убытка бот замолкает навсегда, без автоматического восстановления, без оповещения. `RiskConfig.consecutive_losses_pause_hours` — мёртвая настройка.
**Кто ещё это читает:** `RiskConfig:52-53`, `PortfolioTracker._consecutive_losses`, `RiskStateStore` (персистит счётчик, то есть переживает рестарт).
**Как установлено:** чтением (порядок вызовов в `on_bar`, отсутствие временного условия в breaker).
**Уверенность:** доказано

### A2-040 Частичное исполнение оставляет часть позиции без стопа и вне учёта
**Севирити:** HIGH
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:907`, `922`, `935`, `937-941`; `TimeInForce.IOC` на `1111`
**Что в коде:**
```
907:        is_entry_fill = client_oid in self._pending_sl_params
922:            sl_params = self._pending_sl_params.pop(client_oid)
935:                fill_qty=fill_qty,
937:        else:
938:            # Exit fill (SL or manual close) → fees tracked via close_position
```
**В чём дефект:** признак «это вход» — наличие ключа в словаре, а первый же филл ключ удаляет. Nautilus порождает отдельное `OrderFilled` на каждый частичный филл одного `client_order_id`. Второй и последующие частичные филлы классифицируются как выходные: не попадают в `update_fill`, не получают стопа. Стоп ставится на количество первого филла.
**Как проявляется:** позиция на бирже больше, чем в трекере; часть её без стопа. Ордер отправлен с `IOC`, где частичное исполнение — штатный исход.
**Кто ещё это читает:** `PortfolioTracker._positions` → `get_state()` → все девять фильтров и breaker; расхождение поглощается `sync_equity` без флага (§6.4).
**Как установлено:** чтением + замером. Док-строка `nautilus_trader.model.events.OrderFilled` (v1.221.0) описывает событие как одно исполнение, а не итог ордера:
```
last_qty : Quantity
    The fill quantity for this execution.
last_px : Price
    The fill price for this execution (not average price).
trade_id : TradeId
    The trade match ID (assigned by the venue).
```
То есть на один `client_order_id` приходится по событию на каждый venue trade match. Именно `last_qty` и `last_px` читает `on_order_filled` (892–893).
**Уверенность:** доказано

### A2-041 Комиссия закрывающего ордера не списывается нигде
**Севирити:** HIGH
**Тип:** математика / логика
**Где:** `src/execution/strategies/ml_strategy.py:937-941`, `967-972`
**Что в коде:**
```
970:                fee=0.0,  # fees already accounted in on_order_filled
```
**В чём дефект:** комментарий неверен. `on_order_filled` списывает комиссию только в ветке entry (917); выходные филлы уходят в `else`, где выполняется только `log.debug`. `close_position` получает `fee=0.0`.
**Как проявляется:** ~4.5 bps от нотионала на каждой сделке (≈3.2 USDT на нотионал 7000, при round-trip издержках ≈25 bps — около 18 % полной стоимости) не попадают ни в `_cash`, ни в `realized_pnl`, ни в дневной/недельный счётчик. PnL завышается систематически, в одну сторону.
**Кто ещё это читает:** `_daily_realized_pnl` → `get_daily_pnl` → `_check_daily_loss` и `CircuitBreaker`; `record_loss` вызывается по `realized_pnl < 0`, поэтому пограничные сделки не считаются убыточными.
**Как установлено:** чтением + замером инварианта трекера (§6.1: внутри трекера учёт верен, теряется на уровне стратегии).
**Уверенность:** доказано

### A2-042 Circuit breaker в развёрнутой конфигурации — тождественный no-op
**Севирити:** HIGH
**Тип:** логика / архитектура
**Где:** `src/execution/strategies/ml_strategy.py:2196-2217`, `842-849`; `src/risk/circuit_breaker.py:86-213`
**Что в коде:**
```
2196:        if self._config.dry_run:
...
2217:            return
```
**В чём дефект:** возврат происходит до `seed_from_authoritative_equity` и `sync_equity`. В `--dry-run` `_open_position` не вызывается, значит `update_fill` не вызывается, значит `_positions` пуст и `_cash` неизменен. Все входы `check()` — `drawdown`, `daily_pnl`, `weekly_pnl`, `consecutive_losses` — константы (0, 0, 0, 0). Ветка всплеска волатильности получает `current_atr=0.0, avg_atr=0.0` (`ml_strategy.py:726-727`) и мертва при любом режиме.
**Как проявляется:** breaker логируется как работающий, но не может сработать. Ни одна из его пяти ветвей не исполнялась ни разу с 14.08. Это же означает, что и **A2-012** (посев базы просадки) не проверялась на живых данных.
**Кто ещё это читает:** `on_bar:716-735` — единственный вызывающий.
**Как установлено:** чтением + `deploy/atomicortex-bot.service:26-32` (`--dry-run` в развёрнутом юните).
**Уверенность:** доказано

### A2-043 `_daily_triggered` пишется, персистится, восстанавливается и не читается
**Севирити:** HIGH
**Тип:** мёртвый код / логика
**Где:** `src/risk/circuit_breaker.py:153`, `158`, `299-305`, `247-252`; `src/risk/portfolio_tracker.py:409-429`; `src/risk/risk_state_store.py:116-141`
**Что в коде:**
```
299:        self._daily_triggered = bool(state.get("breaker_daily_triggered", False))
302:            log.warning(
303:                "CircuitBreaker restored as TRIGGERED | reason={r}",
```
**В чём дефект:** `check()` (86–213) не обращается к `self._daily_triggered` ни разу. Липкого дневного стопа нет: после рестарта breaker пересчитывает всё заново из `portfolio_state`. `reset_daily()` не имеет производственных вызовов. Сверх того `PortfolioTracker._persist` вызывает `RiskStateStore.save(self._snapshot())`, а `save` перезаписывает файл целиком (`json.dump(state, f)`), тогда как `_snapshot()` не содержит ключей breaker'а — то есть любой филл, закрытие или откат периода стирает `breaker_daily_triggered` из общего файла. Слияние делает только `CircuitBreaker._persist` (262–267), одностороннее.
**Как проявляется:** док-строка модуля обещает, что «a bot that lost -2.9 % over the morning and was restarted at noon» не сможет потерять ещё 3 %. Механизм, который это обеспечивает, отсутствует; работает только восстановление `_daily_realized_pnl`.
**Кто ещё это читает:** тесты `tests/test_risk_state_persistence.py:227-269` проверяют именно поле, а не поведение — поэтому находка не ловится тестами.
**Как установлено:** замером (`grep -rn "_daily_triggered\|reset_daily\|get_position_size_multiplier" src/ scripts/ tests/` — вне `circuit_breaker.py` только тесты).
**Уверенность:** доказано

### A2-044 Три экземпляра стратегии делят один файл риск-состояния и один файл pending-SL
**Севирити:** HIGH
**Тип:** архитектура
**Где:** `src/execution/strategies/ml_strategy.py:123`, `286-297`, `562-568`; `src/execution/live_trader.py:216-251`
**Что в коде:**
```
123:    signal_db_path: str = "data/atomicortex.db"
290:            _risk_state_path = _db_path.parent / "risk_state_4h.json"
566:            store_path = db_path.parent / "pending_sl_4h.json"
```
**В чём дефект:** `live_trader.build_node` создаёт по экземпляру `MLTradingStrategy` на символ и не переопределяет `signal_db_path`. Оба пути состояния выводятся из него, значит совпадают для BTC/ETH/SOL. Каждый экземпляр держит собственный `PortfolioTracker`, `CircuitBreaker` и `PendingOrdersStore`, и каждый пишет **полный документ** через `os.replace` (`risk_state_store.py:126-130`, `pending_orders_store.py:211-215`). Последняя запись затирает состояние остальных.
**Как проявляется:** (а) `cash`, `peak_equity`, `consecutive_losses` одного символа перезаписываются другим; при рестарте каждый трекер восстанавливает чужие числа. (б) `pending_sl_4h.json` теряет записи: `put` от BTC стирает pending-SL ETH, и после рестарта позиция ETH остаётся без стопа. (в) `_read_nautilus_equity` (2163–2173) даёт каждому трекеру **весь** баланс счёта, поэтому каждый символ сайзится в 1 % от полного капитала — суммарный риск на бар втрое выше объявленного.
**Кто ещё это читает:** `RiskStateStore`, `PendingOrdersStore`, `CircuitBreaker` — все три через один и тот же путь.
**Как установлено:** чтением. Латентно: развёрнут один символ (`deploy/atomicortex-bot.service:30` — `--symbols BTCUSDT-PERP`).
**Уверенность:** доказано (код), последствие — вероятно (не наблюдалось, т.к. один символ)

### A2-045 «Ожидаемая доходность» приравнена к 1×ATR; confidence не влияет ни на что, кроме порога
**Севирити:** HIGH
**Тип:** математика / расхождение с практикой
**Где:** `src/risk/risk_engine.py:354`, `333-365`; `calculate_position_size:233-262`
**Что в коде:**
```
354:        expected_return_bps = signal.atr_pct * 10_000  # ATR% → bps
```
**В чём дефект:** ожидаемая доходность сделки не выводится ни из вероятности модели, ни из геометрии барьеров. Она равна одному ATR всегда — при confidence 0.56 и при 0.95 одинаково. При этом стоп стоит на 1.5·ATR, а объявленный TP — на 2.25·ATR: ни одна из трёх величин не согласована с двумя другими. `signal.confidence` во всём модуле используется ровно один раз — в `_check_confidence` (306–313).
**Как проявляется:** размер позиции не масштабируется краем модели; порог издержек не зависит от силы сигнала. Замер на 8 сделках: `exp_ret_bps` 87.5…153.5 против порога 66.4…76.7, минимальный запас 10.8 bps (12 % от порога) при confidence 0.654.
**Кто ещё это читает:** `RiskDecision.expected_fee_bps` → `_emit_signal` не пишет его в журнал вовсе; поле уходит в никуда.
**Как установлено:** чтением + замером (таблица §4.3).
**Уверенность:** доказано

### A2-046 Хардкод «средней волатильности» 1 % отвергает 44 % баров ETH и 71 % баров SOL
**Севирити:** HIGH
**Тип:** математика / логика
**Где:** `src/risk/risk_engine.py:387-401`, особенно `396`
**Что в коде:**
```
396:        if signal.atr_pct > self._config.vol_spike_multiplier * 0.01:
```
**В чём дефект:** «среднее» — литерал `0.01`, одинаковый для всех инструментов; никакого скользящего среднего ATR не вычисляется. Комментарий сам переадресует настоящее сравнение circuit breaker'у («the actual comparison is done via the circuit breaker»), а там ветка получает нули (**A2-042**). Фильтр вырождается в абсолютный потолок ATR% = 2 %.
**Как проявляется:** замер по `data/features/ml_features/*_4h_features.parquet` (n = 5038 на символ), доля баров выше потолка:
```
BTCUSDT median=0.01288 | above=10.4%
ETHUSDT median=0.01919 | above=44.5%
SOLUSDT median=0.02327 | above=71.4%
```
Совместно с `_check_expected_return` образуется жёсткий коридор ATR% (замер при цене 80 000, equity 10 000): **0.795 % … 1.995 %**. Вне коридора сигнал отвергается независимо от модели. Медиана SOL лежит выше верхней границы — инструмент неторгуем по конструкции, а не по решению.
**Кто ещё это читает:** `regime_detector` пишет `atr_pct` в фичи и в `RegimeState`; `microstructure.py:238-239` и `mtf_context.py:324-325` читают ту же колонку как долю — контракт единиц держится везде.
**Как установлено:** замером (polars по трём parquet + численный скан коридора).
**Уверенность:** доказано

### A2-047 В строке лога `atr_pct` печатается `atr_percentile`
**Севирити:** MEDIUM
**Тип:** семантика (наблюдаемость)
**Где:** `src/execution/strategies/ml_strategy.py:761`
**Что в коде:**
```
761:                f"atr_pct={regime_state.atr_percentile:.2f} | "
```
**В чём дефект:** `RegimeState` имеет оба поля: `atr_pct` (доля ATR/цена, `regime_detector.py:202`) и `atr_percentile` (ранг ∈ [0,1], `:203`). В диагностику выводится второе под именем первого.
**Как проявляется:** значения 0.07–0.69 в журнале прода читаются как «ATR = 7…69 % цены», что для 4H BTC неправдоподобно, и порождают ложную гипотезу о рассогласовании единиц. Настоящий `atr_pct` в проде — 0.0088…0.0154 (замер по 8 записям снимка). Единицы в вычислениях согласованы; дефект — только в строке лога, но именно она формирует картину у оператора.
**Кто ещё это читает:** только человек. Значение из этой строки в код не возвращается.
**Как установлено:** замером (`grep -rn "atr_pct" src/execution/strategies/ src/risk/`; сверка с `regime_detector.py:202-203`, `329`, `448`).
**Уверенность:** доказано

### A2-048 Ветка «POSITION UNPROTECTED» вызывает `cache.order` со строкой
**Севирити:** MEDIUM
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:1200-1212`
**Что в коде:**
```
1201:            try:
1202:                order = self.cache.order(client_oid)
1203:            except Exception:
1204:                order = None
1206:            if order and any(t.startswith("SL-") for t in (order.tags or [])):
```
**В чём дефект:** `client_oid` — `str` (получен как `str(event.client_order_id)` на 1161). `Cache.order` принимает `ClientOrderId`. Исключение или `None` гасится `except`, и управление уходит в `else` с формулировкой «Unknown order rejected».
**Как проявляется:** отклонение стопа биржей регистрируется как неизвестный ордер уровня WARNING вместо ERROR «POSITION UNPROTECTED!». Единственный сигнал о незащищённой позиции теряется в шуме. Повторной отправки стопа нет ни в одной ветке; маршрут в Telegram помечен `TODO(PR-0.2)` и не реализован.
**Кто ещё это читает:** никто — ветка терминальная.
**Как установлено:** замером. Сигнатура (Cython, v1.221.0):
```
Cache.order(self, ClientOrderId client_order_id) -> Order
```
Прямой вызов со строкой:
```
TypeError : Argument 'client_order_id' has incorrect type (expected nautilus_trader.model.identifiers.ClientOrderId, got str)
```
Исключение перехватывается `except Exception` на 1203, `order` становится `None`, условие 1206 ложно — ветка недостижима **всегда**, а не в отдельных случаях.
**Уверенность:** доказано

### A2-049 Оба множителя размера позиции существуют и не применяются (A19 не закрыта)
**Севирити:** MEDIUM
**Тип:** мёртвый код / недоделка
**Где:** `src/risk/circuit_breaker.py:215-245`; `src/features/regime_detector.py:211-219`
**Что в коде:**
```
211:    def position_size_multiplier(self) -> float:
215:    def get_position_size_multiplier(
```
**В чём дефект:** `decision.position_size` нигде не умножается: из `calculate_position_size` он идёт напрямую в `instrument.make_qty(...)` (`ml_strategy.py:1099`).
**Как проявляется:** мягкий уровень breaker'а («-2 % → сократить позиции вдвое», объявленный в док-строке модуля, строки 9 и 14) не существует как поведение: между «полный размер» и «полная остановка» нет промежуточного состояния. Регимный масштаб (HIGH_VOL → 0.5, RANGE → 0.7) также не применяется — сделка №8 снимка открыта в режиме `high_vol` полным размером.
**Кто ещё это читает:** `scripts/analyze_regimes.py:158` — печать; `tests/test_risk_engine.py:274,290` и `tests/test_regime_detector.py:289` — тесты. Производственных потребителей нет.
**Как установлено:** замером (`grep -rn` по `src/ scripts/ tests/`).
**Уверенность:** доказано

### A2-050 Биржевые фильтры не проверяются; плечо на бирже не выставляется; локальный кэп плеча недостижим
**Севирити:** MEDIUM
**Тип:** недоделка / расхождение с практикой
**Где:** `src/risk/risk_engine.py:255-261`; `src/execution/strategies/ml_strategy.py:1099`, `1235-1236`
**Что в коде:**
```
256:        max_notional = equity * self._config.max_leverage
257:        if notional > max_notional:
```
**В чём дефект:** (а) нигде нет проверок `minQty` / `stepSize` / `minNotional` — `grep -rn "min_notional\|min_quantity\|minNotional\|minQty\|step_size" src/execution/ src/risk/` даёт только синтетические инструменты бэктеста в `data_catalog.py`. Единственное приведение — `instrument.make_qty` / `make_price`. (б) Плечо на бирже (`POST /fapi/v1/leverage`) не выставляется — `max_leverage` только локальный расчёт, на реальную маржу не влияет. (в) кэп срабатывает при `atr_pct < 0.01/(1.5·10) = 0.000667`, то есть ATR < 0.067 % цены; измеренный 5-й перцентиль ATR% для BTC на 4H — 0.00793, в 12 раз выше. Ветка недостижима.
**Как проявляется:** отказ биржи по фильтру приходит уже после отправки, в `on_order_rejected`. Реальное плечо счёта — то, что стоит на бирже по умолчанию, а не 10. Когда кэп всё же срабатывает, фактический риск на сделку молча становится **меньше** 1 % и нигде не отражается: `RiskDecision` не несёт поля фактического риска, а `leverage` присваивается константой (260) вместо пересчёта.
**Кто ещё это читает:** `decision.notional` → `_emit_signal` → `signals_log.notional` → `stats_engine`; `decision.leverage` → журнал.
**Как установлено:** замером (аналитический вывод порога + распределение `atr_pct` по parquet; leverage на 8 сделках 0.434…0.762). Дополнительно проверено поведение округления на построенном `CryptoPerpetual` BTCUSDT-PERP (`size_precision=3`, `size_increment=0.001`):
```
0.0865432 -> 0.087
0.0864999 -> 0.086
0.0009 -> 0.001
make_price 82349.22576545 -> 82349.2
min_quantity attr: None | min_notional: None
```
`make_qty` округляет **к ближайшему**, а не вниз: запрошенные 0.0865432 превращаются в 0.087, то есть фактический риск может превысить бюджет на половину шага. `0.0009 -> 0.001` показывает округление вверх через границу минимального лота. Атрибуты `min_quantity` / `min_notional` у инструмента — `None`, то есть на них опереться нельзя даже при желании.
**Уверенность:** доказано

### A2-051 Ни `1.5`, ни `2.25` не обоснованы; `MLStrategyConfig.rr_ratio` мёртв
**Севирити:** MEDIUM
**Тип:** недоделка / математика
**Где:** `src/risk/risk_engine.py:37`, `286`, `184-186`; `src/execution/strategies/ml_strategy.py:119`
**Что в коде:**
```
risk_engine.py:37:    atr_stop_multiplier: float = 1.5    # stop = 1.5 × ATR
risk_engine.py:286:        rr_ratio: float = 1.5,
ml_strategy.py:119:    rr_ratio: float = 1.5
```
**В чём дефект:** `evaluate` вызывает `calculate_take_profit` без `rr_ratio`, поэтому всегда берётся дефолт функции. `MLStrategyConfig.rr_ratio` не читается нигде (`grep -rn "rr_ratio" src/` — потребителей нет). Число 2.25 нигде не записано, оно только произведение двух дефолтов. Обоснования ни одного из значений в репозитории нет: ни из горизонта метки, ни из барьеров triple-barrier, ни из измеренной статистики исходов.
**Как проявляется:** геометрия сделки жёстко зафиксирована и не настраивается через конфиг стратегии; изменение `rr_ratio` в конфиге не даёт никакого эффекта. Подтверждено замером на 8 сделках: `SL/atr = 1.500000`, `TP/atr = 2.250000` во всех восьми.
**Кто ещё это читает:** `src/configs/strategy_1h.py:52` и `strategy_15m.py:62` объявляют отдельное `min_rr_ratio = 1.3` — третье независимое число того же смысла.
**Как установлено:** замером (grep + сверка со снимком).
**Уверенность:** доказано

### A2-052 Нет идемпотентности ни по бару, ни по позиции; трекер усредняет противоположные направления
**Севирити:** MEDIUM
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:212`, `641-843`; `src/risk/portfolio_tracker.py:124-133`
**Что в коде:**
```
212:        self._pending_stops: dict[str, str] = {}  # instrument_id → client_order_id
portfolio_tracker.py:124:        if symbol in self._positions:
portfolio_tracker.py:126:            # Average in
portfolio_tracker.py:127:            total_qty = pos.quantity + quantity
```
**В чём дефект:** (а) `on_bar` не хранит метку последнего обработанного бара; повторная доставка того же бара породит второй вход. (б) поле `_pending_stops`, объявленное как защита «to avoid duplicate orders», не читается и не пишется нигде (`grep -rn "_pending_stops" src/` — одна строка, само объявление). (в) `update_fill` не сверяет `direction`: SHORT поверх LONG **складывает** количества, сохраняя направление первого филла. (г) `_check_max_positions` недостижим (§4.1), поэтому лимита на повторный вход по символу нет.
**Как проявляется:** удвоенная позиция с двумя `reduce_only` стопами на разные количества по разным ценам; при развороте — трекер показывает LONG удвоенного размера там, где биржа свела позицию неттингом почти к нулю, и все последующие `unrealized_pnl`, `equity`, `drawdown` недостоверны.
**Кто ещё это читает:** `get_state()` → девять фильтров + breaker; `sync_equity` замажет расхождение без флага.
**Как установлено:** чтением + замером (`grep -rn "_pending_stops"`, `grep -n "_last_bar\|dedup\|duplicate\|seen"`).
**Уверенность:** доказано

### A2-053 Документированный механизм восстановления незащищённой позиции не имеет исполнителя
**Севирити:** MEDIUM
**Тип:** недоделка / архитектура
**Где:** `src/execution/pending_orders_store.py:20-23`; `src/risk/risk_state_store.py:12-13`
**Что в коде:**
```
20: ``load_all`` ... Fills that occurred
21: *during* the crash window (and that Binance does not re-deliver) leave an
22: unprotected position on the exchange; that scenario is covered by the
23: ``PositionReconciler`` which surfaces it as ORPHAN.
```
**В чём дефект:** `PositionReconciler` мёртв (**A2-001**). `RiskStateStore` док-строка аналогично отсылает сверку позиций к «reconciled separately». Живой `reconciler_signals` работает только с `signals_log` и историческими свечами и никогда не обращается к бирже за позициями.
**Как проявляется:** единственный сценарий, который может оставить открытую позицию без стопа при падении процесса, объявлен покрытым и не покрыт. Обнаружения позиции, открытой не этим процессом, тоже нет.
**Кто ещё это читает:** `_init_pending_store:549-557` повторяет ту же ссылку («covered by the external orphan alert»).
**Как установлено:** чтением + подтверждением A2-001 из прохода 1.
**Уверенность:** доказано

### A2-054 Вся торговая история из 8 сделок — симуляция реконсилятора, и реконсилятор не развёрнут
**Севирити:** HIGH
**Тип:** семантика / архитектура
**Где:** `src/execution/reconciler_signals.py:202-250`, `254-267`; `deploy/units.enabled`
**Что в коде:**
```
231:            if sl_hit or tp_hit:
232:                # Tie on same bar → SL (loss), conservative.
233:                if sl_hit:
234:                    close_px, result = sl, "loss"
235:                else:
236:                    close_px, result = tp, "win"
```
**В чём дефект:** закрытия книжатся ровно по цене SL или TP, по `high`/`low` исторического бара, без комиссий и проскальзывания, и — главное — по TP-ордеру, которого на бирже не бывает (**A2-037**). Это не сверка исполнения, а переигровка гипотетической стратегии.
**Как проявляется:** замер на снимке: `close_price` совпадает с `stop_loss`/`take_profit` до последнего знака во всех 8 строках; `pnl_pct` сделки №1 = 2.090924999999978 против расчётного `2.25 × 0.00929 = 2.0909 %`. `equity` при этом ни разу не сдвинулась с 10000.0 — трекер этих «сделок» не видел. Плюс `deploy/units.enabled` перечисляет 4 юнита и явно исключает реконсилятор: «the reconciler ... not part of the current paper --dry-run deployment». То есть в проде даже эта симуляция не выполняется.
**Кто ещё это читает:** `_update_daily_stats` → `daily_stats` (7 строк в снимке), `_refresh_performance_cache` → `performance_cache` (12 строк), далее `src/api/main.py` и `src/telegram_bot/database.py:527-573`. Вся отображаемая статистика системы построена на этих числах.
**Как установлено:** замером (SQL по `/tmp/ac_snap.db`, сверка формул) + чтением юнитов.
**Уверенность:** доказано

### A2-055 Дневной P&L не включает комиссии; дневной и недельный лимиты считаются от разных баз
**Севирити:** LOW
**Тип:** математика
**Где:** `src/risk/portfolio_tracker.py:331-343`, `345-351`, `122`
**Что в коде:**
```
340:        denom = self._day_start_equity if self._day_start_equity > 0 else self._initial_equity
349:        if self._initial_equity <= 0:
351:        return weekly_total / self._initial_equity
```
**В чём дефект:** (а) `update_fill` списывает комиссию из `_cash` (122), но не добавляет её в `_daily_realized_pnl`; `get_daily_pnl` суммирует `_daily_realized_pnl + Σ unrealized`, поэтому комиссии в дневном проценте не видны. Замер: после филла с комиссией 3.24 и марка +1000 `get_daily_pnl()` вернул ровно 0.009, то есть 900/10000 без вычета комиссии. (б) дневной делитель — `_day_start_equity`, недельный — `_initial_equity`. Один и тот же по смыслу «процент от капитала» измеряется от двух разных величин.
**Как проявляется:** дневной лимит −3 % срабатывает позже, чем должен, на сумму уплаченных комиссий; недельный лимит после посева S0-2 или после компаундирования измеряется от устаревшей базы.
**Кто ещё это читает:** `_check_daily_loss`, `_check_weekly_loss`, `CircuitBreaker.check`, `SignalBridge.update_metrics`.
**Как установлено:** замером (§6.1).
**Уверенность:** доказано

### A2-056 Фандинг не входит в P&L удерживаемой позиции; предторговая оценка занижена до порядка
**Севирити:** MEDIUM
**Тип:** математика
**Где:** `src/risk/portfolio_tracker.py:27-38`, `155-173`, `175-202`; `src/risk/risk_engine.py:58`; `src/execution/cost_model.py:91`
**Что в коде:**
```
risk_engine.py:58:    default_hours_held: float = 8.0
cost_model.py:91:        num_payments = hours_held / 8.0
```
**В чём дефект:** (а) `_Position` не имеет поля накопленного фандинга; `update_price` считает `unrealized_pnl` без него. Единственный путь — `sync_equity`, которая поглощает фандинг как дрейф `_cash` и намеренно не трогает `_daily_realized_pnl` / `_weekly_realized_pnl` (док-строка 184–186). Значит дневной и недельный лимиты потерь фандинг не видят вообще. (б) предторговая оценка использует фиксированные 8 часов = ровно один платёж.
**Как проявляется:** измеренные удержания в снимке — 239…5039 минут (4…84 часа, до 10.5 платежей). Горизонт метки — 6 баров = 24 часа (3 платежа). Оценка издержек занижена в 3–10 раз именно на том компоненте, который растёт со временем удержания. Поскольку выхода по времени нет (§1.1), удержание не ограничено сверху ничем.
**Кто ещё это читает:** `_check_expected_return` (порог `3 × cost_bps`), `RiskDecision.expected_fee_bps`.
**Как установлено:** замером (длительности из снимка) + чтением.
**Уверенность:** доказано

### A2-057 `on_position_changed` не переопределён: частичный выход из позиции невидим для учёта
**Севирити:** HIGH
**Тип:** логика
**Где:** `src/execution/strategies/ml_strategy.py:943`, `950`, `937-941`
**Что в коде:** в файле определены только два позиционных хука:
```
943:    def on_position_opened(self, event: PositionOpened) -> None:
950:    def on_position_closed(self, event: PositionClosed) -> None:
```
**В чём дефект:** интроспекция `Strategy` (Nautilus 1.221.0) даёт три позиционных хука — `on_position_opened`, **`on_position_changed`**, `on_position_closed`. `PositionChanged` — это событие добора и **частичного сокращения** позиции. Оно не обрабатывается. Параллельно `on_order_filled` отправляет любой выходной филл в ветку `else` (937–941), где выполняется только `log.debug`.
**Как проявляется:** при частичном срабатывании `STOP_MARKET` (штатный исход для стоп-маркета на неликвидном уровне) позиция на бирже уменьшилась, а `PortfolioTracker._positions` продолжает держать её в полном размере с прежним `avg_entry_price`. Последующие `unrealized_pnl`, `equity`, `drawdown`, `daily_pnl_pct` вычисляются на несуществующем количестве. `close_position` не вызывается, `_consecutive_losses` не обновляется, комиссия частичного выхода не списывается. Расхождение с биржей молча поглотит `sync_equity` (§6.4).
**Кто ещё это читает:** `get_state()` → все девять фильтров `RiskEngine` и `CircuitBreaker.check`.
**Как установлено:** замером (интроспекция `Strategy`, §11) + чтением (`grep -n "def on_position" ml_strategy.py`).
**Уверенность:** доказано

### A2-058 Отказы отмены и модификации ордеров не обрабатываются
**Севирити:** LOW
**Тип:** недоделка
**Где:** `src/execution/strategies/ml_strategy.py:481-501` (`on_stop`), отсутствие хуков
**Что в коде:**
```
496:            self.cancel_all_orders(self._instrument_id)
497:            self.close_all_positions(self._instrument_id)
```
**В чём дефект:** `on_order_cancel_rejected` и `on_order_modify_rejected` существуют в API `Strategy` (§11) и не переопределены. `on_stop` отменяет все ордера и сразу закрывает позиции, не дожидаясь подтверждения отмены и не реагируя на отказ.
**Как проявляется:** если отмена `reduce_only` стопа отклонена, а `close_all_positions` при этом отработала, на бирже остаётся висящий стоп-ордер без позиции. Обнаружителя нет (`PositionReconciler` мёртв, A2-053).
**Кто ещё это читает:** никто.
**Как установлено:** замером (интроспекция) + чтением.
**Уверенность:** доказано

---

## 10. СВОДКА ПРОХОДА 5

| Севирити | Кол-во | ID |
|---|---|---|
| CRITICAL | 3 | A2-037, A2-038, A2-039 |
| HIGH | 9 | A2-040, A2-041, A2-042, A2-043, A2-044, A2-045, A2-046, A2-054, A2-057 |
| MEDIUM | 8 | A2-047, A2-048, A2-049, A2-050, A2-051, A2-052, A2-053, A2-056 |
| LOW | 2 | A2-055, A2-058 |
| **Всего** | **22** | A2-037 … A2-058 |

Нарастающий итог Аудита-2 после прохода 5: 36 + 22 = **58** находок.

### Статус находок Аудита-1, затронутых этим проходом

| ID | Формулировка | Статус |
|---|---|---|
| A6 | сайзинг / риск-параметры | размерности верны и воспроизводятся точно (§2.2); открыты производные дефекты A2-045, A2-046, A2-050, A2-051 |
| A10 | отклонённый ордер логировался как исполненный | **закрыта** для entry (`result='rejected'`, `signal_bridge.py:280-287`); не закрыта для SL (A2-048) и для окна между `_emit_signal` и `submit_order` |
| A19 | regime-multiplier | **не закрыта** — A2-049 |
| A2-001 | `PositionReconciler` мёртв | подтверждена со стороны исполнения — A2-053 |
| A2-012 | посев базы просадки под `except: pass` | недостижима в развёрнутой конфигурации — A2-042, §5.2 |
| A2-013 | в `--dry-run` у журнала сигналов нет производителя | подтверждена: `_emit_signal` вызывается только из `_open_position`, которое `--dry-run` пропускает (`ml_strategy.py:842`) |
| A2-015 | разметка и исполнение описывают разные события | подтверждена со стороны исполнения: стоп 1.5·ATR, TP отсутствует, выхода по времени нет — A2-037 |

---

## 11. ВЕРИФИКАЦИЯ ПРОТИВ УСТАНОВЛЕННОГО NAUTILUS 1.221.0

Выполнено через `/home/asus/Desktop/AtomiCortex/.venv/bin/python`.

**Хуки `Strategy`** — `sorted(m for m in dir(Strategy) if m.startswith(('on_order','on_position')))`:

```
['on_order_accepted', 'on_order_book', 'on_order_book_deltas', 'on_order_book_depth',
 'on_order_cancel_rejected', 'on_order_canceled', 'on_order_denied', 'on_order_emulated',
 'on_order_event', 'on_order_expired', 'on_order_filled', 'on_order_initialized',
 'on_order_modify_rejected', 'on_order_pending_cancel', 'on_order_pending_update',
 'on_order_rejected', 'on_order_released', 'on_order_submitted', 'on_order_triggered',
 'on_order_updated', 'on_position_changed', 'on_position_closed', 'on_position_event',
 'on_position_opened']
```

Следствия:

1. `on_order_rejected` и `on_order_denied`, переопределённые стратегией
   (1157, 1166), — настоящие хуки. Здесь всё в порядке.
2. **`on_position_changed` не переопределён.** `grep -n "def on_position"`
   по `ml_strategy.py` даёт только `on_position_opened` (943) и
   `on_position_closed` (950). Nautilus шлёт `PositionChanged` при
   доборе и при **частичном** сокращении позиции. Значит частичное
   срабатывание стопа (`STOP_MARKET` может исполниться частями) не
   доходит до `PortfolioTracker` вообще: `close_position` вызывается
   только из `on_position_closed`, а выходной филл в `on_order_filled`
   уходит в ветку `else`, где не делается ничего. Это усиливает A2-040 и
   A2-041: при частичном выходе трекер продолжает считать позицию целой.
3. `on_order_cancel_rejected` и `on_order_modify_rejected` не
   переопределены — отказ биржи отменить или подвинуть стоп нигде не
   обрабатывается. Учитывая `cancel_all_orders` в `on_stop` (496), отказ
   отмены перед `close_all_positions` пройдёт молча.

**`Cache.order`** — сигнатура и поведение приведены в A2-048.

**`OrderFilled`** — параметры приведены в A2-040.

**`CryptoPerpetual.make_qty` / `make_price`** — замер приведён в A2-050.

---

## 12. НЕ ИССЛЕДОВАНО

1. **`ml_strategy_15m` и `meta_strategy`** разобраны только в части,
   унаследованной от `_open_position`. Их собственные пути риска
   (`ml_strategy_15m.py:475-510`, `meta_strategy.py:275`) не проходились
   построчно. A2-004 и A2-005 из прохода 1 остаются в силе.
2. **`paper_trader.py`, `backtest_runner.py`, `binance_rate_limiter.py`,
   `startup_check.py`** в этом проходе не разбирались.
3. **`_compute_features` (`ml_strategy.py:1369`, помечен DEPRECATED)**
   определяет `atr_pct` как однобаровый диапазон
   `(high[-1] - low[-1]) / close[-1]` (1497–1499), тогда как обучающий
   `atr_pct` — это ATR/цена (`regime_detector.py:329`). Одно имя фичи, два
   определения. Достижимость этой ветки не подтверждена (единственный
   вызов в `on_bar` идёт в `_compute_features_unified`), поэтому находка
   не оформлена — предмет прохода по фичам/мёртвому коду.
4. **Поведение `close_all_positions` в `on_stop`** относительно висящих
   `reduce_only` стопов (отменяются ли они до закрытия, `cancel_all_orders`
   на 496 идёт первым) — не проверено против реального движка.
5. **Транзакционность `signals_log`** при одновременной работе бота и
   реконсилятора (WAL заявлен в док-строке `reconciler_signals.py:18`) —
   не проверялась.
6. **`heartbeat` / `watchdog` / `signal_bridge` в целом** — предмет
   прохода 8, здесь затронуты только точки вызова из пути ордера.
