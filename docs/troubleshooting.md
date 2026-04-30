# AtomiCortex — Troubleshooting Journal

Журнал реальных проблем, возникших в ходе разработки.  
Каждая запись содержит точный текст ошибки, первопричину и способ решения.

---

## Формат записи

```
### [XX-NNN] Краткое название
Дата:    YYYY-MM
Фаза:    N.N — Описание фазы
Ошибка:  точный текст из терминала
Причина: первопричина
Решение: что именно изменили
Файл:    путь:строка
```

---

## Фаза 1 — Data Pipeline

---

### [CF-001] Cryptofeed: ImportError — L2Book не существует в types
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка:**
```
ImportError: cannot import name 'L2Book' from 'cryptofeed.types'
(/home/asus/Desktop/AtomiCortex/.venv/lib/python3.11/site-packages/
cryptofeed/types.cpython-311-x86_64-linux-gnu.so)
```

**Причина:**  
В документации Cryptofeed и большинстве примеров в интернете тип книги заявок называется `L2Book`. В действительности канал задаётся строковой константой `L2_BOOK` из `cryptofeed.defines`, а соответствующий класс данных в `cryptofeed.types` называется `OrderBook`. Имена канала и типа не совпадают. Дополнительная сложность: модуль скомпилирован в `.so` через Cython — IDE не подсказывает имена.

**Решение:**  
Вывести полный список экспортируемых имён модуля:
```python
python -c "import cryptofeed.types as t; print(dir(t))"
# -> [..., 'L1Book', 'OrderBook', 'Trade', 'Funding', ...]
```
Использовать `OrderBook` вместо `L2Book`. В финальном коде аннотация параметра `book` в `_on_book()` задана как `Any` — импорт Cython-класса для аннотации не нужен.

**Файл:** `src/ingestion/live_feed.py` — импорт убран, тип `Any` в сигнатуре `_on_book(self, book: Any, ...)`

---

### [CF-002] Cryptofeed: inspect.signature не работает на Cython-типах
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка:**
```
ValueError: no signature found for builtin type <class 'cryptofeed.types.Trade'>
Trade fields: no dataclass
```
*(при попытке `inspect.signature(Trade)` и проверке `Trade.__dataclass_fields__`)*

**Причина:**  
Типы `Trade`, `Funding`, `OrderBook` в Cryptofeed 2.4.1 скомпилированы Cython в нативное расширение `.so`. Такие типы Python видит как `builtin type` — они не имеют `__dataclass_fields__`, `__init__` не интроспектируем через `inspect`, поле `__doc__` может быть пустым. Стандартные инструменты интроспекции Python здесь не работают.

**Решение (workaround):**  
Создать экземпляр через `__new__` (без вызова `__init__`) и применить `dir()`:
```python
t = Trade.__new__(Trade)
print([a for a in dir(t) if not a.startswith('_')])
# Trade:   ['amount', 'exchange', 'id', 'price', 'raw', 'side', 'symbol', 'timestamp', 'type']
# Funding: ['exchange', 'mark_price', 'next_funding_time', 'predicted_rate', 'rate', 'raw', 'symbol', 'timestamp']
```

**Файл:** Информация применена при написании `_on_trade` и `_on_funding` в `src/ingestion/live_feed.py:243–270` и `src/ingestion/live_feed.py:300–325`.

---

### [CF-003] Cryptofeed: двойная вложенность OrderBook.book.bids
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка:**  
Не исключение интерпретатора, но потенциальный `AttributeError` в runtime при обращении `book.bids` вместо `book.book.bids`. Неизвестная структура данных.

**Причина:**  
`OrderBook` содержит атрибут `.book` типа `_OrderBook`, и уже `_OrderBook` содержит `.bids` и `.asks`. Двойная вложенность не отражена в документации. Структура: `book.book.bids[Decimal(price)] = Decimal(qty)`.

**Решение:**  
Проверить атрибуты `_OrderBook` тем же методом `__new__` + `dir()`:
```python
from cryptofeed.types import _OrderBook
_ob = _OrderBook.__new__(_OrderBook)
print([a for a in dir(_ob) if not a.startswith('_')])
# -> ['ask', 'asks', 'bid', 'bids', 'checksum', 'max_depth', 'to_dict']
```
Использовать корректный путь доступа `book.book.bids` и `book.book.asks`.

**Файл:** `src/ingestion/live_feed.py:272–281` — метод `_on_book()`:
```python
bids = sorted(book.book.bids.keys(), reverse=True)[:5]
asks = sorted(book.book.asks.keys())[:5]
```

---

### [CF-004] Cryptofeed: ModuleNotFoundError при поиске setup_signal_handlers
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка:**
```
ModuleNotFoundError: No module named 'cryptofeed.util.async_utils'
```

**Причина:**  
Попытка найти `setup_signal_handlers` по аналогии с другими asyncio-библиотеками, где утилиты обычно живут в `util/` или `utils/`. В Cryptofeed 2.4.1 эта функция находится прямо в `cryptofeed.feedhandler`, а не в подпакете утилит. Структура пакета нестандартная.

**Решение:**  
Поиск по всем подмодулям через `pkgutil`:
```python
import pkgutil, importlib, cryptofeed
for mod in pkgutil.walk_packages(cryptofeed.__path__, prefix='cryptofeed.'):
    try:
        m = importlib.import_module(mod.name)
        if hasattr(m, 'setup_signal_handlers'):
            print('Found in:', mod.name)  # -> cryptofeed.feedhandler
    except Exception:
        pass
```
Выяснено поведение: на Linux `loop.add_signal_handler(SIGINT/SIGTERM, handle_stop_signals)` — callback вызывает `raise SystemExit`, что выбрасывается из `loop.run_forever()`. Это определило архитектуру `try/except SystemExit` в `LiveFeedManager.run()`.

**Файл:** `src/ingestion/live_feed.py:204–212` — блок `try/except SystemExit` в методе `run()`.

---

### [CF-005] Cryptofeed: FeedHandler.run() блокирует event loop — нельзя добавить duration таймер
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка:**  
Не исключение, но архитектурный капкан: при вызове `fh.run()` процесс блокируется навсегда — невозможно добавить `--duration` таймер поверх стандартного вызова.

**Причина:**  
`FeedHandler.run()` вызывает `loop.run_forever()` внутри себя — управление не возвращается вызывающему коду. Единственный способ остановить — SIGINT или callback изнутри loop. Параметр `start_loop=False` существует, но не документирован явно:
```python
# исходник cryptofeed/feedhandler.py
def run(self, start_loop=True, ...):
    for feed in self.feeds:
        feed.start(loop)   # присоединяет feeds к loop
    if not start_loop:
        return             # выходит БЕЗ запуска loop
    loop.run_forever()     # блокирует при start_loop=True
```

**Решение (workaround):**  
Создать собственный event loop, передать управление `start_loop=False`, затем добавить `loop.call_later()` для таймера и вызвать `loop.run_forever()` самостоятельно:
```python
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
self._fh.run(start_loop=False, install_signal_handlers=True)
if duration is not None:
    loop.call_later(duration, loop.stop)
try:
    loop.run_forever()
except SystemExit:
    ...
```

**Файл:** `src/ingestion/live_feed.py:188–213` — метод `LiveFeedManager.run()`.

---

### [CF-006] Cryptofeed: неправильный порядок graceful shutdown — двойное закрытие loop
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка:**  
При наивном вызове `fh.stop()` + `fh.close()` после `loop.run_forever()` — `RuntimeError: This event loop is already running` или `RuntimeError: Event loop is closed` в зависимости от порядка вызовов.

**Причина:**  
`FeedHandler.close()` содержит собственный `loop.stop()` → `loop.run_forever()` → `loop.close()`. При `start_loop=False` мы уже остановили loop через `loop.stop()` (из `call_later`). Если вызвать `fh.close()` после этого — он пытается снова запустить уже остановленный loop, а затем закрыть его — мы теряем контроль над cleanup:
```python
# исходник FeedHandler.close()
def close(self, loop=None):
    loop.stop()        # no-op если уже остановлен
    loop.run_forever() # запускает ещё раз (!), обрабатывает pending callbacks
    # ... отменяет задачи, закрывает loop
    loop.close()       # теперь loop уже наш закрытый
```

**Решение:**  
Вызывать только `fh._stop(loop)` (внутренний метод, возвращает список coroutine-задач завершения), собирать их через `asyncio.gather`, закрывать loop вручную:
```python
def _shutdown(self, loop):
    shutdown_tasks = self._fh._stop(loop=loop)
    loop.run_until_complete(asyncio.gather(*shutdown_tasks))
    if not loop.is_closed():
        loop.close()
```

**Файл:** `src/ingestion/live_feed.py:218–235` — метод `LiveFeedManager._shutdown()`.

---

### [CF-007] Cryptofeed: "Task was destroyed but it is pending!" при остановке
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket

**Ошибка** (вывод при живом тесте):
```
Task was destroyed but it is pending!
task: <Task pending name='Task-3' coro=<ConnectionHandler._watcher()
running at .venv/lib/python3.11/site-packages/cryptofeed/connection_handler.py:48>
wait_for=<Future pending cb=[Task.task_wakeup()]>>
```

**Причина:**  
Cryptofeed создаёт внутреннюю задачу `ConnectionHandler._watcher()`, которая ожидает событий от WebSocket через `asyncio.Future`. При остановке loop через `loop.stop()` эта задача не успевает корректно отменить своё `wait_for` Future — её `cancel()` не вызывается до закрытия loop. Это известный баг в Cryptofeed 2.4.1, воспроизводится при `start_loop=False`.

**Решение (workaround):**  
Принято как косметическая проблема сторонней библиотеки. Поведение не влияет на корректность данных и не вызывает утечек ресурсов — Python GC корректно собирает незавершённые задачи. Программно подавить можно через `asyncio.get_event_loop().set_exception_handler(lambda loop, ctx: None)`, но это скроет реальные ошибки. Исправление требует патча в `cryptofeed/connection_handler.py`.

**Файл:** Не изменялось. Баг в `cryptofeed/connection_handler.py:48`.

---

### [CF-008] Binance WebSocket: timeout при первом подключении
**Дата:** 2026-04  
**Фаза:** 1.6 — Live WebSocket (живой тест)

**Ошибка** (вывод при живом тесте):
```
BINANCE_FUTURES.ws.1: encountered connection issue
timed out during opening handshake - reconnecting in 1.0 seconds...
TimeoutError: timed out during opening handshake
BINANCE_FUTURES.ws.1: encountered connection issue
timed out during opening handshake - reconnecting in 2.0 seconds...
```

**Причина:**  
Первые две попытки подключения к `wss://fstream.binance.com` завершились timeout при TLS handshake. Вероятные причины: кратковременный geo-rate-limiting Binance на новые WebSocket соединения, или сетевая задержка при первом DNS-резолвинге. Asyncio DNS резолвинг (`loop.getaddrinfo`) был отменён через `CancelledError` — это видно в traceback.

**Решение:**  
Cryptofeed самостоятельно повторил подключение с экспоненциальным backoff (1s → 2s → ...). На третьей попытке соединение установилось, начали приходить реальные L2 Book тики:
```
INFO | BOOK BTC-USDT-PERP bid=76069.4000 ask=76069.5000 imbalance=+0.6448
```
Параметр `retries=-1` при создании `BinanceFutures()` обеспечивает бесконечные попытки реконнекта — это правильное поведение для продакшн-системы.

**Файл:** `src/ingestion/live_feed.py:192` — `BinanceFutures(..., retries=-1)`.

---

### [PL-001] Polars: read_csv — параметр dtypes устарел
**Дата:** 2026-04  
**Фаза:** 1.3 — Parquet Converter

**Ошибка:**
```
DeprecationWarning: `dtypes` is deprecated. Use `schema_overrides` instead.
```

**Причина:**  
В Polars 0.20.31 параметр `dtypes=` в `pl.read_csv()` переименован в `schema_overrides=`. Старое имя ещё принимается, но генерирует предупреждение и будет удалено в следующих версиях.

**Решение:**  
Заменить `dtypes=cfg.csv_dtypes` на `schema_overrides=cfg.csv_dtypes` во всех вызовах `pl.read_csv()`.

**Файл:** `src/ingestion/parquet_converter.py:251` — вызов `pl.read_csv()`.

---

### [PL-002] Polars: DuplicateError при чтении Parquet с Hive-партиционированием
**Дата:** 2026-04  
**Фаза:** 1.4 — DuckDB / DataStore

**Ошибка:**
```
polars.exceptions.DuplicateError: invalid Hive partition schema,
column 'symbol' exists in the file and the Hive partitions
```

**Причина:**  
Polars 0.20 автоматически определяет Hive-партиции из пути (`symbol=BTCUSDT` в имени директории) и пытается добавить колонку `symbol` в DataFrame. Одновременно колонка `symbol` уже хранится в самом Parquet-файле. Возникает конфликт дублирующихся имён колонок.

**Решение:**  
Передать `hive_partitioning=False` во все вызовы `pl.read_parquet()`:
```python
df = pl.read_parquet(parquet_path, hive_partitioning=False)
```

**Файл:** `src/ingestion/parquet_converter.py:467` (`validate_parquet`), `tests/test_parquet_converter.py` — все вызовы `pl.read_parquet`.

---

### [PD-001] pydantic-settings: model_fields deprecated на инстансе
**Дата:** 2026-04  
**Фаза:** 1.1 — Config

**Ошибка:**
```
PydanticUserError: `model_fields` is a class attribute and should be
accessed via the class, not an instance.
```

**Причина:**  
В pydantic v2 `model_fields` — атрибут **класса**, не инстанса. Обращение `self.model_fields` внутри метода `safe_dict()` считается некорректным и вызывает ошибку.

**Решение:**  
Заменить `self.model_fields` на `self.__class__.model_fields`:
```python
for field_name in self.__class__.model_fields:
    ...
```

**Файл:** `src/config.py:167` — метод `safe_dict()`.

---

### [LG-001] loguru: Logger.add() — multiple values for argument 'sink'
**Дата:** 2026-04  
**Фаза:** 1.1 — Logger

**Ошибка:**
```
TypeError: Logger.add() got multiple values for argument 'sink'
```

**Причина:**  
В вызове `logger.add()` передавался и позиционный аргумент (путь к файлу как строка) и именованный `sink=lambda ...` одновременно. Python интерпретирует первый позиционный как `sink`, а потом видит ещё один `sink=` — конфликт.

**Решение:**  
Убрать `sink=lambda` и использовать встроенный параметр `serialize=True` для JSON-вывода в файл. Путь к файлу передаётся первым позиционным аргументом:
```python
logger.add(log_path, serialize=True, rotation="1 day", ...)
```

**Файл:** `src/logger.py` — функция `setup_logging()`.

---

### [EX-001] extract_date_from_stem: неправильное извлечение даты для месячных файлов
**Дата:** 2026-04  
**Фаза:** 1.3 — Parquet Converter

**Ошибка:**  
Не исключение, но silent data corruption: для файла `BTCUSDT-fundingRate-2024-01.csv` извлекалась строка `"te-2024-01"` вместо `"2024-01"`. В результате создавалась директория `date=te-2024-01` вместо `date=2024-01`.

**Причина:**  
Оригинальная реализация использовала `stem[-10:]` для вырезания даты. Это работает для дневных файлов (`BTCUSDT-4h-2024-01-01`, длина суффикса = 10). Для месячных файлов (`BTCUSDT-fundingRate-2024-01`, суффикс = 7) `[-10:]` захватывает лишние символы: `"fundingRate-2024-01"[-10:]` = `"te-2024-01"`.

**Решение:**  
Заменить срез на regex с конца строки, обрабатывающий оба формата (дневной и месячный):
```python
import re as _re
_DATE_RE = _re.compile(r"(\d{4}-\d{2}(?:-\d{2})?)$")

def extract_date_from_stem(stem: str) -> str:
    m = _DATE_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot extract date from stem: {stem!r}")
    return m.group(1)
```
Дополнительно: удалены ошибочные директории `date=te-2024-01` и произведена повторная конвертация funding_rate.

**Файл:** `src/ingestion/parquet_converter.py:189–203` — функция `extract_date_from_stem()`.

---

### [DL-001] binance_downloader: _csv_valid не находит данные в CSV с заголовком
**Дата:** 2026-04  
**Фаза:** 1.2 — Binance Downloader

**Ошибка:**  
Тест `test_download_klines` падал с `AssertionError` — скачанный файл считался невалидным несмотря на корректное содержимое.

**Причина:**  
Функция `_csv_valid()` проверяла только **первую строку** файла на наличие цифр. Klines CSV начинается с заголовка `open_time,open,high,...` — первый символ `o`, не цифра. Функция возвращала `False` для валидных файлов.

**Решение:**  
Изменить проверку — сканировать **все строки** в поисках хотя бы одной, начинающейся с цифры:
```python
def _csv_valid(path: Path) -> bool:
    with path.open() as f:
        for line in f:
            if line and line[0].isdigit():
                return True
    return False
```

**Файл:** `tests/test_binance_downloader.py` — вспомогательная функция `_csv_valid()`.

---

### [DS-001] DataStore: funding_rate — несовпадение схемы с реальным API
**Дата:** 2026-04  
**Фаза:** 1.5 — Funding Rate Download

**Ошибка:**  
После конвертации `funding_rate` CSV → Parquet, запрос через `DataStore.get_funding_rate()` возвращал пустой DataFrame, хотя файлы существовали.

**Причина:**  
`FUNDING_SCHEMA` и `_CONFIGS["funding_rate"]` были написаны под формат Binance Data Portal (колонки `calc_time`, `funding_interval_hours`, `last_funding_rate`, `mark_price`). Реальные данные были скачаны через REST API `fapi.binance.com/fapi/v1/fundingRate`, который возвращает совершенно другие колонки: `fundingTime`, `fundingRate`, `markPrice`, `symbol`. CSV содержал правильные данные, но конвертер пытался читать несуществующие колонки — все значения становились `null`, DataFrame пустел.

**Решение:**  
Обновить `FUNDING_SCHEMA` и `_CONFIGS["funding_rate"]`:
```python
FUNDING_SCHEMA = {
    "fundingTime": pl.Int64,
    "fundingRate": pl.Float64,
    "markPrice":   pl.Float64,
    "symbol":      pl.Utf8,
}
# _TypeConfig: timestamp_col="fundingTime", sort_col="fundingTime"
```
Удалить ошибочные Parquet-файлы, запустить повторную конвертацию. Итог: 2193 строки на символ.

**Файл:** `src/ingestion/parquet_converter.py:46–51` (`FUNDING_SCHEMA`) и `src/ingestion/parquet_converter.py:114–124` (`_CONFIGS["funding_rate"]`).

---

## Известные баги библиотек

| Библиотека | Версия | Баг | Workaround |
|---|---|---|---|
| cryptofeed | 2.4.1 | `Task destroyed but pending` при shutdown через `start_loop=False` | Принято как косметика; патч требует изменений в `connection_handler.py` |
| cryptofeed | 2.4.1 | Типы `Trade`/`Funding`/`OrderBook` скомпилированы Cython — `inspect` не работает | `obj = Type.__new__(Type); dir(obj)` |
| pydantic-settings | 2.3.4 | `self.model_fields` deprecated — только через класс | `self.__class__.model_fields` |
| polars | 0.20.31 | `read_csv(dtypes=...)` deprecated | Переименовать в `schema_overrides=` |
| polars | 0.20.31 | `read_parquet` авто-определяет Hive-партиции → DuplicateError если колонка есть в файле | `hive_partitioning=False` |
| loguru | 0.7.2 | `logger.add(path, sink=lambda)` — конфликт позиционного и именованного `sink` | Убрать `sink=`, использовать `serialize=True` |

---

## Полезные команды для отладки

```python
# Список всех имён в Cython-модуле (когда import конкретного имени падает)
python -c "import cryptofeed.types as t; print(dir(t))"

# Атрибуты Cython-объекта без вызова __init__
python -c "
from cryptofeed.types import Trade, Funding, OrderBook, _OrderBook
for cls in [Trade, Funding, OrderBook, _OrderBook]:
    obj = cls.__new__(cls)
    pub = [a for a in dir(obj) if not a.startswith('_')]
    print(f'{cls.__name__}: {pub}')
"

# Поиск функции/класса по всем подмодулям библиотеки
python -c "
import pkgutil, importlib, cryptofeed
target = 'setup_signal_handlers'
for mod in pkgutil.walk_packages(cryptofeed.__path__, prefix='cryptofeed.'):
    try:
        m = importlib.import_module(mod.name)
        if hasattr(m, target):
            print(f'Found in: {mod.name}')
    except Exception:
        pass
"

# Проверить реальные колонки CSV перед написанием схемы
python -c "
import polars as pl
df = pl.read_csv('path/to/file.csv', n_rows=2)
print(df.columns)
print(df.dtypes)
"

# Проверить содержимое Parquet без Hive-интерференции
python -c "
import polars as pl
df = pl.read_parquet('path/to/part-0.parquet', hive_partitioning=False)
print(df.schema)
print(df.head(3))
"

# Интроспекция сигнатуры FeedHandler без документации
python -c "
import inspect
from cryptofeed import FeedHandler
from cryptofeed.exchanges import BinanceFutures
print('add_feed:', inspect.signature(FeedHandler.add_feed))
print('run:     ', inspect.signature(FeedHandler.run))
print('BF init: ', inspect.signature(BinanceFutures.__init__))
"

# Проверить какие каналы/строки используются как define-константы
python -c "
from cryptofeed.defines import TRADES, L2_BOOK, FUNDING
print(repr(TRADES), repr(L2_BOOK), repr(FUNDING))
# -> 'trades' 'l2_book' 'funding'
"
```

---

## Фаза 1 — Шаг 1.7: Data Quality

---

### [DQ-001] clock_drift: false positive ~86 399 417 ms (~24 часа) для agg_trades
**Дата:** 2026-04  
**Фаза:** 1.7 — Data Quality

**Ошибка** (первый запуск `scripts/check_data_quality.py`):
```
❌ BTCUSDT/agg_trades: drift=86399417ms
❌ ETHUSDT/agg_trades: drift=86397069ms
❌ SOLUSDT/agg_trades: drift=86393323ms
Pass: 12/15 checks
```

**Причина:**  
`check_clock_drift` собирал первые 1000 строк из каждого дневного файла и конкатенировал все timestamps в один плоский список:
```python
all_ts: list[int] = []
for f in files:
    df = scan_parquet(f).head(1000).collect()
    all_ts.extend(df["transact_time"].to_list())   # межфайловая граница!

ts_series = pl.Series("ts", all_ts)
diffs = ts_series.diff().drop_nulls().abs()
drift_ms_max = int(diffs.max())   # ← захватывает ~24h между днями
```
Diff между последней строкой файла дня N и первой строкой файла дня N+1 = ~86 400 000 ms (~24 часа). Именно этот межфайловый разрыв попадал в `max()` и перекрывал реальный интервал между сделками (~5-6 секунд).

**Решение:**  
Вычислять `diff()` отдельно внутри каждого файла, не пересекая границы файлов:
```python
max_intra_drift = 0
for f in files:
    df = scan_parquet(f).head(AGG_TRADES_SAMPLE).collect()
    ts = df["transact_time"]
    diffs = ts.diff().drop_nulls().abs()
    if len(diffs) > 0:
        file_max = int(diffs.max())
        if file_max > max_intra_drift:
            max_intra_drift = file_max
```
После исправления: BTCUSDT=5708ms, ETH=6321ms, SOL=6671ms (нормальный межторговый интервал).

**Файл:** `src/ingestion/data_quality.py` — метод `DataQualityChecker.check_clock_drift()`, полная замена тела цикла.

**Что не трогали:** `check_completeness`, `check_gaps`, `check_data_integrity`, `row_passes`, CLI, все 19 тестов.

---

### [DQ-002] row_passes применял 50ms порог к stored data — семантическая ошибка
**Дата:** 2026-04  
**Фаза:** 1.7 — Data Quality

**Ошибка** (второй запуск после DQ-001):
```
❌ BTCUSDT/agg_trades: drift=5708ms
❌ ETHUSDT/agg_trades: drift=6321ms
❌ SOLUSDT/agg_trades: drift=6671ms
Pass: 12/15 checks
```

**Причина:**  
В `row_passes()` было условие:
```python
if clock_drift.get("drift_ms_max", 0) >= THRESHOLD_DRIFT_MS:  # 50ms
    return False
```
Порог 50ms взят из master-спецификации («Clock drift < 50ms»), но там он описывает задержку live-фида: разницу между `receipt_timestamp` и `exchange_timestamp`. Для хранимых данных `drift_ms_max` означает совершенно другое — **максимальный интервал между соседними сделками** (market liquidity metric). Пауза 5–6 секунд между сделками на любом фьючерсном рынке абсолютно нормальна. Применять к ней порог 50ms некорректно.

**Решение:**  
Удалить проверку `drift_ms_max >= threshold` из `row_passes`. Для хранимых данных только `is_monotonic=False` является признаком нарушения:
```python
if clock_drift is not None:
    if not clock_drift.get("is_monotonic", True):
        return False
# drift_ms_max остаётся информационным полем в таблице
```

**Файл:** `src/ingestion/data_quality.py` — функция `row_passes()`, удалена одна ветка `if`.

**Что не трогали:** `check_clock_drift` (уже исправлен в DQ-001), все остальные методы, CLI-скрипт, 18 из 19 тестов.

---

### [DQ-003] Тест сломался после исправления DQ-002
**Дата:** 2026-04  
**Фаза:** 1.7 — Data Quality

**Ошибка** (pytest после изменения `row_passes`):
```
FAILED tests/test_data_quality.py::test_row_passes_fails_on_high_drift

AssertionError: assert True is False
  + where True = row_passes('BTCUSDT', 'agg_trades',
      {'completeness_pct': 100.0},
      {'gap_count': 0, 'skipped': True},
      {'anomaly_count': 0, 'is_valid': True, 'null_count': 0},
      {'drift_ms_max': 100, 'is_monotonic': True})

tests/test_data_quality.py:366: AssertionError
1 failed, 18 passed in 1.61s
```

**Причина:**  
Тест `test_row_passes_fails_on_high_drift` проверял: `drift_ms_max=100 >= 50` → должен вернуть `False`. После исправления DQ-002 это условие было удалено из `row_passes` — функция теперь возвращает `True` для данного входа. Тест задокументировал поведение, которое намеренно изменилось.

**Решение:**  
Удалить исходный тест и заменить двумя, отражающими реальную семантику:
```python
def test_row_passes_fails_on_non_monotonic():
    # is_monotonic=False — единственный hard fail для clock_drift stored data
    non_monotonic = {"is_monotonic": False, "drift_ms_max": 5}
    assert row_passes(..., non_monotonic) is False

def test_row_passes_large_drift_ms_is_informational():
    # 6000ms между сделками — нормальный рынок, не сбой качества данных
    large_drift = {"is_monotonic": True, "drift_ms_max": 6_000}
    assert row_passes(..., large_drift) is True
```
Итог: 20 тестов вместо 19, все зелёные.

**Файл:** `tests/test_data_quality.py` — удалён `test_row_passes_fails_on_high_drift`, добавлено два новых теста.

**Что не трогали:** `src/ingestion/data_quality.py` (код уже исправлен), 18 остальных тестов.

---

### Известные особенности data_quality модуля

| Поле | Тип данных | Значение | Порог применим? |
|---|---|---|---|
| `drift_ms_max` | agg_trades stored | max gap между сделками в выборке | ❌ Нет (рыночная характеристика) |
| `drift_ms_max` | live feed | `receipt_ts - exchange_ts` | ✅ Да, < 50ms |
| `is_monotonic` | agg_trades | порядок transact_time в файле | ✅ Всегда должен быть True |
| `gap_count` | klines/funding | пропуски > ожидаемого интервала | ✅ Должен быть 0 |

---

*Последнее обновление: 2026-04 | Фаза: 1.7 — Data Quality*

---

## Фаза 2 — Шаг 2.1–2.2: Nautilus BacktestEngine

Реализация: `src/execution/data_catalog.py`, `src/execution/strategies/baseline_strategy.py`,
`src/execution/backtest_runner.py`, `scripts/run_backtest.py`, `tests/test_backtest_engine.py`.  
Итог: 5 ошибок → 26/26 тестов зелёные.

---

### [NT-001] Polars: IsADirectoryError при чтении partitioned Parquet-директории
**Дата:** 2026-04  
**Фаза:** 2.1 — Data Catalog

**Ошибка:**
```
IsADirectoryError: expected a file path;
'/mnt/hdd/AtomiCortex/data/features/exchange=BINANCE_UM/
symbol=BTCUSDT/klines_4h/' is a directory
```

**Причина:**  
Данные хранятся в Hive-партициях: `klines_4h/date=YYYY-MM-DD/part-0.parquet`.  
`pl.read_parquet()` ожидает путь до конкретного файла — передача директории вызывает ошибку ещё до выполнения запроса.

**Решение:**  
Перейти на `pl.scan_parquet()` с glob-паттерном `**/*.parquet` и lazy evaluation:
```python
# Было:
df = pl.read_parquet('/mnt/hdd/.../klines_4h/')

# Стало:
df = (
    pl.scan_parquet('/mnt/hdd/.../klines_4h/**/*.parquet', hive_partitioning=False)
    .filter((pl.col("open_time") >= start_ms) & (pl.col("open_time") < end_ms))
    .collect()
)
```

**Файл:** `src/execution/data_catalog.py:90–102`  
**Что не трогали:** структура директорий данных, Parquet-файлы, Фаза 1 pipeline.

---

### [NT-002] Polars: DuplicateError — конфликт Hive-схемы и колонок файла
**Дата:** 2026-04  
**Фаза:** 2.1 — Data Catalog

**Ошибка:**
```
polars.exceptions.DuplicateError: invalid Hive partition schema

Extending the schema with the Hive partition schema would create
duplicate fields.

This error occurred with the following context stack:
    [1] 'parquet scan' failed
    [2] 'slice' input failed to resolve
```

**Причина:**  
Путь `exchange=BINANCE_UM/symbol=BTCUSDT/` содержит Hive-сегменты. Polars по умолчанию
(`hive_partitioning=True`) парсит их как дополнительные колонки. Но в самих `.parquet`-файлах
уже есть колонка `symbol` — возникает дублирование при расширении схемы.

**Решение:**  
`hive_partitioning=False`. Фильтрация по дате — через колонку `open_time` (unix ms) внутри файлов:
```python
pl.scan_parquet(pattern, hive_partitioning=False)
```

**Файл:** `src/execution/data_catalog.py:93` (`load_bar_data`) и `:128` (`load_trade_data`)  
**Что не трогали:** структура Parquet, колонки в файлах, код Фазы 1.

---

### [NT-003] Nautilus: BarType — неверный kwarg `spec` и int вместо AggregationSource enum
**Дата:** 2026-04  
**Фаза:** 2.1 — Data Catalog

**Ошибка:**
```
TypeError: __init__() got an unexpected keyword argument 'spec'
  File "nautilus_trader/model/data.pyx", line 1155, in
  nautilus_trader.model.data.BarType.__init__
```

**Причина:**  
Два независимых несоответствия в одном вызове:

1. Именованный аргумент — `bar_spec`, а не `spec`  
2. `aggregation_source` ожидает `AggregationSource` enum, не целое число  

`inspect.signature(BarType.__init__)` возвращает `(self, /, *args, **kwargs)` — Cython не
экспортирует параметры. Правильные имена раскрывает только `help(BarType)`.

```
# help(BarType) показывает:
BarType(InstrumentId instrument_id,
        BarSpecification bar_spec,
        AggregationSource aggregation_source=AggregationSource.EXTERNAL)
```

**Решение:**
```python
# Было:
bar_type = BarType(
    instrument_id=instrument_id,
    spec=BarSpecification(4, BarAggregation.HOUR, PriceType.LAST),
    aggregation_source=2,
)

# Стало:
from nautilus_trader.model.enums import AggregationSource
bar_spec = BarSpecification(4, BarAggregation.HOUR, PriceType.LAST)
bar_type = BarType(instrument_id, bar_spec, AggregationSource.EXTERNAL)
```

**Правило:** для Cython-классов Nautilus всегда использовать `help()`, а не `inspect.signature()`.

**Файл:** `src/execution/data_catalog.py:74–76`  
**Что не трогали:** `BarSpecification` (аргументы корректны), `InstrumentId`, `PriceType`.

---

### [NT-004] `inspect.signature` возвращает строку вместо класса — `from __future__ import annotations`
**Дата:** 2026-04  
**Фаза:** 2.2 — BacktestRunner

**Ошибка:**
```
TypeError: 'str' object is not callable
  src/execution/backtest_runner.py:119: TypeError
```

Стектрейс указывал на строку:
```python
config_type = sig.parameters["config"].annotation   # -> "BuyAndHoldConfig" (str!)
strategy = strategy_class(config=config_type(**full_strategy_config))
#                                 ^^^^^^^^^^ вызов строки → TypeError
```

**Причина:**  
В `baseline_strategy.py` объявлено `from __future__ import annotations` (PEP 563).
Это делает **все** аннотации в модуле ленивыми строками — они не вычисляются при определении
класса. `inspect.signature().parameters["config"].annotation` возвращает именно строку
`"BuyAndHoldConfig"`, а не класс. Попытка вызвать строку как функцию → `TypeError`.

Это подводный камень: в модулях без `from __future__ import annotations` тот же код работает.

**Решение:**  
`typing.get_type_hints()` вычисляет строки в контексте исходного модуля через `__globals__`:
```python
# Было:
import inspect
sig = inspect.signature(strategy_class.__init__)
config_type = sig.parameters["config"].annotation   # "BuyAndHoldConfig" — строка

# Стало:
import typing
hints = typing.get_type_hints(strategy_class.__init__)
config_type = hints["config"]                        # <class 'BuyAndHoldConfig'>
```

**Файл:** `src/execution/backtest_runner.py:107–111`  
**Что не трогали:** `baseline_strategy.py` (строка `from __future__ import annotations` оставлена),
сигнатура метода `run(strategy_class, strategy_config: dict)`.

---

### [NT-005] pytest.approx: tolerance ±$1.0 поглотил реальную разницу equity $0.50
**Дата:** 2026-04  
**Фаза:** 2.2 — Tests

**Ошибка:**
```
AssertionError: 10000.50585595 != 10000.0 ± 1.0e+00
 +  where 10000.50585595 = BacktestResult(...).end_equity
 +  and   10000.0 ± 1.0e+00 = pytest.approx(10000.0, rel=0.0001)
tests/test_backtest_engine.py:197: AssertionError
```

**Причина:**  
`pytest.approx(start_equity, rel=1e-4)` вычисляет допуск как `rel × value = 1e-4 × 10 000 = ±1.0 USDT`.
Реальное изменение equity составило $0.51 — меньше допуска, тест прошёл как «приблизительно равны».

Корень: тестовая позиция 0.001 BTC при $42 300 = $42.3 номинала. BTC вырос на ~1.25% за период
→ P&L $0.53, минус комиссии $0.04 = $0.49. `leverage=5` не умножает `trade_size` автоматически —
он лишь определяет доступное плечо; фактический объём всегда равен переданному `trade_size`.

**Решение:**  
Проверять абсолютную разницу напрямую:
```python
# Было:
assert result.end_equity != pytest.approx(result.start_equity, rel=1e-4)

# Стало:
assert abs(result.end_equity - result.start_equity) > 0.01
```

**Файл:** `tests/test_backtest_engine.py:197`  
**Что не трогали:** логику расчёта equity, `BacktestRunner`, 25 остальных тестов.

---

### Итоговая таблица Фазы 2 (Шаг 2.1–2.2)

| ID | Файл | Суть | Инструмент диагностики |
|---|---|---|---|
| NT-001 | `data_catalog.py:90` | `read_parquet` на директорию | Traceback |
| NT-002 | `data_catalog.py:93,128` | Hive-схема дублирует колонку `symbol` | Polars DuplicateError |
| NT-003 | `data_catalog.py:74` | BarType: kwarg `spec` → `bar_spec`, int → enum | `help(BarType)` |
| NT-004 | `backtest_runner.py:107` | `inspect.signature` возвращает строку из-за PEP 563 | TypeError: str not callable |
| NT-005 | `tests/test_backtest_engine.py:197` | `pytest.approx rel=1e-4` поглощает $0.50 разницу | AssertionError |

**Общий паттерн Nautilus Trader:** для всех Cython-классов (`BarType`, `CryptoPerpetual`,
`BacktestEngine`, etc.) единственный надёжный способ узнать параметры — `help(ClassName)`.
`inspect.signature()` возвращает `(self, /, *args, **kwargs)` и бесполезен.

---

*Последнее обновление: 2026-04 | Фаза: 2.2 — BacktestEngine*
