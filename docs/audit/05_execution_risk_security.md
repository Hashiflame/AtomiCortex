# 05 — Исполнение, риск, доступность, безопасность

> TL;DR: два системных дефекта исполнения. (1) Доступность: боты умеют зависать в
> RUNNING без данных, а watchdog не запущен — сегодняшний инцидент это доказал.
> (2) Экономика: торгуется не то событие, которое предсказывает модель (геометрия
> SL/TP/holding не совпадает с барьерами лейбла), а cost-model завышает слиппедж ~19×.

## 5.1 🔴 A1. STOP-SHIP: zombie-RUNNING + неработающий watchdog
**Факты (2026-07-02):**
- 06:36:07 — все сервисы перезапущены (деплой/рестарт);
- 06:36:17 — оба бота: `HttpError(… /fapi/v1/exchangeInfo)` на `_connect`
  (сеть в момент старта; сейчас `curl fapi.binance.com/fapi/v1/ping` → 200 за 0.1 c);
- 06:36:47 — `Timed out (30.0s)… DataEngine.check_connected() == False` →
  **`TradingNode: RUNNING`** — и с этого момента ни одной строки от стратегии:
  нет подписок, нет баров, нет heartbeat;
- `systemctl is-active atomicortex-watchdog` → **inactive** (enabled),
  `atomicortex-watchdog-15m` → disabled+inactive. Dead-man's switch не работает.
**Почему важно:** в live-режиме это часы открытых позиций без управления. В paper —
беззвучная остановка сбора доказательств (см. A2).
**Фикс (3 маленьких PR):**
1. `scripts/run_live.py`: после `node.run()`-startup проверять `check_connected()`;
   если false — `sys.exit(1)` (systemd `Restart=always` перезапустит с backoff).
   Плюс исправить unit: `StartLimitIntervalSec` лежит в секции `[Service]` и
   игнорируется systemd (журнал это прямо пишет) — перенести в `[Unit]`.
2. Запустить и заэнейблить оба watchdog-сервиса; алерт в Telegram при stale heartbeat
   (код heartbeat/watchdog уже есть — он просто не активен).
3. Reconciler в стратегии: `Reconciler schedule failed (non-fatal): no running event loop`
   повторяется **каждые 15 минут весь июнь** (`ml_strategy.py:493-533` — clock-таймер
   Nautilus зовёт async-код вне loop). Либо чинить получение loop (Nautilus
   `self.clock`/`asyncio.get_running_loop()` в actor-контексте), либо признать, что
   сверка живёт только в `atomicortex-reconciler.timer` (он active), и убрать мёртвый код.
**Источник:** Breck et al. 2017, *ML Test Score* — Monitor-раздел (dead-man's switch,
data-feed monitor); Nautilus Trader docs/issues о reconnect-политике клиентов.

## 5.2 🔴 A6. Геометрия сделки ≠ предсказываемое событие
**Файл:строка:** лейбл: pt=1.0·ATR / sl=0.8·ATR / h=6 баров по close
(`retrain_v3_selected_results.txt`, `triple_barrier.py`); исполнение:
`RiskConfig.atr_stop_multiplier=1.5` (SL=1.5·ATR), `rr_ratio=1.5` (TP=2.25·ATR),
`risk_engine.py:264-300`; time-exit **отсутствует** (в `ml_strategy.py` нет выхода по
количеству баров; факт: сигнал #5 держался 3.5 суток = 21 бар при h=6).
**Проблема:** модель оценивает P(близко: +1.0·ATR раньше −0.8·ATR в ≤6 баров).
Live-сделка выигрывает/проигрывает совсем другое событие: ±(2.25/1.5)·ATR без
ограничения времени, с интрабарными срабатываниями. Даже идеально откалиброванная
модель не обязана давать положительный P&L в такой сделке — это разные случайные
величины. Плюс asymmetry: лейбл-настройка сильнее штрафует за short-window шум.
**Фикс:** привести исполнение к лейблу: SL=0.8·ATR, TP=1.0·ATR, принудительный выход
на закрытии 6-го бара (или наоборот — перелейблить под 1.5/2.25, но тогда h надо
увеличить и переобучить). Одно из двух, и закрепить инвариант тестом.
**Источник:** AFML ch.3 (лейблы должны отражать торгуемую стратегию); Carver,
*Systematic Trading* — согласованность forecast horizon и trade management.
**Честный outcome:** это кандидат №1 на объяснение разрыва «PF 1.38 в eval → PF<1 в paper»
вместе с A3; количественно не обещаю — после фикса paper покажет.

## 5.3 🟠 A9. Cost model: σ_annual в sqrt-импакте
**Файл:строка:** `cost_model.py:75`: `slippage_fraction = 0.5 · volatility · √(Q/V)`
c `volatility=0.60` (годовая, `risk_engine.py:57`).
**Вывод из первоисточника:** закон импакта I = Y·σ_daily·√(Q/ADV), Y≈0.5–1
(Tóth et al. 2011; Kyle & Obizhaeva 2016; подтверждён для BTC — Donier & Bonart 2014,
arXiv:1412.4503). σ — **дневная** волатильность. σ_annual/σ_daily = √365 ≈ 19.1 —
во столько раз завышен слиппедж. Для $10k при V=$1B: код даёт ~9.5 bps/сторона,
реалистично ~0.5 bps.
**Последствие:** `_check_expected_return` (`risk_engine.py:354-357`) требует
`ATR_bps ≥ max(15, 3×round_trip)`; завышенный round_trip (~28 bps → порог 84 bps)
блокирует сделки при нормальном 4H ATR% 0.4–0.8%. Фильтр «душит» сигналы по фиктивной
причине и одновременно (см. A8) в eval издержки вообще не учитываются — worst of both.
**Фикс:** `sigma_daily = volatility_annual/√365` в формуле; Y оставить 0.5; добавить
тест на порядок величины.

## 5.4 🟠 A10. Paper-исполнение расходится с биржей
Факт: 2026-06-02 20:00 `submit_order` отвергнут (`-2019 Margin is insufficient`,
testnet), но `signals_log` содержит сигнал #8 как открытый и закрытый с pnl −2.30%.
Paper-учёт (SignalBridge/reconciler_signals) живёт своей жизнью от биржевого стейта.
Для валидации это допустимо только если объявить paper «signal-based simulation»
и никогда не смешивать с биржевыми fill'ами; сейчас же метрики выглядят как
исполненные сделки. Фикс: сигнал, чей ордер отвергнут, помечать `rejected`, не `open`.

## 5.5 🟡 A19. Незадействованный regime-sizing и hardcoded vol-фильтр
- `RegimeState.position_size_multiplier()` / `_REGIME_SIZE_MULT`
  (`regime_detector.py:181-213`) не вызываются нигде (`grep` — только circuit_breaker
  имеет свой аналог, тоже не вызываемый из сайзинга). `calculate_position_size`
  (`risk_engine.py:233-262`) игнорирует regime. Гипотеза #13 подтверждена.
- `_check_volatility` (`risk_engine.py:387-401`): `atr_pct > 2.0 × 0.01` — «средний ATR»
  захардкожен как 1%: на 4H BTC (ATR%≈0.5–0.9%) фильтр почти не срабатывает, на SOL
  15m — рубит хвост. Гипотеза #12 подтверждена. Фикс: сравнивать с rolling-медианой
  atr_pct того же символа/TF (данные уже в буфере).

## 5.6 🟡 A18. Pickle
`use_native_save` (H13) есть, но не включён ни одним скриптом (`grep` — пусто);
`ml_strategy_15m.py:261` грузит pickle напрямую, минуя `load_model_bundle`.
Риск RCE локально ограничен (файлы свои), но version-brittleness реальна
(LightGBM/numpy апгрейд молча ломает загрузку). Фикс: включить native save в
retrain-скриптах + перевести 15m-загрузку на `load_model_bundle`.

## 5.7 Безопасность (быстрый проход)
- **API** (`src/api/main.py:109-215`): X-API-Key с `secrets.compare_digest`, эфемерный
  ключ при отсутствии env, CORS-allowlist, rate limiter — ✅ прилично.
- **Telegram Stars** (`payments_stars.py`): проверка payload↔payer, суммы, идемпотентность
  по `telegram_payment_charge_id` — ✅; **Crypto** (`payments_crypto.py:256`) — dedup
  повторной активации есть. RBAC: role-check на owner-callbacks добавлен (a8f4807) ✅.
  Не проверялось глубоко: восстановление подписки при race renewal — в план тестов.
- `.env` в git? — `git check-ignore .env` подтвердить при следующем проходе; ключи
  в журналах не замечены.

## 5.8 Watchlist (не баги, но следить)
- `_fetch_taker_buy_volume_for_bar` — синхронный REST-вызов в `on_bar` (латентность до 5с
  таймаута в горячем пути).
- `equity` для DSR/статистики берётся из PortfolioTracker с известными оговорками
  прошлого ревью (peak на close_position); H5-фикс mark-to-market в on_bar снял главное.
- Funding в backtest теперь биллится по held-часам (262a5d6) — сверено, корректно.
