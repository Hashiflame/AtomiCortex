"""
AtomiCortex — External Watchdog.

Runs as a **separate** process (or on a different server).  Checks the Redis
heartbeat key every ``check_interval`` seconds.  If the bot heartbeat is
missing for longer than ``max_silence_seconds``:

1. Send a Telegram alert.
2. Emergency-close all open positions via Binance REST API.
3. Cancel all open orders.

**Design principle:** the watchdog deliberately does NOT import
``nautilus_trader`` or any heavy trading framework.  It uses only
``aiohttp`` (REST) + ``redis.asyncio`` so that it starts instantly and
has no shared failure mode with the trading bot.

Phase 4 — Step 4.6.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from src.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Binance API URLs
# ---------------------------------------------------------------------------

# H17: canonical map from --strategy short name to (heartbeat_key,
# service_name) so the watchdog launcher cannot drift from the keys the
# strategies actually publish to. Exported so tests / external tools can
# share the same source of truth.
STRATEGY_HEARTBEAT_KEYS: dict[str, str] = {
    "4h":  "atomicortex:heartbeat",
    "1h":  "bot_1h_heartbeat",
    "15m": "bot_15m_heartbeat",
}
DEFAULT_HEARTBEAT_KEY: str = STRATEGY_HEARTBEAT_KEYS["4h"]


_BINANCE_URLS: dict[str, dict[str, str]] = {
    "testnet": {
        "base": "https://testnet.binancefuture.com",
        "position_risk": "/fapi/v2/positionRisk",
        "order": "/fapi/v1/order",
        "all_open_orders": "/fapi/v1/allOpenOrders",
    },
    "live": {
        "base": "https://fapi.binance.com",
        "position_risk": "/fapi/v2/positionRisk",
        "order": "/fapi/v1/order",
        "all_open_orders": "/fapi/v1/allOpenOrders",
    },
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class WatchdogConfig:
    """Configuration for the Watchdog process."""

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    binance_api_key: str = ""
    binance_api_secret: str = ""
    trading_mode: str = "testnet"      # testnet / live
    heartbeat_key: str = "atomicortex:heartbeat"
    # Phase 5 isolation: scope this watchdog instance to ONE service.
    # ``symbol`` empty  → legacy behaviour: emergency-close ALL positions
    #                     (the running 4H watchdog is unchanged).
    # ``symbol`` set    → only close / cancel that symbol's positions, so
    #                     a dead 15m bot never touches the 4H bot's book.
    # ``service_name``  → label for logs / alerts only.
    symbol: str = ""
    service_name: str = "4h"
    check_interval: int = 15           # seconds
    max_silence_seconds: int = 60
    max_bar_silence_seconds: int = 0
    startup_bar_grace_seconds: int = 900
    alert_cooldown_seconds: int = 900
    telegram_token: str = ""
    telegram_admin_id: str = ""


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

class Watchdog:
    """External watchdog that monitors the bot heartbeat and performs
    emergency position closure when the bot becomes unresponsive.
    """

    def __init__(self, config: WatchdogConfig) -> None:
        self._config = config
        self._redis: Any = None
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._incidents: list[dict[str, Any]] = []

        self._last_alert_ts: float = 0.0
        self._incident_active: bool = False
        self._last_close_found_positions: bool = True
        self._legacy_format_logged: bool = False

        # Resolve base URL
        mode = config.trading_mode.lower()
        urls = _BINANCE_URLS.get(mode, _BINANCE_URLS["testnet"])
        self._base_url: str = urls["base"]
        self._urls = urls

        # Normalised symbol scope ("" = all, legacy). Strips venue / -PERP
        # so "BTCUSDT-PERP.BINANCE" and "BTCUSDT" both match Binance's
        # positionRisk "symbol" field ("BTCUSDT").
        self._scope_symbol: str = ""
        if config.symbol:
            s = config.symbol.split(".")[0]
            self._scope_symbol = s.split("-")[0].upper()

        _log.info(
            "Watchdog created | service={svc} | scope={scope} | "
            "mode={mode} | silence_limit={sl}s | check_interval={ci}s",
            svc=config.service_name,
            scope=self._scope_symbol or "ALL",
            mode=config.trading_mode,
            sl=config.max_silence_seconds,
            ci=config.check_interval,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the watchdog check loop."""
        if self._running:
            _log.warning("Watchdog already running")
            return

        self._redis = await self._connect_redis()
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        _log.info("Watchdog started")

    async def stop(self) -> None:
        """Stop the watchdog loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

        _log.info(
            "Watchdog stopped | incidents={n}",
            n=len(self._incidents),
        )

    async def emergency_close_all(self) -> dict[str, Any]:
        """Emergency-close all open positions and cancel all orders via
        Binance REST API (aiohttp, NOT WebSocket).

        Returns a summary dict of actions taken.
        """
        import aiohttp

        cfg = self._config
        result: dict[str, Any] = {
            "positions_closed": [],
            "orders_cancelled": False,
            "errors": [],
        }

        _log.warning("EMERGENCY CLOSE ALL — starting")

        try:
            async with aiohttp.ClientSession() as session:
                # 1. GET position risk
                positions = await self._signed_get(
                    session, self._urls["position_risk"],
                )
                if positions is None:
                    result["errors"].append("Failed to fetch positions")
                    return result

                # H15: cancel BEFORE close.
                #
                # Old order was MARKET close → then cancel. If a resting
                # SL fired during the close, the position was closed
                # twice — the second leg flipped the account into the
                # reverse direction. The correct ordering kills the
                # resting orders first so only our reduceOnly MARKET
                # actually moves the book.
                #
                # 2. Build the set of in-scope symbols once.
                symbols_with_positions = {
                    pos.get("symbol") for pos in positions
                    if abs(float(pos.get("positionAmt", 0))) > 1e-10
                    and (
                        not self._scope_symbol
                        or str(pos.get("symbol", "")).upper() == self._scope_symbol
                    )
                }

                # 3. Cancel all open orders for each symbol FIRST.
                #    Fail-soft: a failed cancel still proceeds to close —
                #    leaving the account open is the bigger risk.
                for symbol in symbols_with_positions:
                    try:
                        cancel_result = await self._signed_delete(
                            session,
                            self._urls["all_open_orders"],
                            {"symbol": symbol},
                        )
                        if cancel_result is not None:
                            _log.warning(
                                "EMERGENCY CANCEL ORDERS | {sym}",
                                sym=symbol,
                            )
                        else:
                            result["errors"].append(
                                f"Cancel orders failed for {symbol} — "
                                "proceeding with close anyway"
                            )
                    except Exception as exc:
                        result["errors"].append(
                            f"Cancel orders raised for {symbol}: {exc} — "
                            "proceeding with close anyway"
                        )

                result["orders_cancelled"] = True

                # 4. Brief pause so the exchange has finished processing
                #    the cancellations before we send the reduceOnly
                #    MARKETs. Half a second matches the upper bound of
                #    Binance's documented order-state propagation.
                await asyncio.sleep(0.5)

                # 5. Close positions with |positionAmt| > 0.
                for pos in positions:
                    amt = float(pos.get("positionAmt", 0))
                    if abs(amt) < 1e-10:
                        continue

                    symbol = pos.get("symbol", "UNKNOWN")

                    # Phase 5: isolated watchdog only closes its own
                    # symbol — a dead 15m bot must not flatten the 4H book.
                    if self._scope_symbol and symbol.upper() != self._scope_symbol:
                        _log.info(
                            "Skip {sym} — out of scope ({scope})",
                            sym=symbol, scope=self._scope_symbol,
                        )
                        continue
                    side = "SELL" if amt > 0 else "BUY"
                    qty = str(abs(amt))

                    # H16: try a Limit-IOC at markPrice ± 0.3% FIRST so
                    # the close doesn't eat 1-5% of slippage on a thin
                    # book (exactly the kind of market that triggered
                    # the watchdog in the first place). MARKET stays as
                    # the unconditional fallback so an unfilled IOC
                    # cannot leave the account exposed.
                    ioc_filled = await self._try_limit_ioc_close(
                        session=session, pos=pos, symbol=symbol,
                        side=side, qty=qty, result=result,
                    )
                    if ioc_filled:
                        continue

                    order_result = await self._signed_post(
                        session,
                        self._urls["order"],
                        {
                            "symbol": symbol,
                            "side": side,
                            "type": "MARKET",
                            "quantity": qty,
                            "reduceOnly": "true",
                        },
                    )
                    if order_result is not None:
                        result["positions_closed"].append({
                            "symbol": symbol,
                            "side": side,
                            "quantity": qty,
                            "method": "MARKET",
                            "response": order_result,
                        })
                        _log.warning(
                            "EMERGENCY CLOSE (MARKET) | {sym} {side} {qty}",
                            sym=symbol, side=side, qty=qty,
                        )
                    else:
                        result["errors"].append(
                            f"Failed to close {symbol} {side} {qty}"
                        )

        except Exception as exc:
            _log.error(
                "Emergency close failed: {err}", err=str(exc),
            )
            result["errors"].append(str(exc))

        _log.warning(
            "EMERGENCY CLOSE ALL — done | closed={n} errors={e}",
            n=len(result["positions_closed"]),
            e=len(result["errors"]),
        )
        return result

    async def send_telegram_alert(self, message: str) -> bool:
        """Send alert to Telegram via Bot API.

        Returns True if sent successfully, False otherwise.
        """
        cfg = self._config
        if not cfg.telegram_token or not cfg.telegram_admin_id:
            _log.warning("Telegram not configured — alert skipped")
            return False

        import aiohttp

        url = f"https://api.telegram.org/bot{cfg.telegram_token}/sendMessage"
        payload = {
            "chat_id": cfg.telegram_admin_id,
            "text": f"🚨 AtomiCortex Watchdog\n\n{message}",
            "parse_mode": "HTML",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        _log.info("Telegram alert sent")
                        return True
                    else:
                        body = await resp.text()
                        _log.warning(
                            "Telegram API error: {status} {body}",
                            status=resp.status, body=body,
                        )
                        return False
        except Exception as exc:
            _log.warning("Telegram send failed: {err}", err=str(exc))
            return False

    # ------------------------------------------------------------------
    # Internal: check loop
    # ------------------------------------------------------------------

    async def _check_loop(self) -> None:
        """Periodically check the heartbeat key in Redis."""
        while self._running:
            try:
                alive, reason = await self._check_heartbeat_detailed()
                if not alive:
                    _log.warning(
                        "HEARTBEAT MISSING — bot may be down! "
                        "Triggering emergency close. Reason: {reason}",
                        reason=reason
                    )
                    incident = {
                        "timestamp": time.time(),
                        "action": "emergency_close",
                        "reason": reason,
                    }

                    # 1. Telegram alert (cooldown)
                    now = time.time()
                    if now - self._last_alert_ts > self._config.alert_cooldown_seconds:
                        if reason == "data_stale":
                            msg = (
                                f"⚠️ DATA STALE (zombie-RUNNING?) | {self._config.service_name}\n"
                                f"Silence > {self._config.max_bar_silence_seconds}s\n"
                                "Emergency closing all positions..."
                            )
                        else:
                            msg = (
                                f"⚠️ Bot heartbeat missing! | {self._config.service_name}\n"
                                f"Silence > {self._config.max_silence_seconds}s\n"
                                "Emergency closing all positions..."
                            )
                        
                        await self.send_telegram_alert(msg)
                        self._last_alert_ts = now

                    # 2. Emergency close (idempotent)
                    if not self._incident_active or self._last_close_found_positions:
                        close_result = await self.emergency_close_all()
                        incident["result"] = close_result
                        self._incidents.append(incident)
                        self._last_close_found_positions = len(close_result.get("positions_closed", [])) > 0
                    
                    self._incident_active = True

                    # 3. Wait before next check to avoid rapid re-triggers
                    await asyncio.sleep(self._config.max_silence_seconds)
                else:
                    if self._incident_active:
                        _log.info("heartbeat recovered")
                        self._incident_active = False
                        self._last_alert_ts = 0.0
                        self._last_close_found_positions = True
                    _log.debug("Heartbeat OK")

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.error(
                    "Watchdog check error: {err}", err=str(exc),
                )

            try:
                await asyncio.sleep(self._config.check_interval)
            except asyncio.CancelledError:
                break

    async def _check_heartbeat_detailed(self) -> tuple[bool, str]:
        """Return (is_alive, reason) where reason is 'ok', 'process_dead', or 'data_stale'."""
        if self._redis is None:
            self._redis = await self._connect_redis()
            if self._redis is None:
                _log.warning("Cannot check heartbeat — Redis unavailable")
                return True, "ok"  # fail-open

        try:
            val = await self._redis.get(self._config.heartbeat_key)
            if val is None:
                return False, "process_dead"

            try:
                data = json.loads(val)
                if not isinstance(data, dict):
                    raise ValueError("Not a dict")
            except (json.JSONDecodeError, ValueError):
                if not getattr(self, "_legacy_format_logged", False):
                    _log.info("legacy heartbeat format")
                    self._legacy_format_logged = True
                
                beat_ts = float(val)
                if time.time() - beat_ts > self._config.max_silence_seconds:
                    return False, "process_dead"
                return True, "ok"

            now = time.time()
            if "process_ts" not in data:
                _log.warning("Heartbeat missing process_ts")
                return True, "ok" # fail-open for invalid JSON payload
                
            if now - data["process_ts"] > self._config.max_silence_seconds:
                return False, "process_dead"

            if self._config.max_bar_silence_seconds > 0:
                last_bar = data.get("last_bar_ts")
                if last_bar is None:
                    if now - data.get("started_ts", now) > self._config.startup_bar_grace_seconds:
                        return False, "data_stale"
                elif now - last_bar > self._config.max_bar_silence_seconds:
                    return False, "data_stale"
                    
            return True, "ok"
        except Exception as exc:
            _log.warning("Heartbeat check error: {err}", err=str(exc))
            return True, "ok"

    async def _check_heartbeat(self) -> bool:
        """Wrapper around _check_heartbeat_detailed for backward compatibility."""
        is_alive, _ = await self._check_heartbeat_detailed()
        return is_alive

    # ------------------------------------------------------------------
    # Internal: Redis
    # ------------------------------------------------------------------

    async def _connect_redis(self) -> Any:
        """Connect to Redis."""
        try:
            import redis.asyncio as aioredis

            kwargs: dict[str, Any] = {
                "host": self._config.redis_host,
                "port": self._config.redis_port,
                "decode_responses": True,
            }
            if self._config.redis_password:
                kwargs["password"] = self._config.redis_password

            client = aioredis.Redis(**kwargs)
            await client.ping()
            _log.info(
                "Watchdog Redis connected | {host}:{port}",
                host=self._config.redis_host,
                port=self._config.redis_port,
            )
            return client
        except Exception as exc:
            _log.warning("Watchdog Redis connect failed: {err}", err=str(exc))
            return None

    # ------------------------------------------------------------------
    # Internal: Binance REST signed requests
    # ------------------------------------------------------------------

    def _sign_params(self, params: dict[str, str]) -> dict[str, str]:
        """Add timestamp + HMAC-SHA256 signature to params."""
        params["timestamp"] = str(int(time.time() * 1000))
        params["recvWindow"] = "5000"
        query_string = urlencode(params)
        signature = hmac.new(
            self._config.binance_api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _auth_headers(self) -> dict[str, str]:
        """Return headers with API key."""
        return {"X-MBX-APIKEY": self._config.binance_api_key}

    # H16: 0.3% slippage tolerance for Limit-IOC emergency close.
    # Wide enough to fill on a reasonably liquid book; narrow enough
    # that a thin book rejects the IOC and we fall back to MARKET
    # rather than ate a 1-5% slip silently.
    _IOC_SLIPPAGE: float = 0.003

    async def _try_limit_ioc_close(
        self,
        session: Any,
        pos: dict[str, Any],
        symbol: str,
        side: str,
        qty: str,
        result: dict[str, Any],
    ) -> bool:
        """Attempt a Limit-IOC reduceOnly close. Return True iff filled.

        Reads ``markPrice`` from the positionRisk record (already in
        memory — no extra HTTP). Falls back silently to ``False`` on any
        error so the caller's MARKET path runs unconditionally.
        """
        try:
            mark_price = float(pos.get("markPrice", 0) or 0)
        except (TypeError, ValueError):
            mark_price = 0.0
        if mark_price <= 0:
            return False

        # SELL (close LONG) accepts price slightly below mark; BUY
        # (close SHORT) accepts price slightly above. The IOC times out
        # if neither makes immediately, so on a thin book the order
        # cancels itself and we fall through to MARKET.
        if side == "SELL":
            limit_price = mark_price * (1.0 - self._IOC_SLIPPAGE)
        else:
            limit_price = mark_price * (1.0 + self._IOC_SLIPPAGE)

        try:
            ioc_result = await self._signed_post(
                session,
                self._urls["order"],
                {
                    "symbol": symbol,
                    "side": side,
                    "type": "LIMIT",
                    "timeInForce": "IOC",
                    "quantity": qty,
                    "price": f"{limit_price:.2f}",
                    "reduceOnly": "true",
                },
            )
        except Exception as exc:
            _log.warning(
                "IOC close raised for {sym}: {e} — falling back to MARKET",
                sym=symbol, e=str(exc),
            )
            return False

        if ioc_result is None:
            return False

        status = str(ioc_result.get("status", "")).upper()
        try:
            executed = float(ioc_result.get("executedQty", 0) or 0)
            requested = float(qty)
        except (TypeError, ValueError):
            executed, requested = 0.0, 0.0

        fully_filled = (
            status == "FILLED"
            or (requested > 0 and executed + 1e-10 >= requested)
        )
        if not fully_filled:
            # Partial / unfilled IOC — MARKET fallback will close the
            # remainder. Easier to over-cancel than to under-close.
            _log.info(
                "IOC partial/unfilled for {sym}: status={st} executed={ex}/"
                "{rq} — fallback to MARKET",
                sym=symbol, st=status, ex=executed, rq=requested,
            )
            return False

        result["positions_closed"].append({
            "symbol": symbol,
            "side": side,
            "quantity": qty,
            "method": "LIMIT_IOC",
            "response": ioc_result,
        })
        _log.warning(
            "EMERGENCY CLOSE (IOC) | {sym} {side} {qty} @ {px:.2f}",
            sym=symbol, side=side, qty=qty, px=limit_price,
        )
        return True

    @staticmethod
    def _binance_weight_for(path: str) -> int:
        """Approximate weight per endpoint (Binance Futures docs).

        Token bucket coordinates several callers; the X-MBX-USED-WEIGHT
        header corrects any per-call misestimate after the fact.
        """
        if path.endswith("/positionRisk"):
            return 5
        if path.endswith("/allOpenOrders"):
            return 1
        if path.endswith("/order"):
            return 1
        return 1

    async def _signed_get(
        self,
        session: Any,
        path: str,
        extra_params: dict[str, str] | None = None,
    ) -> Any:
        """Signed GET request to Binance."""
        import aiohttp

        from src.execution.binance_rate_limiter import BinanceRateLimiter
        limiter = BinanceRateLimiter.instance()
        await limiter.acquire(self._binance_weight_for(path))

        params = extra_params or {}
        params = self._sign_params(params)
        url = self._base_url + path
        try:
            async with session.get(
                url, params=params, headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                limiter.update_from_headers(getattr(resp, "headers", None))
                data = await resp.json()
                if resp.status != 200:
                    _log.error(
                        "Binance GET {path} error: {data}",
                        path=path, data=data,
                    )
                    return None
                return data
        except Exception as exc:
            _log.error("Binance GET {path} failed: {err}", path=path, err=str(exc))
            return None

    async def _signed_post(
        self,
        session: Any,
        path: str,
        params: dict[str, str],
    ) -> Any:
        """Signed POST request to Binance."""
        import aiohttp

        from src.execution.binance_rate_limiter import BinanceRateLimiter
        limiter = BinanceRateLimiter.instance()
        await limiter.acquire(self._binance_weight_for(path))

        params = self._sign_params(params)
        url = self._base_url + path
        try:
            async with session.post(
                url, data=params, headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                limiter.update_from_headers(getattr(resp, "headers", None))
                data = await resp.json()
                if resp.status != 200:
                    _log.error(
                        "Binance POST {path} error: {data}",
                        path=path, data=data,
                    )
                    return None
                return data
        except Exception as exc:
            _log.error("Binance POST {path} failed: {err}", path=path, err=str(exc))
            return None

    async def _signed_delete(
        self,
        session: Any,
        path: str,
        params: dict[str, str],
    ) -> Any:
        """Signed DELETE request to Binance."""
        import aiohttp

        from src.execution.binance_rate_limiter import BinanceRateLimiter
        limiter = BinanceRateLimiter.instance()
        await limiter.acquire(self._binance_weight_for(path))

        params = self._sign_params(params)
        url = self._base_url + path
        try:
            async with session.delete(
                url, params=params, headers=self._auth_headers(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                limiter.update_from_headers(getattr(resp, "headers", None))
                data = await resp.json()
                if resp.status != 200:
                    _log.error(
                        "Binance DELETE {path} error: {data}",
                        path=path, data=data,
                    )
                    return None
                return data
        except Exception as exc:
            _log.error("Binance DELETE {path} failed: {err}", path=path, err=str(exc))
            return None
