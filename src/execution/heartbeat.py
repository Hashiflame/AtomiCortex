"""
AtomiCortex — Dead-Man's Switch (Heartbeat Manager).

Sends periodic heartbeats to Redis.  If the bot process dies, the Redis
key expires after ``heartbeat_ttl`` seconds and the external :class:`Watchdog`
triggers an emergency close of all positions.

Phase 4 — Step 4.6.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.logger import get_logger

_log = get_logger(__name__)


class HeartbeatManager:
    """Periodically writes a heartbeat key into Redis with a TTL.

    Parameters
    ----------
    redis_host:
        Redis hostname.
    redis_port:
        Redis port.
    redis_password:
        Redis AUTH password (empty string for no auth).
    heartbeat_key:
        The Redis key to write to.
    heartbeat_interval:
        Seconds between successive heartbeats.
    heartbeat_ttl:
        TTL (seconds) on the Redis key.  If the bot stops writing,
        the key auto-expires and the watchdog can detect silence.
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = "",
        heartbeat_key: str = "atomicortex:heartbeat",
        heartbeat_interval: int = 30,
        heartbeat_ttl: int = 60,
    ) -> None:
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_password = redis_password
        self._heartbeat_key = heartbeat_key
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_ttl = heartbeat_ttl

        self._started_ts: float = 0.0
        self._last_bar_ts: float | None = None
        self._bars_seen: int = 0

        self._redis: Any = None  # redis.asyncio.Redis instance
        self._task: asyncio.Task | None = None
        self._running: bool = False
        self._last_beat_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the heartbeat background loop."""
        if self._running:
            _log.warning("HeartbeatManager already running")
            return

        self._started_ts = time.time()
        self._redis = await self._connect_redis()
        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        _log.info(
            "HeartbeatManager started | key={key} interval={iv}s ttl={ttl}s",
            key=self._heartbeat_key,
            iv=self._heartbeat_interval,
            ttl=self._heartbeat_ttl,
        )

    async def stop(self) -> None:
        """Stop the heartbeat loop and close Redis connection."""
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
                await self._redis.delete(self._heartbeat_key)
            except Exception:
                pass
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

        _log.info("HeartbeatManager stopped")

    def report_bar(self, bar_ts: float) -> None:
        """Called by the strategy to indicate data is flowing."""
        self._last_bar_ts = bar_ts
        self._bars_seen += 1

    def is_alive(self) -> bool:
        """Check whether the heartbeat loop has written recently.

        Returns True if a beat was sent within the last ``heartbeat_ttl``
        seconds.  This is a *local* check — for external checks, read the
        Redis key directly.
        """
        if not self._running:
            return False
        elapsed = time.time() - self._last_beat_ts
        return elapsed < self._heartbeat_ttl

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Write heartbeat to Redis every ``heartbeat_interval`` seconds."""
        while self._running:
            try:
                ts = time.time()
                payload = json.dumps({
                    "process_ts": ts,
                    "started_ts": self._started_ts,
                    "last_bar_ts": self._last_bar_ts,
                    "bars_seen": self._bars_seen,
                })

                if self._redis is not None:
                    await self._redis.setex(
                        self._heartbeat_key,
                        self._heartbeat_ttl,
                        payload,
                    )
                    self._last_beat_ts = time.time()
                    _log.debug(
                        "Heartbeat sent | key={key} ts={ts}",
                        key=self._heartbeat_key,
                        ts=ts,
                    )
                else:
                    _log.warning("Redis client is None — attempting reconnect")
                    self._redis = await self._connect_redis()
            except asyncio.CancelledError:
                raise  # let cancellation propagate
            except Exception as exc:
                _log.warning(
                    "Heartbeat write failed (will retry): {err}",
                    err=str(exc),
                )

            try:
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break

    async def _connect_redis(self) -> Any:
        """Create a ``redis.asyncio.Redis`` connection.

        Returns None if connection fails — the heartbeat loop will retry.
        """
        try:
            import redis.asyncio as aioredis

            kwargs: dict[str, Any] = {
                "host": self._redis_host,
                "port": self._redis_port,
                "decode_responses": True,
            }
            if self._redis_password:
                kwargs["password"] = self._redis_password

            client = aioredis.Redis(**kwargs)
            await client.ping()
            _log.info(
                "Redis connected | {host}:{port}",
                host=self._redis_host,
                port=self._redis_port,
            )
            return client
        except Exception as exc:
            _log.warning(
                "Redis connection failed: {err}",
                err=str(exc),
            )
            return None
