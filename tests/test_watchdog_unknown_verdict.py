"""Tests for PR-0.8 — the third heartbeat verdict, UNKNOWN.

Before PR-0.8 ``_check_heartbeat_detailed`` returned ``(True, "ok")``
on three paths that carry no information about the bot at all: Redis
unavailable, a JSON payload without ``process_ts``, and any exception
raised while reading. A boolean cannot express "I do not know", so all
three read as "the bot is alive" and the watchdog stood down.

These tests pin the three paths to ``HeartbeatVerdict.UNKNOWN`` and to
one distinguishable reason each, so an operator reading the journal can
tell *why* the watchdog went blind.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.watchdog import (
    REASON_BAD_PAYLOAD,
    REASON_READ_ERROR,
    REASON_REDIS_DOWN,
    HeartbeatVerdict,
    Watchdog,
    WatchdogConfig,
)


@pytest.fixture
def config() -> WatchdogConfig:
    return WatchdogConfig(
        redis_host="localhost",
        redis_port=6379,
        heartbeat_key="test:key",
        max_silence_seconds=60,
        max_bar_silence_seconds=3600,
        startup_bar_grace_seconds=900,
    )


@pytest.fixture
def watchdog(config: WatchdogConfig) -> Watchdog:
    wd = Watchdog(config)
    wd._redis = AsyncMock()
    return wd


# ---------------------------------------------------------------------------
# Path 1 — Redis unavailable
# ---------------------------------------------------------------------------


async def test_redis_unavailable_returns_unknown(watchdog: Watchdog) -> None:
    """No Redis client at all is ignorance, not proof of life."""
    watchdog._redis = None
    with patch.object(watchdog, "_connect_redis", AsyncMock(return_value=None)):
        verdict, _reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.UNKNOWN


async def test_redis_unavailable_reason_is_redis_down(watchdog: Watchdog) -> None:
    """The reason names the blind spot, so the journal is diagnosable."""
    watchdog._redis = None
    with patch.object(watchdog, "_connect_redis", AsyncMock(return_value=None)):
        _verdict, reason = await watchdog._check_heartbeat_detailed()
    assert reason == REASON_REDIS_DOWN


# ---------------------------------------------------------------------------
# Path 2 — payload without process_ts
# ---------------------------------------------------------------------------


async def test_payload_without_process_ts_returns_unknown(
    watchdog: Watchdog,
) -> None:
    """A payload we cannot interpret says nothing about the process."""
    now = time.time()
    watchdog._redis.get.return_value = json.dumps({
        "started_ts": now - 1000,
        "last_bar_ts": now - 10,
        "bars_seen": 10,
    })
    verdict, _reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.UNKNOWN


async def test_payload_without_process_ts_reason_is_bad_payload(
    watchdog: Watchdog,
) -> None:
    now = time.time()
    watchdog._redis.get.return_value = json.dumps({
        "started_ts": now - 1000,
        "last_bar_ts": now - 10,
        "bars_seen": 10,
    })
    _verdict, reason = await watchdog._check_heartbeat_detailed()
    assert reason == REASON_BAD_PAYLOAD


# ---------------------------------------------------------------------------
# Path 3 — exception while reading
# ---------------------------------------------------------------------------


async def test_read_exception_returns_unknown(watchdog: Watchdog) -> None:
    """A failed read is ignorance, whatever the exception was."""
    watchdog._redis.get.side_effect = RuntimeError("connection reset")
    verdict, _reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.UNKNOWN


async def test_read_exception_reason_is_read_error(watchdog: Watchdog) -> None:
    watchdog._redis.get.side_effect = RuntimeError("connection reset")
    _verdict, reason = await watchdog._check_heartbeat_detailed()
    assert reason == REASON_READ_ERROR


# ---------------------------------------------------------------------------
# R4 — a failed read drops the client so the next tick reconnects
# ---------------------------------------------------------------------------


async def test_read_exception_clears_redis_handle(watchdog: Watchdog) -> None:
    """Keeping a dead client makes the blindness permanent."""
    watchdog._redis.get.side_effect = RuntimeError("connection reset")
    await watchdog._check_heartbeat_detailed()
    assert watchdog._redis is None


async def test_next_check_reconnects_after_read_exception(
    watchdog: Watchdog,
) -> None:
    """Having dropped the client, the following tick must dial again."""
    watchdog._redis.get.side_effect = RuntimeError("connection reset")
    await watchdog._check_heartbeat_detailed()

    reconnect = AsyncMock(return_value=None)
    with patch.object(watchdog, "_connect_redis", reconnect):
        await watchdog._check_heartbeat_detailed()
    reconnect.assert_called_once()


# ---------------------------------------------------------------------------
# O3 — bounded socket timeouts, so a tick cannot outlast check_interval
# ---------------------------------------------------------------------------


async def test_connect_redis_passes_socket_timeouts(watchdog: Watchdog) -> None:
    """Without timeouts a tick can hang and the UNKNOWN budget drifts."""
    client = AsyncMock()
    factory = MagicMock(return_value=client)
    with patch("redis.asyncio.Redis", factory):
        await watchdog._connect_redis()

    factory.assert_called_once()
    kwargs = factory.call_args.kwargs
    assert kwargs["socket_connect_timeout"] == Watchdog._REDIS_SOCKET_TIMEOUT
    assert kwargs["socket_timeout"] == Watchdog._REDIS_SOCKET_TIMEOUT
