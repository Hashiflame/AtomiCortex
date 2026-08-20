"""Tests for PR-0.8 — the UNKNOWN budget in ``_check_loop``.

UNKNOWN is not DEAD. Treating the first blind tick as death would flatten
the book every time Redis restarts. Treating it as life is the defect this
PR closes. The middle course is a budget: ``max_unknown_checks`` consecutive
blind ticks, sized so the budget spans ``max_silence_seconds``, after which
the watchdog acts as if the bot were dead. The first blind tick alerts
immediately regardless of the alert cooldown, so a human sees the blindness
a full budget before the machine acts on it.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.watchdog import (
    REASON_PROCESS_DEAD,
    REASON_READ_ERROR,
    REASON_REDIS_DOWN,
    HeartbeatVerdict,
    Watchdog,
    WatchdogConfig,
)

_UNKNOWN = (HeartbeatVerdict.UNKNOWN, REASON_REDIS_DOWN)
_UNKNOWN_OTHER = (HeartbeatVerdict.UNKNOWN, REASON_READ_ERROR)
_ALIVE = (HeartbeatVerdict.ALIVE, "ok")
_DEAD = (HeartbeatVerdict.DEAD, REASON_PROCESS_DEAD)


@pytest.fixture
def watchdog() -> Watchdog:
    config = WatchdogConfig(
        redis_host="localhost",
        heartbeat_key="test:key",
        check_interval=15,
        max_silence_seconds=60,
        max_unknown_checks=4,
        alert_cooldown_seconds=900,
        telegram_token="dummy",
        telegram_admin_id="dummy",
        service_name="test_service",
    )
    wd = Watchdog(config)
    wd._redis = AsyncMock()
    wd.send_telegram_alert = AsyncMock(return_value=True)
    wd.emergency_close_all = AsyncMock(return_value={
        "positions_closed": [{"symbol": "BTCUSDT"}],
        "orders_cancelled": True,
        "errors": [],
    })
    return wd


async def _drive(watchdog: Watchdog, verdicts: list[tuple]) -> None:
    """Run ``_check_loop`` through exactly ``len(verdicts)`` ticks."""
    seq = list(verdicts)
    state = {"i": 0}

    async def _check():
        index = state["i"]
        state["i"] += 1
        if state["i"] >= len(seq):
            watchdog._running = False
        return seq[index]

    watchdog._running = True
    with patch.object(watchdog, "_check_heartbeat_detailed", side_effect=_check):
        with patch("asyncio.sleep", AsyncMock()):
            await watchdog._check_loop()
    assert state["i"] == len(seq), "loop did not consume every scripted tick"


async def test_unknown_below_threshold_does_not_close(watchdog: Watchdog) -> None:
    """Three blind ticks out of a budget of four must not touch the book."""
    await _drive(watchdog, [_UNKNOWN, _UNKNOWN, _UNKNOWN])
    watchdog.emergency_close_all.assert_not_called()


async def test_unknown_at_threshold_closes(watchdog: Watchdog) -> None:
    """The fourth consecutive blind tick spends the budget and closes."""
    await _drive(watchdog, [_UNKNOWN, _UNKNOWN, _UNKNOWN, _UNKNOWN])
    watchdog.emergency_close_all.assert_called_once()


async def test_first_unknown_alerts_immediately(watchdog: Watchdog) -> None:
    """The first blind tick bypasses the cooldown — a human must know now."""
    watchdog._last_alert_ts = time.time()  # deep inside the 900s cooldown
    await _drive(watchdog, [_UNKNOWN])
    watchdog.send_telegram_alert.assert_called_once()
    watchdog.emergency_close_all.assert_not_called()


async def test_unknown_streak_resets_on_alive(watchdog: Watchdog) -> None:
    """One informed ALIVE breaks the streak; the budget starts over."""
    await _drive(
        watchdog,
        [_UNKNOWN, _UNKNOWN, _UNKNOWN, _ALIVE,
         _UNKNOWN, _UNKNOWN_OTHER, _UNKNOWN],
    )
    watchdog.emergency_close_all.assert_not_called()


async def test_dead_closes_without_waiting_for_threshold(
    watchdog: Watchdog,
) -> None:
    """The budget must never delay the response to an informed DEAD."""
    await _drive(watchdog, [_DEAD])
    watchdog.emergency_close_all.assert_called_once()


async def test_unknown_does_not_reset_incident_state(watchdog: Watchdog) -> None:
    """A blind tick mid-incident is not a recovery."""
    watchdog._incident_active = True
    watchdog._last_alert_ts = 12345.0
    watchdog._last_close_found_positions = False

    await _drive(watchdog, [_UNKNOWN])

    assert watchdog._incident_active is True
    assert watchdog._last_close_found_positions is False
    assert watchdog._last_alert_ts != 0.0
