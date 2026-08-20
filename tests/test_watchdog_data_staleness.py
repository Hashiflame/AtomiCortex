import json
import time
from unittest.mock import AsyncMock

import pytest

from src.execution.watchdog import (
    REASON_BAD_PAYLOAD,
    HeartbeatVerdict,
    Watchdog,
    WatchdogConfig,
)


@pytest.fixture
def watchdog_config():
    return WatchdogConfig(
        redis_host="localhost",
        redis_port=6379,
        heartbeat_key="test:key",
        max_silence_seconds=60,
        max_bar_silence_seconds=3600,
        startup_bar_grace_seconds=900,
    )


@pytest.fixture
def watchdog(watchdog_config):
    wd = Watchdog(watchdog_config)
    wd._redis = AsyncMock()
    return wd


@pytest.mark.asyncio
async def test_fresh_process_fresh_bar_ok(watchdog):
    now = time.time()
    payload = json.dumps({
        "process_ts": now - 10,
        "started_ts": now - 5000,
        "last_bar_ts": now - 100,
        "bars_seen": 10,
    })
    watchdog._redis.get.return_value = payload

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.ALIVE
    assert reason == "ok"


@pytest.mark.asyncio
async def test_fresh_process_stale_bar(watchdog):
    now = time.time()
    payload = json.dumps({
        "process_ts": now - 10,  # fresh process
        "started_ts": now - 10000,
        "last_bar_ts": now - 4000,  # stale bar (> 3600)
        "bars_seen": 10,
    })
    watchdog._redis.get.return_value = payload

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.DEAD
    assert reason == "data_stale"


@pytest.mark.asyncio
async def test_stale_process(watchdog):
    now = time.time()
    payload = json.dumps({
        "process_ts": now - 100,  # stale process (> 60)
        "started_ts": now - 5000,
        "last_bar_ts": now - 10,
        "bars_seen": 10,
    })
    watchdog._redis.get.return_value = payload

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.DEAD
    assert reason == "process_dead"


@pytest.mark.asyncio
async def test_legacy_float_fresh(watchdog):
    now = time.time()
    watchdog._redis.get.return_value = str(now - 10)  # fresh process, legacy float

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.ALIVE
    assert reason == "ok"
    assert watchdog._legacy_format_logged is True

    # Call again to test single logging (would need to patch log, but we just check the flag)
    await watchdog._check_heartbeat_detailed()
    assert watchdog._legacy_format_logged is True


@pytest.mark.asyncio
async def test_legacy_float_stale(watchdog):
    now = time.time()
    watchdog._redis.get.return_value = str(now - 100)  # stale process (> 60)

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.DEAD
    assert reason == "process_dead"


@pytest.mark.asyncio
async def test_none_bar_within_grace_ok(watchdog):
    now = time.time()
    payload = json.dumps({
        "process_ts": now - 10,
        "started_ts": now - 800,  # < 900 grace
        "last_bar_ts": None,
        "bars_seen": 0,
    })
    watchdog._redis.get.return_value = payload

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.ALIVE
    assert reason == "ok"


@pytest.mark.asyncio
async def test_none_bar_beyond_grace_stale(watchdog):
    now = time.time()
    payload = json.dumps({
        "process_ts": now - 10,
        "started_ts": now - 1000,  # > 900 grace
        "last_bar_ts": None,
        "bars_seen": 0,
    })
    watchdog._redis.get.return_value = payload

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.DEAD
    assert reason == "data_stale"


@pytest.mark.asyncio
async def test_max_bar_silence_zero_disables_data_check(watchdog):
    watchdog._config.max_bar_silence_seconds = 0
    now = time.time()
    payload = json.dumps({
        "process_ts": now - 10,
        "started_ts": now - 10000,
        "last_bar_ts": now - 5000,  # very stale
        "bars_seen": 10,
    })
    watchdog._redis.get.return_value = payload

    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.ALIVE
    assert reason == "ok"


@pytest.mark.asyncio
async def test_missing_process_ts_is_unknown(watchdog):
    now = time.time()
    # Missing process_ts
    payload = json.dumps({
        "started_ts": now - 1000,
        "last_bar_ts": now - 10,
        "bars_seen": 10,
    })
    watchdog._redis.get.return_value = payload

    # A payload we cannot interpret is ignorance, not proof of life.
    verdict, reason = await watchdog._check_heartbeat_detailed()
    assert verdict is HeartbeatVerdict.UNKNOWN
    assert reason == REASON_BAD_PAYLOAD
