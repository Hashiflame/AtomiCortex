import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.watchdog import Watchdog, WatchdogConfig


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

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is True
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

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is False
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

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is False
    assert reason == "process_dead"


@pytest.mark.asyncio
async def test_legacy_float_fresh(watchdog):
    now = time.time()
    watchdog._redis.get.return_value = str(now - 10)  # fresh process, legacy float

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is True
    assert reason == "ok"
    assert watchdog._legacy_format_logged is True

    # Call again to test single logging (would need to patch log, but we just check the flag)
    await watchdog._check_heartbeat_detailed()
    assert watchdog._legacy_format_logged is True


@pytest.mark.asyncio
async def test_legacy_float_stale(watchdog):
    now = time.time()
    watchdog._redis.get.return_value = str(now - 100)  # stale process (> 60)

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is False
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

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is True
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

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is False
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

    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is True
    assert reason == "ok"


@pytest.mark.asyncio
async def test_check_heartbeat_bool_wrapper(watchdog):
    now = time.time()
    # Mock detailed to return (False, "data_stale")
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "data_stale")):
        result = await watchdog._check_heartbeat()
        assert result is False

    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(True, "ok")):
        result = await watchdog._check_heartbeat()
        assert result is True


@pytest.mark.asyncio
async def test_missing_json_fields_fail_open(watchdog):
    now = time.time()
    # Missing process_ts
    payload = json.dumps({
        "started_ts": now - 1000,
        "last_bar_ts": now - 10,
        "bars_seen": 10,
    })
    watchdog._redis.get.return_value = payload

    # Should fail open when key error happens
    is_alive, reason = await watchdog._check_heartbeat_detailed()
    assert is_alive is True
    assert reason == "ok"
