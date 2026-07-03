import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.heartbeat import HeartbeatManager
from src.execution.strategies.ml_strategy import MLTradingStrategy
from src.execution.strategies.ml_strategy_15m import MLTradingStrategy15M


@pytest.fixture
def mock_redis():
    mock = AsyncMock()
    return mock


@pytest.fixture
def heartbeat(mock_redis):
    manager = HeartbeatManager(
        redis_host="localhost",
        redis_port=6379,
        redis_password="",
        heartbeat_key="test:key",
    )
    manager._redis = mock_redis
    return manager


def test_started_ts_set_on_start(heartbeat):
    with patch("time.time", return_value=12345.0):
        # We only call start up to the point it sets _started_ts,
        # but since start() is async and enters a loop, we just mock the loop.
        # Actually start() first sets _started_ts, then connects to Redis, then runs _heartbeat_loop.
        # We can just manually check initialization.
        pass

@pytest.mark.asyncio
async def test_started_ts_set_on_start(heartbeat):
    with patch("time.time", return_value=12345.0):
        with patch.object(heartbeat, "_connect_redis", return_value=AsyncMock()):
            with patch.object(heartbeat, "_heartbeat_loop"):
                await heartbeat.start()
                assert heartbeat._started_ts == 12345.0


def test_report_bar_updates_fields(heartbeat):
    assert heartbeat._bars_seen == 0
    assert heartbeat._last_bar_ts is None

    heartbeat.report_bar(100.0)
    assert heartbeat._bars_seen == 1
    assert heartbeat._last_bar_ts == 100.0

    heartbeat.report_bar(200.0)
    assert heartbeat._bars_seen == 2
    assert heartbeat._last_bar_ts == 200.0


@pytest.mark.asyncio
async def test_loop_writes_valid_json_with_4_keys(heartbeat):
    heartbeat._started_ts = 1000.0
    heartbeat._running = True

    # Run exactly one iteration of the loop
    async def mock_sleep(*args, **kwargs):
        heartbeat._running = False

    with patch("time.time", return_value=1500.0):
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await heartbeat._heartbeat_loop()

    heartbeat._redis.setex.assert_called_once()
    args, kwargs = heartbeat._redis.setex.call_args
    assert args[0] == "test:key"
    payload = json.loads(args[2])

    assert payload["process_ts"] == 1500.0
    assert payload["started_ts"] == 1000.0
    assert payload["last_bar_ts"] is None
    assert payload["bars_seen"] == 0

    # Now with report_bar
    heartbeat.report_bar(1200.0)
    heartbeat._running = True
    heartbeat._redis.setex.reset_mock()

    with patch("time.time", return_value=1600.0):
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await heartbeat._heartbeat_loop()

    args, kwargs = heartbeat._redis.setex.call_args
    payload = json.loads(args[2])
    assert payload["last_bar_ts"] == 1200.0
    assert payload["bars_seen"] == 1


def test_4h_on_bar_reports_heartbeat():
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price, Quantity
    from nautilus_trader.model.identifiers import InstrumentId
    from src.execution.strategies.ml_strategy import MLStrategyConfig
    strategy = MLTradingStrategy(config=MLStrategyConfig())
    strategy._heartbeat = MagicMock()
    strategy._bars = []

    bar = MagicMock(spec=Bar)
    bar.ts_event = int(1234567890 * 1e9)
    bar.close = Price(50000.0, precision=2)
    bar.bar_type = "test_bar"

    # Patch the rest of on_bar to not crash
    with patch.object(strategy, "_detect_regime", return_value=None):
        strategy.on_bar(bar)

    strategy._heartbeat.report_bar.assert_called_once_with(1234567890.0)

    # Test heartbeat=None doesn't crash
    strategy._heartbeat = None
    with patch.object(strategy, "_detect_regime", return_value=None):
        strategy.on_bar(bar)  # Should not raise


def test_15m_on_bar_reports_heartbeat():
    from nautilus_trader.model.data import Bar
    from nautilus_trader.model.objects import Price
    from src.execution.strategies.ml_strategy_15m import MLStrategy15MConfig
    strategy = MLTradingStrategy15M(config=MLStrategy15MConfig())
    strategy._heartbeat = MagicMock()
    strategy._bars = []
    strategy._max_bars = 100
    strategy._bar_count = 0
    strategy._record_equity = MagicMock()

    bar = MagicMock(spec=Bar)
    bar.ts_event = int(1234567890 * 1e9)
    bar.close = Price(50000.0, precision=2)

    with patch.object(strategy, "_detect_regime", return_value=None):
        strategy.on_bar(bar)

    strategy._heartbeat.report_bar.assert_called_once_with(1234567890.0)

    # Test heartbeat=None doesn't crash
    strategy._heartbeat = None
    with patch.object(strategy, "_detect_regime", return_value=None):
        strategy.on_bar(bar)
