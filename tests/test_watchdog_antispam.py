import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.watchdog import Watchdog, WatchdogConfig


@pytest.fixture
def watchdog():
    config = WatchdogConfig(
        redis_host="localhost",
        heartbeat_key="test:key",
        max_silence_seconds=60,
        alert_cooldown_seconds=900,
        telegram_token="dummy",
        telegram_admin_id="dummy",
        service_name="test_service",
    )
    wd = Watchdog(config)
    wd._redis = AsyncMock()
    
    # Mock actions
    wd.send_telegram_alert = AsyncMock(return_value=True)
    wd.emergency_close_all = AsyncMock(return_value={"positions_closed": [{"symbol": "BTC"}], "orders_cancelled": True})
    return wd


@pytest.mark.asyncio
async def test_first_incident_alerts_immediately(watchdog):
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
            
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    watchdog.send_telegram_alert.assert_called_once()
    watchdog.emergency_close_all.assert_called_once()
    assert watchdog._incident_active is True
    assert watchdog._last_alert_ts > 0


@pytest.mark.asyncio
async def test_second_incident_within_cooldown_no_alert(watchdog):
    # Setup state as if an incident just happened
    now = time.time()
    watchdog._incident_active = True
    watchdog._last_alert_ts = now - 100  # Within 900s cooldown
    watchdog._last_close_found_positions = False  # To not trigger close

    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
            
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    watchdog.send_telegram_alert.assert_not_called()
    watchdog.emergency_close_all.assert_not_called()


@pytest.mark.asyncio
async def test_alert_after_cooldown_expires(watchdog):
    now = time.time()
    watchdog._incident_active = True
    watchdog._last_alert_ts = now - 1000  # Past 900s cooldown
    watchdog._last_close_found_positions = False

    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
            
        with patch("asyncio.sleep", side_effect=mock_sleep):
            with patch("time.time", return_value=now):
                await watchdog._check_loop()
            
    watchdog.send_telegram_alert.assert_called_once()
    # Close should not be called because it's not the first incident and last found no positions
    watchdog.emergency_close_all.assert_not_called()


@pytest.mark.asyncio
async def test_close_called_on_first_incident(watchdog):
    # Covered by test_first_incident_alerts_immediately, but isolated here
    assert watchdog._incident_active is False
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    watchdog.emergency_close_all.assert_called_once()


@pytest.mark.asyncio
async def test_close_skipped_when_prev_found_no_positions(watchdog):
    watchdog._incident_active = True
    watchdog._last_close_found_positions = False
    
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    watchdog.emergency_close_all.assert_not_called()


@pytest.mark.asyncio
async def test_close_repeats_while_positions_found(watchdog):
    watchdog._incident_active = True
    watchdog._last_close_found_positions = True
    
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    watchdog.emergency_close_all.assert_called_once()


@pytest.mark.asyncio
async def test_recovery_resets_state_and_logs_once(watchdog):
    # Start in incident state
    watchdog._incident_active = True
    watchdog._last_alert_ts = 12345.0
    watchdog._last_close_found_positions = False
    
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(True, "ok")):
        watchdog._running = True
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    assert watchdog._incident_active is False
    assert watchdog._last_alert_ts == 0.0
    assert watchdog._last_close_found_positions is True

    # Next incident will alert and close immediately
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "data_stale")):
        watchdog._running = True
        async def mock_sleep2(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep2):
            await watchdog._check_loop()
            
    watchdog.send_telegram_alert.assert_called_once()
    watchdog.emergency_close_all.assert_called_once()


@pytest.mark.asyncio
async def test_alert_text_differs_by_reason(watchdog):
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "data_stale")):
        watchdog._running = True
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    alert_text = watchdog.send_telegram_alert.call_args[0][0]
    assert "DATA STALE" in alert_text
    assert "test_service" in alert_text
    
    # Reset
    watchdog._incident_active = False
    watchdog._last_alert_ts = 0.0
    watchdog.send_telegram_alert.reset_mock()
    
    with patch.object(watchdog, "_check_heartbeat_detailed", return_value=(False, "process_dead")):
        watchdog._running = True
        async def mock_sleep(*args, **kwargs):
            watchdog._running = False
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await watchdog._check_loop()
            
    alert_text = watchdog.send_telegram_alert.call_args[0][0]
    assert "heartbeat missing" in alert_text.lower() or "bot heartbeat missing" in alert_text.lower()
    assert "test_service" in alert_text
