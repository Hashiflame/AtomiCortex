"""Tests — EngineConnectionChecker (fail-fast on engine connect timeout).

All tests use MagicMock / AsyncMock / monkeypatch.  No network, no Redis.
``os.kill`` is never called for real — ``kill_fn`` is a mock.
``reporter.send_alert`` is an ``AsyncMock`` (checker calls it via
``asyncio.run()``).
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(*, data_connected: bool = True, exec_connected: bool = True):
    """Build a mock TradingNode with controllable check_connected()."""
    node = MagicMock()
    node.kernel.data_engine.check_connected.return_value = data_connected
    node.kernel.exec_engine.check_connected.return_value = exec_connected
    return node


def _make_reporter(*, side_effect=None):
    """Build a mock reporter whose ``send_alert`` is an AsyncMock."""
    reporter = MagicMock()
    reporter.send_alert = AsyncMock(return_value=True, side_effect=side_effect)
    return reporter


def _run_checker(
    node,
    grace_sec: float = 0.0,
    reporter=None,
    kill_fn=None,
    sleep_fn=None,
):
    """Instantiate, start, and join the checker thread."""
    from src.execution.startup_check import EngineConnectionChecker

    kill = kill_fn or MagicMock()
    sleep = sleep_fn or MagicMock()

    checker = EngineConnectionChecker(
        node=node,
        grace_sec=grace_sec,
        reporter=reporter,
        kill_fn=kill,
        sleep_fn=sleep,
    )
    checker.start()
    # The thread is daemon but we need to wait for it to finish.
    checker._thread.join(timeout=5)
    return checker, kill, sleep


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestBothConnected:
    def test_kill_not_called(self):
        node = _make_node(data_connected=True, exec_connected=True)
        checker, kill, _ = _run_checker(node)
        kill.assert_not_called()

    def test_engines_failed_false(self):
        node = _make_node(data_connected=True, exec_connected=True)
        checker, _, _ = _run_checker(node)
        assert checker.engines_failed is False


class TestDataDisconnected:
    def test_kill_called_once_with_sigterm(self):
        node = _make_node(data_connected=False, exec_connected=True)
        reporter = _make_reporter()
        checker, kill, _ = _run_checker(node, reporter=reporter)
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_engines_failed_true(self):
        node = _make_node(data_connected=False, exec_connected=True)
        checker, _, _ = _run_checker(node)
        assert checker.engines_failed is True

    def test_alert_sent_with_data_engine_name(self):
        node = _make_node(data_connected=False, exec_connected=True)
        reporter = _make_reporter()
        checker, _, _ = _run_checker(node, reporter=reporter)
        reporter.send_alert.assert_called_once()
        msg = reporter.send_alert.call_args[0][0]
        assert "DataEngine" in msg


class TestExecDisconnected:
    def test_kill_called_once_with_sigterm(self):
        node = _make_node(data_connected=True, exec_connected=False)
        reporter = _make_reporter()
        checker, kill, _ = _run_checker(node, reporter=reporter)
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_engines_failed_true(self):
        node = _make_node(data_connected=True, exec_connected=False)
        checker, _, _ = _run_checker(node)
        assert checker.engines_failed is True

    def test_alert_sent_with_exec_engine_name(self):
        node = _make_node(data_connected=True, exec_connected=False)
        reporter = _make_reporter()
        checker, _, _ = _run_checker(node, reporter=reporter)
        reporter.send_alert.assert_called_once()
        msg = reporter.send_alert.call_args[0][0]
        assert "ExecEngine" in msg


class TestAlertExceptionDoesNotBlockKill:
    def test_kill_still_called(self):
        node = _make_node(data_connected=False, exec_connected=False)
        reporter = _make_reporter(side_effect=RuntimeError("Telegram down"))
        checker, kill, _ = _run_checker(node, reporter=reporter)
        kill.assert_called_once_with(os.getpid(), signal.SIGTERM)

    def test_engines_failed_true(self):
        node = _make_node(data_connected=False, exec_connected=False)
        reporter = _make_reporter(side_effect=RuntimeError("Telegram down"))
        checker, _, _ = _run_checker(node, reporter=reporter)
        assert checker.engines_failed is True


class TestCheckConnectedException:
    def test_kill_not_called(self):
        node = _make_node()
        node.kernel.data_engine.check_connected.side_effect = RuntimeError("boom")
        checker, kill, _ = _run_checker(node)
        kill.assert_not_called()

    def test_engines_failed_false(self):
        node = _make_node()
        node.kernel.data_engine.check_connected.side_effect = RuntimeError("boom")
        checker, _, _ = _run_checker(node)
        assert checker.engines_failed is False


class TestGracePeriod:
    def test_sleep_called_with_grace_sec(self):
        node = _make_node()
        checker, _, sleep = _run_checker(node, grace_sec=42.5)
        sleep.assert_called_once_with(42.5)


class TestEnginesFailedSetBeforeKill:
    def test_flag_true_at_kill_time(self):
        node = _make_node(data_connected=False, exec_connected=True)
        flag_at_kill_time = {}

        def _capturing_kill(pid, sig):
            flag_at_kill_time["engines_failed"] = checker.engines_failed

        from src.execution.startup_check import EngineConnectionChecker

        checker = EngineConnectionChecker(
            node=node,
            grace_sec=0.0,
            reporter=None,
            kill_fn=_capturing_kill,
            sleep_fn=MagicMock(),
        )
        checker.start()
        checker._thread.join(timeout=5)

        assert flag_at_kill_time.get("engines_failed") is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
class TestSigtermFromThreadReachesHandler:
    def test_signal_delivered_across_threads(self):
        """Integration smoke: a SIGTERM sent from a child thread IS
        delivered to a handler registered with ``signal.signal()`` in
        the main thread.
        """
        received = threading.Event()
        original_handler = signal.getsignal(signal.SIGTERM)

        def _handler(sig, frame):
            received.set()

        try:
            signal.signal(signal.SIGTERM, _handler)

            def _send():
                os.kill(os.getpid(), signal.SIGTERM)

            t = threading.Thread(target=_send, daemon=True)
            t.start()
            t.join(timeout=2)

            assert received.wait(timeout=2), (
                "SIGTERM from child thread was NOT received by the main-thread handler"
            )
        finally:
            signal.signal(signal.SIGTERM, original_handler)
