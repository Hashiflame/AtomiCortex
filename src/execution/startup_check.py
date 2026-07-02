"""
AtomiCortex — Engine Connection Checker (fail-fast, fix A1).

Daemon thread that verifies DataEngine and ExecEngine are connected
after a grace period.  If either engine is not connected:

1. Sets ``engines_failed = True`` (before kill, so callers can read it).
2. Sends a CRITICAL log + Telegram alert (best-effort).
3. Sends SIGTERM to the current process → graceful shutdown via Nautilus
   or script-level handler → ``sys.exit(1)`` → systemd ``Restart=on-failure``.

If ``check_connected()`` itself raises — ERROR log, NO kill (fail-soft:
don't kill a healthy process because the check broke).
"""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from typing import Any, Callable

from src.logger import get_logger

_log = get_logger(__name__)


class EngineConnectionChecker:
    """Background checker for DataEngine / ExecEngine connectivity.

    Parameters
    ----------
    node :
        A ``TradingNode`` instance (must have ``node.kernel.data_engine``
        and ``node.kernel.exec_engine`` with ``check_connected()``).
    grace_sec :
        Seconds to wait after start before checking connectivity.
    reporter :
        Optional ``TelegramReporter`` (with async ``send_alert(msg)``).
    kill_fn :
        Callable matching ``os.kill(pid, sig)`` signature; injected for
        testability.
    sleep_fn :
        Callable matching ``time.sleep(sec)``; injected for testability.
    """

    def __init__(
        self,
        node: Any,
        grace_sec: float,
        reporter: Any | None = None,
        kill_fn: Callable[[int, int], None] = os.kill,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._node = node
        self._grace_sec = grace_sec
        self._reporter = reporter
        self._kill_fn = kill_fn
        self._sleep_fn = sleep_fn
        self._engines_failed = False
        self._thread: threading.Thread | None = None

    @property
    def engines_failed(self) -> bool:
        """Whether the post-startup check detected disconnected engines."""
        return self._engines_failed

    def start(self) -> None:
        """Launch the checker in a daemon thread."""
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="engine-conn-check",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        self._sleep_fn(self._grace_sec)

        try:
            data_ok = self._node.kernel.data_engine.check_connected()
            exec_ok = self._node.kernel.exec_engine.check_connected()
        except Exception as exc:
            _log.error(
                "Startup connection check raised (fail-soft, NOT killing): {err}",
                err=str(exc),
            )
            return

        if data_ok and exec_ok:
            _log.info(
                "Startup check passed — DataEngine and ExecEngine connected",
            )
            return

        # --- At least one engine is disconnected --------------------------
        failed: list[str] = []
        if not data_ok:
            failed.append("DataEngine")
        if not exec_ok:
            failed.append("ExecEngine")

        # Set flag BEFORE kill_fn so callers can read it immediately.
        self._engines_failed = True

        failed_str = ", ".join(failed)
        _log.critical(
            "Startup check FAILED — engines not connected after "
            "{grace}s grace period: {engines}. Sending SIGTERM for "
            "systemd restart.",
            grace=self._grace_sec,
            engines=failed_str,
        )

        # Telegram alert (best-effort).
        if self._reporter is not None:
            msg = (
                f"🚨 AtomiCortex Engine Connect FAILED\n\n"
                f"Engines not connected: {failed_str}\n"
                f"Grace period: {self._grace_sec}s\n"
                f"Action: sending SIGTERM for systemd restart"
            )
            try:
                asyncio.run(self._reporter.send_alert(msg))
            except Exception as exc:
                _log.warning(
                    "Telegram alert failed (non-blocking): {err}",
                    err=str(exc),
                )

        self._kill_fn(os.getpid(), signal.SIGTERM)
