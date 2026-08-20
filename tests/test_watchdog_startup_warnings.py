"""Tests for PR-0.8 — start-up warnings about a mis-sized watchdog.

Two silent misconfigurations can make a running watchdog useless:

* ``max_bar_silence_seconds == 0`` disables the data-staleness check
  entirely, so a zombie-RUNNING bot is never noticed. The default stays 0
  because the safe value depends on the timeframe — but it must be loud.
* ``max_unknown_checks x check_interval`` is the wall-clock budget the
  watchdog allows itself to stay blind. If it drifts far from
  ``max_silence_seconds``, the operator has silently changed the meaning
  of the threshold without changing its name.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.execution.watchdog import Watchdog, WatchdogConfig


def _warnings_of(config: WatchdogConfig) -> list[str]:
    """Construct a Watchdog and return the WARNING templates it emitted."""
    log = MagicMock()
    with patch("src.execution.watchdog._log", log):
        Watchdog(config)
    return [call.args[0] for call in log.warning.call_args_list]


def test_zero_max_bar_silence_warns_on_start() -> None:
    """A disabled staleness check must announce itself."""
    warnings = _warnings_of(WatchdogConfig(
        check_interval=15,
        max_silence_seconds=60,
        max_unknown_checks=4,
        max_bar_silence_seconds=0,
    ))
    assert any("data-staleness check disabled" in text for text in warnings)


def test_unknown_budget_mismatch_warns_on_start() -> None:
    """4 x 60s = 240s of blindness against a 60s silence limit is a drift."""
    warnings = _warnings_of(WatchdogConfig(
        check_interval=60,
        max_silence_seconds=60,
        max_unknown_checks=4,
        max_bar_silence_seconds=3600,
    ))
    assert any("unknown-verdict budget" in text for text in warnings)


def test_unknown_budget_match_does_not_warn() -> None:
    """4 x 15s = 60s against a 60s silence limit is exactly right."""
    warnings = _warnings_of(WatchdogConfig(
        check_interval=15,
        max_silence_seconds=60,
        max_unknown_checks=4,
        max_bar_silence_seconds=3600,
    ))
    assert not any("unknown-verdict budget" in text for text in warnings)
