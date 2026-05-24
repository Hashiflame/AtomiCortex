"""Tests for Step H17 — explicit ``--strategy`` flag on run_watchdog.

Pre-H17: launchers without ``--heartbeat-key`` silently fell through
to the 4H default. Operators starting a 15m watchdog "by reflex"
would actually monitor the 4H bot and on alert flatten the wrong book.
Post-H17: ``--strategy`` picks the canonical key; bare invocation
emits a WARNING; conflicts surface explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_watchdog import parse_args, resolve_strategy_args
from src.execution.watchdog import (
    DEFAULT_HEARTBEAT_KEY,
    STRATEGY_HEARTBEAT_KEYS,
)


# ---------------------------------------------------------------------------
# Canonical mapping shared with strategies
# ---------------------------------------------------------------------------


class TestStrategyKeyMap:
    def test_4h_key(self):
        assert STRATEGY_HEARTBEAT_KEYS["4h"] == "atomicortex:heartbeat"

    def test_1h_key(self):
        assert STRATEGY_HEARTBEAT_KEYS["1h"] == "bot_1h_heartbeat"

    def test_15m_key(self):
        assert STRATEGY_HEARTBEAT_KEYS["15m"] == "bot_15m_heartbeat"

    def test_default_is_4h(self):
        assert DEFAULT_HEARTBEAT_KEY == STRATEGY_HEARTBEAT_KEYS["4h"]


# ---------------------------------------------------------------------------
# resolve_strategy_args — pure logic, exhaustive
# ---------------------------------------------------------------------------


class TestResolveStrategyArgs:
    def test_strategy_4h(self):
        key, svc, warn = resolve_strategy_args(
            strategy="4h",
            heartbeat_key=DEFAULT_HEARTBEAT_KEY,
            service_name="4h",
        )
        assert key == "atomicortex:heartbeat"
        assert svc == "4h"
        assert warn is None

    def test_strategy_15m_derives_key_and_service(self):
        key, svc, warn = resolve_strategy_args(
            strategy="15m",
            heartbeat_key=DEFAULT_HEARTBEAT_KEY,
            service_name="4h",  # left at default
        )
        assert key == "bot_15m_heartbeat"
        assert svc == "15m"
        assert warn is None

    def test_strategy_1h_derives_key_and_service(self):
        key, svc, warn = resolve_strategy_args(
            strategy="1h",
            heartbeat_key=DEFAULT_HEARTBEAT_KEY,
            service_name="4h",
        )
        assert key == "bot_1h_heartbeat"
        assert svc == "1h"
        assert warn is None

    def test_custom_service_name_preserved(self):
        """An explicitly-set --service-name must not be overwritten
        by --strategy auto-derivation."""
        key, svc, warn = resolve_strategy_args(
            strategy="15m",
            heartbeat_key=DEFAULT_HEARTBEAT_KEY,
            service_name="paper-15m",
        )
        assert key == "bot_15m_heartbeat"
        assert svc == "paper-15m"
        assert warn is None

    def test_explicit_heartbeat_key_alone(self):
        """Backward compat: ops scripts that already pass
        --heartbeat-key directly keep working without warnings."""
        key, svc, warn = resolve_strategy_args(
            strategy=None,
            heartbeat_key="custom:key",
            service_name="custom",
        )
        assert key == "custom:key"
        assert svc == "custom"
        assert warn is None

    def test_no_flags_warns_and_falls_through_to_4h(self):
        key, svc, warn = resolve_strategy_args(
            strategy=None,
            heartbeat_key=DEFAULT_HEARTBEAT_KEY,
            service_name="4h",
        )
        assert key == DEFAULT_HEARTBEAT_KEY
        assert svc == "4h"
        assert warn is not None
        assert "--strategy" in warn

    def test_conflict_explicit_key_wins_with_warning(self):
        key, svc, warn = resolve_strategy_args(
            strategy="15m",
            heartbeat_key="manual:key",
            service_name="manual",
        )
        # Explicit key wins.
        assert key == "manual:key"
        assert svc == "manual"
        # And the conflict is surfaced.
        assert warn is not None
        assert "--strategy=15m" in warn
        assert "manual:key" in warn


# ---------------------------------------------------------------------------
# argparse layer
# ---------------------------------------------------------------------------


def _argv(extra: list[str]):
    """Temporarily set sys.argv for parse_args()."""
    saved = sys.argv
    sys.argv = ["run_watchdog.py"] + extra
    try:
        return parse_args()
    finally:
        sys.argv = saved


class TestArgparse:
    def test_strategy_choices_accept_known(self):
        for s in ("4h", "1h", "15m"):
            args = _argv(["--strategy", s])
            assert args.strategy == s

    def test_unknown_strategy_rejected(self):
        with pytest.raises(SystemExit):
            _argv(["--strategy", "30m"])  # not in map

    def test_strategy_default_is_none(self):
        args = _argv([])
        assert args.strategy is None

    def test_heartbeat_key_default_pulled_from_constant(self):
        args = _argv([])
        assert args.heartbeat_key == DEFAULT_HEARTBEAT_KEY

    def test_explicit_heartbeat_key_passes_through(self):
        args = _argv(["--heartbeat-key", "explicit:key"])
        assert args.heartbeat_key == "explicit:key"


# ---------------------------------------------------------------------------
# End-to-end: parse + resolve matches expectations
# ---------------------------------------------------------------------------


class TestArgparseResolveIntegration:
    @pytest.mark.parametrize("strategy,expected_key,expected_svc", [
        ("4h",  "atomicortex:heartbeat", "4h"),
        ("1h",  "bot_1h_heartbeat",      "1h"),
        ("15m", "bot_15m_heartbeat",     "15m"),
    ])
    def test_strategy_flag_full_pipeline(
        self, strategy, expected_key, expected_svc,
    ):
        args = _argv(["--strategy", strategy])
        key, svc, warn = resolve_strategy_args(
            strategy=args.strategy,
            heartbeat_key=args.heartbeat_key,
            service_name=args.service_name,
        )
        assert key == expected_key
        assert svc == expected_svc
        assert warn is None

    def test_bare_invocation_warns(self):
        args = _argv([])
        key, svc, warn = resolve_strategy_args(
            strategy=args.strategy,
            heartbeat_key=args.heartbeat_key,
            service_name=args.service_name,
        )
        assert key == DEFAULT_HEARTBEAT_KEY
        assert warn is not None
