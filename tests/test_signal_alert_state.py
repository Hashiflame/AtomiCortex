"""Tests — ``SignalAlertState``, the dedup ledger for starvation alerts.

The freshness checker is a oneshot unit: it holds no memory between the
hourly timer firings, so "do not repeat the same alert within N hours"
needs a file.  Same shape as ``RiskStateStore``: path from the
constructor, ``{}`` on a missing or corrupt file, atomic writes,
fail-soft on an unwritable target.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.monitoring.signal_alert_state import SignalAlertState


_NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """A first run has no ledger; that is not an error."""
    store = SignalAlertState(tmp_path / "signal_check_state.json")
    assert store.load() == {}


def test_load_returns_empty_on_corrupt_json(tmp_path: Path) -> None:
    """A truncated / garbage file degrades to 'no history', never raises."""
    path = tmp_path / "signal_check_state.json"
    path.write_text("{not json at all", encoding="utf-8")
    store = SignalAlertState(path)
    assert store.load() == {}


def test_save_is_atomic_and_leaves_no_tmp(tmp_path: Path) -> None:
    """After save() the directory holds the target file and nothing else."""
    path = tmp_path / "nested" / "signal_check_state.json"
    store = SignalAlertState(path)
    store.save({"atomicortex.db|stale": _NOW.isoformat()})

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "atomicortex.db|stale": _NOW.isoformat()
    }
    leftovers = [p.name for p in path.parent.iterdir() if p.name != path.name]
    assert not leftovers, f"temporary files left behind: {leftovers}"


def test_save_is_failsoft_on_unwritable_dir(tmp_path: Path) -> None:
    """An unwritable target logs and returns; it never propagates."""
    ro_dir = tmp_path / "readonly"
    ro_dir.mkdir()
    path = ro_dir / "signal_check_state.json"
    os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR)
    try:
        store = SignalAlertState(path)
        store.save({"atomicortex.db|stale": _NOW.isoformat()})  # must not raise
        assert store.load() == {}
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)


def test_clear_db_removes_all_events_of_that_db(tmp_path: Path) -> None:
    """Recovery drops every event type of that DB and nothing else (R11)."""
    path = tmp_path / "signal_check_state.json"
    store = SignalAlertState(path)
    store.record("atomicortex.db", "stale", _NOW)
    store.record("atomicortex.db", "never", _NOW)
    store.record("atomicortex_15m.db", "stale", _NOW)

    store.clear_db("atomicortex.db")

    remaining = store.load()
    assert list(remaining) == ["atomicortex_15m.db|stale"], remaining


def test_should_alert_honours_the_window(tmp_path: Path) -> None:
    """Inside the window: suppressed. Past it: allowed again."""
    store = SignalAlertState(tmp_path / "signal_check_state.json")
    store.record("atomicortex.db", "stale", _NOW)

    assert store.should_alert("atomicortex.db", "stale", _NOW + timedelta(hours=1), 24.0) is False
    assert store.should_alert("atomicortex.db", "stale", _NOW + timedelta(hours=25), 24.0) is True


def test_should_alert_with_zero_window_never_suppresses(tmp_path: Path) -> None:
    """cooldown_hours <= 0 disables de-duplication entirely."""
    store = SignalAlertState(tmp_path / "signal_check_state.json")
    store.record("atomicortex.db", "stale", _NOW)
    assert store.should_alert("atomicortex.db", "stale", _NOW, 0.0) is True


def test_should_alert_ignores_a_future_timestamp(tmp_path: Path) -> None:
    """A backwards clock jump must not freeze the window forever."""
    store = SignalAlertState(tmp_path / "signal_check_state.json")
    store.record("atomicortex.db", "stale", _NOW + timedelta(days=365))
    assert store.should_alert("atomicortex.db", "stale", _NOW, 24.0) is True
