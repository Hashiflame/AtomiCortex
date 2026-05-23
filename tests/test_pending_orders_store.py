"""Crash-safety tests for ``PendingOrdersStore``.

The store is the difference between "bot died mid-trade and the position
has no stop" and "bot recovers cleanly". These tests pin down:

* writes survive a process restart (new instance, same path)
* deletes survive a process restart
* corrupted file at startup → empty state, no raise
* writes are atomic (no half-written file is ever visible)
* TTL drops stale entries on load
* TradeSignal datetime round-trips losslessly
* RiskDecision round-trips losslessly
* unwritable path / construction errors are non-fatal
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.execution.pending_orders_store import (
    PendingOrdersStore,
    _decision_from_dict,
    _decision_to_dict,
    _signal_from_dict,
    _signal_to_dict,
)
from src.risk.risk_engine import RiskDecision, TradeSignal


# ---------------------------------------------------------------------------
# Fixtures — canonical Signal + Decision used across tests
# ---------------------------------------------------------------------------

def _signal() -> TradeSignal:
    return TradeSignal(
        symbol="BTCUSDT-PERP.BINANCE",
        direction=1,
        confidence=0.72,
        regime="trend_up",
        entry_price=50_000.0,
        atr=500.0,
        atr_pct=0.01,
        funding_rate=0.00012,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _decision() -> RiskDecision:
    return RiskDecision(
        approved=True,
        reason="",
        position_size=0.05,
        stop_loss=49_250.0,
        take_profit=51_125.0,
        notional=2_500.0,
        leverage=0.25,
        expected_fee_bps=4.0,
        risk_reward_ratio=1.5,
    )


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "pending_sl_4h.json"


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_signal_round_trip_preserves_datetime(self) -> None:
        sig = _signal()
        d = _signal_to_dict(sig)
        # JSON-serialisable
        assert isinstance(d["timestamp"], str)
        json.dumps(d)
        back = _signal_from_dict(d)
        assert back == sig
        assert back.timestamp == sig.timestamp
        assert back.timestamp.tzinfo is not None

    def test_decision_round_trip(self) -> None:
        dec = _decision()
        d = _decision_to_dict(dec)
        json.dumps(d)
        assert _decision_from_dict(d) == dec


# ---------------------------------------------------------------------------
# Persistence across "process restart" (new instance, same path)
# ---------------------------------------------------------------------------

class TestPersistenceAcrossRestart:
    def test_put_survives_restart(self, store_path: Path) -> None:
        s1 = PendingOrdersStore(store_path)
        s1.put("AC-L-1001", _decision(), _signal())

        s2 = PendingOrdersStore(store_path)
        restored = s2.load_all()
        assert "AC-L-1001" in restored
        assert restored["AC-L-1001"]["decision"] == _decision()
        assert restored["AC-L-1001"]["signal"] == _signal()

    def test_pop_removes_from_disk(self, store_path: Path) -> None:
        s1 = PendingOrdersStore(store_path)
        s1.put("AC-L-1001", _decision(), _signal())
        s1.put("AC-S-1002", _decision(), _signal())

        popped = s1.pop("AC-L-1001")
        assert popped is not None
        assert popped["signal"] == _signal()

        # Reload — only the surviving entry should remain
        s2 = PendingOrdersStore(store_path)
        restored = s2.load_all()
        assert "AC-L-1001" not in restored
        assert "AC-S-1002" in restored

    def test_pop_unknown_returns_none(self, store_path: Path) -> None:
        s = PendingOrdersStore(store_path)
        assert s.pop("nonexistent") is None


# ---------------------------------------------------------------------------
# Crash-safety — corrupted file, atomic write
# ---------------------------------------------------------------------------

class TestCrashSafety:
    def test_corrupted_json_yields_empty_state(self, store_path: Path) -> None:
        store_path.write_text("{not valid json", encoding="utf-8")
        s = PendingOrdersStore(store_path)
        # Must not raise; load_all returns {}
        assert s.load_all() == {}

    def test_non_object_json_yields_empty_state(self, store_path: Path) -> None:
        store_path.write_text("[1, 2, 3]", encoding="utf-8")
        s = PendingOrdersStore(store_path)
        assert s.load_all() == {}

    def test_missing_file_yields_empty_state(self, store_path: Path) -> None:
        assert not store_path.exists()
        s = PendingOrdersStore(store_path)
        assert s.load_all() == {}

    def test_atomic_write_no_tmp_files_left(self, store_path: Path) -> None:
        s = PendingOrdersStore(store_path)
        for i in range(5):
            s.put(f"AC-L-{i}", _decision(), _signal())
        tmp_files = list(store_path.parent.glob("pending_sl_4h.json.*.tmp"))
        assert tmp_files == [], (
            f"atomic write leaked temp files: {tmp_files}"
        )

    def test_replace_failure_does_not_corrupt_existing_file(
        self, store_path: Path
    ) -> None:
        """Even if os.replace fails mid-write, the old file content must
        remain readable (no half-written content)."""
        s = PendingOrdersStore(store_path)
        s.put("AC-L-1001", _decision(), _signal())
        good_content = store_path.read_text(encoding="utf-8")

        # Force a flush failure
        with patch(
            "src.execution.pending_orders_store.os.replace",
            side_effect=OSError("disk full"),
        ):
            s.put("AC-L-1002", _decision(), _signal())  # fail-soft, no raise

        # Original file still intact
        assert store_path.read_text(encoding="utf-8") == good_content


# ---------------------------------------------------------------------------
# TTL — stale entries dropped on load
# ---------------------------------------------------------------------------

class TestTTL:
    def test_expired_entries_dropped_on_load(self, store_path: Path) -> None:
        # 0.1s TTL — write, sleep, reload
        s1 = PendingOrdersStore(store_path, ttl_seconds=0.05)
        s1.put("AC-L-OLD", _decision(), _signal())
        import time
        time.sleep(0.1)
        s2 = PendingOrdersStore(store_path, ttl_seconds=0.05)
        assert s2.load_all() == {}

    def test_fresh_entries_kept_on_load(self, store_path: Path) -> None:
        s1 = PendingOrdersStore(store_path, ttl_seconds=3600)
        s1.put("AC-L-NEW", _decision(), _signal())
        s2 = PendingOrdersStore(store_path, ttl_seconds=3600)
        assert "AC-L-NEW" in s2.load_all()

    def test_unparseable_entry_dropped_silently(self, store_path: Path) -> None:
        # Hand-craft a file with one good + one malformed entry
        good_entry = {
            "created_at": __import__("time").time(),
            "decision": _decision_to_dict(_decision()),
            "signal": _signal_to_dict(_signal()),
        }
        bad_entry = {"created_at": __import__("time").time(), "decision": {}, "signal": {}}
        store_path.write_text(
            json.dumps({"good": good_entry, "bad": bad_entry}),
            encoding="utf-8",
        )
        s = PendingOrdersStore(store_path)
        restored = s.load_all()
        assert "good" in restored
        assert "bad" not in restored


# ---------------------------------------------------------------------------
# Fail-soft — bad paths
# ---------------------------------------------------------------------------

class TestFailSoft:
    def test_put_on_unwritable_path_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        # Path under a file (parent is not a directory) — mkdir + write will fail
        bad_parent = tmp_path / "afile"
        bad_parent.write_text("hi")
        store_path = bad_parent / "store.json"
        s = PendingOrdersStore(store_path)
        # Must not raise
        s.put("AC-L-1", _decision(), _signal())

    def test_len_and_contains(self, store_path: Path) -> None:
        s = PendingOrdersStore(store_path)
        assert len(s) == 0
        assert "x" not in s
        s.put("x", _decision(), _signal())
        assert len(s) == 1
        assert "x" in s
        s.pop("x")
        assert len(s) == 0
        assert "x" not in s
