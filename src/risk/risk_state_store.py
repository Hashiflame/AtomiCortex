"""Crash-safe persistence for ``PortfolioTracker`` + ``CircuitBreaker`` state.

Why this module exists
----------------------
Both classes keep risk-critical counters in memory only. On every restart
they reset to zero, so a bot that lost -2.9 % over the morning and was
restarted at noon could still lose another -3 % before the daily hard
breaker kicked in — defeating the very limit it was meant to enforce.

What it persists
----------------
A flat JSON document with primitive scalars (no positions; those are
reconciled separately). Each mutation writes a full checkpoint via
``tempfile.mkstemp`` + ``fsync`` + ``os.replace`` — POSIX rename is
atomic, so a crash mid-write can never produce a half-written file.

Temporal-reset semantics
------------------------
``load()`` honours daily / weekly boundaries: if the persisted
``day_start`` is older than today's UTC midnight, the daily counter is
dropped (treated as a fresh day) and ``day_start`` is bumped. Same for
``week_start`` vs the most recent Monday-00:00 UTC. This means a bot
that was stopped *across* a reset still starts each new period clean
— losses from yesterday don't leak into today's limit.

Fail-soft on every path
-----------------------
A corrupted / unreadable / unwritable file logs a warning and returns
an empty state. Callers run with in-memory accounting (the pre-fix
behaviour) — degraded but never a hard stop.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Time helpers — keep day / week boundaries computed in exactly one place.
# ---------------------------------------------------------------------------

def _today_start_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _week_start_utc() -> datetime:
    today = _today_start_utc()
    return today - timedelta(days=today.weekday())


def _parse_dt(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class RiskStateStore:
    """JSON-backed persistence for portfolio + breaker risk counters."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Read the persisted state, applying daily / weekly resets.

        Returns ``{}`` if the file is missing / corrupted / unreadable.
        """
        raw = self._read_raw()
        if not raw:
            return {}

        today_start = _today_start_utc()
        week_start = _week_start_utc()

        # --- Daily reset ---
        persisted_day = _parse_dt(raw.get("day_start"))
        if persisted_day is None or persisted_day < today_start:
            # Boundary crossed (or unparseable) → start fresh for this day.
            raw["daily_realized_pnl"] = 0.0
            raw["day_start"] = today_start.isoformat()
            # Breaker daily-trigger flag is per-day; clear it too.
            raw["breaker_daily_triggered"] = False
            raw["breaker_daily_trigger_reason"] = ""

        # --- Weekly reset ---
        persisted_week = _parse_dt(raw.get("week_start"))
        if persisted_week is None or persisted_week < week_start:
            raw["weekly_realized_pnl"] = 0.0
            raw["week_start"] = week_start.isoformat()

        return raw

    def save(self, state: dict[str, Any]) -> None:
        """Atomically persist ``state`` to ``self._path``. Fail-soft."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=self._path.name + ".",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            _log.warning(
                "risk_state_store: save failed for {p}: {err}",
                p=str(self._path), err=str(exc),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_raw(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("top-level JSON is not an object")
            return raw
        except Exception as exc:
            _log.warning(
                "risk_state_store: corrupted/unreadable {p} ({err}) — "
                "starting with empty state",
                p=str(self._path), err=str(exc),
            )
            return {}
