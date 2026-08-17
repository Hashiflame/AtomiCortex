"""Persistent de-duplication ledger for signal-starvation alerts.

Why this module exists
----------------------
``scripts/check_signal_freshness.py`` runs as a ``Type=oneshot`` unit
fired by ``atomicortex-signal-check.timer`` once an hour. Nothing of the
process survives to the next firing, so the in-memory ``_last_alert_ts``
cooldown that ``src/execution/watchdog.py`` uses is not available here:
a starving database would produce 24 identical Telegram messages a day
until someone noticed. This file is that cooldown, moved to disk.

What it stores
--------------
A flat JSON object mapping ``"<db name>|<event>"`` to the ISO-8601 UTC
timestamp of the last alert sent for that pair. The key is a *pair*, not
just the database, so that a change in failure mode — "no signals_log
table" becoming "table exists but empty" — is reported immediately
instead of being swallowed by a window opened for the previous mode.

Recovery semantics
------------------
``clear_db`` drops every event of one database. It is called the moment
a database is found fresh again, mirroring ``self._last_alert_ts = 0.0``
on the watchdog's recovery path: a relapse after a recovery is news, and
must not be muted by a window opened before the recovery.

Fail-soft on every path
-----------------------
A missing, corrupt or unwritable file degrades to "no history", which
means alerts are sent rather than suppressed. Over-alerting is the safe
direction for a monitor; silence is the failure mode PR-0.6 exists to
remove.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.logger import get_logger

_log = get_logger(__name__)

# Separator between the two halves of a ledger key. Not a legal character
# in the ``atomicortex*.db`` filenames this ledger is keyed by, so a key
# can always be split back apart unambiguously.
_KEY_SEP = "|"


def _parse_dt(s: object) -> datetime | None:
    """Parse a persisted ISO-8601 stamp, or return None if unusable."""
    if not isinstance(s, str):
        return None
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class SignalAlertState:
    """JSON-backed record of which starvation alerts were already sent."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_alert(
        self,
        db_name: str,
        event: str,
        now: datetime,
        cooldown_hours: float,
    ) -> bool:
        """True if this (db, event) may be reported again right now.

        A window of zero or less disables suppression entirely — that is
        what callers pass when they have no ledger to consult.
        """
        if cooldown_hours <= 0:
            return True

        last = _parse_dt(self.load().get(self._key(db_name, event)))
        if last is None:
            return True

        elapsed_hours = (now - last).total_seconds() / 3600.0
        if elapsed_hours < 0:
            # Stamp is in the future: the clock moved backwards, or the
            # file was copied from another host. Treating that as "inside
            # the window" would mute this alert until the stamp aged out
            # — potentially forever. Alert, and let the write below
            # correct the stamp.
            return True
        return elapsed_hours >= cooldown_hours

    def record(self, db_name: str, event: str, now: datetime) -> None:
        """Stamp this (db, event) as reported and persist immediately.

        The write is not deferred: ``main()`` may ``sys.exit(1)`` right
        after a monitoring-failure alert, and a buffered ledger would be
        lost exactly in the case that repeats every hour.
        """
        state = self.load()
        state[self._key(db_name, event)] = now.astimezone(timezone.utc).isoformat()
        self.save(state)

    def clear_db(self, db_name: str) -> None:
        """Forget every event recorded for ``db_name`` (recovery)."""
        state = self.load()
        prefix = f"{db_name}{_KEY_SEP}"
        remaining = {k: v for k, v in state.items() if not k.startswith(prefix)}
        if len(remaining) != len(state):
            self.save(remaining)

    def load(self) -> dict[str, str]:
        """Read the ledger. ``{}`` if missing, corrupted or unreadable."""
        return self._read_raw()

    def save(self, state: dict[str, str]) -> None:
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
                "signal_alert_state: save failed for {p}: {err} — "
                "de-duplication is degraded to 'always alert'",
                p=str(self._path), err=str(exc),
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _key(db_name: str, event: str) -> str:
        return f"{db_name}{_KEY_SEP}{event}"

    def _read_raw(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("top-level JSON is not an object")
            return {str(k): v for k, v in raw.items()}
        except Exception as exc:
            _log.warning(
                "signal_alert_state: corrupted/unreadable {p} ({err}) — "
                "starting with an empty ledger",
                p=str(self._path), err=str(exc),
            )
            return {}
