#!/usr/bin/env python3
"""
AtomiCortex Signal Freshness Checker.

Reads MAX(created_at) from SQLite DBs in readonly mode and sends a
Telegram alert if any DB has not recorded a signal for a prolonged
period (starvation).

Six distinct failure modes are reported separately, because the repair
is different for each one:

  no_file     the expected .db is not on disk at all
  no_table    the file exists but signals_log was never created
  never       signals_log exists and has never held a row
  stale       rows exist, but the newest is older than the threshold
  no_database data/ contains no atomicortex*.db whatsoever
  read_error  the file is there but could not be opened or queried

Three of them mean the state of a database is UNKNOWN — the check could
not do its job — and those exit non-zero, leaving the oneshot unit in
``failed`` where ``systemctl status`` shows it:

  no_file, read_error, no_database   ->  exit 1

The other three mean the state is known and bad: the check worked, the
bot did not. They alert, but the unit still succeeds, because failing it
would hide a real monitoring outage behind an ordinary starvation:

  no_table, never, stale             ->  exit 0

Because the unit is ``Type=oneshot`` fired hourly by a timer, repeat
alerts are suppressed per (database, failure mode) through an on-disk
ledger — see ``src/monitoring/signal_alert_state.py``.  Suppression
applies to delivery only: the journal always records the ERROR.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.logger import get_logger, setup_logging
from src.monitoring.signal_alert_state import SignalAlertState
from src.monitoring.telegram_reporter import TelegramReporter

_log = get_logger("signal_freshness")

# Failure modes.  Each is its own de-duplication key, so a database that
# moves from one mode to another is reported at once instead of being
# muted by the window opened for the previous mode.
#
# Three of the six say the state of a database is unknown and make main()
# exit 1: _EVENT_NO_FILE, _EVENT_READ_ERROR and _EVENT_NO_DB.  The other
# three — _EVENT_NO_TABLE, _EVENT_NEVER, _EVENT_STALE — say the state is
# known and bad, which is a starvation report, not a monitoring outage.
_EVENT_NO_FILE = "no_file"
_EVENT_NO_TABLE = "no_table"
_EVENT_NEVER = "never"
_EVENT_STALE = "stale"
_EVENT_NO_DB = "no_database"
_EVENT_READ_ERROR = "read_error"

# Ledger key for the "no database at all" event.  Not a filename — there
# is no file to name — so it is bracketed to keep it out of the namespace
# of real ``atomicortex*.db`` keys.
_NO_DB_KEY = "<no-db>"

# Lives beside the logs: it is the only directory the signal-check unit
# has in ReadWritePaths= under ProtectSystem=strict.
_STATE_FILENAME = "signal_check_state.json"


def _send_if_due(
    reporter: Any,
    state: "SignalAlertState | None",
    cooldown_hours: float,
    db_name: str,
    event: str,
    msg: str,
    now: datetime,
) -> bool:
    """Deliver ``msg`` unless the same alert is still inside its window.

    Returns True if the alert was actually sent.  Callers log the ERROR
    themselves before calling: the journal must show every occurrence,
    only the Telegram traffic is rate-limited.
    """
    if state is not None and not state.should_alert(db_name, event, now, cooldown_hours):
        _log.info(
            f"Alert for '{db_name}' ({event}) suppressed — already reported "
            f"within the last {cooldown_hours}h"
        )
        return False

    asyncio.run(reporter.send_alert(msg))
    if state is not None:
        state.record(db_name, event, now)
    return True


def _inspect_db(
    path: Path,
    threshold_hours: float,
    now: datetime,
) -> tuple[str, str] | None:
    """Open one DB readonly and classify it.

    Returns ``(event, message)`` for the failure found, or None if the
    database is fresh.
    """
    uri = f"file:{path.absolute()}?mode=ro"

    # Must use uri=True for readonly access
    conn = sqlite3.connect(uri, uri=True)
    try:
        # Check if signals_log exists
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals_log'")
        if not cur.fetchone():
            return _EVENT_NO_TABLE, (
                f"⚠️ Starvation Alert: DB '{path.name}' has no signals_log table!\n"
                f"The schema was never applied here, so no signal could ever "
                f"be written. This database has not been initialised by a bot."
            )

        cur.execute("SELECT MAX(created_at) FROM signals_log")
        row = cur.fetchone()
        if not row or not row[0]:
            return _EVENT_NEVER, (
                f"⚠️ Starvation Alert: DB '{path.name}' has no signals ever recorded!\n"
                f"The signals_log table exists but is empty — the bot reached "
                f"this database and has not produced a single signal since."
            )

        last_time_str = row[0]
        # Parse datetime with timezone correctly
        last_time = datetime.fromisoformat(last_time_str)
        if not last_time.tzinfo:
            last_time = last_time.replace(tzinfo=timezone.utc)

        diff = now - last_time
        diff_hours = diff.total_seconds() / 3600.0

        if diff_hours > threshold_hours:
            return _EVENT_STALE, (
                f"⚠️ Starvation Alert: DB '{path.name}' has not had a signal "
                f"in {diff_hours:.1f} hours! (Threshold: {threshold_hours}h)\n"
                f"Last signal: {last_time_str}"
            )

        _log.info(f"DB '{path.name}' is fresh. Last signal: {diff_hours:.1f} hours ago.")
        return None
    finally:
        conn.close()


def check_freshness(
    db_paths: list[str],
    thresholds: dict[str, float],
    reporter: Any,
    now_fn: Callable[[], datetime],
    state: "SignalAlertState | None" = None,
    cooldown_hours: float = 0.0,
    failures: list[str] | None = None,
) -> int:
    """Core logic to check freshness of signals in given DBs.

    Returns the number of alerts SENT.  An alert suppressed by the
    cooldown window is logged but not counted — the counter tracks
    delivery, which is what the caller reports.

    ``state`` defaults to None, which disables de-duplication entirely.

    ``failures`` collects ``"<db>: <event>"`` for the modes that mean the
    state of a database is unknown — ``no_file`` and ``read_error``.  The
    caller turns a non-empty list into a non-zero exit.  It is filled
    before de-duplication is consulted: suppression is about Telegram
    traffic, never about whether the unit failed.  Defaults to None,
    which records nothing.
    """
    alerts = 0
    now = now_fn()

    for db_path_str in db_paths:
        path = Path(db_path_str)
        threshold_hours = thresholds.get(path.name, thresholds.get("default", 48.0))

        # R7: a database that is not on disk is a failure of monitoring,
        # not something to skip quietly.  The bot never created it, or it
        # was deleted — either way nothing here is being watched.
        if not path.exists():
            msg = (
                f"🚨 Monitoring Failure: DB file '{path.name}' is missing from disk!\n"
                f"Expected at {path.absolute()}. Starvation cannot be detected "
                f"for it — this check is blind, not reassuring."
            )
            _log.error(msg)
            if failures is not None:
                failures.append(f"{path.name}: {_EVENT_NO_FILE}")
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_NO_FILE, msg, now):
                alerts += 1
            continue

        try:
            finding = _inspect_db(path, threshold_hours, now)
        except Exception as e:
            # R12: the read failed, so freshness is unknown — and unknown
            # is not the same as fine.  Swallowing this was the hole PR-0.6
            # left: the journal got an ERROR, the operator got nothing, and
            # the unit reported success.
            msg = (
                f"🚨 Monitoring Failure: DB '{path.name}' could not be read!\n"
                f"{type(e).__name__}: {e}\n"
                f"Freshness is UNKNOWN for this database — the check reached "
                f"the file but could not open or query it, so starvation here "
                f"would go unnoticed."
            )
            _log.error(msg)
            if failures is not None:
                failures.append(f"{path.name}: {_EVENT_READ_ERROR}")
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_READ_ERROR, msg, now):
                alerts += 1
            continue

        if finding is None:
            # R11: freshness restored — drop every recorded event for this
            # database so a relapse is reported immediately rather than
            # being muted by a window opened before the recovery.
            if state is not None:
                state.clear_db(path.name)
            continue

        event, msg = finding
        _log.error(msg)
        if _send_if_due(reporter, state, cooldown_hours, path.name, event, msg, now):
            alerts += 1

    return alerts


class _NullReporter:
    async def send_alert(self, msg: str) -> bool:
        _log.error(f"ALERT (telegram off): {msg}")
        return False


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Signal freshness checker")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--threshold-4h", type=float, default=None, help="Override threshold for atomicortex.db")
    ap.add_argument("--threshold-15m", type=float, default=None, help="Override threshold for atomicortex_15m.db")
    ap.add_argument("--threshold-default", type=float, default=None, help="Override default threshold")
    ap.add_argument(
        "--alert-cooldown-hours",
        type=float,
        default=None,
        help="Override the repeat-alert suppression window (hours; 0 disables)",
    )
    return ap


def main() -> None:
    ap = get_parser()
    args = ap.parse_args()
    setup_logging(level_console=args.log_level)

    settings = get_settings()

    if not settings.telegram_bot_token or not settings.telegram_admin_id:
        _log.warning("Telegram not configured, alerts will be skipped.")
        reporter = _NullReporter()
    else:
        reporter = TelegramReporter(
            bot_token=settings.telegram_bot_token,
            admin_id=settings.telegram_admin_id,
        )

    cooldown_hours = (
        args.alert_cooldown_hours
        if args.alert_cooldown_hours is not None
        else settings.signal_alert_cooldown_hours
    )
    state = SignalAlertState(_ROOT / "logs" / _STATE_FILENAME)

    def _now() -> datetime:
        return datetime.now(timezone.utc)

    data_dir = _ROOT / "data"
    db_paths = sorted(str(p) for p in data_dir.glob("atomicortex*.db"))
    if not db_paths:
        # P3/R6: no database means the monitor has nothing to watch. That
        # is an outage of monitoring, and it has to be visible in
        # `systemctl status`, not just in the journal — hence exit 1 on a
        # Type=oneshot unit. Delivery is de-duplicated (the timer fires
        # hourly); the non-zero exit is not.
        msg = (
            f"🚨 Monitoring Failure: no atomicortex*.db found in {data_dir}!\n"
            f"There is nothing to check, so signal starvation cannot be "
            f"detected at all. Freshness monitoring is DOWN."
        )
        _log.error(msg)
        _send_if_due(reporter, state, cooldown_hours, _NO_DB_KEY,
                     _EVENT_NO_DB, msg, _now())
        sys.exit(1)

    thresholds = {
        "atomicortex.db": args.threshold_4h if args.threshold_4h is not None else settings.signal_stale_hours_4h,
        "atomicortex_15m.db": args.threshold_15m if args.threshold_15m is not None else settings.signal_stale_hours_15m,
        "default": args.threshold_default if args.threshold_default is not None else settings.signal_stale_hours_default
    }

    failures: list[str] = []
    alerts = check_freshness(db_paths, thresholds, reporter, _now, state,
                             cooldown_hours, failures)
    _log.info(f"Freshness check completed. Alerts generated: {alerts}")

    if failures:
        # R12: same reasoning as the no-database branch above, one level
        # down — a database whose state could not be established leaves
        # this check blind, and blind must be visible in `systemctl
        # status`.  The INFO line above is emitted first so the journal
        # always carries the alert count, failed run or not.  Delivery is
        # de-duplicated; the non-zero exit is not.
        _log.error(
            "Monitoring failure(s): "
            + "; ".join(failures)
            + " — the check could not confirm freshness"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
