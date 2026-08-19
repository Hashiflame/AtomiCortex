#!/usr/bin/env python3
"""
AtomiCortex Signal Freshness Checker.

Freshness is read from two sources, in this order:

  1. the bot's heartbeat in Redis, field ``last_signal_ts`` — the bot
     publishes the moment it persisted a signal, because the bot is the
     process that knows;
  2. ``MAX(created_at)`` in the SQLite signal ledger, opened readonly.

Redis is first on purpose.  Reading the ledger file means reaching into
another process's private storage across the filesystem, and under the
unit's ``ProtectHome=read-only`` a WAL database cannot create its ``-shm``
sidecar, so the read fails while the bot is perfectly healthy.  When the
heartbeat answers with a published timestamp it decides freshness on its
own and SQLite is consulted only to corroborate it.

``last_signal_ts`` carries three different answers, and PR-0.10 stopped
treating them as one:

  the key is absent       nothing is claimed.  A build from before the
                          field existed, or a payload written between the
                          merge and the restart that introduced it.  The
                          state is UNKNOWN and the SQLite rules below
                          apply unchanged, exactly as before this split;
  the key holds null      the bot is stating, about itself, that it has
                          emitted no signal since ``started_ts``.  That is
                          a fact, not a gap: the ledger is not consulted
                          at all, and the claim is aged against the same
                          threshold every other verdict uses;
  the key holds a number  the moment the bot persisted its last signal.

Anything else in that field is none of the three.  The bot's own
serialiser writes epoch seconds or the contract is gone, so an
unparseable value is a broken first source rather than a claim, and it is
reported as one.

Ten distinct failure modes are reported separately, because the repair
is different for each one:

  no_file      the expected .db is not on disk at all
  no_table     the file exists but signals_log was never created
  never        signals_log exists and has never held a row
  stale        the newest signal is older than the threshold
  no_database  data/ contains no atomicortex*.db whatsoever
  read_error   a source is there but could not be opened or queried —
               either the .db file or the Redis heartbeat
  no_heartbeat the heartbeat key is absent from Redis: the bot is dead,
               or its 60s TTL lapsed
  bridge_lag   the heartbeat published a signal that never reached the
               ledger, by more than the tolerance
  never_since_start
               the heartbeat claims no signal at all since the process
               came up, and the process has been up longer than the
               threshold
  bad_payload  ``last_signal_ts`` is present but is not a number: the
               heartbeat's contract with this checker is broken

Some of them mean the state is UNKNOWN — the check could not do its job
— and those exit non-zero, leaving the oneshot unit in ``failed`` where
``systemctl status`` shows it:

  no_file, no_database, no_heartbeat, bad_payload  ->  exit 1
  read_error                                       ->  exit 1 ONLY when
      the heartbeat did not establish freshness.  On the Redis heartbeat
      it is always fatal — that IS the first source failing.  On the .db
      file it is fatal only while no published timestamp is in hand;
      with one, freshness is known and just the corroboration is blind,
      so the alert stands and the unit succeeds.

The rest mean the state is known and bad: the check worked, the bot did
not.  They alert, but the unit still succeeds, because failing it would
hide a real monitoring outage behind an ordinary starvation:

  no_table, never, stale, bridge_lag,
  never_since_start                                ->  exit 0

``never_since_start`` sits on the exit-0 side for the same reason: the
bot answered, and the answer was bad.  ``bad_payload`` sits on the
exit-1 side because the bot answered with something that is not an
answer, which leaves freshness unestablished — the definition of UNKNOWN
used everywhere above.

``bridge_lag`` sits deliberately on the exit-0 side.  Two sources that
disagree are a known, contradictory state, not an unknown one, and the
divergence is one-directional: only Redis running ahead of the ledger is
reported.  The ledger running ahead is normal — the Telegram bot and the
reconciler write into the same file.

OPERATIONAL NOTE — stopping the bot while the timer is enabled produces
``no_heartbeat`` and a failed unit within the hour: ``HeartbeatManager``
deletes its key on shutdown and the key would expire in 60s anyway,
while the timer keeps firing hourly.  That is the alert working, not
misfiring.  Stop the timer together with the bot for planned downtime
(``systemctl stop atomicortex-signal-check.timer``) and start it back
with it.

Because the unit is ``Type=oneshot`` fired hourly by a timer, repeat
alerts are suppressed per (database, failure mode) through an on-disk
ledger — see ``src/monitoring/signal_alert_state.py``.  Suppression
applies to delivery only: the journal always records the ERROR, and it
never affects the exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any
from urllib.parse import quote

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
# Of the ten, these four always say the state is unknown and make main()
# exit 1 — the check could not establish freshness at all:
#
#   _EVENT_NO_FILE, _EVENT_NO_DB, _EVENT_NO_HEARTBEAT, _EVENT_BAD_PAYLOAD
#
# _EVENT_READ_ERROR says so conditionally: exit 1 only while no heartbeat
# has established freshness, and exit 0 once one has, because then just
# the corroboration is blind.
#
# These five say the state is known and bad — a starvation report, not a
# monitoring outage — and leave the unit successful:
#
#   _EVENT_NO_TABLE, _EVENT_NEVER, _EVENT_STALE, _EVENT_BRIDGE_LAG,
#   _EVENT_NEVER_SINCE_START
#
# _EVENT_NEVER_SINCE_START and _EVENT_BAD_PAYLOAD land on opposite sides
# of that line although both come from the same field: a null is an
# answer the bot is entitled to give, and a non-number is not an answer.
_EVENT_NO_FILE = "no_file"
_EVENT_NO_TABLE = "no_table"
_EVENT_NEVER = "never"
_EVENT_STALE = "stale"
_EVENT_NO_DB = "no_database"
_EVENT_READ_ERROR = "read_error"
_EVENT_NO_HEARTBEAT = "no_heartbeat"
_EVENT_BRIDGE_LAG = "bridge_lag"
_EVENT_NEVER_SINCE_START = "never_since_start"
_EVENT_BAD_PAYLOAD = "bad_payload"

# Ledger key for the "no database at all" event.  Not a filename — there
# is no file to name — so it is bracketed to keep it out of the namespace
# of real ``atomicortex*.db`` keys.
_NO_DB_KEY = "<no-db>"

# Lives beside the logs: it is the only directory the signal-check unit
# has in ReadWritePaths= under ProtectSystem=strict.
_STATE_FILENAME = "signal_check_state.json"

# Outcomes of reading the heartbeat.  ``disabled`` is not a failure: it
# says no Redis source was configured for this call, which is how every
# pre-PR-0.9 caller behaves and how the SQLite-only rules are reached.
_HB_OK = "ok"
_HB_ABSENT = "absent"
_HB_ERROR = "error"
_HB_DISABLED = "disabled"

# One checker instance serves one strategy, so the key is a flag rather
# than a lookup from database filename — see --heartbeat-key.
_DEFAULT_HEARTBEAT_KEY = "atomicortex:heartbeat"

# Redis logical database index.  Settings carries host, port and password
# but no index, and the bot's clients never select one, so 0 it is.
_REDIS_DB_INDEX = 0

# Hard bound on the Redis round trip.  This is a Type=oneshot unit under
# a timer: a client left hanging on the OS connect timeout would block
# the next firing, so both halves of the round trip are capped.
_REDIS_TIMEOUT_SEC = 2

# How far the published freshness may run ahead of the newest ledger row
# before it is a divergence.  Overridden from Settings / CLI by main();
# the literal here is what a direct caller gets.
_DEFAULT_LAG_TOLERANCE_SEC = 300.0


def _settings_redis_url(settings: Any) -> str:
    """Build the Redis URL from the same fields the bot's clients use.

    The address is never hardcoded here: Settings owns it (REDIS_HOST /
    REDIS_PORT / REDIS_PASSWORD), and ``--redis-url`` overrides the whole
    thing for a one-off run.  Only the logical database index is a
    literal, because no field describes it.
    """
    password = settings.redis_password or ""
    auth = f":{quote(password, safe='')}@" if password else ""
    return (
        f"redis://{auth}{settings.redis_host}:{settings.redis_port}"
        f"/{_REDIS_DB_INDEX}"
    )


def _read_heartbeat(redis_url: str, heartbeat_key: str) -> tuple[str, dict | None]:
    """Read the bot's heartbeat over TCP.

    Returns ``(_HB_OK, payload)``, ``(_HB_ABSENT, None)`` when the key is
    not there, or ``(_HB_ERROR, None)`` when it could not be read or does
    not parse.

    Synchronous on purpose.  This process already spends one
    ``asyncio.run`` per delivered alert, and a second event loop beside
    it in a short-lived oneshot is a source of races for no gain: there
    is exactly one GET to make and nothing to overlap it with.

    ``decode_responses=False`` with an explicit ``.decode()`` mirrors the
    watchdog, the other reader of this key, so both see the same bytes
    and the same failure when they are not UTF-8.
    """
    try:
        import redis

        client = redis.Redis.from_url(
            redis_url,
            decode_responses=False,
            socket_timeout=_REDIS_TIMEOUT_SEC,
            socket_connect_timeout=_REDIS_TIMEOUT_SEC,
        )
        try:
            raw = client.get(heartbeat_key)
        finally:
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        _log.warning(
            f"Heartbeat read failed | key={heartbeat_key}: "
            f"{type(exc).__name__}: {exc}"
        )
        return _HB_ERROR, None

    if raw is None:
        return _HB_ABSENT, None

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        _log.warning(
            f"Heartbeat payload did not parse | key={heartbeat_key}: "
            f"{type(exc).__name__}: {exc}"
        )
        return _HB_ERROR, None

    if not isinstance(payload, dict):
        _log.warning(
            f"Heartbeat payload is not an object | key={heartbeat_key}"
        )
        return _HB_ERROR, None

    return _HB_OK, payload


def _epoch_to_dt(value: Any) -> datetime | None:
    """Epoch seconds (the heartbeat's unit) → aware UTC, or None.

    None covers three cases the caller treats alike: the field is
    missing, it is null, or it is not a number.  All three mean the
    heartbeat has nothing to say about signals.
    """
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


# What ``last_signal_ts`` was found to be.  Before PR-0.10 the first
# three collapsed into a single None, which is how a bot that had never
# signalled was mistaken for a bot with nothing to say about signals.
_LS_ABSENT = "absent"      # the key is not in the payload
_LS_NEVER = "never"        # the key is there and holds null
_LS_AT = "at"              # the key is there and holds epoch seconds
_LS_CORRUPT = "corrupt"    # the key is there and holds something else

# Sentinel for "the key is not in the payload".  A plain ``.get()``
# default cannot express it: None is itself one of the four outcomes.
_MISSING = object()


def _classify_last_signal(payload: dict) -> tuple[str, datetime | None]:
    """Tell the four meanings of ``last_signal_ts`` apart.

    Returns ``(outcome, published_at)``; ``published_at`` is not None
    only for ``_LS_AT``.

    Takes a dict and never None on purpose: whether there is a heartbeat
    to read at all is decided one level up, from the read's status, and
    duplicating that check here would give the same question two
    answers.
    """
    raw = payload.get("last_signal_ts", _MISSING)
    if raw is _MISSING:
        return _LS_ABSENT, None
    if raw is None:
        return _LS_NEVER, None

    published_at = _epoch_to_dt(raw)
    if published_at is None:
        # _epoch_to_dt already ruled out None above, so the only way back
        # is a value it could not convert.
        return _LS_CORRUPT, None
    return _LS_AT, published_at


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
) -> tuple[str | None, str | None, datetime | None]:
    """Open one DB readonly and classify it.

    Returns ``(event, message, last_signal_at)``.  ``event`` is None when
    the database is fresh on its own terms.  ``last_signal_at`` is the
    parsed ``MAX(created_at)`` whenever there is one — it is returned for
    every outcome, fresh or stale, because the caller needs it to
    reconcile against the freshness published in the heartbeat, and a
    verdict alone cannot answer that.
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
            ), None

        cur.execute("SELECT MAX(created_at) FROM signals_log")
        row = cur.fetchone()
        if not row or not row[0]:
            return _EVENT_NEVER, (
                f"⚠️ Starvation Alert: DB '{path.name}' has no signals ever recorded!\n"
                f"The signals_log table exists but is empty — the bot reached "
                f"this database and has not produced a single signal since."
            ), None

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
            ), last_time

        _log.info(f"DB '{path.name}' is fresh. Last signal: {diff_hours:.1f} hours ago.")
        return None, None, last_time
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
    *,
    heartbeat_key: str = _DEFAULT_HEARTBEAT_KEY,
    redis_url: str | None = None,
    lag_tolerance_sec: float = _DEFAULT_LAG_TOLERANCE_SEC,
) -> int:
    """Core logic to check freshness of signals in given DBs.

    Returns the number of alerts SENT.  An alert suppressed by the
    cooldown window is logged but not counted — the counter tracks
    delivery, which is what the caller reports.

    ``state`` defaults to None, which disables de-duplication entirely.

    ``failures`` collects ``"<db>: <event>"`` for the modes that mean the
    state of a database is unknown — ``no_file``, ``read_error`` and
    ``no_heartbeat``.  The caller turns a non-empty list into a non-zero
    exit.  It is filled before de-duplication is consulted: suppression
    is about Telegram traffic, never about whether the unit failed.
    Defaults to None, which records nothing.

    Everything after ``failures`` is keyword-only, so the positional
    contract the existing callers rely on cannot shift under them.

    ``redis_url`` defaults to None, meaning *no first source for this
    call*: the six SQLite events then decide alone, which is precisely
    the behaviour of every caller written before the heartbeat carried
    freshness.  ``main()`` always passes one, built from Settings.

    ``lag_tolerance_sec`` governs one comparison: how far the published
    timestamp may run ahead of the newest ledger row before it is a
    ``bridge_lag``.  A restart never excuses that gap — a signal the bot
    says it emitted has to be in the ledger no matter how young the
    process is — so ``started_ts`` takes no part in the decision.
    """
    alerts = 0
    now = now_fn()

    # One read per run, applied to each database in the loop: the events
    # it produces are keyed by the database being checked, because that
    # is what the operator has to go and look at.
    hb_status, hb_payload = _HB_DISABLED, None
    if redis_url is not None:
        hb_status, hb_payload = _read_heartbeat(redis_url, heartbeat_key)

    for db_path_str in db_paths:
        path = Path(db_path_str)
        threshold_hours = thresholds.get(path.name, thresholds.get("default", 48.0))

        # ------------------------------------------------------------------
        # First source: the heartbeat.
        # ------------------------------------------------------------------
        if hb_status == _HB_ABSENT:
            # The bot is gone, or its 60s TTL lapsed between two hourly
            # firings.  Either way the first source is silent and the
            # second one cannot be trusted to speak for it, so this ends
            # the checks for this database exactly as read_error does.
            msg = (
                f"🚨 Monitoring Failure: heartbeat key '{heartbeat_key}' is not "
                f"in Redis!\n"
                f"The bot that feeds '{path.name}' is not publishing — it is "
                f"stopped, or it has been silent longer than the key's TTL. "
                f"Freshness is UNKNOWN."
            )
            _log.error(msg)
            if failures is not None:
                failures.append(f"{path.name}: {_EVENT_NO_HEARTBEAT}")
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_NO_HEARTBEAT, msg, now):
                alerts += 1
            continue

        if hb_status == _HB_ERROR:
            # Deliberately unlike Watchdog._check_heartbeat_detailed,
            # which fails open here.  Fail-open costs the watchdog a
            # false emergency close; it costs this check its own
            # blindness, silently.
            msg = (
                f"🚨 Monitoring Failure: heartbeat key '{heartbeat_key}' could "
                f"not be read!\n"
                f"Redis is unreachable or returned something unusable, so the "
                f"first freshness source is blind for '{path.name}'. "
                f"Freshness is UNKNOWN."
            )
            _log.error(msg)
            if failures is not None:
                failures.append(f"{path.name}: {_EVENT_READ_ERROR}")
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_READ_ERROR, msg, now):
                alerts += 1
            continue

        published_at: datetime | None = None
        if hb_status == _HB_OK and hb_payload is not None:
            outcome, published_at = _classify_last_signal(hb_payload)

            if outcome == _LS_CORRUPT:
                # The field is written by the bot's own serialiser: epoch
                # seconds, or the contract is gone.  Falling back to the
                # ledger would hide a broken schema for as long as the
                # ledger keeps answering — and under this unit's
                # ProtectHome=read-only it usually does not answer at
                # all, so nobody would ever find out.
                #
                # Its own event and not _EVENT_READ_ERROR's: a Redis
                # outage already inside its window would otherwise
                # swallow the first report of this one, and two unrelated
                # faults must not share a de-duplication key.
                raw = hb_payload.get("last_signal_ts")
                msg = (
                    f"🚨 Monitoring Failure: heartbeat key '{heartbeat_key}' "
                    f"carries an unusable last_signal_ts!\n"
                    f"The field is present but is not a number ({raw!r} of "
                    f"type {type(raw).__name__}), so the first freshness "
                    f"source cannot be parsed for '{path.name}'. "
                    f"Freshness is UNKNOWN."
                )
                _log.error(msg)
                if failures is not None:
                    failures.append(f"{path.name}: {_EVENT_BAD_PAYLOAD}")
                if _send_if_due(reporter, state, cooldown_hours, path.name,
                                _EVENT_BAD_PAYLOAD, msg, now):
                    alerts += 1
                continue

            if outcome == _LS_NEVER:
                # Not a gap in the payload but a statement about the bot
                # itself: it has emitted nothing since it came up.  The
                # ledger cannot overrule a bot reporting on its own
                # output, so it is not consulted at all — which is also
                # what makes this branch survive an unreadable database.
                # The claim is aged against the same threshold every
                # other verdict here uses; a second knob would only be a
                # second thing to get wrong.
                started_at = _epoch_to_dt(hb_payload.get("started_ts"))
                if started_at is not None:
                    silent_hours = (now - started_at).total_seconds() / 3600.0
                    if silent_hours > threshold_hours:
                        msg = (
                            f"⚠️ Starvation Alert: the bot feeding "
                            f"'{path.name}' has emitted nothing since it "
                            f"started {silent_hours:.1f} hours ago! "
                            f"(Threshold: {threshold_hours}h)\n"
                            f"Reported by the bot itself, which came up at "
                            f"{started_at.isoformat()} — the ledger was not "
                            f"consulted."
                        )
                        _log.error(msg)
                        if _send_if_due(reporter, state, cooldown_hours,
                                        path.name, _EVENT_NEVER_SINCE_START,
                                        msg, now):
                            alerts += 1
                        continue

                    # A young process that has not signalled yet is not a
                    # fault.  R11 lives behind the ledger and this branch
                    # never reaches it, so the recovery has to happen
                    # here: without it the window opened by an earlier
                    # claim would outlive the restart that ended it and
                    # mute the next relapse.
                    _log.info(
                        f"Heartbeat '{heartbeat_key}' reports no signal yet, "
                        f"{silent_hours:.1f}h since start — inside the "
                        f"{threshold_hours}h threshold for '{path.name}'"
                    )
                    if state is not None:
                        state.clear_db(path.name)
                    continue

                # A claim that cannot be aged is no verdict.  started_ts
                # is missing, null or unparseable, so this degrades to
                # exactly what the checker did before PR-0.10 rather than
                # inventing an age.
                _log.debug(
                    f"Heartbeat '{heartbeat_key}' reports no signal since "
                    f"start but carries no usable started_ts — falling back "
                    f"to the ledger for '{path.name}'"
                )

            elif outcome == _LS_ABSENT:
                # M9: the field is not there at all — a build from before
                # it existed, or a payload written between the merge and
                # the restart.  Nothing is claimed, so nothing is a
                # fault, and this is DEBUG rather than an alert: an
                # hourly message about a missing field would only teach
                # the operator to ignore alerts.
                _log.debug(
                    f"Heartbeat '{heartbeat_key}' carries no last_signal_ts — "
                    f"falling back to the ledger for '{path.name}'"
                )

        # The heartbeat decides freshness only when it actually published
        # a timestamp.  Reaching here with published_at None means the
        # payload claimed nothing that could be aged — no key, or a claim
        # with no usable started_ts — and everything below this line is
        # then the pre-PR-0.9 behaviour, unchanged.
        if published_at is not None:
            age_hours = (now - published_at).total_seconds() / 3600.0
            if age_hours > threshold_hours:
                msg = (
                    f"⚠️ Starvation Alert: DB '{path.name}' has not had a signal "
                    f"in {age_hours:.1f} hours! (Threshold: {threshold_hours}h)\n"
                    f"Published by the bot itself at {published_at.isoformat()}."
                )
                _log.error(msg)
                if _send_if_due(reporter, state, cooldown_hours, path.name,
                                _EVENT_STALE, msg, now):
                    alerts += 1
                continue

        # ------------------------------------------------------------------
        # Second source: the ledger.
        # ------------------------------------------------------------------
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
            event, msg, last_signal_at = _inspect_db(path, threshold_hours, now)
        except Exception as e:
            # R12: the read failed, so freshness is unknown — and unknown
            # is not the same as fine.  Swallowing this was the hole PR-0.6
            # left: the journal got an ERROR, the operator got nothing, and
            # the unit reported success.
            #
            # PR-0.9 narrows "unknown": with an authoritative heartbeat
            # freshness IS established and only the corroboration is
            # blind, so the alert stands but the unit does not fail.  That
            # is the whole point of the split — under the unit's
            # ProtectHome=read-only this read fails on a healthy bot.
            if published_at is None:
                blind = True
                tail = (
                    f"Freshness is UNKNOWN for this database — the check reached "
                    f"the file but could not open or query it, so starvation here "
                    f"would go unnoticed."
                )
            else:
                blind = False
                tail = (
                    f"Freshness itself is known — the heartbeat published "
                    f"{published_at.isoformat()} — but the two sources cannot be "
                    f"reconciled while the ledger is unreadable."
                )
            msg = (
                f"🚨 Monitoring Failure: DB '{path.name}' could not be read!\n"
                f"{type(e).__name__}: {e}\n" + tail
            )
            _log.error(msg)
            if failures is not None and blind:
                failures.append(f"{path.name}: {_EVENT_READ_ERROR}")
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_READ_ERROR, msg, now):
                alerts += 1
            continue

        if published_at is None:
            # No first source: the ledger's verdict is the verdict.
            if event is None:
                # R11: freshness restored — drop every recorded event for
                # this database so a relapse is reported immediately
                # rather than being muted by a window opened before the
                # recovery.
                if state is not None:
                    state.clear_db(path.name)
                continue

            _log.error(msg)
            if _send_if_due(reporter, state, cooldown_hours, path.name, event,
                            msg, now):
                alerts += 1
            continue

        # ------------------------------------------------------------------
        # Both sources spoke: reconcile them.
        # ------------------------------------------------------------------
        if event == _EVENT_NO_TABLE:
            # Not a divergence — there is no ledger to diverge from. The
            # schema was never applied, which is its own repair.
            _log.error(msg)
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_NO_TABLE, msg, now):
                alerts += 1
            continue

        if last_signal_at is None:
            lag_msg = (
                f"⚠️ Bridge Lag: the bot published a signal at "
                f"{published_at.isoformat()}, but '{path.name}' holds no rows "
                f"at all!\n"
                f"The heartbeat and the ledger disagree: the write never "
                f"landed. Freshness is fine, the record of it is not."
            )
            _log.error(lag_msg)
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_BRIDGE_LAG, lag_msg, now):
                alerts += 1
            continue

        lag_sec = (published_at - last_signal_at).total_seconds()
        if lag_sec > lag_tolerance_sec:
            lag_msg = (
                f"⚠️ Bridge Lag: the bot's published freshness runs "
                f"{lag_sec:.0f}s ahead of '{path.name}'! "
                f"(Tolerance: {lag_tolerance_sec:.0f}s)\n"
                f"Published {published_at.isoformat()}, newest row "
                f"{last_signal_at.isoformat()} — a signal the bot considers "
                f"emitted never reached the ledger."
            )
            _log.error(lag_msg)
            if _send_if_due(reporter, state, cooldown_hours, path.name,
                            _EVENT_BRIDGE_LAG, lag_msg, now):
                alerts += 1
            continue

        # Only the divergence direction that means a lost write is
        # reported.  The ledger running ahead is normal: the Telegram bot
        # and the reconciler write into the same file.
        if state is not None:
            state.clear_db(path.name)

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
    ap.add_argument(
        "--heartbeat-key",
        default=_DEFAULT_HEARTBEAT_KEY,
        help="Redis key the bot publishes freshness to (one strategy per run)",
    )
    ap.add_argument(
        "--redis-url",
        default=None,
        help="Override the Redis URL; None builds it from Settings",
    )
    ap.add_argument(
        "--bridge-lag-tolerance",
        type=float,
        default=None,
        help="Override the heartbeat/ledger divergence tolerance (seconds)",
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
    lag_tolerance_sec = (
        args.bridge_lag_tolerance
        if args.bridge_lag_tolerance is not None
        else settings.signal_bridge_lag_tolerance_sec
    )
    redis_url = (
        args.redis_url
        if args.redis_url is not None
        else _settings_redis_url(settings)
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
                             cooldown_hours, failures,
                             heartbeat_key=args.heartbeat_key,
                             redis_url=redis_url,
                             lag_tolerance_sec=lag_tolerance_sec)
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
