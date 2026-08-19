import pytest
import sqlite3
import sys
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta

from src.monitoring.telegram_reporter import TelegramReporter
from scripts.check_signal_freshness import check_freshness, get_parser, main


@pytest.fixture
def mock_reporter():
    reporter = MagicMock(spec=TelegramReporter)
    reporter.send_alert = AsyncMock(return_value=True)
    return reporter


@pytest.fixture
def now_fn():
    def _now():
        return datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
    return _now


def setup_db(db_path: str, created_at: str | None = None, create_table: bool = True):
    conn = sqlite3.connect(db_path)
    if create_table:
        conn.execute("CREATE TABLE IF NOT EXISTS signals_log (id INTEGER PRIMARY KEY, created_at TIMESTAMP)")
        conn.execute("DELETE FROM signals_log")
        if created_at is not None:
            conn.execute("INSERT INTO signals_log (created_at) VALUES (?)", (created_at,))
    conn.commit()
    conn.close()


def test_fresh_signal_no_alert(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    # 1 hour ago
    fresh_time = (now_fn() - timedelta(hours=1)).isoformat()
    setup_db(db_path, created_at=fresh_time)

    thresholds = {"atomicortex.db": 48.0}
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert alerts == 0
    mock_reporter.send_alert.assert_not_called()


def test_stale_signal_triggers_alert(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    # 72 hours ago
    stale_time = (now_fn() - timedelta(hours=72)).isoformat()
    setup_db(db_path, created_at=stale_time)

    thresholds = {"atomicortex.db": 48.0}
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert alerts == 1
    mock_reporter.send_alert.assert_called_once()
    call_arg = mock_reporter.send_alert.call_args[0][0]
    assert "atomicortex.db" in call_arg
    assert "72.0" in call_arg or "72" in call_arg


def test_empty_table_triggers_alert(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=None)

    thresholds = {"atomicortex.db": 48.0}
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert alerts == 1
    mock_reporter.send_alert.assert_called_once()
    assert "no signals ever recorded" in mock_reporter.send_alert.call_args[0][0]


def test_no_table_triggers_alert(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, create_table=False)

    thresholds = {"atomicortex.db": 48.0}
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert alerts == 1
    mock_reporter.send_alert.assert_called_once()
    assert "signals_log" in mock_reporter.send_alert.call_args[0][0]


def test_missing_db_file_triggers_alert(tmp_path, mock_reporter, now_fn):
    """R7: a DB file that is not on disk is a monitoring failure, not a skip.

    Replaces test_missing_db_skipped_with_warning, whose assertions were the
    exact opposite: silence was the bug PR-0.6 exists to remove.
    """
    db_path = str(tmp_path / "missing.db")
    thresholds = {"missing.db": 48.0}
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert alerts == 1
    mock_reporter.send_alert.assert_called_once()
    assert "missing.db" in mock_reporter.send_alert.call_args[0][0]


def test_per_db_thresholds(tmp_path, mock_reporter, now_fn):
    db1 = str(tmp_path / "atomicortex_4h.db")
    db2 = str(tmp_path / "atomicortex_15m.db")
    
    time_4h_ago = (now_fn() - timedelta(hours=4)).isoformat()
    time_20h_ago = (now_fn() - timedelta(hours=20)).isoformat()
    
    setup_db(db1, created_at=time_20h_ago) # Stale for 15m but fine for 48h
    setup_db(db2, created_at=time_4h_ago)  # Fine for 15m (6h threshold)
    
    thresholds = {
        "atomicortex_4h.db": 48.0,
        "atomicortex_15m.db": 6.0
    }
    
    alerts = check_freshness([db1, db2], thresholds, mock_reporter, now_fn)
    assert alerts == 0
    
    # Now make db2 stale
    setup_db(db2, created_at=(now_fn() - timedelta(hours=10)).isoformat())
    alerts = check_freshness([db1, db2], thresholds, mock_reporter, now_fn)
    assert alerts == 1
    assert "atomicortex_15m.db" in mock_reporter.send_alert.call_args[0][0]


def test_readonly_access(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    
    thresholds = {"atomicortex.db": 48.0}
    
    # Actually, the check_freshness function connects in ro mode inside itself.
    # We can test that by intercepting sqlite3.connect and checking uri=True.
    
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)
    assert alerts == 0
    
    # To really test sqlite ro, open a connection to the db file with uri=True
    # and mode=ro, and try to INSERT.
    import sqlite3
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO signals_log (created_at) VALUES ('something')")
    conn.close()


def test_null_reporter_still_counts_alerts(tmp_path, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    
    thresholds = {"atomicortex.db": 48.0}
    
    from scripts.check_signal_freshness import _NullReporter
    reporter = _NullReporter()
    
    alerts = check_freshness([db_path], thresholds, reporter, now_fn)
    assert alerts == 1


# ===========================================================================
# PR-0.6
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers shared by the PR-0.6 blocks
# ---------------------------------------------------------------------------

class _StubSettings:
    """Just the fields main() reads — no .env, no pydantic, no banner."""

    telegram_bot_token = "stub-token"
    telegram_admin_id = "42"
    signal_stale_hours_4h = 48.0
    signal_stale_hours_15m = 6.0
    signal_stale_hours_default = 48.0
    signal_alert_cooldown_hours = 24.0
    signal_bridge_lag_tolerance_sec = 300.0
    redis_host = "localhost"
    redis_port = 6379
    redis_password = ""


@pytest.fixture
def main_env(tmp_path, monkeypatch):
    """main() rooted at tmp_path, with Telegram and logging stubbed out.

    Returns ``(module, reporter)``.  ``data/`` exists but is empty, which is
    the P3 case; tests that need a database create one inside it.
    """
    import scripts.check_signal_freshness as mod

    reporter = MagicMock()
    reporter.send_alert = AsyncMock(return_value=True)

    monkeypatch.setattr(mod, "_ROOT", tmp_path)
    monkeypatch.setattr(mod, "get_settings", lambda: _StubSettings())
    monkeypatch.setattr(mod, "setup_logging", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "TelegramReporter", lambda *a, **kw: reporter)
    monkeypatch.setattr(sys, "argv", ["check_signal_freshness.py"])

    # PR-0.9: main() reads Redis.  The default stub is a live heartbeat
    # that has not published a signal yet — Redis is reachable but not
    # authoritative, so every pre-existing main() test keeps exercising
    # the SQLite rules unchanged and none of them opens a socket.  Tests
    # that care about the heartbeat re-stub it via _stub_heartbeat().
    # raising defaults to True: if the name ever moves, every main() test
    # fails loudly instead of quietly talking to a real Redis.
    #
    # PR-0.10: "not authoritative" is now expressed by the ABSENCE of
    # last_signal_ts.  A key holding null became a claim the bot makes
    # about itself, so leaving one here would silently change what these
    # tests exercise and make the paragraph above untrue.  started_ts is
    # a real recent stamp for the same reason — the old 0.0 meant 1970,
    # which would hand any future test an age of half a century.
    _stub_now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        mod,
        "_read_heartbeat",
        lambda url, key: ("ok", {
            "process_ts": _stub_now.timestamp(),
            "started_ts": (_stub_now - timedelta(hours=10)).timestamp(),
            "last_bar_ts": None,
            "bars_seen": 0,
        }),
    )

    (tmp_path / "data").mkdir()
    (tmp_path / "logs").mkdir()
    return mod, reporter


def _capture_error_logs():
    """Context-manager-free loguru ERROR sink: returns (records, remove_fn)."""
    from loguru import logger as _loguru_logger

    records: list[str] = []
    sink_id = _loguru_logger.add(records.append, level="ERROR")
    return records, lambda: _loguru_logger.remove(sink_id)


def _messages(reporter):
    """Every message passed to send_alert, in call order."""
    return [call.args[0] for call in reporter.send_alert.call_args_list]


# ---------------------------------------------------------------------------
# P3 / R6 — main(): no database at all is a monitoring failure
# ---------------------------------------------------------------------------

def test_main_exits_1_when_no_db(main_env):
    """R6: absence of any atomicortex*.db must leave the unit failed."""
    mod, _reporter = main_env

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1


def test_main_alerts_when_no_db(main_env):
    """P3: the checker must shout, not return silently."""
    mod, reporter = main_env

    with pytest.raises(SystemExit):
        mod.main()

    reporter.send_alert.assert_called_once()
    assert "data" in reporter.send_alert.call_args[0][0]


def test_main_logs_error_not_warning_when_no_db(main_env):
    """A monitoring outage belongs at ERROR; WARNING scrolls past unread."""
    mod, _reporter = main_env
    records, remove = _capture_error_logs()
    try:
        with pytest.raises(SystemExit):
            mod.main()
    finally:
        remove()

    assert records, "no ERROR record emitted for a missing database"


def test_main_exits_0_when_db_present(main_env):
    """R6 is scoped to the no-database case; a healthy run still returns 0."""
    mod, reporter = main_env
    fresh = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    setup_db(str(mod._ROOT / "data" / "atomicortex.db"), created_at=fresh)

    mod.main()  # must not raise SystemExit

    reporter.send_alert.assert_not_called()


# ---------------------------------------------------------------------------
# R8 — argument parser
# ---------------------------------------------------------------------------

def test_parser_threshold_flags_default_to_none():
    """The existing overrides mean 'take it from Settings', not 'zero'."""
    args = get_parser().parse_args([])
    assert args.threshold_4h is None
    assert args.threshold_15m is None
    assert args.threshold_default is None


def test_parser_accepts_alert_cooldown_hours():
    """R8: the de-duplication window is overridable from the command line."""
    args = get_parser().parse_args(["--alert-cooldown-hours", "6"])
    assert args.alert_cooldown_hours == 6.0
    assert isinstance(args.alert_cooldown_hours, float)


def test_parser_alert_cooldown_defaults_to_none():
    """Same convention as the thresholds: None means 'use Settings'."""
    args = get_parser().parse_args([])
    assert args.alert_cooldown_hours is None


# ---------------------------------------------------------------------------
# P4 — de-duplication
# ---------------------------------------------------------------------------

def test_repeated_call_alerts_twice_without_store(tmp_path, mock_reporter, now_fn):
    """CONTROL for the whole de-duplication block.

    Without a store the same starvation alerts on every call.  If this ever
    goes green for the wrong reason — a cached connection, a memoised
    result, a reused mock — the suppression tests below would pass without
    the store doing anything.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    thresholds = {"atomicortex.db": 48.0}

    first = check_freshness([db_path], thresholds, mock_reporter, now_fn)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert first == 1
    assert second == 1
    assert mock_reporter.send_alert.call_count == 2


def test_cooldown_suppresses_second_alert(tmp_path, mock_reporter, now_fn):
    """P4: the same event on the same DB stays quiet inside the window."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")

    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)

    assert first == 1
    assert second == 0
    assert mock_reporter.send_alert.call_count == 1


def test_cooldown_expires_after_window(tmp_path, mock_reporter, now_fn):
    """Past the window the alert fires again — suppression is not a mute."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")

    check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)
    later = lambda: now_fn() + timedelta(hours=25)  # noqa: E731
    second = check_freshness([db_path], thresholds, mock_reporter, later, state, 24.0)

    assert second == 1
    assert mock_reporter.send_alert.call_count == 2


def test_event_type_change_breaks_window(tmp_path, mock_reporter, now_fn):
    """R10: the key is (db, event) — a different failure gets through at once."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, create_table=False)          # event: no signals_log table
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")

    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)
    setup_db(db_path, created_at=None)             # event: table exists, empty
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)

    assert first == 1
    assert second == 1
    assert mock_reporter.send_alert.call_count == 2


def test_same_event_suppressed_across_store_instances(tmp_path, mock_reporter, now_fn):
    """The unit is oneshot: suppression must survive process exit."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    path = tmp_path / "signal_check_state.json"

    check_freshness([db_path], thresholds, mock_reporter, now_fn,
                    SignalAlertState(path), 24.0)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                             SignalAlertState(path), 24.0)

    assert second == 0
    assert mock_reporter.send_alert.call_count == 1


def test_recovery_clears_state(tmp_path, mock_reporter, now_fn):
    """R11: freshness restored drops the record, so a relapse alerts at once."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")

    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0) == 1

    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0) == 0
    assert state.load() == {}, "recovery must clear the ledger entry"

    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0) == 1
    assert mock_reporter.send_alert.call_count == 2


def test_future_timestamp_does_not_suppress_forever(tmp_path, mock_reporter, now_fn):
    """A clock that jumped backwards must not mute alerts indefinitely."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    state.record("atomicortex.db", "stale", now_fn() + timedelta(days=365))

    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)

    assert alerts == 1


# ---------------------------------------------------------------------------
# R9 — the alert texts must be tellable apart without reading the code
# ---------------------------------------------------------------------------

def _build_four_dbs(tmp_path, now_fn):
    """One DB per failure mode, in a fixed order, plus a path that does not exist."""
    no_table = str(tmp_path / "atomicortex_a.db")
    never = str(tmp_path / "atomicortex_b.db")
    stale = str(tmp_path / "atomicortex_c.db")
    no_file = str(tmp_path / "atomicortex_d.db")

    setup_db(no_table, create_table=False)
    setup_db(never, created_at=None)
    setup_db(stale, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    return [no_table, never, stale, no_file]


def test_no_table_text_names_the_schema(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, create_table=False)
    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    msg = mock_reporter.send_alert.call_args[0][0]
    assert "atomicortex.db" in msg
    assert "signals_log" in msg


def test_never_recorded_text_is_distinct(tmp_path, mock_reporter, now_fn):
    """R9/V8: keeps 'no signals ever recorded' and adds its own context."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=None)
    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    msg = mock_reporter.send_alert.call_args[0][0]
    assert "no signals ever recorded" in msg
    assert "has no signals_log table" not in msg


def test_stale_text_carries_hours_and_threshold(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    msg = mock_reporter.send_alert.call_args[0][0]
    assert "72.0" in msg
    assert "48.0" in msg


def test_four_event_texts_are_pairwise_distinct(tmp_path, mock_reporter, now_fn):
    """No two failure modes may produce the same, or a nested, message."""
    paths = _build_four_dbs(tmp_path, now_fn)
    thresholds = {"default": 48.0}

    alerts = check_freshness(paths, thresholds, mock_reporter, now_fn)
    assert alerts == 4

    msgs = _messages(mock_reporter)
    assert len(set(msgs)) == 4, f"duplicate alert texts: {msgs}"
    for i, a in enumerate(msgs):
        for j, b in enumerate(msgs):
            if i != j:
                assert a not in b, f"message {i} is a substring of message {j}"


def test_no_file_text_differs_from_all_three(tmp_path, mock_reporter, now_fn):
    """R7's missing-file text is its own event, not a reused starvation text."""
    paths = _build_four_dbs(tmp_path, now_fn)
    check_freshness(paths, {"default": 48.0}, mock_reporter, now_fn)

    msgs = _messages(mock_reporter)
    no_file_msg = msgs[-1]
    assert "atomicortex_d.db" in no_file_msg
    assert "no signals ever recorded" not in no_file_msg
    assert "has no signals_log table" not in no_file_msg


# ===========================================================================
# PR-0.7
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers shared by the PR-0.7 blocks
# ---------------------------------------------------------------------------

def _break_db_reads(monkeypatch, exc=None):
    """Make the checker's readonly open fail, leaving setup_db working.

    Only the ``uri=True`` connect — the one _inspect_db uses — is broken,
    so a test can still build and repair databases around the failure.
    Returns a one-element list: set ``flag[0] = False`` to let readonly
    opens succeed again inside the same test.
    """
    import scripts.check_signal_freshness as mod

    broken = [True]
    real_connect = sqlite3.connect
    failure = exc if exc is not None else sqlite3.OperationalError(
        "unable to open database file"
    )

    def _connect(*args, **kwargs):
        if broken[0] and kwargs.get("uri"):
            raise failure
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(mod.sqlite3, "connect", _connect)
    return broken


# ---------------------------------------------------------------------------
# R12 — a database that cannot be read is its own event, not a silent skip
# ---------------------------------------------------------------------------

def test_read_error_triggers_alert(tmp_path, mock_reporter, now_fn, monkeypatch):
    """The defect this PR closes: the read failed, so the check must shout.

    Before the fix check_freshness logged the exception and continued, so
    this asserted `assert 0 == 1` — nothing was sent and nothing counted.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _break_db_reads(monkeypatch)

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    assert alerts == 1
    assert mock_reporter.send_alert.call_count == 1
    msg = mock_reporter.send_alert.call_args[0][0]
    assert "atomicortex.db" in msg
    assert "could not be read" in msg
    assert "OperationalError" in msg
    assert "unable to open database file" in msg


def test_read_error_control_inspect_db_really_raises(tmp_path, now_fn, monkeypatch):
    """CONTROL for the whole read_error block.

    Green before the fix and after it.  It proves the scenario really does
    reach the `except` in check_freshness: _inspect_db raises rather than
    returning a finding.  Without it, a zero alert count in the test above
    could be blamed on a broken monkeypatch instead of the defect.
    """
    from scripts.check_signal_freshness import _inspect_db

    db_path = tmp_path / "atomicortex.db"
    setup_db(str(db_path), created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _break_db_reads(monkeypatch)

    with pytest.raises(sqlite3.OperationalError):
        _inspect_db(db_path, 48.0, now_fn())


def test_read_error_suppressed_by_cooldown(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R12: read_error de-duplicates on (db, event) like the other five."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    _break_db_reads(monkeypatch)

    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)

    assert first == 1
    assert second == 0
    assert mock_reporter.send_alert.call_count == 1


def test_read_error_lands_in_failures(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R12: an unreadable DB is a monitoring failure, so main() must see it."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _break_db_reads(monkeypatch)
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, failures=failures)

    assert alerts == 1
    assert failures == ["atomicortex.db: read_error"]


def test_no_file_lands_in_failures(tmp_path, mock_reporter, now_fn):
    """R7's event joins read_error in the exit-code list: state unknown."""
    db_path = str(tmp_path / "missing.db")
    failures: list[str] = []

    alerts = check_freshness([db_path], {"missing.db": 48.0}, mock_reporter,
                             now_fn, failures=failures)

    assert alerts == 1
    assert failures == ["missing.db: no_file"]


def test_starvation_events_stay_out_of_failures(tmp_path, mock_reporter, now_fn):
    """The boundary: no_table/never/stale are known-bad, not unknown.

    They alert, but they must not fail the unit — the check did its job.
    """
    no_table = str(tmp_path / "atomicortex_a.db")
    never = str(tmp_path / "atomicortex_b.db")
    stale = str(tmp_path / "atomicortex_c.db")
    setup_db(no_table, create_table=False)
    setup_db(never, created_at=None)
    setup_db(stale, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    failures: list[str] = []

    alerts = check_freshness([no_table, never, stale], {"default": 48.0},
                             mock_reporter, now_fn, failures=failures)

    assert alerts == 3
    assert failures == []


def test_read_error_recorded_in_failures_even_when_alert_suppressed(
    tmp_path, mock_reporter, now_fn, monkeypatch
):
    """Suppression is about Telegram traffic, never about the exit code.

    A DB unreadable for days sends one message per window but must leave
    the unit failed on every single run.
    """
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    _break_db_reads(monkeypatch)

    check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0)
    failures: list[str] = []
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                             state, 24.0, failures)

    assert second == 0
    assert failures == ["atomicortex.db: read_error"]


def test_main_exits_1_when_read_error(main_env, monkeypatch):
    """R12: an unreadable DB has to leave the oneshot unit failed."""
    mod, _reporter = main_env
    fresh = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    setup_db(str(mod._ROOT / "data" / "atomicortex.db"), created_at=fresh)
    _break_db_reads(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1


def test_main_exits_1_when_db_file_vanishes(main_env, monkeypatch):
    """The no_file half of the same rule, through the real check_freshness.

    main() globs data/, so the file exists by construction; the proxy adds
    a path that does not, which is the TOCTOU the glob cannot rule out.
    """
    mod, _reporter = main_env
    fresh = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    setup_db(str(mod._ROOT / "data" / "atomicortex.db"), created_at=fresh)

    real_check = mod.check_freshness
    vanished = str(mod._ROOT / "data" / "atomicortex_gone.db")

    def _proxy(db_paths, *args, **kwargs):
        return real_check([*db_paths, vanished], *args, **kwargs)

    monkeypatch.setattr(mod, "check_freshness", _proxy)

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1


def test_read_error_state_cleared_after_successful_read(
    tmp_path, mock_reporter, now_fn, monkeypatch
):
    """R11 covers the sixth event too: a repaired DB forgets the read_error."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    broken = _break_db_reads(monkeypatch)

    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0) == 1

    broken[0] = False
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0) == 0
    assert state.load() == {}, "a readable DB must clear its read_error record"

    broken[0] = True
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state, 24.0) == 1
    assert mock_reporter.send_alert.call_count == 2


def test_read_error_text_distinct_from_all_five(tmp_path, mock_reporter, now_fn,
                                                monkeypatch):
    """R9 extended to six: the sixth text carries none of the other markers.

    Pure string comparison against the literal markers of the five existing
    texts — no four-database fixture is rebuilt for it.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _break_db_reads(monkeypatch)

    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    msgs = _messages(mock_reporter)
    assert len(msgs) == 1
    read_error_msg = msgs[0]

    assert "could not be read" in read_error_msg
    for marker in (
        "Starvation Alert",                  # no_table, never and stale
        "has no signals_log table",          # no_table
        "has no signals ever recorded",      # never
        "has not had a signal",              # stale
        "is missing from disk",              # no_file
        "no atomicortex*.db found",          # no_database
    ):
        assert marker not in read_error_msg, f"read_error reuses '{marker}'"


# ===========================================================================
# PR-0.9 — Redis is the first source, SQLite the second
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers shared by the PR-0.9 blocks
# ---------------------------------------------------------------------------

_REDIS_URL = "redis://localhost:6379/0"


def _stub_heartbeat(monkeypatch, status: str, payload: dict | None = None):
    """Replace the checker's Redis read with a fixed answer.

    ``status`` is one of 'ok' / 'absent' / 'error'.  No socket is opened
    anywhere in this module: check_freshness() is handed a URL, and the
    only thing that URL reaches is this stub.
    """
    import scripts.check_signal_freshness as mod

    monkeypatch.setattr(mod, "_read_heartbeat", lambda url, key: (status, payload))


_UNSET = object()


def _hb(now, *, last_signal_delta=None, started_delta=timedelta(hours=10),
        omit_last_signal=False, last_signal_raw=_UNSET) -> dict:
    """Build a heartbeat payload with epoch-second floats (P5).

    ``last_signal_raw`` writes an arbitrary value into the field, skipping
    the epoch conversion.  It is the only way to build the third case
    PR-0.10 has to tell apart: the key is present and holds something that
    is not a number.  A sentinel and not ``None``, because ``None`` is
    itself one of the three cases.
    """
    payload = {
        "process_ts": now.timestamp(),
        "started_ts": (now - started_delta).timestamp(),
        "last_bar_ts": (now - timedelta(minutes=5)).timestamp(),
        "bars_seen": 42,
    }
    if last_signal_raw is not _UNSET:
        payload["last_signal_ts"] = last_signal_raw
    elif not omit_last_signal:
        payload["last_signal_ts"] = (
            None if last_signal_delta is None
            else (now - last_signal_delta).timestamp()
        )
    return payload


# ---------------------------------------------------------------------------
# The three new command-line knobs (R1, M8, R9)
# ---------------------------------------------------------------------------

def test_parser_heartbeat_key_default():
    """R1: the key is a literal default, not None — there is no Settings field."""
    args = get_parser().parse_args([])
    assert args.heartbeat_key == "atomicortex:heartbeat"


def test_parser_accepts_heartbeat_key():
    """One checker serves one strategy; the key says which one."""
    args = get_parser().parse_args(["--heartbeat-key", "bot_15m_heartbeat"])
    assert args.heartbeat_key == "bot_15m_heartbeat"


def test_parser_redis_url_defaults_to_none():
    """M8: no address is baked into the parser — None means 'ask Settings'."""
    args = get_parser().parse_args([])
    assert args.redis_url is None


def test_parser_accepts_redis_url():
    """M8: a non-local Redis must be reachable without editing the script."""
    args = get_parser().parse_args(["--redis-url", "redis://10.0.0.5:6380/1"])
    assert args.redis_url == "redis://10.0.0.5:6380/1"


def test_parser_accepts_bridge_lag_tolerance():
    """R9: the divergence tolerance is overridable from the command line."""
    args = get_parser().parse_args(["--bridge-lag-tolerance", "600"])
    assert args.bridge_lag_tolerance == 600.0
    assert isinstance(args.bridge_lag_tolerance, float)


def test_parser_bridge_lag_tolerance_defaults_to_none():
    """Same convention as the thresholds: None means 'use Settings'."""
    args = get_parser().parse_args([])
    assert args.bridge_lag_tolerance is None


# ---------------------------------------------------------------------------
# R2 — no_heartbeat: the key is gone, so the first source says nothing
# ---------------------------------------------------------------------------

def test_no_heartbeat_when_key_absent(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R2: an absent key is its own event — the process is dead or the TTL expired."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "absent")

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msg = mock_reporter.send_alert.call_args[0][0]
    assert "heartbeat" in msg.lower()
    assert "atomicortex:heartbeat" in msg


def test_no_heartbeat_lands_in_failures(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R2 + M7: state unknown → exit 1, keyed by the database being checked."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "absent")
    failures: list[str] = []

    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn,
                    failures=failures, redis_url=_REDIS_URL)

    assert failures == ["atomicortex.db: no_heartbeat"]


def test_no_heartbeat_skips_the_rest_of_that_db(tmp_path, mock_reporter, now_fn,
                                                monkeypatch):
    """M6: like read_error, it ends the checks for this database.

    The database here is also stale.  Without the skip the operator gets
    two alerts for one outage, the second of them meaningless: with no
    heartbeat there is nothing to compare the ledger against.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    _stub_heartbeat(monkeypatch, "absent")

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msgs = _messages(mock_reporter)
    assert "has not had a signal" not in msgs[0], "stale leaked past the skip"


def test_no_heartbeat_control_same_db_alone_is_stale(tmp_path, mock_reporter, now_fn):
    """CONTROL for the skip above: without Redis that database does alert stale."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    assert alerts == 1
    assert "has not had a signal" in _messages(mock_reporter)[0]


def test_no_heartbeat_deduplicated_across_runs(tmp_path, mock_reporter, now_fn,
                                               monkeypatch):
    """The timer fires hourly; the ledger keeps Telegram quiet, not the exit code."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    _stub_heartbeat(monkeypatch, "absent")
    thresholds = {"atomicortex.db": 48.0}

    first_failures: list[str] = []
    second_failures: list[str] = []
    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                            24.0, first_failures, redis_url=_REDIS_URL)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                             24.0, second_failures, redis_url=_REDIS_URL)

    assert first == 1
    assert second == 0
    assert mock_reporter.send_alert.call_count == 1
    assert first_failures == second_failures == ["atomicortex.db: no_heartbeat"]


# ---------------------------------------------------------------------------
# R3 — an unreadable Redis is read_error, and the checker is fail-closed
# ---------------------------------------------------------------------------

def test_redis_error_is_a_read_error(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R3: unlike the watchdog, the checker never treats a blind read as fine."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "error")

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msg = mock_reporter.send_alert.call_args[0][0]
    assert "could not be read" in msg
    assert "atomicortex:heartbeat" in msg


def test_redis_error_lands_in_failures(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R3 + M7: the event name is read_error, the ledger key is the database."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "error")
    failures: list[str] = []

    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn,
                    failures=failures, redis_url=_REDIS_URL)

    assert failures == ["atomicortex.db: read_error"]


# ---------------------------------------------------------------------------
# R8/R9/R12 — bridge_lag: Redis newer than SQLite beyond the tolerance
# ---------------------------------------------------------------------------

def test_bridge_lag_when_redis_newer_than_sqlite(tmp_path, mock_reporter, now_fn,
                                                 monkeypatch):
    """R8: the bot published a signal the ledger never received."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1) - timedelta(seconds=301)))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msg = mock_reporter.send_alert.call_args[0][0]
    assert "atomicortex.db" in msg
    assert "301" in msg or "300" in msg


def test_bridge_lag_control_sqlite_alone_is_silent(tmp_path, mock_reporter, now_fn):
    """CONTROL for the whole bridge_lag block — green before the fix and after.

    It proves the database used above is, on its own, perfectly fresh: any
    alert in those tests comes from the reconciliation and cannot be an
    ordinary `stale` in disguise.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn)

    assert alerts == 0
    assert mock_reporter.send_alert.call_count == 0


def test_bridge_lag_control_inside_tolerance_is_silent(tmp_path, mock_reporter,
                                                       now_fn, monkeypatch):
    """R9: 299s of divergence is the heartbeat tick, not a lost signal."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1) - timedelta(seconds=299)))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 0
    assert mock_reporter.send_alert.call_count == 0


def test_bridge_lag_not_in_failures(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R8: the state is known and contradictory, not unknown — exit stays 0."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1) - timedelta(seconds=600)))
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 1
    assert failures == []


def test_no_bridge_lag_when_sqlite_newer(tmp_path, mock_reporter, now_fn, monkeypatch):
    """R12: the Telegram bot and the reconciler also write there — normal."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=6)))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 0


def test_bridge_lag_reported_after_a_recent_restart(tmp_path, mock_reporter, now_fn,
                                                    monkeypatch):
    """A restart is no excuse: a published signal missing from the ledger is lag.

    The bot came up five minutes ago and published a signal two minutes
    ago, while the newest row is ten hours old.  That signal went
    somewhere other than the ledger, which is exactly the divergence this
    check exists to catch — the recency of the restart says nothing about
    it either way.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=10)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(minutes=2),
                        started_delta=timedelta(minutes=5)))
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 1
    assert "Bridge Lag" in _messages(mock_reporter)[0]
    assert failures == []


def test_empty_table_with_authoritative_redis_is_bridge_lag(tmp_path, mock_reporter,
                                                            now_fn, monkeypatch):
    """Redis knows a signal happened and signals_log holds nothing at all."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=None)
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(minutes=2)))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msg = mock_reporter.send_alert.call_args[0][0]
    assert "has no signals ever recorded" not in msg, "never leaked instead of bridge_lag"


def test_bridge_lag_deduplicated_and_cleared_on_recovery(tmp_path, mock_reporter,
                                                         now_fn, monkeypatch):
    """The hourly timer must not turn one divergence into 24 messages."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    thresholds = {"atomicortex.db": 48.0}
    lagging = _hb(now_fn(), last_signal_delta=timedelta(hours=1) - timedelta(seconds=600))

    _stub_heartbeat(monkeypatch, "ok", lagging)
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                           24.0, redis_url=_REDIS_URL) == 1
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                           24.0, redis_url=_REDIS_URL) == 0

    # Sources agree again → the record is dropped, so a relapse is immediate.
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1)))
    assert check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                           24.0, redis_url=_REDIS_URL) == 0
    assert state.load() == {}, "a reconciled pair must clear its bridge_lag record"


# ---------------------------------------------------------------------------
# The boundary between the two sources
# ---------------------------------------------------------------------------

def test_sqlite_read_error_not_fatal_when_redis_fresh(tmp_path, mock_reporter,
                                                      now_fn, monkeypatch):
    """The VM case: freshness is established, only the reconciliation is blind."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1)))
    _break_db_reads(monkeypatch)
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 1
    assert failures == [], "an authoritative Redis must not leave the unit failed"


def test_sqlite_read_error_still_fatal_when_redis_silent(tmp_path, mock_reporter,
                                                         now_fn, monkeypatch):
    """CONTROL for the case above: without a first source, blind is fatal again.

    PR-0.10 moved which payload means "no first source": a null
    last_signal_ts became a claim of its own, so the absence of the key
    is now the only shape that says the heartbeat has nothing to offer.
    The assertion below is unchanged — only the literal expressing "no
    first source" moved with the definition.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), omit_last_signal=True))
    _break_db_reads(monkeypatch)
    failures: list[str] = []

    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn,
                    failures=failures, redis_url=_REDIS_URL)

    assert failures == ["atomicortex.db: read_error"]


def test_absent_redis_url_preserves_today_behaviour(tmp_path, mock_reporter, now_fn,
                                                    monkeypatch):
    """No Redis source configured → the six original events rule alone."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "absent")
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, failures=failures)

    assert alerts == 0
    assert failures == []


def test_stale_verdict_comes_from_redis_when_authoritative(tmp_path, mock_reporter,
                                                           now_fn, monkeypatch):
    """The first source decides starvation; the database only corroborates."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=72),
                        started_delta=timedelta(hours=100)))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msg = mock_reporter.send_alert.call_args[0][0]
    assert "has not had a signal" in msg
    assert "72.0" in msg


def test_two_new_event_texts_are_distinct_from_the_six(tmp_path, mock_reporter,
                                                       now_fn, monkeypatch):
    """Eight events, eight texts: neither newcomer reuses an older marker."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 48.0}

    _stub_heartbeat(monkeypatch, "absent")
    check_freshness([db_path], thresholds, mock_reporter, now_fn, redis_url=_REDIS_URL)

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1) - timedelta(seconds=600)))
    check_freshness([db_path], thresholds, mock_reporter, now_fn, redis_url=_REDIS_URL)

    msgs = _messages(mock_reporter)
    assert len(msgs) == 2
    no_heartbeat_msg, bridge_lag_msg = msgs
    assert no_heartbeat_msg != bridge_lag_msg

    for msg in (no_heartbeat_msg, bridge_lag_msg):
        for marker in (
            "has no signals_log table",      # no_table
            "has no signals ever recorded",  # never
            "has not had a signal",          # stale
            "is missing from disk",          # no_file
            "no atomicortex*.db found",      # no_database
            "could not be read",             # read_error
        ):
            assert marker not in msg, f"a new event reuses '{marker}'"


# ---------------------------------------------------------------------------
# M9 — the transition window: the bot has not been restarted yet
# ---------------------------------------------------------------------------

def test_missing_last_signal_ts_field_is_silent(tmp_path, mock_reporter, now_fn,
                                                monkeypatch):
    """M9: a four-key payload is a legitimate transitional state, not a fault.

    Between the merge and the bot restart the field simply is not there.
    An hourly alert about it would teach the operator to ignore alerts.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), omit_last_signal=True))
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 0
    assert failures == []
    assert mock_reporter.send_alert.call_count == 0


def test_missing_last_signal_ts_still_uses_sqlite_verdict(tmp_path, mock_reporter,
                                                          now_fn, monkeypatch):
    """M9 does not blind the checker: the second source still rules.

    CONTROL for the silence above — the same four-key payload over a
    starving database does alert, so the silence is about the field and
    not about the check having been switched off.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), omit_last_signal=True))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    assert "has not had a signal" in _messages(mock_reporter)[0]


# ---------------------------------------------------------------------------
# main() — the two new events and the exit code
# ---------------------------------------------------------------------------

def test_main_exits_1_when_no_heartbeat(main_env, monkeypatch):
    """R2 through the real entry point: freshness unknown leaves the unit failed."""
    mod, _reporter = main_env
    db_path = mod._ROOT / "data" / "atomicortex.db"
    setup_db(str(db_path), created_at=datetime.now(timezone.utc).isoformat())
    _stub_heartbeat(monkeypatch, "absent")

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1


def test_main_exits_0_when_bridge_lag(main_env, monkeypatch):
    """R8 through the real entry point: contradictory is not unknown."""
    mod, reporter = main_env
    now = datetime.now(timezone.utc)
    db_path = mod._ROOT / "data" / "atomicortex.db"
    setup_db(str(db_path), created_at=(now - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now, last_signal_delta=timedelta(hours=1) - timedelta(seconds=600)))

    mod.main()

    assert reporter.send_alert.call_count == 1


def test_main_builds_redis_url_from_settings(main_env, monkeypatch):
    """M8 + R3: the address comes from Settings, never from a literal here."""
    mod, _reporter = main_env
    now = datetime.now(timezone.utc)
    db_path = mod._ROOT / "data" / "atomicortex.db"
    setup_db(str(db_path), created_at=(now - timedelta(hours=1)).isoformat())

    class _RemoteSettings(_StubSettings):
        redis_host = "10.9.8.7"
        redis_port = 6380
        redis_password = "p@ss word"

    monkeypatch.setattr(mod, "get_settings", lambda: _RemoteSettings())

    seen: list[tuple[str, str]] = []

    def _spy(url, key):
        seen.append((url, key))
        return "ok", _hb(now, last_signal_delta=timedelta(hours=1))

    monkeypatch.setattr(mod, "_read_heartbeat", _spy)

    mod.main()

    assert seen == [
        ("redis://:p%40ss%20word@10.9.8.7:6380/0", "atomicortex:heartbeat")
    ]


def test_main_passes_cli_redis_url_through(main_env, monkeypatch):
    """M8: the address main() uses is the one on the command line."""
    mod, _reporter = main_env
    now = datetime.now(timezone.utc)
    db_path = mod._ROOT / "data" / "atomicortex.db"
    setup_db(str(db_path), created_at=(now - timedelta(hours=1)).isoformat())

    seen: list[tuple[str, str]] = []

    def _spy(url, key):
        seen.append((url, key))
        return "ok", _hb(now, last_signal_delta=timedelta(hours=1))

    monkeypatch.setattr(mod, "_read_heartbeat", _spy)
    monkeypatch.setattr(sys, "argv", [
        "check_signal_freshness.py",
        "--redis-url", "redis://10.0.0.5:6380/1",
        "--heartbeat-key", "bot_15m_heartbeat",
    ])

    mod.main()

    assert seen == [("redis://10.0.0.5:6380/1", "bot_15m_heartbeat")]


# ===========================================================================
# PR-0.10 — a present-but-null last_signal_ts is a claim, not a gap
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers shared by the PR-0.10 blocks
# ---------------------------------------------------------------------------

def _spy_db_opens(monkeypatch):
    """Record every readonly open, without changing whether it succeeds.

    Only the ``uri=True`` connect is counted — the one _inspect_db uses —
    so ``setup_db`` and the repairs around it stay invisible.  Returns the
    list the spy appends to.

    Install it AFTER ``_break_db_reads`` when both are needed: the spy then
    wraps the breaker and records the attempt the breaker turns into an
    error, which is what "the ledger was not even touched" has to be
    distinguished from.
    """
    import scripts.check_signal_freshness as mod

    opens: list[str] = []
    real_connect = sqlite3.connect

    def _connect(*args, **kwargs):
        if kwargs.get("uri"):
            opens.append(str(args[0]) if args else "")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(mod.sqlite3, "connect", _connect)
    return opens


# Every marker of the eight events that existed before PR-0.10.
_EIGHT_MARKERS = (
    "has no signals_log table",      # no_table
    "has no signals ever recorded",  # never
    "has not had a signal",          # stale
    "is missing from disk",          # no_file
    "no atomicortex*.db found",      # no_database
    "could not be read",             # read_error
    "is not in Redis",               # no_heartbeat
    "Bridge Lag",                    # bridge_lag
)

_NEVER_SINCE_START_MARKER = "has emitted nothing since it started"
_BAD_PAYLOAD_MARKER = "carries an unusable last_signal_ts"


# ---------------------------------------------------------------------------
# The claim itself
# ---------------------------------------------------------------------------

def test_null_last_signal_ts_is_a_claim_not_a_gap(tmp_path, mock_reporter, now_fn,
                                                  monkeypatch):
    """A present-but-null field is the bot reporting silence, not absence of news.

    The process has been up 26 hours against a 24-hour threshold and says
    it has emitted nothing in all that time.  The ledger holds a row from
    an hour ago — a leftover from a previous process — and must not be
    allowed to overrule the bot's own report.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 24.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 1
    assert failures == [], "a known-and-bad state must not fail the unit"
    msg = _messages(mock_reporter)[0]
    assert _NEVER_SINCE_START_MARKER in msg
    assert "26.0" in msg
    assert "24.0" in msg


def test_null_last_signal_ts_below_threshold_is_silent(tmp_path, mock_reporter,
                                                       now_fn, monkeypatch):
    """CONTROL for the test above: the alert comes from the age, not from the null.

    Same payload shape, same database, only the process is two hours old
    against the same 24-hour threshold.  Without this, an alert fired for
    every present-but-null field would pass the test above unnoticed.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=2)))
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 24.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 0
    assert failures == []
    assert mock_reporter.send_alert.call_count == 0


def test_null_last_signal_ts_does_not_open_the_ledger(tmp_path, mock_reporter,
                                                      now_fn, monkeypatch):
    """The claim is decided on its own: the second source is never queried.

    Counting alerts cannot show this — the verdicts happen to agree here.
    Only the absence of a readonly open does.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    opens = _spy_db_opens(monkeypatch)

    check_freshness([db_path], {"atomicortex.db": 24.0}, mock_reporter, now_fn,
                    redis_url=_REDIS_URL)

    assert opens == [], f"the ledger was opened {len(opens)} time(s)"


def test_null_last_signal_ts_survives_an_unreadable_ledger(tmp_path, mock_reporter,
                                                           now_fn, monkeypatch):
    """The production case of 2026-08-19, reproduced end to end.

    A live bot that has never signalled, a database the unit cannot open
    under ProtectHome=read-only, and a threshold the process age has
    already passed.  The verdict comes from the heartbeat alone, so the
    unreadable ledger is not a monitoring failure and the oneshot unit
    must not be left failed by it.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    _break_db_reads(monkeypatch)
    opens = _spy_db_opens(monkeypatch)
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 24.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 1
    assert failures == []
    assert opens == [], "the unreadable ledger must not even be reached"
    msg = _messages(mock_reporter)[0]
    assert _NEVER_SINCE_START_MARKER in msg
    assert "could not be read" not in msg


def test_missing_key_and_null_take_different_paths(tmp_path, mock_reporter, now_fn,
                                                   monkeypatch):
    """The split itself: two payloads that were indistinguishable before.

    Same database, same threshold, same process age.  The four-key payload
    says nothing and leaves the verdict to the ledger; the five-key one
    with a null says the bot has been silent for 26 hours.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 24.0}

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26),
                        omit_last_signal=True))
    absent_key = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                                 redis_url=_REDIS_URL)

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    present_null = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                                   redis_url=_REDIS_URL)

    assert absent_key == 0, "no key means unknown, and unknown defers to the ledger"
    assert present_null == 1, "a null key is a claim, and the claim is stale"


def test_never_since_start_text_distinct_from_the_eight(tmp_path, mock_reporter,
                                                        now_fn, monkeypatch):
    """Nine events, nine texts: the newcomer reuses no older marker."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))

    check_freshness([db_path], {"atomicortex.db": 24.0}, mock_reporter, now_fn,
                    redis_url=_REDIS_URL)

    msgs = _messages(mock_reporter)
    assert len(msgs) == 1
    for marker in _EIGHT_MARKERS:
        assert marker not in msgs[0], f"never_since_start reuses '{marker}'"


def test_never_since_start_deduplicated_across_runs(tmp_path, mock_reporter, now_fn,
                                                    monkeypatch):
    """The timer fires hourly; the operator must not hear about it hourly."""
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 24.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")
    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))

    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                            24.0, redis_url=_REDIS_URL)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                             24.0, redis_url=_REDIS_URL)

    assert first == 1
    assert second == 0
    assert mock_reporter.send_alert.call_count == 1


def test_never_since_start_cleared_when_a_signal_arrives(tmp_path, mock_reporter,
                                                         now_fn, monkeypatch):
    """R11 has to hold in the new branch too, or a relapse stays muted.

    The recovery that drops recorded events lives behind the ledger, and
    this branch never reaches it — so the branch has to clear the state
    itself.  Three runs: silent bot, a real signal, silent bot again.
    """
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 24.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                            24.0, redis_url=_REDIS_URL)

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), last_signal_delta=timedelta(hours=1)))
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                             24.0, redis_url=_REDIS_URL)

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    third = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                            24.0, redis_url=_REDIS_URL)

    assert first == 1
    assert second == 0
    assert third == 1, "the window from the first run must not survive a recovery"
    assert mock_reporter.send_alert.call_count == 2


def test_restart_resets_the_claim_age(tmp_path, mock_reporter, now_fn, monkeypatch):
    """A restart moves started_ts forward, and the claim's age with it.

    Documented limitation, pinned here as a contract rather than left to be
    rediscovered: the age is measured from the process start, so a bot
    restarted more often than the threshold never accumulates enough
    silence to be reported.  Measuring it otherwise needs a source that
    outlives the process, which the heartbeat is not.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 24.0}

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(hours=26)))
    before_restart = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                                     redis_url=_REDIS_URL)

    _stub_heartbeat(monkeypatch, "ok",
                    _hb(now_fn(), started_delta=timedelta(minutes=5)))
    after_restart = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                                    redis_url=_REDIS_URL)

    assert before_restart == 1
    assert after_restart == 0


def test_unusable_started_ts_falls_back_to_the_ledger(tmp_path, mock_reporter,
                                                      now_fn, monkeypatch):
    """CONTROL: without a usable started_ts the claim cannot be aged.

    Green before this PR and after it — that is the point.  A claim whose
    age is unknown is no verdict at all, so the checker degrades to
    exactly what it did before: the ledger decides.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=72)).isoformat())
    payload = _hb(now_fn())
    payload.pop("started_ts")
    _stub_heartbeat(monkeypatch, "ok", payload)
    failures: list[str] = []

    alerts = check_freshness([db_path], {"atomicortex.db": 24.0}, mock_reporter,
                             now_fn, failures=failures, redis_url=_REDIS_URL)

    assert alerts == 1
    assert failures == []
    assert "has not had a signal" in _messages(mock_reporter)[0]


# ---------------------------------------------------------------------------
# The third case the old code glued to the other two: not a number at all
# ---------------------------------------------------------------------------

def test_non_numeric_last_signal_ts_is_a_bad_payload(tmp_path, mock_reporter,
                                                     now_fn, monkeypatch):
    """A field the bot could never legitimately write means the source is broken.

    Not a claim and not a transitional gap: the serialiser produces epoch
    seconds or the contract is gone.  Degrading to the ledger would hide a
    schema break for as long as the ledger keeps answering.
    """
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), last_signal_raw="yesterday"))

    alerts = check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter,
                             now_fn, redis_url=_REDIS_URL)

    assert alerts == 1
    msg = _messages(mock_reporter)[0]
    assert _BAD_PAYLOAD_MARKER in msg
    assert "yesterday" in msg, "the operator has to see what was actually there"


def test_non_numeric_last_signal_ts_lands_in_failures(tmp_path, mock_reporter,
                                                      now_fn, monkeypatch):
    """Shouting is half of it: an unparseable first source has to fail the unit."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), last_signal_raw=[1, 2, 3]))
    failures: list[str] = []

    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn,
                    failures=failures, redis_url=_REDIS_URL)

    assert failures == ["atomicortex.db: bad_payload"]


def test_bad_payload_text_distinct_from_the_nine(tmp_path, mock_reporter, now_fn,
                                                 monkeypatch):
    """Ten events, ten texts — including against the ninth added by this PR."""
    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), last_signal_raw="yesterday"))

    check_freshness([db_path], {"atomicortex.db": 48.0}, mock_reporter, now_fn,
                    redis_url=_REDIS_URL)

    msgs = _messages(mock_reporter)
    assert len(msgs) == 1
    for marker in _EIGHT_MARKERS + (_NEVER_SINCE_START_MARKER,):
        assert marker not in msgs[0], f"bad_payload reuses '{marker}'"


def test_bad_payload_deduplicated_and_distinct_from_read_error(tmp_path,
                                                               mock_reporter,
                                                               now_fn, monkeypatch):
    """Its own key: suppressed on repeat, and it breaks an older window.

    R10 keys de-duplication on (db, event).  Sharing read_error's key would
    have meant that a Redis outage already inside its window silently
    swallows the first report of a corrupted payload — a second, unrelated
    fault drowning in the window of the first.
    """
    from src.monitoring.signal_alert_state import SignalAlertState

    db_path = str(tmp_path / "atomicortex.db")
    setup_db(db_path, created_at=(now_fn() - timedelta(hours=1)).isoformat())
    thresholds = {"atomicortex.db": 48.0}
    state = SignalAlertState(tmp_path / "signal_check_state.json")

    _stub_heartbeat(monkeypatch, "error")
    read_error = check_freshness([db_path], thresholds, mock_reporter, now_fn,
                                 state, 24.0, redis_url=_REDIS_URL)

    _stub_heartbeat(monkeypatch, "ok", _hb(now_fn(), last_signal_raw="yesterday"))
    first = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                            24.0, redis_url=_REDIS_URL)
    second = check_freshness([db_path], thresholds, mock_reporter, now_fn, state,
                             24.0, redis_url=_REDIS_URL)

    assert read_error == 1
    assert first == 1, "a different event must get through the older window"
    assert second == 0, "the same event inside the window must not"
    assert mock_reporter.send_alert.call_count == 2


# ---------------------------------------------------------------------------
# main() — the exit code each of the two new events leaves behind
# ---------------------------------------------------------------------------

def test_main_exits_0_when_the_bot_never_signalled(main_env, monkeypatch):
    """Known and bad is a starvation report, not a monitoring outage.

    The main_env stub is overridden on purpose: its default payload is the
    one with no last_signal_ts key at all, and this test needs the
    opposite — the key present and null.
    """
    mod, reporter = main_env
    now = datetime.now(timezone.utc)
    db_path = mod._ROOT / "data" / "atomicortex.db"
    setup_db(str(db_path), created_at=(now - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now, started_delta=timedelta(hours=72)))

    mod.main()  # must not raise SystemExit

    assert reporter.send_alert.call_count == 1


def test_main_exits_1_when_last_signal_ts_is_corrupt(main_env, monkeypatch):
    """An unparseable first source leaves the oneshot unit failed.

    The main_env stub is overridden on purpose: its default payload omits
    the field entirely, and this test needs the field present holding
    something that is not a number.
    """
    mod, _reporter = main_env
    now = datetime.now(timezone.utc)
    db_path = mod._ROOT / "data" / "atomicortex.db"
    setup_db(str(db_path), created_at=(now - timedelta(hours=1)).isoformat())
    _stub_heartbeat(monkeypatch, "ok", _hb(now, last_signal_raw="yesterday"))

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert excinfo.value.code == 1
