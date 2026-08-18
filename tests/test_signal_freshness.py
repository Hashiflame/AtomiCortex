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
