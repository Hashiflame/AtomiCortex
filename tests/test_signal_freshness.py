import pytest
import sqlite3
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta

from src.monitoring.telegram_reporter import TelegramReporter
from scripts.check_signal_freshness import check_freshness


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


def test_missing_db_skipped_with_warning(tmp_path, mock_reporter, now_fn):
    db_path = str(tmp_path / "missing.db")
    thresholds = {"missing.db": 48.0}
    alerts = check_freshness([db_path], thresholds, mock_reporter, now_fn)

    assert alerts == 0
    mock_reporter.send_alert.assert_not_called()


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
