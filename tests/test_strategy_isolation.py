"""
tests/test_strategy_isolation.py

Phase 5 — isolation guarantees between the 4H / 1H / 15m bots.

Each bot must be fully independent: its own SQLite DB, its own Redis
heartbeat key, and a watchdog that can only ever touch its own symbol.
These tests lock those invariants so a future change that accidentally
shares state (the classic "15m bot flattens the 4H book") fails CI.

Note on ``test_watchdog_monitors_only_existing_dbs``: the original plan
sketched a ``WATCHED_SERVICES`` + db-existence design. The implemented
design (decision #4) is instead a per-instance, symbol-scoped watchdog
with a backward-compatible global default. The test keeps its name but
asserts the real invariant: a 15m-scoped watchdog never targets the 4H
symbol/heartbeat, and the legacy default stays global (4H unchanged).

Run:
    pytest tests/test_strategy_isolation.py -v
"""

from __future__ import annotations

try:
    import sqlite3
except ImportError:  # pragma: no cover
    import pysqlite3 as sqlite3  # type: ignore[no-redef]

import pytest

from src.configs.strategy_1h import MLStrategyConfig1H
from src.configs.strategy_15m import MLStrategyConfig15M
from src.execution.live_trader import LiveTraderConfig
from src.execution.signal_bridge import SignalBridge
from src.execution.strategies.ml_strategy import MLStrategyConfig
from src.execution.strategies.ml_strategy_15m import (
    MLStrategy15MConfig,
    MLTradingStrategy15M,
)
from src.execution.watchdog import Watchdog, WatchdogConfig

# Canonical, intentionally-distinct identities.
_DB_4H = "data/atomicortex.db"
_DB_1H = "data/atomicortex_1h.db"
_DB_15M = "data/atomicortex_15m.db"
_HB_4H = "atomicortex:heartbeat"
_HB_1H = "bot_1h_heartbeat"
_HB_15M = "bot_15m_heartbeat"


# ──────────────────────────────────────────────────────────────────────
# 1-2. Database isolation
# ──────────────────────────────────────────────────────────────────────

def test_15m_uses_different_db_than_4h() -> None:
    """The 15m Nautilus config must not share the 4H SQLite file."""
    db_4h = MLStrategyConfig().signal_db_path
    db_15m = MLStrategy15MConfig().signal_db_path
    assert db_4h == _DB_4H
    assert db_15m == _DB_15M
    assert db_15m != db_4h
    # The src.configs dataclass (used by tooling) must agree.
    assert MLStrategyConfig15M().signal_db_path == _DB_15M


def test_1h_uses_different_db_than_4h() -> None:
    """The 1H config must not share the 4H SQLite file."""
    db_4h = MLStrategyConfig().signal_db_path
    db_1h = MLStrategyConfig1H().signal_db_path
    assert db_1h == _DB_1H
    assert db_1h != db_4h
    # All three are pairwise distinct.
    assert len({_DB_4H, db_1h, MLStrategy15MConfig().signal_db_path}) == 3


# ──────────────────────────────────────────────────────────────────────
# 3-4. Heartbeat key isolation
# ──────────────────────────────────────────────────────────────────────

def test_15m_heartbeat_key_unique() -> None:
    """15m heartbeat key differs from both 4H and 1H."""
    hb_15m = MLStrategy15MConfig().heartbeat_key
    assert hb_15m == _HB_15M
    assert hb_15m != MLStrategyConfig().heartbeat_key      # vs 4H
    assert hb_15m != MLStrategyConfig1H().heartbeat_key     # vs 1H
    assert MLStrategyConfig15M().heartbeat_key == _HB_15M


def test_1h_heartbeat_key_unique() -> None:
    """1H heartbeat key differs from both 4H and 15m."""
    hb_1h = MLStrategyConfig1H().heartbeat_key
    assert hb_1h == _HB_1H
    assert hb_1h != MLStrategyConfig().heartbeat_key            # vs 4H
    assert hb_1h != MLStrategy15MConfig().heartbeat_key          # vs 15m
    assert len({_HB_4H, hb_1h, MLStrategy15MConfig().heartbeat_key}) == 3


# ──────────────────────────────────────────────────────────────────────
# 5. SignalBridge writes to the configured (isolated) DB only
# ──────────────────────────────────────────────────────────────────────

def test_signal_bridge_15m_writes_to_correct_db(tmp_path) -> None:
    """A SignalBridge pointed at the 15m DB writes there and nowhere
    else — in particular it must not create / touch the 4H DB."""
    db_15m = tmp_path / "atomicortex_15m.db"
    db_4h = tmp_path / "atomicortex.db"

    bridge = SignalBridge(db_path=str(db_15m))
    sid = bridge.log_signal(
        symbol="BTCUSDT-PERP.BINANCE",
        direction="long",
        entry_price=40_000.0,
        stop_loss=39_500.0,
        take_profit=41_000.0,
        confidence=0.76,
        regime="orb:trend_up",
    )
    assert sid > 0
    assert db_15m.exists()
    assert not db_4h.exists(), "4H DB must not be created by the 15m bridge"

    conn = sqlite3.connect(str(db_15m))
    try:
        rows = conn.execute(
            "SELECT symbol, regime, confidence FROM signals_log"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("BTCUSDT-PERP.BINANCE", "orb:trend_up", 0.76)]


# ──────────────────────────────────────────────────────────────────────
# 6. Watchdog isolation (per-instance, symbol-scoped)
# ──────────────────────────────────────────────────────────────────────

def test_watchdog_monitors_only_existing_dbs() -> None:
    """Isolation invariant for the per-instance watchdog design:

    * legacy default = global (4H watchdog unchanged: scope '');
    * a 15m-scoped instance only ever targets its own symbol and its
      own heartbeat key — it can never close the 4H book.
    """
    legacy = Watchdog(WatchdogConfig())  # 4H default
    assert legacy._scope_symbol == ""                   # closes ALL (legacy)
    assert legacy._config.heartbeat_key == _HB_4H

    wd_15m = Watchdog(WatchdogConfig(
        symbol="BTCUSDT-PERP.BINANCE",
        service_name="15m",
        heartbeat_key=_HB_15M,
    ))
    assert wd_15m._scope_symbol == "BTCUSDT"            # normalized
    assert wd_15m._config.heartbeat_key == _HB_15M
    # The 15m watchdog watches a different key and a scoped symbol than 4H.
    assert wd_15m._config.heartbeat_key != legacy._config.heartbeat_key
    assert wd_15m._scope_symbol != legacy._scope_symbol


# ──────────────────────────────────────────────────────────────────────
# 7. The 4H bot is unaffected by the new configs/code
# ──────────────────────────────────────────────────────────────────────

def test_4h_strategy_unaffected_by_new_configs() -> None:
    """Importing/instantiating the 15m machinery must not mutate the 4H
    defaults, and LiveTrader's default path stays the 4H one."""
    # Touch the 15m machinery (would surface any import-time mutation).
    _ = MLStrategy15MConfig()
    assert issubclass(MLStrategy15MConfig, MLStrategyConfig)
    assert MLTradingStrategy15M is not None

    cfg_4h = MLStrategyConfig()
    assert cfg_4h.signal_db_path == _DB_4H
    assert cfg_4h.heartbeat_key == _HB_4H
    assert cfg_4h.bar_type == "BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL"
    assert cfg_4h.interval == "4h"

    # Backward compat: no factory ⇒ LiveTrader builds the 4H strategy.
    assert LiveTraderConfig().strategy_factory is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
