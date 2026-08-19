"""
PR-0.9 — the bot publishes signal freshness itself.

Everything here pins the one place where a signal becomes real for the
outside world: ``MLTradingStrategy._emit_signal``.  Three things have to
hold there, and two of them are invisible today:

  * the row is written to SQLite through ``SignalBridge`` (unchanged);
  * a ``signal_id`` of 0 — meaning ``log_signal`` swallowed an exception
    and returned zero — is logged as an ERROR instead of passing
    silently into ``_pending_signal_ids``;
  * a non-zero ``signal_id`` publishes the *moment of the write* into
    the heartbeat, so the freshness checker can read it from Redis
    instead of opening the bot's private database file.

M1: every strategy that can open a position routes through this single
``_emit_signal``.  The 4H base owns it; the 15m and meta subclasses
inherit ``_open_position`` unchanged, so there is exactly one copy of
the log-then-publish sequence in the codebase and no way for the three
to drift apart.

M2: the publish call is made whenever ``signal_id != 0`` — the presence
of a heartbeat manager is decided *inside* the publisher, never by the
caller, so a bot running without Redis follows the same code path.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.execution.strategies.meta_strategy import (
    MetaMLStrategyConfig,
    MetaMLTradingStrategy,
)
from src.execution.strategies.ml_strategy import (
    MLStrategyConfig,
    MLTradingStrategy,
)
from src.execution.strategies.ml_strategy_15m import (
    MLStrategy15MConfig,
    MLTradingStrategy15M,
)
from src.risk.risk_engine import RiskDecision, TradeSignal


_BASE = "src.execution.strategies.ml_strategy.MLTradingStrategy."


# ---------------------------------------------------------------------------
# Harness — drives the real _open_position with Nautilus plumbing mocked.
# ---------------------------------------------------------------------------

def _config_4h() -> MLStrategyConfig:
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        dry_run=True,
    )


def _config_15m() -> MLStrategy15MConfig:
    return MLStrategy15MConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        dry_run=True,
    )


def _config_meta() -> MetaMLStrategyConfig:
    return MetaMLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        dry_run=True,
        meta_enabled=False,
    )


@contextmanager
def _harness(strategy_cls, config, signal_id: int = 77):
    """Yield ``(strategy, log, submit)`` ready to run _open_position."""
    with patch(_BASE + "cache", new_callable=PropertyMock) as mock_cache, \
         patch(_BASE + "order_factory", new_callable=PropertyMock) as mock_factory, \
         patch(_BASE + "log", new_callable=PropertyMock) as mock_log, \
         patch(_BASE + "submit_order") as mock_submit:

        strategy = strategy_cls(config=config)
        strategy._signal_bridge = MagicMock()
        strategy._signal_bridge.log_signal.return_value = signal_id
        strategy._pending_store = MagicMock()
        strategy._heartbeat = MagicMock()

        mock_cache.return_value.instrument.return_value = MagicMock()
        mock_factory.return_value.market.return_value.client_order_id = "oid-1"

        yield strategy, mock_log.return_value, mock_submit


def _make_signal() -> TradeSignal:
    return TradeSignal(
        symbol="BTCUSDT-PERP.BINANCE",
        direction=1,
        confidence=0.72,
        regime="trend",
        entry_price=50_000.0,
        atr=750.0,
        atr_pct=0.015,
        funding_rate=0.0001,
        timestamp=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )


def _make_decision() -> RiskDecision:
    decision = MagicMock(spec=RiskDecision)
    decision.entry_price = 50_000.0
    decision.stop_loss = 49_000.0
    decision.take_profit = 52_000.0
    decision.position_size = 0.1
    decision.notional = 5_000.0
    decision.leverage = 2.0
    return decision


# ---------------------------------------------------------------------------
# Publishing the write moment (P4/P5 — time.time(), float epoch seconds)
# ---------------------------------------------------------------------------

def test_emit_signal_publishes_write_moment_to_heartbeat():
    """A persisted signal puts its write moment into the heartbeat."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=77) as (s, _log, _sub):
        with patch("time.time", return_value=1_800_000_000.0):
            s._open_position(_make_decision(), _make_signal())

    s._heartbeat.report_signal.assert_called_once_with(1_800_000_000.0)


def test_publish_uses_write_moment_not_bar_timestamp():
    """P4: the published value is time.time(), never signal.timestamp."""
    signal = _make_signal()
    bar_epoch = signal.timestamp.timestamp()

    with _harness(MLTradingStrategy, _config_4h(), signal_id=5) as (s, _log, _sub):
        with patch("time.time", return_value=1_800_000_000.0):
            s._open_position(_make_decision(), signal)

    published = s._heartbeat.report_signal.call_args[0][0]
    assert published == 1_800_000_000.0
    assert published != bar_epoch
    assert isinstance(published, float)


# ---------------------------------------------------------------------------
# M3/P7 — signal_id == 0 is a persisted-nothing case
# ---------------------------------------------------------------------------

def test_zero_signal_id_does_not_publish():
    """P7: nothing reached the database, so nothing may be published."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=0) as (s, _log, _sub):
        s._open_position(_make_decision(), _make_signal())

    s._heartbeat.report_signal.assert_not_called()


def test_zero_signal_id_logs_error():
    """M3: the swallowed write becomes visible in the journal — ERROR only."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=0) as (s, log, _sub):
        s._open_position(_make_decision(), _make_signal())

    assert log.error.call_count >= 1
    text = " ".join(str(c.args[0]) for c in log.error.call_args_list)
    assert "BTCUSDT-PERP.BINANCE" in text
    assert "0" in text


def test_zero_signal_id_not_recorded_as_pending():
    """D-14: a row that was never written owns no pending id.

    Zero is not a signal_id — it is the bridge's way of saying it wrote
    nothing.  Stored under the symbol it becomes a booby trap for
    ``on_order_rejected`` and ``on_position_closed``, which read that map
    and would hand 0 to ``mark_rejected`` / ``close_signal``.
    """
    with _harness(MLTradingStrategy, _config_4h(), signal_id=0) as (s, _log, _sub):
        s._open_position(_make_decision(), _make_signal())

    assert "BTCUSDT-PERP.BINANCE" not in s._pending_signal_ids


def test_zero_signal_id_still_submits_the_order():
    """CONTROL: the ERROR is a report, not a new veto on trading."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=0) as (s, _log, submit):
        s._open_position(_make_decision(), _make_signal())

    submit.assert_called_once()


# ---------------------------------------------------------------------------
# M2 — the call does not depend on a heartbeat manager existing
# ---------------------------------------------------------------------------

def test_publish_called_even_without_heartbeat_manager():
    """M2: the None check lives inside the publisher, not at the call site."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=77) as (s, _log, _sub):
        s._heartbeat = None
        with patch.object(MLTradingStrategy, "_publish_signal_ts") as pub:
            s._open_position(_make_decision(), _make_signal())

    pub.assert_called_once()


def test_publish_without_heartbeat_manager_does_not_raise():
    """A bot with no Redis follows the same path and simply does nothing."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=77) as (s, _log, submit):
        s._heartbeat = None
        s._open_position(_make_decision(), _make_signal())

    submit.assert_called_once()


def test_report_signal_failure_does_not_block_entry():
    """Fail-soft, exactly like report_bar: a broken heartbeat never trades."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=77) as (s, log, submit):
        s._heartbeat.report_signal.side_effect = RuntimeError("redis gone")
        s._open_position(_make_decision(), _make_signal())

    submit.assert_called_once()
    assert log.warning.call_count >= 1


# ---------------------------------------------------------------------------
# M1 — one _emit_signal shared by every strategy that can open a position
# ---------------------------------------------------------------------------

def test_emit_signal_records_pending_id_and_returns_it():
    """_emit_signal owns the whole bridge block, including _pending_signal_ids."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=77) as (s, _log, _sub):
        returned = s._emit_signal(_make_decision(), _make_signal())

    assert returned == 77
    assert s._pending_signal_ids["BTCUSDT-PERP.BINANCE"] == 77


def test_15m_routes_through_the_inherited_emit_signal():
    """M1: the 15m subclass gets publishing for free, with no code of its own."""
    with _harness(MLTradingStrategy15M, _config_15m(), signal_id=77) as (s, _log, _sub):
        with patch("time.time", return_value=1_800_000_000.0):
            s._open_position(_make_decision(), _make_signal())

    s._heartbeat.report_signal.assert_called_once_with(1_800_000_000.0)


def test_all_strategies_share_one_emit_signal():
    """M1: 4H, 15m and meta reach the same method — one copy, no drift."""
    cases = [
        (MLTradingStrategy, _config_4h()),
        (MLTradingStrategy15M, _config_15m()),
        (MetaMLTradingStrategy, _config_meta()),
    ]

    for strategy_cls, config in cases:
        with _harness(strategy_cls, config, signal_id=77) as (s, _log, _sub):
            with patch.object(MLTradingStrategy, "_emit_signal") as emit:
                s._open_position(_make_decision(), _make_signal())

            assert emit.call_count == 1, f"{strategy_cls.__name__} bypassed _emit_signal"


def test_emit_signal_defined_once_in_the_hierarchy():
    """M1 as structure: no subclass may carry its own copy of the sequence."""
    assert "_emit_signal" in vars(MLTradingStrategy)
    assert "_emit_signal" not in vars(MLTradingStrategy15M)
    assert "_emit_signal" not in vars(MetaMLTradingStrategy)


def test_missing_bridge_does_not_publish_or_crash():
    """No bridge → no signal_id → nothing to publish, and no exception."""
    with _harness(MLTradingStrategy, _config_4h(), signal_id=77) as (s, _log, submit):
        s._signal_bridge = None
        s._open_position(_make_decision(), _make_signal())

    s._heartbeat.report_signal.assert_not_called()
    submit.assert_called_once()
