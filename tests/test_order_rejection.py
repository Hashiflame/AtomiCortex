from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from datetime import datetime, timezone

from src.execution.strategies.ml_strategy import MLTradingStrategy, MLStrategyConfig
from src.execution.strategies.ml_strategy_15m import MLTradingStrategy15M
from src.execution.signal_bridge import SignalBridge
from src.risk.risk_engine import RiskDecision, TradeSignal

@pytest.fixture
def default_strategy_config():
    return MLStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
        initial_equity=10_000.0,
        warmup_bars=10,
        dry_run=True,
    )

@pytest.fixture
def strategy(default_strategy_config):
    with patch("src.execution.strategies.ml_strategy.MLTradingStrategy.cache", new_callable=PropertyMock) as mock_cache, \
         patch("src.execution.strategies.ml_strategy.MLTradingStrategy.order_factory", new_callable=PropertyMock) as mock_factory, \
         patch("src.execution.strategies.ml_strategy.MLTradingStrategy.log", new_callable=PropertyMock) as mock_log, \
         patch("src.execution.strategies.ml_strategy.MLTradingStrategy.submit_order") as mock_submit:
             
        strat = MLTradingStrategy(config=default_strategy_config)
        strat._signal_bridge = MagicMock()
        strat._pending_store = MagicMock()
        
        strat.mock_cache = mock_cache.return_value
        strat.mock_factory = mock_factory.return_value
        strat.mock_log = mock_log.return_value
        strat.mock_submit = mock_submit
        yield strat

def make_mock_event(client_oid: str, reason: str = "insufficient margin", instrument_id: str = "BTCUSDT-PERP.BINANCE"):
    event = MagicMock()
    event.client_order_id = client_oid
    event.reason = reason
    event.instrument_id = instrument_id
    return event

def test_entry_reject_marks_signal_rejected(strategy):
    strategy._pending_sl_params["oid_123"] = {"decision": MagicMock(), "signal": MagicMock()}
    strategy._pending_signal_ids["BTCUSDT-PERP.BINANCE"] = 42
    
    event = make_mock_event("oid_123", "margin")
    strategy.on_order_rejected(event)
    
    strategy._signal_bridge.mark_rejected.assert_called_once_with(42, "margin")
    assert "oid_123" not in strategy._pending_sl_params
    strategy._pending_store.pop.assert_called_once_with("oid_123")
    assert "BTCUSDT-PERP.BINANCE" not in strategy._pending_signal_ids

def test_entry_denied_same_path(strategy):
    strategy._pending_sl_params["oid_123"] = {"decision": MagicMock(), "signal": MagicMock()}
    strategy._pending_signal_ids["BTCUSDT-PERP.BINANCE"] = 42
    
    event = make_mock_event("oid_123", "risk_limit")
    strategy.on_order_denied(event)
    
    strategy._signal_bridge.mark_rejected.assert_called_once_with(42, "risk_limit")
    assert "oid_123" not in strategy._pending_sl_params
    strategy._pending_store.pop.assert_called_once_with("oid_123")
    assert "BTCUSDT-PERP.BINANCE" not in strategy._pending_signal_ids

def test_sl_reject_logs_critical(strategy):
    mock_order = MagicMock()
    mock_order.tags = ["SL-attempt-1"]
    strategy.mock_cache.order.return_value = mock_order
    
    event = make_mock_event("sl_oid_456", "margin_sl")
    strategy.on_order_rejected(event)
    
    strategy._signal_bridge.mark_rejected.assert_not_called()
    
    # We used error instead of critical since Nautilus Logger has no critical
    strategy.mock_log.error.assert_called_with(
        "POSITION UNPROTECTED! Stop-loss order rejected: margin_sl | oid=sl_oid_456"
    )

def test_unknown_oid_warning(strategy):
    mock_order = MagicMock()
    mock_order.tags = ["SOME-TAG"]
    strategy.mock_cache.order.return_value = mock_order
    
    event = make_mock_event("unknown_oid", "unknown_reason")
    strategy.on_order_rejected(event)
        
    strategy._signal_bridge.mark_rejected.assert_not_called()
    strategy.mock_log.warning.assert_called_with(
        "Unknown order rejected | oid=unknown_oid | reason=unknown_reason"
    )

def test_signal_id_missing_warning(strategy):
    strategy._pending_sl_params["oid_123"] = {"decision": MagicMock(), "signal": MagicMock()}
    
    event = make_mock_event("oid_123", "margin")
    strategy.on_order_rejected(event)
        
    strategy._signal_bridge.mark_rejected.assert_not_called()
    assert "oid_123" not in strategy._pending_sl_params
    strategy._pending_store.pop.assert_called_once_with("oid_123")
    strategy.mock_log.warning.assert_any_call("Entry reject for unknown signal_id | symbol=BTCUSDT-PERP.BINANCE")

def test_submit_order_exception_marks_rejected(strategy):
    strategy.mock_submit.side_effect = Exception("Exchange timeout")
    
    decision = MagicMock()
    decision.entry_price = 10000
    decision.stop_loss = 9000
    decision.take_profit = 11000
    decision.position_size = 0.1
    decision.notional = 1000
    decision.leverage = 1.0
    signal = TradeSignal(
        symbol="BTCUSDT-PERP.BINANCE", direction=1, confidence=0.8, regime="trend",
        timestamp=datetime.now(timezone.utc), atr=100, entry_price=10000,
        atr_pct=0.01, funding_rate=0.0001
    )
    
    strategy.mock_cache.instrument.return_value = MagicMock()
    strategy.mock_factory.market.return_value.client_order_id = "test_oid_123"
    
    strategy._signal_bridge.log_signal.return_value = 99
    
    strategy._open_position(decision, signal)
    
    strategy._signal_bridge.mark_rejected.assert_called_once_with(99, "Exchange timeout")
    assert "test_oid_123" not in strategy._pending_sl_params
    strategy._pending_store.pop.assert_called_once_with("test_oid_123")
    assert "BTCUSDT-PERP.BINANCE" not in strategy._pending_signal_ids

def test_log_signal_before_submit_order(strategy):
    call_order = []
    strategy._signal_bridge.log_signal.side_effect = lambda *args, **kwargs: call_order.append("log_signal") or 77
    strategy.mock_submit.side_effect = lambda *args, **kwargs: call_order.append("submit_order")
    
    decision = MagicMock()
    decision.entry_price = 10000
    decision.stop_loss = 9000
    decision.take_profit = 11000
    decision.position_size = 0.1
    decision.notional = 1000
    decision.leverage = 1.0
    signal = TradeSignal(
        symbol="BTCUSDT-PERP.BINANCE", direction=1, confidence=0.8, regime="trend",
        timestamp=datetime.now(timezone.utc), atr=100, entry_price=10000,
        atr_pct=0.01, funding_rate=0.0001
    )
    
    strategy.mock_cache.instrument.return_value = MagicMock()
    strategy.mock_factory.market.return_value.client_order_id = "test_oid_123"
    
    strategy._open_position(decision, signal)
    
    assert call_order == ["log_signal", "submit_order"]
    assert strategy._pending_signal_ids["BTCUSDT-PERP.BINANCE"] == 77

def test_mark_rejected_sql(tmp_path):
    db_path = str(tmp_path / "test_signals.db")
    bridge = SignalBridge(db_path=db_path)
    
    sid = bridge.log_signal(
        symbol="BTC", direction="long", entry_price=100, stop_loss=90, take_profit=120,
        confidence=0.8, regime="range"
    )
    
    conn = bridge._connect()
    res = conn.execute("SELECT result FROM signals_log WHERE id = ?", (sid,)).fetchone()
    assert res[0] == "open"
    conn.close()
    
    bridge.mark_rejected(sid, "test_reject")
    
    conn = bridge._connect()
    row = conn.execute("SELECT result, closed_at, pnl_pct FROM signals_log WHERE id = ?", (sid,)).fetchone()
    conn.close()
    
    assert row[0] == "rejected"
    assert row[1] is not None
    assert row[2] is None

def test_15m_inherits_handlers():
    assert issubclass(MLTradingStrategy15M, MLTradingStrategy)
    assert MLTradingStrategy15M.on_order_rejected is MLTradingStrategy.on_order_rejected
    assert MLTradingStrategy15M.on_order_denied is MLTradingStrategy.on_order_denied

def test_store_pop_exception_failsoft(strategy):
    strategy._pending_sl_params["oid_123"] = {"decision": MagicMock(), "signal": MagicMock()}
    strategy._pending_signal_ids["BTCUSDT-PERP.BINANCE"] = 42
    
    strategy._pending_store.pop.side_effect = Exception("Disk error")
    
    event = make_mock_event("oid_123", "margin")
    strategy.on_order_rejected(event)
    
    strategy._signal_bridge.mark_rejected.assert_called_once_with(42, "margin")
    assert "oid_123" not in strategy._pending_sl_params
    strategy.mock_log.warning.assert_any_call("Pending-SL store pop failed on reject: Disk error")
