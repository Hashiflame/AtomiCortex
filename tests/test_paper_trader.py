"""
Tests for Phase 5: Paper Trading, MetricsCollector, TelegramReporter.

Covers:
- PaperTrader simulate_fill (LONG, SHORT, fees, slippage)
- PaperTrader equity tracking
- MetricsCollector (collect, save, reports)
- TelegramReporter (format, mock send)
- PaperTradingStrategy construction
- Signal logging to SQLite
- Sharpe ratio calculation

Minimum 15 tests.
"""

from __future__ import annotations

import asyncio
import math
try:
    import sqlite3
except ImportError:
    import pysqlite3 as sqlite3  # type: ignore[no-redef]
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.execution.paper_trader import PaperFill, PaperTrader, PaperTraderConfig
from src.monitoring.metrics_collector import MetricsCollector, TradingMetrics
from src.monitoring.telegram_reporter import TelegramReporter
from src.risk.portfolio_tracker import PortfolioTracker


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def paper_config() -> PaperTraderConfig:
    return PaperTraderConfig(
        initial_equity=10_000.0,
        maker_fee=0.0002,
        taker_fee=0.0005,
        slippage_bps=2.0,
    )


@pytest.fixture
def paper_trader(paper_config: PaperTraderConfig) -> PaperTrader:
    return PaperTrader(paper_config)


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_metrics.db")


@pytest.fixture
def metrics_collector(tmp_db: str) -> MetricsCollector:
    return MetricsCollector(db_path=tmp_db, initial_equity=10_000.0)


@pytest.fixture
def telegram_reporter() -> TelegramReporter:
    return TelegramReporter(bot_token="test_token_123", admin_id="12345")


# ═══════════════════════════════════════════════════════════════════════════
# PAPER TRADER — FILLS
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderFills:
    """Test paper fill simulation logic."""

    def test_simulate_fill_long_price_above_market(
        self, paper_trader: PaperTrader,
    ) -> None:
        """LONG fill_price should be > current_price (positive slippage)."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        assert fill.fill_price > 50_000.0
        # 2 bps slippage: 50000 * (1 + 2/10000) = 50010
        assert fill.fill_price == pytest.approx(50_010.0, rel=1e-6)

    def test_simulate_fill_short_price_below_market(
        self, paper_trader: PaperTrader,
    ) -> None:
        """SHORT fill_price should be < current_price (adverse slippage)."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=-1, quantity=0.1,
            current_price=50_000.0,
        )
        assert fill.fill_price < 50_000.0
        # 2 bps: 50000 * (1 - 2/10000) = 49990
        assert fill.fill_price == pytest.approx(49_990.0, rel=1e-6)

    def test_fee_calculation_taker(self, paper_trader: PaperTrader) -> None:
        """Taker fee should be quantity × fill_price × 0.0005."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0, is_maker=False,
        )
        expected_notional = 0.1 * fill.fill_price
        expected_fee = expected_notional * 0.0005
        assert fill.fee == pytest.approx(expected_fee, rel=1e-4)

    def test_fee_calculation_maker(self, paper_trader: PaperTrader) -> None:
        """Maker fee should be quantity × fill_price × 0.0002."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0, is_maker=True,
        )
        expected_notional = 0.1 * fill.fill_price
        expected_fee = expected_notional * 0.0002
        assert fill.fee == pytest.approx(expected_fee, rel=1e-4)

    def test_slippage_bps_recorded(self, paper_trader: PaperTrader) -> None:
        """Fill should record the slippage in bps."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        assert fill.slippage_bps == 2.0

    def test_fill_returns_paperfill(self, paper_trader: PaperTrader) -> None:
        """simulate_fill should return a PaperFill instance."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        assert isinstance(fill, PaperFill)
        assert fill.symbol == "BTCUSDT"
        assert fill.direction == 1
        assert fill.quantity == 0.1


# ═══════════════════════════════════════════════════════════════════════════
# PAPER TRADER — EQUITY & PNL
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTraderEquity:
    """Test equity and PnL tracking."""

    def test_equity_decreases_after_fee(self, paper_trader: PaperTrader) -> None:
        """Equity should decrease after a fill due to fees."""
        initial = paper_trader.get_equity()
        paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        assert paper_trader.get_equity() < initial

    def test_close_with_profit(self, paper_trader: PaperTrader) -> None:
        """Close at higher price → positive PnL."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        pnl = paper_trader.simulate_close(
            symbol="BTCUSDT", order_id=fill.order_id,
            current_price=52_000.0,
        )
        assert pnl > 0

    def test_close_with_loss(self, paper_trader: PaperTrader) -> None:
        """Close at lower price → negative PnL."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        pnl = paper_trader.simulate_close(
            symbol="BTCUSDT", order_id=fill.order_id,
            current_price=48_000.0,
        )
        assert pnl < 0

    def test_pnl_history_populated(self, paper_trader: PaperTrader) -> None:
        """After close, pnl_history should have one entry."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        paper_trader.simulate_close(
            symbol="BTCUSDT", order_id=fill.order_id,
            current_price=51_000.0,
        )
        history = paper_trader.get_pnl_history()
        assert len(history) == 1
        ts, pnl = history[0]
        assert isinstance(ts, datetime)
        assert pnl > 0

    def test_trade_log_has_entries(self, paper_trader: PaperTrader) -> None:
        """Trade log should contain entry + exit."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        paper_trader.simulate_close(
            symbol="BTCUSDT", order_id=fill.order_id,
            current_price=51_000.0,
        )
        log = paper_trader.get_trade_log()
        assert len(log) == 2
        assert log[0]["type"] == "ENTRY"
        assert log[1]["type"] == "EXIT"

    def test_get_stats(self, paper_trader: PaperTrader) -> None:
        """Stats should reflect trade history."""
        fill = paper_trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        paper_trader.simulate_close(
            symbol="BTCUSDT", order_id=fill.order_id,
            current_price=51_000.0,
        )
        stats = paper_trader.get_stats()
        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1
        assert stats["win_rate"] == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════


class TestMetricsCollector:
    """Test MetricsCollector compute and persist logic."""

    def test_collect_returns_trading_metrics(
        self, metrics_collector: MetricsCollector,
    ) -> None:
        """collect() should return a TradingMetrics instance."""
        tracker = PortfolioTracker(10_000)
        metrics = metrics_collector.collect(tracker, regime="trend_up")
        assert isinstance(metrics, TradingMetrics)
        assert metrics.equity == 10_000.0

    def test_daily_report_contains_fields(
        self, metrics_collector: MetricsCollector,
    ) -> None:
        """Daily report string should contain key fields."""
        tracker = PortfolioTracker(10_000)
        metrics = metrics_collector.collect(tracker, regime="trend_up")
        report = metrics_collector.get_daily_report(metrics)
        assert "Daily Report" in report
        assert "Equity" in report
        assert "Win rate" in report
        assert "Sharpe" in report
        assert "Drawdown" in report

    def test_weekly_report_contains_fields(
        self, metrics_collector: MetricsCollector,
    ) -> None:
        """Weekly report string should contain key fields."""
        tracker = PortfolioTracker(10_000)
        metrics = metrics_collector.collect(tracker)
        report = metrics_collector.get_weekly_report(metrics)
        assert "Weekly Report" in report
        assert "Equity" in report

    def test_save_to_db_persists(
        self, metrics_collector: MetricsCollector, tmp_db: str,
    ) -> None:
        """save_to_db should create a row in the metrics table."""
        tracker = PortfolioTracker(10_000)
        metrics = metrics_collector.collect(tracker)
        metrics_collector.save_to_db(metrics)

        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM metrics")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 1

    def test_save_signal_to_db(
        self, metrics_collector: MetricsCollector, tmp_db: str,
    ) -> None:
        """save_signal_to_db should persist signal records."""
        metrics_collector.save_signal_to_db(
            symbol="BTCUSDT", direction=1, confidence=0.75,
            regime="trend_up", entry_price=94_000.0,
            approved=True, reason="",
        )
        metrics_collector.save_signal_to_db(
            symbol="ETHUSDT", direction=-1, confidence=0.60,
            regime="range", entry_price=3_400.0,
            approved=False, reason="Low confidence",
        )

        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM signals")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 2

    def test_sharpe_calculation(self, metrics_collector: MetricsCollector) -> None:
        """Sharpe ratio should be calculated from PnL returns."""
        # Record some trades
        for _ in range(10):
            metrics_collector.record_trade(50.0)   # $50 win
        for _ in range(5):
            metrics_collector.record_trade(-30.0)  # $30 loss

        tracker = PortfolioTracker(10_000)
        metrics = metrics_collector.collect(tracker)
        assert metrics.sharpe_ratio != 0.0
        # Positive expected return, so Sharpe should be positive
        assert metrics.sharpe_ratio > 0

    def test_profit_factor_calculation(
        self, metrics_collector: MetricsCollector,
    ) -> None:
        """Profit factor = gross_wins / |gross_losses|."""
        metrics_collector.record_trade(100.0)
        metrics_collector.record_trade(-50.0)

        tracker = PortfolioTracker(10_000)
        metrics = metrics_collector.collect(tracker)
        # PF = 100 / 50 = 2.0
        assert metrics.profit_factor == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# TELEGRAM REPORTER
# ═══════════════════════════════════════════════════════════════════════════


class TestTelegramReporter:
    """Test Telegram message formatting (no real sends)."""

    def test_format_trade_alert(self) -> None:
        """format_trade_alert should produce a readable string."""
        msg = TelegramReporter.format_trade_alert(
            direction=1,
            symbol="BTC/USDT",
            entry_price=94_250.0,
            quantity=0.044,
            notional=4_147.0,
            stop_loss=92_000.0,
            take_profit=97_625.0,
            rr_ratio=1.5,
            regime="trend_up",
            confidence=0.73,
            funding_rate=0.0001,
        )
        assert "LONG" in msg
        assert "BTC/USDT" in msg
        assert "94,250" in msg
        assert "TREND_UP" in msg
        assert "1:1.5" in msg

    def test_send_alert_mock(self, telegram_reporter: TelegramReporter) -> None:
        """send_alert with valid credentials should attempt HTTP call.

        We verify it does not crash and the internal plumbing is correct
        by testing with empty credentials (returns False) and valid
        credentials with a network error (also returns False gracefully).
        """
        async def _run() -> bool:
            # With valid credentials but no real network, it should
            # catch the exception and return False
            return await telegram_reporter.send_alert("Test message")

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_run())
            # Without a real Telegram bot, this will fail gracefully
            assert result is False
        finally:
            loop.close()

    def test_send_alert_no_credentials(self) -> None:
        """send_alert without credentials should return False."""
        reporter = TelegramReporter(bot_token="", admin_id="")

        async def _run() -> bool:
            return await reporter.send_alert("Test")

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result is False

    def test_reporter_init(self, telegram_reporter: TelegramReporter) -> None:
        """Reporter should store token and admin_id."""
        assert telegram_reporter._bot_token == "test_token_123"
        assert telegram_reporter._admin_id == "12345"


# ═══════════════════════════════════════════════════════════════════════════
# PAPER TRADING STRATEGY
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTradingStrategy:
    """Test PaperTradingStrategy construction."""

    def test_strategy_constructs(self) -> None:
        """PaperTradingStrategy should construct without errors."""
        from src.execution.strategies.paper_strategy import PaperTradingStrategy

        config = MLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
            initial_equity=10_000.0,
            dry_run=True,
        )
        strategy = PaperTradingStrategy(config=config)
        assert strategy._paper_trader is not None
        assert strategy._metrics is not None
        assert strategy._signals_total == 0

    def test_strategy_has_paper_trader(self) -> None:
        """Strategy should have a PaperTrader instance."""
        from src.execution.strategies.paper_strategy import PaperTradingStrategy

        config = MLStrategyConfig(dry_run=True)
        strategy = PaperTradingStrategy(config=config)
        assert isinstance(strategy._paper_trader, PaperTrader)
        assert strategy._paper_trader.get_equity() == 10_000.0


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case coverage."""

    def test_zero_slippage(self) -> None:
        """Zero slippage should give fill_price == current_price."""
        config = PaperTraderConfig(slippage_bps=0.0)
        trader = PaperTrader(config)
        fill = trader.simulate_fill(
            symbol="BTCUSDT", direction=1, quantity=0.1,
            current_price=50_000.0,
        )
        assert fill.fill_price == 50_000.0

    def test_close_unknown_position(self) -> None:
        """Closing unknown position should return 0.0."""
        trader = PaperTrader()
        pnl = trader.simulate_close(
            symbol="BTCUSDT", order_id="nonexistent",
            current_price=50_000.0,
        )
        assert pnl == 0.0

    def test_multiple_fills(self) -> None:
        """Multiple fills should track independently."""
        trader = PaperTrader()
        f1 = trader.simulate_fill("BTCUSDT", 1, 0.1, 50_000.0)
        f2 = trader.simulate_fill("ETHUSDT", -1, 1.0, 3_000.0)
        assert f1.order_id != f2.order_id
        stats = trader.get_stats()
        assert stats["open_positions"] == 2

    def test_daily_report_no_metrics(
        self, metrics_collector: MetricsCollector,
    ) -> None:
        """Daily report with None metrics should return fallback."""
        report = metrics_collector.get_daily_report(None)
        assert "No metrics" in report


# Import after definitions to avoid circular import
from src.execution.strategies.ml_strategy import MLStrategyConfig
