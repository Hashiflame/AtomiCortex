"""Tests for Phase 2 backtest engine (Steps 2.1 + 2.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.execution.data_catalog import AtomiCortexCatalog
from src.execution.backtest_runner import BacktestConfig, BacktestResult, BacktestRunner
from src.execution.strategies.baseline_strategy import BuyAndHoldConfig, BuyAndHoldStrategy

DATA_DIR = Path("/mnt/hdd/AtomiCortex/data/features")
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END_30D = datetime(2024, 1, 31, tzinfo=timezone.utc)
END_6M = datetime(2024, 6, 30, tzinfo=timezone.utc)

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="External data drive not mounted",
)


# ──────────────────────────────────────────────────────────────────────────────
# Catalog / Instrument
# ──────────────────────────────────────────────────────────────────────────────

class TestInstrumentCreation:
    def test_btcusdt_instrument(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        instr = catalog.get_instrument("BTCUSDT")
        assert str(instr.id) == "BTCUSDT-PERP.BINANCE"
        assert instr.price_precision == 1
        assert instr.size_precision == 3

    def test_ethusdt_instrument(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        instr = catalog.get_instrument("ETHUSDT")
        assert str(instr.id) == "ETHUSDT-PERP.BINANCE"
        assert instr.price_precision == 2
        assert instr.size_precision == 3

    def test_solusdt_instrument(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        instr = catalog.get_instrument("SOLUSDT")
        assert str(instr.id) == "SOLUSDT-PERP.BINANCE"
        assert instr.price_precision == 3
        assert instr.size_precision == 0

    def test_instrument_with_fees(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        instr = catalog.get_instrument(
            "BTCUSDT",
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0005"),
        )
        assert instr.maker_fee == Decimal("0.0002")
        assert instr.taker_fee == Decimal("0.0005")


# ──────────────────────────────────────────────────────────────────────────────
# Bar data loading
# ──────────────────────────────────────────────────────────────────────────────

class TestBarDataLoading:
    def test_loads_btcusdt_4h_bars(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        bars = catalog.load_bar_data("BTCUSDT", "4h", START, END_30D)
        assert len(bars) > 0
        # Jan 2024 has 31 days × 6 bars/day = 186 bars
        assert len(bars) >= 180

    def test_bar_timestamps_ascending(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        bars = catalog.load_bar_data("BTCUSDT", "4h", START, END_30D)
        timestamps = [b.ts_event for b in bars]
        assert timestamps == sorted(timestamps)

    def test_bar_ohlcv_valid(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        bars = catalog.load_bar_data("BTCUSDT", "4h", START, END_30D)
        for bar in bars[:10]:
            assert bar.high.as_double() >= bar.low.as_double()
            assert bar.high.as_double() >= bar.close.as_double()
            assert bar.low.as_double() <= bar.close.as_double()
            assert bar.volume.as_double() > 0

    def test_bar_timestamps_in_nanoseconds(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        bars = catalog.load_bar_data("BTCUSDT", "4h", START, END_30D)
        # 2024-01-01 in ns: 1704067200 * 1e9 ≈ 1.7e18
        assert bars[0].ts_event > 1_700_000_000_000_000_000

    def test_1d_bars(self):
        catalog = AtomiCortexCatalog(DATA_DIR)
        bars = catalog.load_bar_data("BTCUSDT", "1d", START, END_30D)
        assert 28 <= len(bars) <= 31


# ──────────────────────────────────────────────────────────────────────────────
# Buy-and-Hold Strategy on 30 days
# ──────────────────────────────────────────────────────────────────────────────

def _run_30d(trade_size: float = 0.001, **extra) -> BacktestResult:
    cfg = BacktestConfig(
        symbol="BTCUSDT",
        interval="4h",
        start=START,
        end=END_30D,
        initial_capital=10_000.0,
        leverage=5,
        maker_fee=0.0002,
        taker_fee=0.0005,
        data_dir=DATA_DIR,
    )
    cfg.__dict__.update(extra)
    runner = BacktestRunner(cfg)
    return runner.run(BuyAndHoldStrategy, {"trade_size": trade_size})


class TestBuyAndHoldStrategy:
    def test_strategy_runs_without_error(self):
        result = _run_30d()
        assert isinstance(result, BacktestResult)

    def test_orders_were_placed(self):
        result = _run_30d()
        # buy + close-position orders = at least 1
        assert result.total_trades >= 1

    def test_equity_curve_populated(self):
        result = _run_30d()
        assert len(result.equity_curve) > 0

    def test_equity_curve_datetime_type(self):
        result = _run_30d()
        dt, val = result.equity_curve[0]
        assert isinstance(dt, datetime)
        assert isinstance(val, float)

    def test_equity_curve_monotone_timestamps(self):
        result = _run_30d()
        timestamps = [dt for dt, _ in result.equity_curve]
        assert timestamps == sorted(timestamps)

    def test_btc_up_january_2024(self):
        """BTC rose ~30%+ in Jan 2024; leveraged buy-and-hold should be profitable."""
        result = _run_30d(trade_size=0.001)
        assert result.total_return_pct > 0, (
            f"Expected positive return in Jan 2024, got {result.total_return_pct:.2f}%"
        )

    def test_start_end_equity(self):
        result = _run_30d()
        assert result.start_equity == pytest.approx(10_000.0)
        assert result.end_equity > 0

    def test_deterministic_results(self):
        """Same config must produce identical results on two runs."""
        r1 = _run_30d()
        r2 = _run_30d()
        assert r1.total_return_pct == pytest.approx(r2.total_return_pct, abs=1e-6)
        assert r1.end_equity == pytest.approx(r2.end_equity, abs=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# P&L correctness with fees
# ──────────────────────────────────────────────────────────────────────────────

class TestPnLWithFees:
    def test_fees_reduce_pnl(self):
        """Run with high fees vs no fees — fees must reduce net return."""
        result_fees = _run_30d()

        # Reproduce with zero fees
        cfg_no_fee = BacktestConfig(
            symbol="BTCUSDT",
            interval="4h",
            start=START,
            end=END_30D,
            initial_capital=10_000.0,
            leverage=5,
            maker_fee=0.0,
            taker_fee=0.0,
            data_dir=DATA_DIR,
        )
        result_no_fee = BacktestRunner(cfg_no_fee).run(
            BuyAndHoldStrategy, {"trade_size": 0.001}
        )
        assert result_no_fee.end_equity >= result_fees.end_equity

    def test_end_equity_not_equal_start_equity(self):
        """A buy-and-hold over 30 days should move equity by at least 1 cent."""
        result = _run_30d()
        assert abs(result.end_equity - result.start_equity) > 0.01


# ──────────────────────────────────────────────────────────────────────────────
# BacktestResult metrics
# ──────────────────────────────────────────────────────────────────────────────

class TestBacktestResultMetrics:
    def test_total_return_pct_type(self):
        result = _run_30d()
        assert isinstance(result.total_return_pct, float)

    def test_sharpe_ratio_is_float(self):
        result = _run_30d()
        assert isinstance(result.sharpe_ratio, float)

    def test_max_drawdown_non_negative(self):
        result = _run_30d()
        assert result.max_drawdown_pct >= 0.0

    def test_max_drawdown_bounded(self):
        result = _run_30d()
        assert result.max_drawdown_pct <= 100.0

    def test_max_drawdown_computed_from_equity_curve(self):
        """Manually verify max-drawdown against equity curve."""
        result = _run_30d()
        equity = [e for _, e in result.equity_curve]
        if not equity:
            pytest.skip("No equity curve data")
        peak = equity[0]
        manual_mdd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100 if peak > 0 else 0.0
            if dd > manual_mdd:
                manual_mdd = dd
        assert result.max_drawdown_pct == pytest.approx(manual_mdd, abs=1e-6)

    def test_profit_factor_type(self):
        result = _run_30d()
        assert isinstance(result.profit_factor, float)

    def test_win_rate_bounded(self):
        result = _run_30d()
        assert 0.0 <= result.win_rate <= 1.0
