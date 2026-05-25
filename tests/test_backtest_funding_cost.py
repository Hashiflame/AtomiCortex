"""Tests for Steps H18 + H19 — realistic funding cost in backtest.

H19: ``_TYPICAL_FUNDING_RATE = 0.0001`` was 3-10× too low for real BTC
perpetuals (typical 0.0003-0.0010). Default bumped + real funding now
read from parquet when available.

H18: funding was billed for the entire backtest window. New formula:
``num_round_trips × avg_holding_hours``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.execution.backtest_runner import (
    _TYPICAL_FUNDING_RATE,
    BacktestConfig,
    _estimate_costs,
    _load_actual_funding_rate,
    _normalize_symbol,
)


T_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
T_END = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _fake_bar(price: float):
    """Bare-minimum bar shape used by _avg_price."""
    return SimpleNamespace(
        open=SimpleNamespace(as_double=lambda: price),
        close=SimpleNamespace(as_double=lambda: price),
    )


def _config(tmp_path: Path, **overrides) -> BacktestConfig:
    kwargs = dict(
        symbol="BTCUSDT-PERP",
        interval="4h",
        start=T_START,
        end=T_END,
        data_dir=tmp_path,
    )
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)


def _seed_funding_parquet(
    tmp_path: Path, symbol_dir: str, rates_with_ts: list[tuple[int, float]],
) -> None:
    target = (
        tmp_path / "exchange=BINANCE_UM" / f"symbol={symbol_dir}"
        / "funding_rate" / "date=2026-01"
    )
    target.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "fundingTime": [t for t, _ in rates_with_ts],
        "fundingRate": [r for _, r in rates_with_ts],
    })
    df.write_parquet(target / "part-0.parquet")


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_legacy_alias_bumped(self):
        """Old _TYPICAL_FUNDING_RATE constant now sits at the realistic
        0.0003 (was 0.0001) so external importers get the fix too."""
        assert _TYPICAL_FUNDING_RATE == 0.0003

    def test_config_typical_funding_rate_default(self, tmp_path):
        cfg = _config(tmp_path)
        assert cfg.typical_funding_rate == 0.0003

    def test_config_avg_holding_hours_default(self, tmp_path):
        cfg = _config(tmp_path)
        assert cfg.avg_holding_hours == 24.0


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


class TestSymbolNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("BTCUSDT", "BTCUSDT"),
        ("BTCUSDT-PERP", "BTCUSDT"),
        ("ETHUSDT-PERP.BINANCE", "ETHUSDT"),
        ("btcusdt", "BTCUSDT"),
    ])
    def test_normalize(self, raw, expected):
        assert _normalize_symbol(raw) == expected


# ---------------------------------------------------------------------------
# Real-funding loader
# ---------------------------------------------------------------------------


class TestLoadActualFundingRate:
    def test_no_parquet_returns_none(self, tmp_path):
        cfg = _config(tmp_path)
        assert _load_actual_funding_rate(cfg) is None

    def test_loads_mean_abs_in_window(self, tmp_path):
        # Mix of positive/negative rates; mean(abs) = 0.0005.
        ts0 = int(T_START.timestamp() * 1000) + 3_600_000
        rates = [
            (ts0 + 8 * 3_600_000 * i, sign * 0.0005)
            for i, sign in enumerate([1, -1, 1, -1, 1, -1])
        ]
        _seed_funding_parquet(tmp_path, "BTCUSDT", rates)
        cfg = _config(tmp_path)
        assert _load_actual_funding_rate(cfg) == pytest.approx(0.0005)

    def test_window_filter_excludes_out_of_range(self, tmp_path):
        in_window = (int(T_START.timestamp() * 1000) + 3_600_000, 0.001)
        before = (int(T_START.timestamp() * 1000) - 86_400_000, 0.999)
        after = (int(T_END.timestamp() * 1000) + 86_400_000, 0.999)
        _seed_funding_parquet(tmp_path, "BTCUSDT", [before, in_window, after])
        cfg = _config(tmp_path)
        # Only the in-window entry should drive the mean.
        assert _load_actual_funding_rate(cfg) == pytest.approx(0.001)

    def test_normalizes_symbol_for_lookup(self, tmp_path):
        ts0 = int(T_START.timestamp() * 1000) + 3_600_000
        _seed_funding_parquet(tmp_path, "BTCUSDT", [(ts0, 0.0004)])
        cfg = _config(tmp_path, symbol="BTCUSDT-PERP.BINANCE")
        assert _load_actual_funding_rate(cfg) == pytest.approx(0.0004)

    def test_empty_window_returns_none(self, tmp_path):
        # All entries outside the backtest window.
        out_of_range = int(T_END.timestamp() * 1000) + 86_400_000
        _seed_funding_parquet(tmp_path, "BTCUSDT", [(out_of_range, 0.001)])
        cfg = _config(tmp_path)
        assert _load_actual_funding_rate(cfg) is None

    def test_corrupt_parquet_returns_none(self, tmp_path):
        target = (
            tmp_path / "exchange=BINANCE_UM" / "symbol=BTCUSDT"
            / "funding_rate" / "date=2026-01"
        )
        target.mkdir(parents=True, exist_ok=True)
        (target / "part-0.parquet").write_bytes(b"not a parquet file")
        cfg = _config(tmp_path)
        assert _load_actual_funding_rate(cfg) is None


# ---------------------------------------------------------------------------
# _estimate_costs: H18 (hours-in-position) + H19 (real funding)
# ---------------------------------------------------------------------------


class TestEstimateCostsFunding:
    def test_zero_round_trips_no_funding(self, tmp_path):
        cfg = _config(tmp_path)
        bars = [_fake_bar(50_000.0)]
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=0,
        )
        assert fund == 0.0

    def test_billing_uses_position_hours_not_window(self, tmp_path):
        """4 round trips × 24h holding = 96h, regardless of the
        backtest window (which is ~744h here)."""
        cfg = _config(tmp_path)  # default avg_holding_hours=24
        bars = [_fake_bar(50_000.0)]
        total_orders = 8  # → 4 round trips
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=total_orders,
        )
        # Expected: avg_notional = 0.001 × 50000 = 50
        # position_hours = 4 × 24 = 96
        # num_payments = 96 / 8 = 12
        # funding_cost = 50 × 0.0003 × 12 = 0.18
        assert fund == pytest.approx(0.18, rel=1e-6)

    def test_falls_back_to_config_default_when_no_parquet(self, tmp_path):
        cfg = _config(tmp_path)  # no parquet seeded
        bars = [_fake_bar(50_000.0)]
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=2,
        )
        # 50 × 0.0003 × (24/8) = 0.045
        assert fund == pytest.approx(0.045, rel=1e-6)

    def test_uses_real_parquet_funding_when_available(self, tmp_path):
        # Real funding 0.0008 (8 bps/8h) drives the calculation.
        ts0 = int(T_START.timestamp() * 1000) + 3_600_000
        _seed_funding_parquet(
            tmp_path, "BTCUSDT",
            [(ts0 + i * 8 * 3_600_000, 0.0008) for i in range(5)],
        )
        cfg = _config(tmp_path)
        bars = [_fake_bar(50_000.0)]
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=2,
        )
        # 50 × 0.0008 × 3 = 0.12 (not the default 0.045).
        assert fund == pytest.approx(0.12, rel=1e-6)

    def test_explicit_typical_funding_rate_override(self, tmp_path):
        cfg = _config(tmp_path, typical_funding_rate=0.0008)
        bars = [_fake_bar(50_000.0)]
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=2,
        )
        # 50 × 0.0008 × 3 = 0.12
        assert fund == pytest.approx(0.12, rel=1e-6)

    def test_avg_holding_hours_override(self, tmp_path):
        cfg = _config(tmp_path, avg_holding_hours=48.0)
        bars = [_fake_bar(50_000.0)]
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=2,
        )
        # position_hours = 1 × 48 = 48; num_payments = 6
        # 50 × 0.0003 × 6 = 0.09
        assert fund == pytest.approx(0.09, rel=1e-6)

    def test_parquet_beats_explicit_default_override(self, tmp_path):
        """Loader return value wins over cfg.typical_funding_rate
        (real data is more authoritative than the fallback)."""
        ts0 = int(T_START.timestamp() * 1000) + 3_600_000
        _seed_funding_parquet(tmp_path, "BTCUSDT", [(ts0, 0.0005)])
        cfg = _config(tmp_path, typical_funding_rate=0.0010)
        bars = [_fake_bar(50_000.0)]
        _, _, fund = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=2,
        )
        # Uses parquet value 0.0005, not the override 0.0010.
        # 50 × 0.0005 × 3 = 0.075
        assert fund == pytest.approx(0.075, rel=1e-6)


# ---------------------------------------------------------------------------
# Pre-H18/H19 sanity: hours-billed difference vs full window
# ---------------------------------------------------------------------------


class TestPreVsPostFix:
    def test_post_fix_strictly_less_than_pre_fix_when_low_holding(self, tmp_path):
        """Same scenario, billed two ways:
          pre-fix: hours = (end-start) ~ 744h
          post-fix: hours = num_rt × 24h (much smaller for short trades)
        The H18 fix produces a strictly lower funding cost when the
        bot doesn't sit in a position the whole window."""
        cfg = _config(tmp_path)
        bars = [_fake_bar(50_000.0)]
        _, _, fund_post = _estimate_costs(
            cfg, strategy_config={"trade_size": 0.001},
            bars=bars, total_orders=4,  # 2 round trips → 48h held
        )
        full_window_hours = (T_END - T_START).total_seconds() / 3600
        # Pre-fix billing on the same rate.
        pre_fix = 50.0 * 0.0003 * (full_window_hours / 8.0)
        assert fund_post < pre_fix
        # Sanity: order of magnitude smaller for a low-frequency strat.
        assert pre_fix / max(fund_post, 1e-12) > 5.0
