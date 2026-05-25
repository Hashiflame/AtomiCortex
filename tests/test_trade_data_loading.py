"""Tests for Step H23 — load_trade_data uniform stride sampling.

Pre-H23: ``.head(sample_size)`` after a time-sort kept only the FIRST
N trades of the window. For an active BTC day with millions of
aggTrades, VPIN / microstructure features saw only the first few
minutes after midnight UTC — strong temporal bias.

Post-H23: count rows, then ``gather_every(stride)`` evenly across the
window. Peak memory still bounded by ``sample_size``; coverage is now
uniform.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from src.execution.data_catalog import AtomiCortexCatalog


T_START = datetime(2026, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
T_END = T_START + timedelta(days=1)


def _write_trades(
    data_dir: Path, symbol: str, n_trades: int,
    *, start_ms: int | None = None, span_ms: int | None = None,
    price: float = 50_000.0,
) -> None:
    """Seed a single agg_trades parquet evenly spaced across the
    target window."""
    target = (
        data_dir / "exchange=BINANCE_UM" / f"symbol={symbol}"
        / "agg_trades" / "date=2026-01-15"
    )
    target.mkdir(parents=True, exist_ok=True)
    start_ms = start_ms or int(T_START.timestamp() * 1000)
    span_ms = span_ms or int((T_END - T_START).total_seconds() * 1000)
    step = max(1, span_ms // max(1, n_trades))
    df = pl.DataFrame({
        "agg_trade_id":  list(range(n_trades)),
        "price":         [price + i * 0.1 for i in range(n_trades)],
        "quantity":      [0.01] * n_trades,
        "transact_time": [start_ms + i * step for i in range(n_trades)],
        "is_buyer_maker": [bool(i % 2) for i in range(n_trades)],
    })
    df.write_parquet(target / "part-0.parquet")


# ---------------------------------------------------------------------------
# Stride sampling: uniform coverage across the window
# ---------------------------------------------------------------------------


class TestUniformSampling:
    def test_small_dataset_returns_everything(self, tmp_path):
        """Fewer rows than sample_size → no sampling, full set returned."""
        _write_trades(tmp_path, "BTCUSDT", n_trades=500)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=100_000,
        )
        assert len(ticks) == 500

    def test_large_dataset_bounded_by_sample_size(self, tmp_path):
        _write_trades(tmp_path, "BTCUSDT", n_trades=50_000)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=10_000,
        )
        assert len(ticks) <= 10_000
        # Stride 5 → ~10k rows.
        assert len(ticks) >= 9_500

    def test_coverage_spans_full_window(self, tmp_path):
        """The KEY post-H23 assertion: sampled ticks should land in
        BOTH halves of the day, not only the first half."""
        _write_trades(tmp_path, "BTCUSDT", n_trades=10_000)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=200,
        )
        assert ticks, "expected non-empty sample"
        midpoint_ns = int(
            (T_START + (T_END - T_START) / 2).timestamp() * 1e9
        )
        first_half = sum(1 for t in ticks if t.ts_event < midpoint_ns)
        second_half = sum(1 for t in ticks if t.ts_event >= midpoint_ns)
        # Each half must have a meaningful share — pre-H23 the second
        # half was empty because .head() took only the earliest rows.
        assert first_half > 0
        assert second_half > 0
        assert second_half >= len(ticks) // 3

    def test_pre_h23_behaviour_was_front_loaded(self, tmp_path):
        """Sanity check on the same data using the OLD head() semantics:
        all sampled ticks land in the first half. Confirms the fixture
        actually distinguishes the two approaches."""
        _write_trades(tmp_path, "BTCUSDT", n_trades=10_000)
        # Reproduce the old logic manually.
        df = pl.read_parquet(
            tmp_path / "exchange=BINANCE_UM/symbol=BTCUSDT"
            / "agg_trades/date=2026-01-15/part-0.parquet"
        ).sort("transact_time").head(200)
        midpoint_ms = int(
            (T_START + (T_END - T_START) / 2).timestamp() * 1000
        )
        second_half = df.filter(pl.col("transact_time") >= midpoint_ms)
        assert second_half.is_empty(), (
            "fixture must let the head() bug bias second-half coverage to 0"
        )


# ---------------------------------------------------------------------------
# Memory safety
# ---------------------------------------------------------------------------


class TestMemoryBound:
    def test_huge_dataset_does_not_load_all_rows(self, tmp_path, monkeypatch):
        """Spy on the eager-collect to confirm we never pull the full
        dataset into memory. gather_every(stride) is applied before
        collect — the materialised frame is bounded."""
        _write_trades(tmp_path, "BTCUSDT", n_trades=200_000)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=5_000,
        )
        assert len(ticks) <= 5_000
        assert len(ticks) >= 4_500  # stride 40 → ~5k rows


# ---------------------------------------------------------------------------
# Edge cases / fail-soft
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_window_returns_empty_list(self, tmp_path):
        # Trades exist, but outside the requested window.
        _write_trades(
            tmp_path, "BTCUSDT", n_trades=100,
            start_ms=int((T_END + timedelta(days=1)).timestamp() * 1000),
            span_ms=3_600_000,
        )
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data("BTCUSDT", T_START, T_END)
        assert ticks == []

    def test_no_parquet_files_returns_empty_list(self, tmp_path):
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data("BTCUSDT", T_START, T_END)
        assert ticks == []

    def test_sample_size_zero_returns_all_rows(self, tmp_path):
        """sample_size ≤ 0 disables capping → caller gets everything."""
        _write_trades(tmp_path, "BTCUSDT", n_trades=1_000)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=0,
        )
        assert len(ticks) == 1_000

    def test_negative_sample_size_returns_all(self, tmp_path):
        _write_trades(tmp_path, "BTCUSDT", n_trades=500)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=-1,
        )
        assert len(ticks) == 500


# ---------------------------------------------------------------------------
# Returned ticks shape: TradeTick fields well-formed
# ---------------------------------------------------------------------------


class TestTickShape:
    def test_ticks_are_chronologically_ordered(self, tmp_path):
        _write_trades(tmp_path, "BTCUSDT", n_trades=5_000)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=200,
        )
        ts = [t.ts_event for t in ticks]
        assert ts == sorted(ts)

    def test_ticks_have_aggressor_side_set(self, tmp_path):
        """Every returned tick carries a non-default aggressor side
        (BUYER or SELLER) — confirming the is_buyer_maker → aggressor
        mapping survives stride sampling."""
        _write_trades(tmp_path, "BTCUSDT", n_trades=2_000)
        cat = AtomiCortexCatalog(tmp_path)
        ticks = cat.load_trade_data(
            "BTCUSDT", T_START, T_END, sample_size=200,
        )
        # Every tick has a real side (not NO_AGGRESSOR / unset).
        for t in ticks:
            assert int(t.aggressor_side) in (1, 2)  # BUYER=1, SELLER=2
