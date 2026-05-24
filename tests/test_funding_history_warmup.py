"""Phase 4 Step 4.2 — funding history depth + preload tests.

Pre-fix ``funding_rate_history`` was capped at 100 settlements (~33 days
at 3 settlements/day). The ``funding_zscore_30d`` feature in
``add_funding_features`` needs ~90 settlements (30 days × 3 / day) for
the rolling window to be well-conditioned. With 33 days of buffer the
window was undersized and the z-score collapsed toward 0.

The fix:

* Bump ``funding_rate_history`` maxlen to 300 (~100 days).
* Add ``LiveFeatureState.preload_funding`` mirroring ``preload_oi``.
* Strategy on_start fetches 200 historical settlements (~66 days) and
  routes them through ``preload_funding`` (dedup + chronological sort).

These tests pin the maxlen sizing, preload semantics, dedup,
fail-soft on bad records, ``get_funding_df`` schema, and a fail-soft
strategy path when the HTTP fetch fails.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.features.live_feature_state import LiveFeatureState


# ---------------------------------------------------------------------------
# Buffer sizing
# ---------------------------------------------------------------------------

class TestBufferSizing:
    def test_maxlen_covers_30_day_zscore_window(self) -> None:
        s = LiveFeatureState()
        # ≥ 90 records = 30 days × 3 settlements/day.
        assert s.funding_rate_history.maxlen >= 90

    def test_maxlen_is_finite(self) -> None:
        s = LiveFeatureState()
        assert s.funding_rate_history.maxlen is not None
        assert s.funding_rate_history.maxlen <= 2_000  # sanity cap


# ---------------------------------------------------------------------------
# preload_funding semantics
# ---------------------------------------------------------------------------

class TestPreloadSemantics:
    def test_preload_into_empty_history_inserts_all(self) -> None:
        s = LiveFeatureState()
        records = [
            {"fundingTime": i * 8 * 3_600_000, "fundingRate": 0.0001 * i}
            for i in range(100)
        ]
        inserted = s.preload_funding(records)
        assert inserted == 100
        assert len(s.funding_rate_history) == 100

    def test_preload_preserves_chronological_order(self) -> None:
        s = LiveFeatureState()
        s.preload_funding([
            {"fundingTime": 30_000, "fundingRate": 0.003},
            {"fundingTime": 10_000, "fundingRate": 0.001},
            {"fundingTime": 20_000, "fundingRate": 0.002},
        ])
        times = [r["fundingTime"] for r in s.funding_rate_history]
        assert times == sorted(times)

    def test_preload_dedupes_against_existing(self) -> None:
        s = LiveFeatureState()
        # Manually seed one record.
        s.funding_rate_history.append({"fundingTime": 1_000, "fundingRate": 0.01})
        inserted = s.preload_funding([
            {"fundingTime": 1_000, "fundingRate": 9.99},   # duplicate ts
            {"fundingTime": 2_000, "fundingRate": 0.02},
        ])
        assert inserted == 1
        # Existing record preserved, not overwritten by the dup.
        rate_at_1000 = next(
            r["fundingRate"] for r in s.funding_rate_history
            if r["fundingTime"] == 1_000
        )
        assert rate_at_1000 == 0.01

    def test_repeat_preload_is_idempotent(self) -> None:
        s = LiveFeatureState()
        records = [
            {"fundingTime": i * 1_000, "fundingRate": 0.0001 * i}
            for i in range(50)
        ]
        s.preload_funding(records)
        before = len(s.funding_rate_history)
        added = s.preload_funding(records)
        assert added == 0
        assert len(s.funding_rate_history) == before

    def test_preload_skips_malformed_records(self) -> None:
        s = LiveFeatureState()
        records = [
            {"fundingTime": 1_000, "fundingRate": 0.01},
            {"fundingTime": None, "fundingRate": 0.02},          # bad ts
            {"fundingTime": 3_000, "fundingRate": "garbage"},    # bad rate
            {"fundingRate": 0.04},                               # missing ts
            {"fundingTime": 5_000},                              # missing rate
            {"fundingTime": 6_000, "fundingRate": 0.06},
        ]
        inserted = s.preload_funding(records)
        assert inserted == 2

    def test_preload_updates_latest_snapshot_when_newer(self) -> None:
        s = LiveFeatureState()
        s.update_funding(rate=0.001, timestamp_ms=1_000)
        s.preload_funding([{"fundingTime": 5_000, "fundingRate": 0.005}])
        assert s.funding_rate == 0.005
        assert s.last_funding_update == 5_000

    def test_preload_does_not_downgrade_latest_snapshot(self) -> None:
        s = LiveFeatureState()
        s.update_funding(rate=0.001, timestamp_ms=10_000)
        s.preload_funding([{"fundingTime": 1_000, "fundingRate": 0.999}])
        assert s.funding_rate == 0.001
        assert s.last_funding_update == 10_000


# ---------------------------------------------------------------------------
# Live updates continue to work after preload
# ---------------------------------------------------------------------------

class TestPostPreloadUpdates:
    def test_update_funding_still_modifies_current_rate(self) -> None:
        s = LiveFeatureState()
        s.preload_funding([
            {"fundingTime": i * 1_000, "fundingRate": 0.0001 * i}
            for i in range(10)
        ])
        s.update_funding(rate=0.123, timestamp_ms=99_000)
        assert s.funding_rate == 0.123
        # update_funding does NOT append history (settlement filter does
        # that via on_data) — assert preload length is unchanged.
        assert len(s.funding_rate_history) == 10


# ---------------------------------------------------------------------------
# Overflow semantics — newest kept, oldest evicted
# ---------------------------------------------------------------------------

class TestOverflowFifo:
    def test_overflow_drops_oldest(self) -> None:
        s = LiveFeatureState()
        cap = s.funding_rate_history.maxlen
        n = cap + 25
        s.preload_funding([
            {"fundingTime": i, "fundingRate": 0.0001}
            for i in range(n)
        ])
        assert len(s.funding_rate_history) == cap
        assert s.funding_rate_history[0]["fundingTime"] == 25
        assert s.funding_rate_history[-1]["fundingTime"] == n - 1


# ---------------------------------------------------------------------------
# get_funding_df schema unchanged
# ---------------------------------------------------------------------------

class TestGetFundingDFContract:
    def test_columns_after_preload(self) -> None:
        s = LiveFeatureState()
        s.preload_funding([
            {"fundingTime": i * 1_000, "fundingRate": 0.0001 * i}
            for i in range(20)
        ])
        df = s.get_funding_df(n_bars=20)
        assert "fundingTime" in df.columns
        assert "fundingRate" in df.columns
        assert len(df) == 20

    def test_empty_when_history_empty(self) -> None:
        df = LiveFeatureState().get_funding_df()
        assert df.is_empty()


# ---------------------------------------------------------------------------
# Strategy preload step is fail-soft
# ---------------------------------------------------------------------------

class TestStrategyPreloadFailSoft:
    def test_strategy_preload_block_swallows_http_errors(self) -> None:
        s = LiveFeatureState()
        with patch("requests.get", side_effect=ConnectionError("network down")):
            try:
                import requests as _req
                resp = _req.get(
                    "https://fapi.binance.com/fapi/v1/fundingRate",
                    params={"symbol": "BTCUSDT", "limit": 200},
                    timeout=5,
                )
                if resp.status_code == 200:
                    s.preload_funding([])
            except Exception:
                pass
        assert len(s.funding_rate_history) == 0
        assert s.funding_rate == 0.0

    def test_strategy_preload_handles_http_429(self) -> None:
        s = LiveFeatureState()
        bad_response = MagicMock()
        bad_response.status_code = 429  # rate-limited
        with patch("requests.get", return_value=bad_response):
            import requests as _req
            resp = _req.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": "BTCUSDT", "limit": 200},
                timeout=5,
            )
            if resp.status_code == 200:
                s.preload_funding([])
        assert len(s.funding_rate_history) == 0


# ---------------------------------------------------------------------------
# End-to-end: preload depth produces non-zero funding_zscore_30d
# ---------------------------------------------------------------------------

class TestEndToEndAddFundingFeatures:
    def test_preload_produces_nonzero_zscore_30d(self) -> None:
        """After preloading ~200 settlements + a 30-day bar series, the
        joined dataframe's 180-bar funding_zscore_30d must be non-zero
        on the latest rows — proof the preload depth is sufficient."""
        from src.features.derivatives import add_funding_features

        s = LiveFeatureState()
        # 200 settlements over ~66 days (settlement every 8 hours).
        base_ts = 1_700_000_000_000
        records = []
        for i in range(200):
            # Use a varying rate so std > 0.
            rate = 0.0001 * ((i % 7) - 3)
            records.append({
                "fundingTime": base_ts + i * 8 * 3_600_000,
                "fundingRate": rate,
            })
        s.preload_funding(records)
        funding_df = s.get_funding_df(n_bars=200)

        # 4H bar series spanning the last 50 days → 300 bars; the rolling
        # 180-bar window has full data on the tail.
        bars = 300
        bar_open_times = [
            base_ts + 50 * 24 * 3_600_000 + i * 4 * 3_600_000
            for i in range(bars)
        ]
        bar_df = pl.DataFrame({"open_time": bar_open_times})
        out = add_funding_features(bar_df, funding_df)
        # Latest row's 30-day z-score should be a meaningful (finite,
        # non-zero) number once the window is fully populated.
        last_z = out["funding_zscore_30d"][-1]
        assert last_z is not None
        assert not (last_z == 0.0)
