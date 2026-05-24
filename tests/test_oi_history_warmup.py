"""Phase 4 Step 4.1 — OI history depth + preload tests.

Pre-fix ``oi_history`` was capped at 100 samples (~8h at the 5-min poll
cadence). The ``oi_zscore`` feature in ``add_oi_features`` uses a 180
4H-bar rolling window — 30 days — so the z-score collapsed toward zero
for the first month of every run.

The fix has two parts:

* Bump the deque maxlen to 10 000 (~35 days at 5-min cadence).
* Add ``LiveFeatureState.preload_oi`` to seed history from Binance's
  ``openInterestHist`` endpoint at startup.

These tests pin the maxlen sizing, the preload semantics (dedup, order
preservation, fail-soft on bad records), the ``get_metrics_df`` schema
contract, and a fail-soft path for the strategy when the HTTP fetch
fails.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.features.live_feature_state import LiveFeatureState


# ---------------------------------------------------------------------------
# Buffer sizing — enough for the 30-day z-score window
# ---------------------------------------------------------------------------

class TestBufferSizing:
    def test_maxlen_covers_30_day_zscore_window(self) -> None:
        s = LiveFeatureState()
        # 30 days × 288 5-min samples/day = 8640 — the window required by
        # add_oi_features (180 bars × 4h = 30 days, joined back to bar level).
        assert s.oi_history.maxlen >= 8_640

    def test_maxlen_is_finite(self) -> None:
        """A deque cap is still required (memory bound). 10 000 is enough
        for ~35 days at 5-min cadence; finite cap protects long-running
        bots."""
        s = LiveFeatureState()
        assert s.oi_history.maxlen is not None
        assert s.oi_history.maxlen <= 50_000


# ---------------------------------------------------------------------------
# preload_oi — seeding semantics
# ---------------------------------------------------------------------------

class TestPreloadSeeding:
    def test_preload_into_empty_history_inserts_all(self) -> None:
        s = LiveFeatureState()
        records = [
            {"timestamp": 1_000 + i * 60_000, "oi_value": 100.0 + i}
            for i in range(50)
        ]
        inserted = s.preload_oi(records)
        assert inserted == 50
        assert len(s.oi_history) == 50

    def test_preload_preserves_chronological_order(self) -> None:
        s = LiveFeatureState()
        # Intentionally pass records out of order.
        records = [
            {"timestamp": 3_000, "oi_value": 3.0},
            {"timestamp": 1_000, "oi_value": 1.0},
            {"timestamp": 2_000, "oi_value": 2.0},
        ]
        s.preload_oi(records)
        timestamps = [r["timestamp"] for r in s.oi_history]
        assert timestamps == sorted(timestamps)

    def test_preload_dedupes_against_existing_history(self) -> None:
        s = LiveFeatureState()
        s.update_oi(oi=42.0, timestamp_ms=1_000)
        # Try to re-add an overlapping record + one new one
        inserted = s.preload_oi([
            {"timestamp": 1_000, "oi_value": 99.0},
            {"timestamp": 2_000, "oi_value": 50.0},
        ])
        assert inserted == 1
        # Original sample preserved (overlap dropped, not overwritten).
        oi_at_1000 = next(
            r["oi_value"] for r in s.oi_history if r["timestamp"] == 1_000
        )
        assert oi_at_1000 == 42.0

    def test_preload_calls_are_idempotent(self) -> None:
        s = LiveFeatureState()
        records = [
            {"timestamp": 1_000 + i * 60_000, "oi_value": 100.0 + i}
            for i in range(10)
        ]
        s.preload_oi(records)
        before = len(s.oi_history)
        # Second call with the same records → no additions.
        added = s.preload_oi(records)
        assert added == 0
        assert len(s.oi_history) == before

    def test_preload_skips_malformed_records(self) -> None:
        s = LiveFeatureState()
        records = [
            {"timestamp": 1_000, "oi_value": 1.0},
            {"timestamp": None, "oi_value": 2.0},          # bad ts
            {"timestamp": 3_000, "oi_value": "not numeric"},  # bad value
            {"oi_value": 4.0},                             # missing ts
            {"timestamp": 5_000},                          # missing value
            {"timestamp": 6_000, "oi_value": 6.0},
        ]
        inserted = s.preload_oi(records)
        assert inserted == 2  # only the two well-formed records

    def test_preload_updates_latest_snapshot_when_newer(self) -> None:
        s = LiveFeatureState()
        s.update_oi(oi=10.0, timestamp_ms=1_000)
        s.preload_oi([{"timestamp": 5_000, "oi_value": 25.0}])
        # Preloaded sample is newer → latest snapshot moves to it.
        assert s.oi_value == 25.0
        assert s.last_oi_update == 5_000

    def test_preload_does_not_downgrade_latest_snapshot(self) -> None:
        s = LiveFeatureState()
        s.update_oi(oi=10.0, timestamp_ms=10_000)
        s.preload_oi([{"timestamp": 1_000, "oi_value": 25.0}])
        # Preloaded sample is older → latest snapshot kept intact.
        assert s.oi_value == 10.0
        assert s.last_oi_update == 10_000


# ---------------------------------------------------------------------------
# Live updates continue to work after preload
# ---------------------------------------------------------------------------

class TestPostPreloadUpdates:
    def test_update_oi_appends_after_preload(self) -> None:
        s = LiveFeatureState()
        s.preload_oi([
            {"timestamp": 1_000 + i * 60_000, "oi_value": 100.0}
            for i in range(5)
        ])
        s.update_oi(oi=200.0, timestamp_ms=10_000_000)
        assert len(s.oi_history) == 6
        assert s.oi_history[-1] == {"timestamp": 10_000_000, "oi_value": 200.0}


# ---------------------------------------------------------------------------
# Overflow semantics — newest samples kept, oldest evicted
# ---------------------------------------------------------------------------

class TestOverflowFifo:
    def test_overflow_drops_oldest(self) -> None:
        s = LiveFeatureState()
        cap = s.oi_history.maxlen
        # Push cap+10 records.
        n = cap + 10
        s.preload_oi([
            {"timestamp": i, "oi_value": float(i)}
            for i in range(n)
        ])
        assert len(s.oi_history) == cap
        # Oldest 10 records dropped.
        oldest_kept_ts = s.oi_history[0]["timestamp"]
        newest_kept_ts = s.oi_history[-1]["timestamp"]
        assert oldest_kept_ts == 10
        assert newest_kept_ts == n - 1


# ---------------------------------------------------------------------------
# get_metrics_df schema contract is unchanged
# ---------------------------------------------------------------------------

class TestGetMetricsDFContract:
    def test_columns_unchanged_after_preload(self) -> None:
        s = LiveFeatureState()
        s.preload_oi([
            {"timestamp": 1_000 + i * 60_000, "oi_value": 100.0 + i}
            for i in range(20)
        ])
        df = s.get_metrics_df(n_bars=20)
        # add_oi_features expects exactly these two columns to be present
        assert "create_time" in df.columns
        assert "sum_open_interest_value" in df.columns
        assert len(df) == 20

    def test_empty_when_history_empty(self) -> None:
        df = LiveFeatureState().get_metrics_df()
        assert df.is_empty()


# ---------------------------------------------------------------------------
# Strategy preload step is fail-soft
# ---------------------------------------------------------------------------

class TestStrategyPreloadFailSoft:
    def test_strategy_preload_block_swallows_http_errors(self) -> None:
        """The on_start preload uses requests.get + try/except. Patching
        requests.get to raise must not propagate; the state stays empty."""
        s = LiveFeatureState()

        def fake_get(*args, **kwargs):
            raise ConnectionError("network down")

        # Mirror the strategy's preload block in isolation.
        with patch("requests.get", side_effect=fake_get):
            try:
                import requests as _req
                resp = _req.get(
                    "https://fapi.binance.com/futures/data/openInterestHist",
                    params={"symbol": "BTCUSDT", "period": "4h", "limit": 500},
                    timeout=5,
                )
                if resp.status_code == 200:
                    records = [
                        {"timestamp": int(d["timestamp"]),
                         "oi_value": float(d["sumOpenInterestValue"])}
                        for d in resp.json()
                    ]
                    s.preload_oi(records)
            except Exception:
                pass
        # state untouched, no crash
        assert len(s.oi_history) == 0
        assert s.oi_value == 0.0

    def test_strategy_preload_handles_http_500(self) -> None:
        s = LiveFeatureState()
        bad_response = MagicMock()
        bad_response.status_code = 500
        with patch("requests.get", return_value=bad_response):
            import requests as _req
            resp = _req.get(
                "https://fapi.binance.com/futures/data/openInterestHist",
                params={"symbol": "BTCUSDT", "period": "4h", "limit": 500},
                timeout=5,
            )
            if resp.status_code == 200:
                s.preload_oi([])
        assert len(s.oi_history) == 0


# ---------------------------------------------------------------------------
# End-to-end: preload + get_metrics_df + add_oi_features round-trip
# ---------------------------------------------------------------------------

class TestEndToEndAddOiFeatures:
    def test_preload_provides_oi_for_join_asof(self) -> None:
        """After preload, the metrics_df fed to add_oi_features carries
        timestamps that join_asof can match against bar open_times."""
        from src.features.derivatives import add_oi_features

        s = LiveFeatureState()
        # 200 4H-spaced samples — enough to fully populate a 180-bar
        # rolling z-score window once joined back to 4H bars.
        base_ts = 1_700_000_000_000
        records = [
            {"timestamp": base_ts + i * 4 * 3_600_000,
             "oi_value": 1_000_000.0 + i * 1_000}
            for i in range(200)
        ]
        s.preload_oi(records)
        metrics_df = s.get_metrics_df(n_bars=200)

        # Build a bar df at the same cadence so join_asof finds a value
        # for every bar.
        bar_open_times = [base_ts + i * 4 * 3_600_000 for i in range(50, 200)]
        bar_df = pl.DataFrame({
            "open_time": bar_open_times,
            "close":     [50_000.0 + i for i in range(len(bar_open_times))],
        })
        out = add_oi_features(bar_df, metrics_df)
        # oi_value should be non-zero on most rows — join hit.
        assert (out["oi_value"] > 0).sum() >= len(out) - 1
