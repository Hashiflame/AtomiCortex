"""
Live feature state manager.

Maintains rolling buffers and latest derivative data for use in
``build_from_buffer()`` during live inference.

Solves train/serve skew:
- ``funding_rate``: updated from ``BinanceFuturesMarkPriceUpdate`` events
- ``oi_value``: updated from periodic REST poll
- ``bar_buffer``: rolling window of recent OHLCV bars

Thread-safe for Nautilus actor model (single-threaded event loop).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from src.logger import get_logger

_log = get_logger(__name__)


# Interval → duration in ms (for close_time → open_time conversion)
_INTERVAL_MS: dict[str, int] = {
    "4h":  4 * 3_600_000,
    "1h":  1 * 3_600_000,
    "15m": 15 * 60_000,
    "5m":  5 * 60_000,
    "1m":  1 * 60_000,
}


@dataclass
class LiveFeatureState:
    """Rolling state for live feature computation.

    Updated by strategy's ``on_data()`` and periodic tasks.

    The ``get_funding_df()`` / ``get_metrics_df()`` methods return
    DataFrames whose schemas match what ``add_funding_features()`` and
    ``add_oi_features()`` in ``src.features.derivatives`` expect.
    """

    # Bar buffers (maxlen keeps memory bounded)
    bar_buffer_4h: deque = field(default_factory=lambda: deque(maxlen=400))
    bar_buffer_1h: deque = field(default_factory=lambda: deque(maxlen=500))
    bar_buffer_15m: deque = field(default_factory=lambda: deque(maxlen=600))

    # Latest derivatives (updated from live feed)
    funding_rate: float = 0.0
    # Phase 4 Step 4.2 — funding_zscore_30d in add_funding_features uses a
    # 180 4H-bar rolling window (~30 days). At 3 settlements/day that's
    # ≥ 90 records; bumped to 300 (~100 days) so the window stays
    # populated even after preloaded samples age out. Memory ≈ 15 KB.
    funding_rate_history: deque = field(
        default_factory=lambda: deque(maxlen=300)
    )

    oi_value: float = 0.0
    # Phase 4 Step 4.1 — oi_zscore in add_oi_features uses a 180 4H-bar
    # rolling window (~30 days). At 5-min poll cadence that's ~8 640
    # samples; bumped to 10 000 (~35 days) so live z-score is stable
    # even when the bot has been up long enough for preloaded samples
    # to age out. Memory cost ≈ 500 KB — negligible.
    oi_history: deque = field(
        default_factory=lambda: deque(maxlen=10_000)
    )

    # Timestamps (unix ms)
    last_funding_update: int = 0
    last_oi_update: int = 0

    # One-shot WARNING flag: emitted the first time get_bar_df() falls back
    # to volume*0.5 because add_bar() was called without taker_buy_volume.
    # In live this means CVD ≡ 0 — a real train/serve skew. The flag keeps
    # the log from being spammed every inference tick.
    _tbv_fallback_warned: bool = False

    # ---------------------------------------------------------------
    # Updates
    # ---------------------------------------------------------------

    def update_funding(self, rate: float, timestamp_ms: int) -> None:
        """Update latest funding rate (called from ``on_data``).

        .. note:: History is managed separately — preloaded from REST and
           appended only at settlement times (every 8h) by the strategy's
           ``on_data()`` handler.  This method only updates the current rate.
        """
        self.funding_rate = rate
        self.last_funding_update = timestamp_ms

    def update_oi(self, oi: float, timestamp_ms: int) -> None:
        """Called from periodic REST poll (every 5 min)."""
        self.oi_value = oi
        self.last_oi_update = timestamp_ms
        self.oi_history.append({
            "timestamp": timestamp_ms,
            "oi_value": oi,
        })

    def preload_funding(self, records: list[dict]) -> int:
        """Seed ``funding_rate_history`` with historical settlements.

        Same shape and semantics as ``preload_oi``: idempotent (dedup
        against existing entries by ``fundingTime``), preserves
        chronological order, skips malformed records, and updates the
        latest snapshot only if a preloaded sample is fresher than the
        current one. Returns the number of records inserted.

        Expected record schema (mirrors what ``/fapi/v1/fundingRate``
        returns after the strategy parses it):
            {"fundingTime": <epoch ms int>, "fundingRate": <float>}
        """
        existing_ts = {r.get("fundingTime") for r in self.funding_rate_history}
        new_clean: list[dict] = []
        for rec in records:
            try:
                ts = int(rec["fundingTime"])
                rate = float(rec["fundingRate"])
            except (KeyError, TypeError, ValueError):
                continue
            if ts in existing_ts:
                continue
            new_clean.append({"fundingTime": ts, "fundingRate": rate})
            existing_ts.add(ts)
        if not new_clean:
            return 0
        merged = sorted(
            list(self.funding_rate_history) + new_clean,
            key=lambda r: r["fundingTime"],
        )
        self.funding_rate_history.clear()
        self.funding_rate_history.extend(merged)
        latest = merged[-1]
        if latest["fundingTime"] >= self.last_funding_update:
            self.funding_rate = latest["fundingRate"]
            self.last_funding_update = latest["fundingTime"]
        return len(new_clean)

    def preload_oi(self, records: list[dict]) -> int:
        """Seed ``oi_history`` with historical OI samples.

        Called once at strategy startup (after fetching Binance's
        ``openInterestHist`` endpoint) so the 30-day ``oi_zscore``
        rolling window is meaningful from the very first inference,
        rather than degenerating to ~0 for the first few days of
        run-time. Each record must carry ``timestamp`` (epoch ms)
        and ``oi_value``; malformed records are skipped silently
        (fail-soft — strategy still operates).

        Idempotent w.r.t. already-present samples: records whose
        timestamp matches an existing one are dropped, so calling
        ``preload_oi`` twice with overlapping data does not duplicate
        history. Buffer is left chronologically sorted.

        Returns the number of records actually inserted.
        """
        existing_ts = {r.get("timestamp") for r in self.oi_history}
        new_clean: list[dict] = []
        for rec in records:
            try:
                ts = int(rec["timestamp"])
                oi = float(rec["oi_value"])
            except (KeyError, TypeError, ValueError):
                continue
            if ts in existing_ts:
                continue
            new_clean.append({"timestamp": ts, "oi_value": oi})
            existing_ts.add(ts)
        if not new_clean:
            return 0
        # Merge + sort + re-seed deque so chronological order is preserved.
        merged = sorted(
            list(self.oi_history) + new_clean,
            key=lambda r: r["timestamp"],
        )
        self.oi_history.clear()
        self.oi_history.extend(merged)
        # Update latest snapshot if the preloaded samples are fresher.
        latest = merged[-1]
        if latest["timestamp"] >= self.last_oi_update:
            self.oi_value = latest["oi_value"]
            self.last_oi_update = latest["timestamp"]
        return len(new_clean)

    # ---------------------------------------------------------------
    # DataFrame builders (compatible with derivatives.py expectations)
    # ---------------------------------------------------------------

    def get_funding_df(self, n_bars: int = 100) -> pl.DataFrame:
        """Build funding DataFrame compatible with ``add_funding_features()``.

        Returns columns: ``fundingTime`` (Int64, ms), ``fundingRate`` (Float64).
        Empty DataFrame when no history is available (zero-fill fallback
        in ``add_funding_features`` handles this gracefully).

        Records may use either ``fundingTime``/``fundingRate`` (new format
        from preload + settlement) or ``timestamp``/``funding_rate`` (legacy).
        """
        if not self.funding_rate_history:
            return pl.DataFrame()
        records = list(self.funding_rate_history)[-n_bars:]
        df = pl.DataFrame(records)
        # Normalize column names to what add_funding_features() expects
        rename_map = {}
        if "timestamp" in df.columns and "fundingTime" not in df.columns:
            rename_map["timestamp"] = "fundingTime"
        if "funding_rate" in df.columns and "fundingRate" not in df.columns:
            rename_map["funding_rate"] = "fundingRate"
        if rename_map:
            df = df.rename(rename_map)
        return df

    def get_metrics_df(self, n_bars: int = 100) -> pl.DataFrame:
        """Build metrics DataFrame compatible with ``add_oi_features()``.

        Returns columns: ``create_time`` (Int64, ms),
        ``sum_open_interest_value`` (Float64).
        Empty DataFrame when no history is available (zero-fill fallback
        in ``add_oi_features`` handles this gracefully).
        """
        if not self.oi_history:
            return pl.DataFrame()
        records = list(self.oi_history)[-n_bars:]
        return pl.DataFrame(records).rename({
            "timestamp": "create_time",
            "oi_value": "sum_open_interest_value",
        })

    # ---------------------------------------------------------------
    # Bar management
    # ---------------------------------------------------------------

    def add_bar(
        self,
        bar,
        interval: str,
        taker_buy_volume: float | None = None,
    ) -> None:
        """Add a closed bar to the appropriate buffer.

        Timestamp handling
        ------------------
        Nautilus ``bar.ts_event`` = bar **close** time (nanoseconds).
        Parquet ``open_time`` = bar **open** time (milliseconds).
        We convert: ``open_time = ts_event - bar_duration``.

        Parameters
        ----------
        bar:
            Nautilus ``Bar`` object (or any object with ``ts_event``,
            ``open``, ``high``, ``low``, ``close``, ``volume`` attrs).
        interval:
            One of ``"4h"``, ``"1h"``, ``"15m"``.
        taker_buy_volume:
            Real taker-buy base-asset volume for the bar (matches Binance
            ``taker_buy_base_asset_volume`` used in training). When ``None``,
            ``get_bar_df()`` will fall back to ``volume*0.5`` so CVD ≡ 0
            (the historical live behavior) — and emit a one-shot WARNING.
        """
        bar_duration_ms = _INTERVAL_MS.get(interval, 4 * 3_600_000)
        close_time_ms = bar.ts_event // 1_000_000   # ns → ms
        open_time_ms = close_time_ms - bar_duration_ms

        record = {
            "open_time": open_time_ms,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "taker_buy_volume": (
                float(taker_buy_volume) if taker_buy_volume is not None else None
            ),
        }
        if interval == "4h":
            self.bar_buffer_4h.append(record)
        elif interval == "1h":
            self.bar_buffer_1h.append(record)
        elif interval == "15m":
            self.bar_buffer_15m.append(record)

    def get_bar_df(self, interval: str) -> pl.DataFrame:
        """Get bar buffer as DataFrame sorted by time.

        Returns columns: ``open_time``, ``open``, ``high``, ``low``,
        ``close``, ``volume``, ``taker_buy_volume``.

        ``taker_buy_volume`` is filled with ``volume * 0.5`` for any record
        added without a real value — matches the historical fallback in
        ``FeaturePipeline._ensure_taker_buy_volume``. A WARNING is emitted
        once per state instance when this fallback triggers.
        """
        if interval == "4h":
            buf = self.bar_buffer_4h
        elif interval == "1h":
            buf = self.bar_buffer_1h
        else:
            buf = self.bar_buffer_15m

        if not buf:
            return pl.DataFrame()

        records = [dict(r) for r in buf]
        missing_tbv = sum(1 for r in records if r.get("taker_buy_volume") is None)
        if missing_tbv:
            if not self._tbv_fallback_warned:
                _log.warning(
                    "live_feature_state[%s]: %d/%d bars missing real "
                    "taker_buy_volume — falling back to volume*0.5. "
                    "CVD will be ≡ 0 until add_bar() is called with a "
                    "real taker_buy_volume (e.g. from Binance klines "
                    "taker_buy_base_asset_volume). Train/serve skew.",
                    interval, missing_tbv, len(records),
                )
                self._tbv_fallback_warned = True
            for r in records:
                if r.get("taker_buy_volume") is None:
                    r["taker_buy_volume"] = r["volume"] * 0.5

        return pl.DataFrame(records).sort("open_time")

    # ---------------------------------------------------------------
    # REST helper — fetch real taker_buy_volume from Binance klines.
    # Strategy may call this on bar-close and pass the result into
    # add_bar(taker_buy_volume=...). Kept here (not in the strategy)
    # so live and tests share one source of truth.
    # ---------------------------------------------------------------

    @staticmethod
    def fetch_taker_buy_volume(
        symbol: str,
        interval: str,
        open_time_ms: int,
        *,
        base_url: str = "https://fapi.binance.com",
        timeout: float = 5.0,
        session: Any | None = None,
    ) -> float | None:
        """Fetch ``taker_buy_base_asset_volume`` for one closed kline.

        Returns ``None`` on any failure (network, parse, empty response,
        timestamp mismatch). Caller decides whether to retry or fall back.

        ``session`` is an optional object exposing ``get(url, params, timeout)``
        — used by tests to inject a mock; defaults to ``requests``.
        """
        try:
            if session is None:
                import requests as session  # type: ignore[no-redef]
            resp = session.get(
                f"{base_url}/fapi/v1/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": int(open_time_ms),
                    "limit": 1,
                },
                timeout=timeout,
            )
            if getattr(resp, "status_code", 200) != 200:
                return None
            data = resp.json()
            if not data:
                return None
            row = data[0]
            # Binance kline schema: [open_time, o, h, l, c, vol, close_time,
            # quote_vol, n_trades, taker_buy_base_vol, taker_buy_quote_vol, _]
            if int(row[0]) != int(open_time_ms):
                return None
            return float(row[9])
        except Exception as exc:
            _log.debug("fetch_taker_buy_volume failed (non-critical): %s", exc)
            return None
