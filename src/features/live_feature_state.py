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

import polars as pl


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
    funding_rate_history: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )  # last 100 funding marks

    oi_value: float = 0.0
    oi_history: deque = field(
        default_factory=lambda: deque(maxlen=100)
    )

    # Timestamps (unix ms)
    last_funding_update: int = 0
    last_oi_update: int = 0

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

    def add_bar(self, bar, interval: str) -> None:
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
        ``close``, ``volume``.
        """
        if interval == "4h":
            buf = self.bar_buffer_4h
        elif interval == "1h":
            buf = self.bar_buffer_1h
        else:
            buf = self.bar_buffer_15m

        if not buf:
            return pl.DataFrame()
        return pl.DataFrame(list(buf)).sort("open_time")
