"""
src/features/feature_pipeline.py

Main feature engineering pipeline: loads OHLCV + derivatives data from
DataStore and produces a single Parquet-ready feature DataFrame.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from src.features.derivatives import (
    add_basis_features,
    add_funding_features,
    add_oi_features,
)
from src.features.microstructure import (
    add_cvd_features,
    add_price_features,
    add_volume_features,
)
from src.features.regime_detector import RegimeDetector
from src.logger import get_logger

if TYPE_CHECKING:
    from src.ingestion.data_store import DataStore

_log = get_logger(__name__)

# Rows to drop at the head of the DataFrame to eliminate NaN from rolling ops
_WARMUP_ROWS = 100

FEATURE_GROUPS: dict[str, list[str]] = {
    "microstructure": [
        "cvd",
        "cvd_cum",
        "cvd_slope_3",
        "cvd_slope_6",
        "cvd_slope_12",
        "taker_buy_ratio",
        "volume_sma_20",
        "volume_ratio",
        "volume_zscore",
        "large_volume",
        "vwap_4h",
        "price_to_vwap",
        "returns_1",
        "returns_3",
        "returns_6",
        "returns_12",
        "returns_24",
        "body_ratio",
        "upper_wick",
        "lower_wick",
        "gap",
    ],
    "derivatives": [
        "funding_rate",
        "funding_abs",
        "funding_zscore_7d",
        "funding_zscore_30d",
        "funding_extreme",
        "funding_positive",
        "funding_cum_24h",
        "oi_value",
        "oi_delta_4h",
        "oi_delta_12h",
        "oi_zscore",
        "oi_quadrant",
        "ls_ratio",
        "ls_ratio_zscore",
        "taker_vol_ratio",
        "basis_approx",
        "basis_extreme",
    ],
    "regime": [
        "hurst",
        "adx",
        "atr_pct",
        "atr_percentile",
        "trend_strength",
        "regime_confidence",
    ],
}


class FeaturePipeline:
    """Build a feature matrix for one symbol / interval from the DataStore.

    Parameters
    ----------
    data_store:
        Loaded DataStore instance pointing at the Parquet feature tree.
    symbol:
        Binance symbol, e.g. ``"BTCUSDT"``.
    interval:
        Kline interval, default ``"4h"``.
    """

    def __init__(
        self,
        data_store: "DataStore",
        symbol: str,
        interval: str = "4h",
    ) -> None:
        self.data_store = data_store
        self.symbol = symbol
        self.interval = interval

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        start: datetime,
        end: datetime,
        save_to: Path | None = None,
    ) -> pl.DataFrame:
        """Build all features for *symbol* over [*start*, *end*].

        Steps
        -----
        1-3.  Load klines, funding_rate, metrics from DataStore.
        4-6.  Add CVD, volume, price features (microstructure).
        7-9.  Add funding, OI, basis features (derivatives).
        10.   Detect market regime (Hurst, ADX, ATR).
        11.   Drop the first ``_WARMUP_ROWS`` rows (NaN from rolling).
        12.   Warn if any NaN remains in feature columns.
        13.   Optionally save to Parquet (ZSTD compression).

        Returns the feature DataFrame.
        """
        _log.info(
            f"FeaturePipeline.build: {self.symbol} {self.interval} "
            f"{start.date()} → {end.date()}"
        )

        # 1. Klines
        df = self.data_store.get_klines(self.symbol, self.interval, start, end)
        if df.is_empty():
            _log.error("No klines data available — aborting build")
            return pl.DataFrame()

        # 2. Funding rate (graceful on DataStore error / missing data)
        try:
            funding_df = self.data_store.get_funding_rate(self.symbol, start, end)
        except Exception as exc:
            _log.warning(f"get_funding_rate failed ({exc}); derivative features will be zero")
            funding_df = pl.DataFrame()

        # 3. Metrics (graceful on DataStore error / missing data)
        try:
            metrics_df = self.data_store.get_metrics(self.symbol, start, end)
        except Exception as exc:
            _log.warning(f"get_metrics failed ({exc}); OI features will be zero")
            metrics_df = pl.DataFrame()

        # 4-6. Microstructure
        df = add_cvd_features(df)
        df = add_volume_features(df)
        df = add_price_features(df)

        # 7-9. Derivatives
        df = add_funding_features(df, funding_df)
        df = add_oi_features(df, metrics_df)
        df = add_basis_features(df)

        # 10. Regime detection (Hurst, ADX, ATR)
        detector = RegimeDetector()
        df = detector.detect_all(df)

        # 11. Drop warmup rows
        df = df.slice(_WARMUP_ROWS)
        _log.info(f"After warmup trim: {len(df)} rows")

        # 12. NaN audit
        feature_cols = self.get_feature_names()
        present = [c for c in feature_cols if c in df.columns]
        nan_counts = {
            c: df[c].null_count() + (df[c].is_nan().sum() if df[c].dtype.is_float() else 0)
            for c in present
        }
        bad = {c: n for c, n in nan_counts.items() if n > 0}
        if bad:
            _log.warning(f"NaN/null detected after build: {bad}")
        else:
            _log.info("NaN audit: clean")

        # 13. Persist
        if save_to is not None:
            save_to = Path(save_to)
            save_to.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(save_to, compression="zstd", compression_level=3)
            _log.info(f"Saved features to {save_to} ({save_to.stat().st_size / 1024:.1f} KB)")

        return df

    def get_feature_names(self) -> list[str]:
        """Return all feature column names (excludes raw OHLCV and timestamps)."""
        return [feat for group in FEATURE_GROUPS.values() for feat in group]
