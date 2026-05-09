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

# Rows to drop at the head of the DataFrame to eliminate NaN from rolling ops.
# Must exceed the longest rolling window (180 for funding_zscore_30d).
_WARMUP_ROWS = 200

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

# Features added only for MTF timeframes (1h, 15m, 5m, 1m).
FEATURE_GROUPS_MTF: dict[str, list[str]] = {
    "session": [
        "trading_session", "session_hour_sin", "session_hour_cos",
        "is_overlap", "is_dead_zone", "hours_to_session_end",
        "session_vwap", "price_vs_session_vwap", "session_vwap_std",
        "vwap_upper_band", "vwap_lower_band", "vwap_band_position",
        "session_volume_pct",
        "avwap_weekly", "price_vs_avwap_weekly", "avwap_weekly_slope",
        "price_above_avwap",
        "hours_to_funding_mark", "pre_funding_window",
        "post_funding_window", "funding_window_urgency",
        "day_of_week_sin", "day_of_week_cos", "is_weekend",
        "is_monday_asia_window", "is_high_vol_day",
    ],
    "orb": [
        "orb_high_asia", "orb_low_asia", "orb_range_asia",
        "orb_range_asia_atr_pct",
        "orb_high_london", "orb_low_london", "orb_range_london",
        "orb_range_london_atr_pct",
        "orb_high_ny", "orb_low_ny", "orb_range_ny",
        "orb_range_ny_atr_pct",
        "current_session", "price_vs_current_orb",
        "dist_to_orb_high_pct", "dist_to_orb_low_pct",
        "orb_breakout_bull", "orb_breakout_bear",
        "bars_since_session_open", "is_session_trap_zone",
    ],
    "mtf_context": [
        "htf_4h_regime", "htf_4h_adx", "htf_4h_trend_dir",
        "htf_4h_hurst", "htf_4h_atr_pct",
        "price_vs_4h_ema20", "mtf_1h_4h_aligned",
        "mtf_alignment_score", "htf_4h_last_n_bars_dir",
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

    # ------------------------------------------------------------------
    # MTF Feature Building (Phase 2)
    # ------------------------------------------------------------------

    def build_mtf(
        self,
        df: pl.DataFrame,
        *,
        df_htf_4h: pl.DataFrame | None = None,
        df_htf_1h: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Add MTF-specific features to an already-loaded DataFrame.

        This method is used for building features for 1H, 15m, 5m, 1m
        intervals.  It does NOT touch DataStore — the caller provides
        the raw DataFrames.

        For ``interval='4h'`` this method is a no-op (returns df unchanged),
        preserving full backward compatibility.

        Parameters
        ----------
        df:
            OHLCV DataFrame with at least ``open_time``, ``open``, ``high``,
            ``low``, ``close``, ``volume`` columns.
        df_htf_4h:
            Optional 4H DataFrame (with regime columns) for MTF context.
        df_htf_1h:
            Optional 1H DataFrame for 15m MTF context.
        """
        if self.interval == "4h":
            return df

        # Session features: applicable to all MTF intervals.
        if self.interval in ("1h", "15m", "5m", "1m"):
            from src.features.session_features import (
                AnchoredVWAP,
                MondayAsiaEffect,
                PreFundingDetector,
                SessionEncoder,
                SessionVWAP,
            )
            df = SessionEncoder().encode(df)
            df = SessionVWAP().calculate(df)
            df = AnchoredVWAP().calculate(df)
            df = PreFundingDetector().detect(df)
            df = MondayAsiaEffect().detect(df)
            _log.info(f"Added session features for {self.interval}")

        # 1H: add 4H HTF context.
        if self.interval == "1h" and df_htf_4h is not None:
            from src.features.mtf_context import MTFContextBuilder
            df = MTFContextBuilder().build_for_1h(df, df_htf_4h)
            _log.info("Added 4H HTF context for 1H")

        # 15m: add ORB features + 1H/4H HTF context.
        if self.interval == "15m":
            from src.features.orb_features import ORBDetector
            df = ORBDetector().calculate(df)
            _log.info("Added ORB features for 15m")

            if df_htf_4h is not None and df_htf_1h is not None:
                from src.features.mtf_context import MTFContextBuilder
                df = MTFContextBuilder().build_for_15m(
                    df, df_htf_1h, df_htf_4h
                )
                _log.info("Added 1H+4H HTF context for 15m")

        return df

    def get_feature_names(self) -> list[str]:
        """Return all feature column names (excludes raw OHLCV and timestamps)."""
        names = [feat for group in FEATURE_GROUPS.values() for feat in group]
        if self.interval in ("1h", "15m", "5m", "1m"):
            names += FEATURE_GROUPS_MTF.get("session", [])
        if self.interval == "15m":
            names += FEATURE_GROUPS_MTF.get("orb", [])
        if self.interval in ("1h", "15m"):
            names += FEATURE_GROUPS_MTF.get("mtf_context", [])
        return names
