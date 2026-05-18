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
    add_cvd_derived_features,
    add_funding_features,
    add_funding_momentum_features,
    add_oi_derived_features,
    add_oi_features,
)
from src.features.microstructure import (
    add_candle_structure_features,
    add_cvd_features,
    add_ema_slope_features,
    add_fractal_features,
    add_price_features,
    add_volume_features,
    add_volume_session_features,
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
    # Scalar features computed at signal-emit time from external sources
    # (Binance allForceOrders, aggTrades, alternative.me F&G). Not part of
    # the vectorized polars pipeline — populated by SignalBridge / live
    # enrichers. Listed here so downstream consumers (model trainers,
    # NaN audits) can opt in once data is being persisted.
    "live_enrichment": [
        "liq_cluster_long_pct",
        "liq_cluster_short_pct",
        "liq_imbalance",
        "liq_cascade_risk",
        "liq_volume_1h",
        "vpin",
        "basis_bps",
        "basis_annualized",
        "basis_zscore_30d",
        "oi_velocity",
        "oi_acceleration",
        "oi_price_divergence",
        "oi_exhaustion",
        "oi_velocity_zscore",
        "fear_greed_norm",
        "fear_greed_extreme",
        "ls_divergence",
        "sentiment_score",
    ],
}

# Features added only for MTF timeframes (1h, 15m, 5m, 1m).
FEATURE_GROUPS_MTF: dict[str, list[str]] = {
    "session": [
        "trading_session", "session_hour_sin", "session_hour_cos",
        "is_overlap", "is_dead_zone", "hours_to_session_end",
        "session_vwap", "price_vs_session_vwap", "session_vwap_std",
        "vwap_upper_band", "vwap_lower_band", "vwap_band_position",
        "session_cumulative_volume_ratio",
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
    # Research-backed derived features (1H / 15m only, post-lookahead
    # expansion). Vectorized OI features use _vec suffix to avoid
    # colliding with the scalar live-enrichment names.
    "alpha_v2": [
        # derivatives — funding momentum
        "funding_rate_change_1", "funding_rate_change_3",
        "funding_rate_zscore_rolling", "funding_rate_acceleration",
        # derivatives — OI derived
        "oi_delta_1h", "oi_accel_vec", "oi_price_div_vec",
        # derivatives — CVD derived
        "cvd_slope_3bar", "cvd_slope_6bar", "cvd_divergence",
        # microstructure — EMA slopes
        "ema9", "ema21", "ema9_slope_normalized",
        "ema21_slope_normalized", "ema9_cross_ema21",
        "ema9_cross_ema21_change",
        # microstructure — volume vs session
        "volume_vs_session_avg", "volume_momentum_3bar",
        # microstructure — fractal efficiency
        "efficiency_ratio_10", "efficiency_ratio_20",
        # microstructure — candle structure
        "candle_range", "candle_body", "candle_body_pct",
        "upper_wick_pct", "lower_wick_pct", "candle_direction",
        # session momentum + VWAP slope
        "session_open_return", "session_momentum_3bar",
        "session_return_cumulative", "vwap_slope_3bar",
        "vwap_slope_6bar", "price_above_vwap",
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

        # Alpha-v2 derived features: 1H / 15m only (research-backed,
        # post-lookahead expansion). Base columns (funding_rate,
        # oi_value, cvd, atr_pct, returns_1) are produced upstream by
        # the build scripts before build_mtf().
        if self.interval in ("1h", "15m"):
            from src.features.session_features import SessionMomentum

            df = add_funding_momentum_features(df)
            df = add_oi_derived_features(df)
            df = add_cvd_derived_features(df)
            df = add_ema_slope_features(df)
            df = add_volume_session_features(df)
            df = add_fractal_features(df)
            df = add_candle_structure_features(df)
            df = SessionMomentum().calculate(df)
            _log.info(f"Added alpha-v2 features for {self.interval}")

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

    @staticmethod
    def _ensure_taker_buy_volume(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure a ``taker_buy_volume`` column exists (mirrors build scripts).

        Uses ``taker_buy_base_vol`` when present, otherwise a 50% proxy —
        identical fallback to ``build_15m_dataset._ensure_taker_buy_volume``
        so live inference matches training.
        """
        if "taker_buy_volume" in df.columns:
            return df
        if "taker_buy_base_vol" in df.columns:
            return df.rename({"taker_buy_base_vol": "taker_buy_volume"})
        return df.with_columns((pl.col("volume") * 0.5).alias("taker_buy_volume"))

    def build_from_buffer(
        self,
        df: pl.DataFrame,
        *,
        df_htf_4h: pl.DataFrame | None = None,
        df_htf_1h: pl.DataFrame | None = None,
        funding_df: pl.DataFrame | None = None,
        metrics_df: pl.DataFrame | None = None,
        single_row: bool = True,
    ) -> pl.DataFrame:
        """Build features from in-memory OHLCV buffers (live inference).

        Replicates the offline ``build_feature_matrix()`` sequence used by
        ``scripts/build_{15m,1h}_dataset.py`` **exactly** — same transforms,
        same order, same regime detector per interval — but:

        * reads nothing from DataStore (caller supplies every frame);
        * does NOT drop the head warmup rows (offline slices the first
          ``_WARMUP_ROWS`` to remove rolling-NaN; live we keep the buffer
          and instead return only the final, fully-warmed row);
        * leaves the existing :meth:`build` / :meth:`build_mtf` untouched
          (pure additive, backward compatible).

        The caller is responsible for passing a buffer long enough that
        the last row's rolling features are valid (>= ``_WARMUP_ROWS`` +
        the longest lookback; the 15m/1h strategies use ``warmup_bars``).

        Parameters
        ----------
        df:
            Primary-interval OHLCV buffer (oldest→newest). ``self.interval``
            selects the regime detector and MTF chain ('15m' or '1h').
        df_htf_4h:
            4H OHLCV buffer — required for ``htf_4h_*`` / ``mtf`` features
            on both 1h and 15m. When ``None`` those columns stay at their
            ``build_mtf`` defaults (train/serve skew — caller must supply).
        df_htf_1h:
            1H OHLCV buffer — required for 15m's ``htf_1h_*`` features.
        funding_df, metrics_df:
            Optional derivative frames; ``add_*`` zero-fill when empty
            (fail-soft, same as offline).
        single_row:
            When True (default) return only the last row (the current
            bar's feature vector). When False return the full enriched
            frame (used by tests to compare against the offline build).
        """
        from src.features.regime_detector import (
            RegimeDetector1H,
            RegimeDetector15M,
        )

        if self.interval not in ("15m", "1h"):
            raise ValueError(
                f"build_from_buffer supports '15m'/'1h', got '{self.interval}'"
            )

        empty = pl.DataFrame()

        # 1-2. taker_buy_volume + microstructure (CVD, volume, price).
        df = self._ensure_taker_buy_volume(df)
        df = add_cvd_features(df)
        df = add_volume_features(df)
        df = add_price_features(df)

        # 2b. Derivatives — base columns for MTF momentum features.
        df = add_funding_features(df, funding_df if funding_df is not None else empty)
        df = add_oi_features(df, metrics_df if metrics_df is not None else empty)

        # 3. Regime detection (interval-tuned, same call as build scripts).
        if self.interval == "15m":
            detector: RegimeDetector = RegimeDetector15M()
        else:
            detector = RegimeDetector1H()
        df = detector.detect_all(df, min_bars=detector.hurst_window)

        # 4a. 1H HTF prep (15m only) — microstructure + regime + session.
        htf_1h = None
        if df_htf_1h is not None and not df_htf_1h.is_empty():
            h1 = self._ensure_taker_buy_volume(df_htf_1h)
            h1 = add_cvd_features(h1)
            h1 = add_volume_features(h1)
            h1 = add_price_features(h1)
            det_1h = RegimeDetector1H()
            h1 = det_1h.detect_all(h1, min_bars=det_1h.hurst_window)
            from src.features.session_features import SessionEncoder, SessionVWAP
            h1 = SessionEncoder().encode(h1)
            h1 = SessionVWAP().calculate(h1)
            htf_1h = h1

        # 4b. 4H HTF prep — regime detection only (same as build scripts).
        htf_4h = None
        if df_htf_4h is not None and not df_htf_4h.is_empty():
            h4 = self._ensure_taker_buy_volume(df_htf_4h)
            h4 = RegimeDetector().detect_all(h4)
            htf_4h = h4

        # 5. MTF features (session + ORB + 1H/4H HTF context).
        df = self.build_mtf(df, df_htf_4h=htf_4h, df_htf_1h=htf_1h)

        # 6. No warmup slice — return the final warmed row for inference.
        if single_row:
            return df.tail(1)
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
            names += FEATURE_GROUPS_MTF.get("alpha_v2", [])
        return names
