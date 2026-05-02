"""
src/models/dataset_builder.py

Data preparation utilities for LightGBM training:
  - Load and combine multi-symbol feature Parquets
  - Create target variables (UP / FLAT / DOWN) with ATR-based threshold
  - Extract ML-ready feature columns (exclude leaky / non-numeric columns)

Phase 3 — Step 3.4.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from src.logger import get_logger

_log = get_logger(__name__)

# Columns that must NOT be used as ML features
_EXCLUDE_COLUMNS: set[str] = {
    # Timestamps
    "datetime",
    "open_time",
    "close_time",
    # Raw OHLCV (use engineered features instead)
    "open",
    "high",
    "low",
    "close",
    "volume",
    # Non-numeric / metadata
    "regime",
    "symbol",
    "date",
    "exchange",
    "ignore",
    # Intermediate join columns
    "_f_time",
    "_m_time",
    # Leak columns (target-derived)
    "future_return",
    "target",
    # Raw derivatives that are replaced by engineered features
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
}


class DatasetBuilder:
    """Prepare multi-symbol feature data for LightGBM training.

    Parameters
    ----------
    data_dir:
        Root Parquet data directory (used by FeaturePipeline via DataStore).
    symbols:
        List of Binance symbols, e.g. ``["BTCUSDT", "ETHUSDT", "SOLUSDT"]``.
    interval:
        Kline interval, default ``"4h"``.
    """

    def __init__(
        self,
        data_dir: Path,
        symbols: list[str],
        interval: str = "4h",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.symbols = symbols
        self.interval = interval

    # ------------------------------------------------------------------
    # Build features for all symbols
    # ------------------------------------------------------------------

    def build_features_all_symbols(
        self,
        start: datetime,
        end: datetime,
        output_dir: Path,
    ) -> None:
        """Build feature parquets for every symbol via FeaturePipeline.

        Saves one file per symbol:
            output_dir/{SYMBOL}_{interval}_features.parquet
        """
        from src.ingestion.data_store import DataStore
        from src.features.feature_pipeline import FeaturePipeline

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with DataStore(self.data_dir) as store:
            for symbol in self.symbols:
                out_path = output_dir / f"{symbol}_{self.interval}_features.parquet"
                _log.info(f"Building features for {symbol} → {out_path}")
                pipeline = FeaturePipeline(store, symbol, self.interval)
                df = pipeline.build(start, end, save_to=out_path)
                if df.is_empty():
                    _log.warning(f"No data produced for {symbol}")
                else:
                    _log.info(f"{symbol}: {len(df)} rows saved")

    # ------------------------------------------------------------------
    # Load and combine
    # ------------------------------------------------------------------

    def load_and_combine(
        self,
        features_dir: Path,
        symbols: list[str] | None = None,
    ) -> pl.DataFrame:
        """Load feature parquets and concatenate (preserving time order).

        1. Read each ``{SYMBOL}_{interval}_features.parquet`` file.
        2. Add a ``symbol`` column (categorical string).
        3. Concat all DataFrames **without** shuffling — temporal order
           is critical for walk-forward validation.

        Returns the combined DataFrame.
        """
        symbols = symbols or self.symbols
        features_dir = Path(features_dir)
        frames: list[pl.DataFrame] = []

        for symbol in symbols:
            path = features_dir / f"{symbol}_{self.interval}_features.parquet"
            if not path.exists():
                _log.warning(f"Feature file not found: {path}")
                continue

            df = pl.read_parquet(path)

            # Ensure symbol column is present and correct
            if "symbol" in df.columns:
                df = df.drop("symbol")
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

            _log.info(f"Loaded {symbol}: {len(df)} rows, {len(df.columns)} cols")
            frames.append(df)

        if not frames:
            _log.error("No feature files loaded!")
            return pl.DataFrame()

        combined = pl.concat(frames, how="diagonal")
        _log.info(f"Combined dataset: {len(combined)} rows, {len(combined.columns)} cols")
        return combined

    # ------------------------------------------------------------------
    # Target creation
    # ------------------------------------------------------------------

    def create_target(
        self,
        df: pl.DataFrame,
        forward_bars: int = 1,
        threshold_atr_multiplier: float = 0.5,
    ) -> pl.DataFrame:
        """Create classification target: UP(1) / FLAT(0) / DOWN(-1).

        Logic
        -----
        * ``future_return = (close[t+forward_bars] - close[t]) / close[t]``
        * ``atr_threshold = atr_pct × threshold_atr_multiplier``
        * ``target = 1``  if future_return > atr_threshold
        * ``target = -1`` if future_return < -atr_threshold
        * ``target = 0``  otherwise (FLAT)

        Rows where target cannot be computed (last *forward_bars*) are dropped.
        """
        if "close" not in df.columns or "atr_pct" not in df.columns:
            raise ValueError("DataFrame must contain 'close' and 'atr_pct' columns")

        # Future return
        future_close = df["close"].shift(-forward_bars)
        future_return = (future_close - df["close"]) / df["close"]

        # ATR-based threshold
        atr_threshold = df["atr_pct"] * threshold_atr_multiplier

        # Target: 1=UP, -1=DOWN, 0=FLAT
        target = (
            pl.when(future_return > atr_threshold)
            .then(pl.lit(1))
            .when(future_return < -atr_threshold)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
        )

        df = df.with_columns([
            future_return.alias("future_return"),
            target.alias("target"),
        ])

        # Drop rows where target is undefined (last forward_bars rows)
        df = df.head(len(df) - forward_bars)

        _log.info(
            f"Target created: {len(df)} rows, "
            f"UP={df['target'].eq(1).sum()}, "
            f"FLAT={df['target'].eq(0).sum()}, "
            f"DOWN={df['target'].eq(-1).sum()}"
        )
        return df

    # ------------------------------------------------------------------
    # Feature column selection
    # ------------------------------------------------------------------

    def get_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Return feature column names suitable for ML.

        Excludes timestamps, raw prices, regime string, target/leak columns,
        and non-numeric columns.
        """
        feature_cols: list[str] = []
        for col in df.columns:
            if col in _EXCLUDE_COLUMNS:
                continue
            dtype = df[col].dtype
            # Only numeric columns
            if dtype.is_float() or dtype.is_integer():
                feature_cols.append(col)

        return sorted(feature_cols)
