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

import numpy as np
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
    "label",  # triple-barrier intermediate (renamed → target)
    "t1_bar",   # AFML Ch.4 exit-bar index (used for uniqueness weights)
    "_bar_idx", # original per-symbol bar position (used for uniqueness weights)
    # Raw derivatives that are replaced by engineered features
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
}

# Features excluded from training due to zero importance on BTCUSDT-only
# datasets.  Kept in the feature pipeline so they remain available if
# multi-symbol training is added later.
_TRAINING_EXCLUDE: set[str] = {
    "mtf_alignment_score",   # zero importance in both 1H models
    "symbol_encoded",        # only one symbol = no information
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
        threshold_atr_multiplier: float = 0.5,  # kept for API compat; unused
    ) -> pl.DataFrame:
        """Create binary classification target: UP(+1) / DOWN(-1).

        ML-017: dropped the FLAT(0) class to fix severe class imbalance
        (FLAT was ~62-65% of bars, causing the multiclass model to predict
        FLAT almost always with confidence < 0.35 on directional signals).

        * ``future_return = (close[t+forward_bars] - close[t]) / close[t]``
        * ``target = +1`` if future_return > 0
        * ``target = -1`` otherwise (return <= 0)
        """
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain a 'close' column")

        future_close = df["close"].shift(-forward_bars)
        future_return = (future_close - df["close"]) / df["close"]

        target = (
            pl.when(future_return > 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(-1))
        )

        df = df.with_columns([
            future_return.alias("future_return"),
            target.alias("target"),
        ])

        # Drop rows where future_return is undefined (last forward_bars rows)
        df = df.head(len(df) - forward_bars)

        _log.info(
            f"Target created (binary): {len(df)} rows, "
            f"UP={df['target'].eq(1).sum()}, "
            f"DOWN={df['target'].eq(-1).sum()}"
        )
        return df

    # ------------------------------------------------------------------
    # Triple-barrier target (v3, López de Prado AFML Ch.3)
    # ------------------------------------------------------------------

    def create_target_triple_barrier(
        self,
        df: pl.DataFrame,
        pt_multiplier: float = 1.0,
        sl_multiplier: float = 1.0,
        max_holding: int = 6,
        atr_col: str = "atr_pct",
        drop_timeout: bool = True,
    ) -> pl.DataFrame:
        """Vol-scaled symmetric triple-barrier labels (v3).

        Thin wrapper around :func:`src.features.triple_barrier.apply_triple_barrier`
        that renames ``label`` → ``target`` (LGBMTrainer contract) and
        optionally drops timeout rows (``target == 0``) so the binary
        booster only sees decisive outcomes.

        Barriers are ATR-scaled (atr_pct = ATR/close, dimensionless):
            upper = close × (1 + pt_multiplier × atr_pct)
            lower = close × (1 - sl_multiplier × atr_pct)
            vertical = t + max_holding bars

        ``future_return`` is the realized close-to-close return at the
        bar the trade actually exits (real path, not the barrier constant).
        """
        from src.features.triple_barrier import apply_triple_barrier

        if atr_col not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{atr_col}'; available cols sample: "
                f"{df.columns[:10]}"
            )

        labeled = apply_triple_barrier(
            df,
            close_col="close",
            atr_col=atr_col,
            pt_multiplier=pt_multiplier,
            sl_multiplier=sl_multiplier,
            max_holding_bars=max_holding,
        )

        # Stamp each row with its per-symbol bar index BEFORE filtering, so
        # ``t1_bar`` (also a per-symbol bar index) remains comparable after
        # timeout rows are dropped. Sample-uniqueness concurrency is
        # computed in this original bar-index coordinate system.
        labeled = labeled.with_columns(
            pl.Series("_bar_idx", np.arange(len(labeled), dtype=np.int64))
        )

        # Rename label → target; keep future_return as-is (already emitted).
        labeled = labeled.rename({"label": "target"}).with_columns(
            pl.col("target").cast(pl.Int64)
        )

        n_total = len(labeled)
        if drop_timeout:
            labeled = labeled.filter(pl.col("target") != 0)

        n_pos = int((labeled["target"] == 1).sum())
        n_neg = int((labeled["target"] == -1).sum())
        n_kept = len(labeled)
        pos_pct = 100.0 * n_pos / n_kept if n_kept else 0.0
        _log.info(
            f"Triple-barrier target (pt={pt_multiplier}, sl={sl_multiplier}, "
            f"h={max_holding}): kept {n_kept}/{n_total} rows | "
            f"UP={n_pos} ({pos_pct:.1f}%), DOWN={n_neg} ({100-pos_pct:.1f}%)"
        )
        return labeled

    # ------------------------------------------------------------------
    # Sample uniqueness weights (López de Prado AFML Ch.4)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_uniqueness_weights(
        n_samples: int,
        max_holding: int,
        t1_bars: np.ndarray | None = None,
        bar_idxs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample uniqueness weights for overlapping triple-barrier labels.

        Two regimes, switched by whether ``t1_bars`` is supplied:

        * **Real exit-bar mode (preferred — AFML Ch.4)**: pass ``t1_bars``
          from ``apply_triple_barrier`` so each label's span is its
          ACTUAL exit bar (PT/SL early exits cut the span short). This
          gives correct concurrency; the legacy mode systematically
          over-counts concurrent labels for early-exit trades.
        * **Legacy fixed-span fallback**: when ``t1_bars`` is None each
          label is assumed to hold the full ``max_holding`` window
          ``[i, i + max_holding - 1]``. Kept for backward compat.

        For each bar t, c_t = number of active labels at t. The
        uniqueness of label i is::

            u_i = mean( 1 / c_t  for t in [bar_idx_i, t1_bar_i] )

        Weights are normalized so the mean weight = 1 — this preserves
        LightGBM's effective sample count and keeps the balanced-class
        weight multiplier interpretable.

        Parameters
        ----------
        n_samples:
            Number of labels in the (per-symbol) DataFrame.
        max_holding:
            Vertical-barrier horizon in bars; used by the legacy fallback.
        t1_bars:
            Per-label exit-bar indices (in input-df coordinates). Same
            length as ``bar_idxs`` and corresponds to the surviving labels.
        bar_idxs:
            Per-label entry-bar indices (in input-df coordinates). When
            ``t1_bars`` is given but ``bar_idxs`` is None, defaults to
            ``arange(n_samples)`` (no rows dropped between labeling and
            here).

        Returns
        -------
        np.ndarray of shape (n_samples,) — non-negative weights, mean ≈ 1.
        """
        if n_samples == 0:
            return np.zeros(0, dtype=np.float64)
        if max_holding < 1:
            raise ValueError("max_holding must be >= 1")

        if t1_bars is None:
            # ---- Legacy fixed-span mode ---------------------------------
            h = max_holding
            t = np.arange(n_samples, dtype=np.float64)
            concur = np.minimum(t + 1.0, float(h))
            concur = np.maximum(concur, 1.0)
            inv_c = 1.0 / concur

            cum = np.concatenate(([0.0], np.cumsum(inv_c)))
            end = np.minimum(np.arange(n_samples) + h, n_samples)
            span = end - np.arange(n_samples)
            u = (cum[end] - cum[np.arange(n_samples)]) / span
        else:
            # ---- Real exit-bar mode -------------------------------------
            t1 = np.asarray(t1_bars, dtype=np.int64)
            if t1.shape[0] != n_samples:
                raise ValueError(
                    f"t1_bars length {t1.shape[0]} != n_samples {n_samples}"
                )
            if bar_idxs is None:
                idx = np.arange(n_samples, dtype=np.int64)
            else:
                idx = np.asarray(bar_idxs, dtype=np.int64)
                if idx.shape[0] != n_samples:
                    raise ValueError(
                        f"bar_idxs length {idx.shape[0]} != n_samples {n_samples}"
                    )

            # Work in original-bar coordinates. Difference-array trick:
            # add +1 at the entry bar, -1 just past the exit bar, cumsum
            # to get concurrency at every bar in [0, T_max].
            t_max = int(t1.max()) + 1
            diff = np.zeros(t_max + 1, dtype=np.int64)
            np.add.at(diff, idx, 1)
            np.add.at(diff, np.minimum(t1 + 1, t_max), -1)
            concur = np.cumsum(diff)[:t_max].astype(np.float64)
            concur = np.maximum(concur, 1.0)
            inv_c = 1.0 / concur

            # u_i = mean of inv_c over [idx[i], t1[i]] via cumulative sum.
            cum = np.concatenate(([0.0], np.cumsum(inv_c)))
            end = t1 + 1
            span = np.maximum(end - idx, 1)
            u = (cum[end] - cum[idx]) / span

        # Normalize: mean weight = 1 so balanced class weights still mean
        # what they say when multiplied in.
        u_mean = u.mean()
        if u_mean > 0:
            u = u / u_mean
        return u.astype(np.float64)

    def compute_uniqueness_weights_by_symbol(
        self,
        df: pl.DataFrame,
        max_holding: int,
        symbol_col: str = "symbol",
        t1_col: str = "t1_bar",
        bar_idx_col: str = "_bar_idx",
    ) -> np.ndarray:
        """Per-symbol uniqueness weights aligned with df row order.

        Concurrent labels only overlap within a single symbol's time
        series, so weights are computed per-symbol then concatenated in
        the original row order. When ``t1_col`` and ``bar_idx_col`` are
        both present (added by ``create_target_triple_barrier``), the
        real exit-bar mode is used — labels that exit early via PT/SL
        contribute less concurrency, fixing AFML Ch.4 under-counting.
        Otherwise falls back to the legacy fixed-span mode.
        """
        has_real_t1 = t1_col in df.columns and bar_idx_col in df.columns

        if symbol_col not in df.columns:
            if has_real_t1:
                return self.compute_uniqueness_weights(
                    len(df), max_holding,
                    t1_bars=df[t1_col].to_numpy(),
                    bar_idxs=df[bar_idx_col].to_numpy(),
                )
            return self.compute_uniqueness_weights(len(df), max_holding)

        weights = np.zeros(len(df), dtype=np.float64)
        idx = np.arange(len(df))
        symbols = df[symbol_col].to_numpy()
        t1_all = df[t1_col].to_numpy() if has_real_t1 else None
        bidx_all = df[bar_idx_col].to_numpy() if has_real_t1 else None
        for sym in np.unique(symbols):
            mask = symbols == sym
            n_sym = int(mask.sum())
            if has_real_t1:
                w = self.compute_uniqueness_weights(
                    n_sym, max_holding,
                    t1_bars=t1_all[mask],
                    bar_idxs=bidx_all[mask],
                )
            else:
                w = self.compute_uniqueness_weights(n_sym, max_holding)
            weights[idx[mask]] = w
        return weights

    # ------------------------------------------------------------------
    # Feature column selection
    # ------------------------------------------------------------------

    def get_feature_columns(self, df: pl.DataFrame) -> list[str]:
        """Return feature column names suitable for ML.

        Excludes timestamps, raw prices, regime string, target/leak columns,
        non-numeric columns, and zero-importance training exclusions.
        """
        exclude = _EXCLUDE_COLUMNS | _TRAINING_EXCLUDE
        feature_cols: list[str] = []
        for col in df.columns:
            if col in exclude:
                continue
            dtype = df[col].dtype
            # Only numeric columns
            if dtype.is_float() or dtype.is_integer():
                feature_cols.append(col)

        return sorted(feature_cols)
