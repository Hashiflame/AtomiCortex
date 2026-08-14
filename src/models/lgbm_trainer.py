"""
src/models/lgbm_trainer.py

LightGBM binary trainer for regime-specific directional prediction.

ML-017: switched from 3-class (UP/FLAT/DOWN) to binary (UP/DOWN) to fix
class imbalance — FLAT dominated ~62-65% of bars and the multiclass model
collapsed to predicting FLAT almost always.

Classes
-------
ModelConfig   — Training hyperparameters + regime filter.
EvaluationResult — Model quality & trading-relevant metrics.
LGBMTrainer   — End-to-end: data preparation → train → evaluate → signal.

Phase 3 — Step 3.4.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from src.logger import get_logger
from src.models.dataset_builder import DatasetBuilder
from src.models.temporal_split import (
    compute_default_oos_start_ms,
    temporal_split_multi,
)

_log = get_logger(__name__)

# PR-G: version of the ``manifest`` dict embedded in every model bundle.
# Bump whenever a key is renamed or removed (adding keys is backwards
# compatible for readers that use .get()).
MANIFEST_SCHEMA_VERSION: int = 1

# PR-K: below this many OOS rows a symbol's per-symbol WR/PF is noise.
# A warning, never a skip — dropping the symbol silently would hide the
# very thing the operator needs to see.
MIN_TEST_ROWS_WARN: int = 50

# PR-K: bar duration per ModelConfig.interval, used to express the
# train→val embargo in wall-clock time. An interval that is not listed
# falls back to the legacy row-count embargo (with a warning) rather than
# guessing a duration.
_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 3_600_000,
    "2h": 2 * 3_600_000,
    "4h": 4 * 3_600_000,
    "6h": 6 * 3_600_000,
    "8h": 8 * 3_600_000,
    "12h": 12 * 3_600_000,
    "1d": 24 * 3_600_000,
}

# Repo root — used only to resolve the git SHA stamped into the manifest.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Paths already warned about for a missing manifest. Strategies reload
# bundles (MetaSignalGate is constructed per strategy instance), and one
# WARNING per load would drown the log.
_MANIFEST_WARNED_PATHS: set[str] = set()


class ModelRejectedError(Exception):
    """PR-H: ``save_bundle`` refused to put this artifact on disk.

    Its own class, not a bare ``Exception``, so a caller can tell "the
    model is bad" apart from "the disk is unwritable" — the two need
    opposite responses.

    Attributes
    ----------
    reason:
        ``"thresholds"`` — failed the go-live gate;
        ``"no_feature_columns"`` — the bundle would be unusable for
        inference. The latter is not waivable by ``allow_failing``.
    path:
        The file that was NOT written.
    regime:
        Regime of the rejected model (callers loop over regimes).
    result:
        The evaluation behind the verdict, or None when none was given.
    """

    def __init__(
        self,
        reason: str,
        path: Path,
        regime: str,
        result: EvaluationResult | None = None,
        detail: str = "",
    ) -> None:
        self.reason = reason
        self.path = Path(path)
        self.regime = regime
        self.result = result

        if detail:
            body = detail
        elif result is not None:
            # Thresholds are quoted here for the operator's benefit only —
            # the verdict itself comes solely from
            # EvaluationResult.passes_minimum_thresholds().
            body = (
                f"failed go-live thresholds: "
                f"WR={result.win_rate} (need >= 52.0), "
                f"PF={result.profit_factor} (need >= 1.3), "
                f"sig={result.signal_rate} (need >= 0.10)"
            )
        else:
            body = "no EvaluationResult was supplied — nothing to verify"

        super().__init__(
            f"[{regime}] model rejected ({reason}): {body}; "
            f"not written: {self.path}"
        )


# Symbol → integer encoding (deterministic)
SYMBOL_ENCODING: dict[str, int] = {
    "BTCUSDT": 0,
    "ETHUSDT": 1,
    "SOLUSDT": 2,
}

# Label mapping: original target → LightGBM class (binary, ML-017)
# -1 (DOWN) → 0,  +1 (UP) → 1
LABEL_TO_CLASS: dict[int, int] = {-1: 0, 1: 1}
CLASS_TO_LABEL: dict[int, int] = {v: k for k, v in LABEL_TO_CLASS.items()}


# Stricter regularization profile for MTF (1H / 15m) models.
# These models overfit with the default params (train WR ~67% vs OOS ~50%
# after the lookahead fix), so constrain tree complexity, add L1/L2, slow
# the learning rate and lean on early stopping. 4H keeps ModelConfig
# defaults — do NOT route 4H through this profile.
MTF_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    # Tree complexity ceiling
    "num_leaves": 25,
    "max_depth": 5,
    # Feature subsampling — strongest regularizer vs noisy financial
    # features (75% of features considered per tree).
    "feature_fraction": 0.75,
    "feature_fraction_seed": 42,
    # Data subsampling (bagging) — 80% of rows per tree, every iteration.
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "bagging_seed": 42,
    # Regularization
    "lambda_l1": 0.05,
    "lambda_l2": 0.05,
    "min_gain_to_split": 0.01,
    # More trees + slower LR — early stopping (100 rounds) decides the
    # actual stopping point, so n_estimators is just an upper bound.
    "n_estimators": 2000,      # was 500 — early stopping caps it
    "learning_rate": 0.02,     # was 0.03 — slightly slower
    "early_stopping_rounds": 100,  # patient — financial data is noisy
    # Default leaf size (1H ≈ 8k train rows). Overridden per-timeframe
    # via LGBMTrainer(min_child_samples=...): 1H=30, 15m=20.
    "min_child_samples": 30,
    "random_state": 42,
}


def _git_commit() -> str | None:
    """Short SHA of the working tree, or None if it cannot be determined.

    Best-effort provenance only: no git, no repo, a timeout or any other
    failure must never take a training run down.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:  # noqa: BLE001 — provenance is optional
        _log.debug(f"git_commit unavailable: {exc}")
        return None
    if proc.returncode != 0:
        _log.debug(f"git_commit unavailable: git exited {proc.returncode}")
        return None
    return proc.stdout.strip() or None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single LightGBM model."""

    regime: str  # "trend", "range", "high_vol", "all"
    symbols: list[str]
    # Bar interval the features were built on. Not used by training —
    # it exists so the model bundle's manifest records which timeframe
    # the weights belong to (PR-G). 4H is the production default.
    interval: str = "4h"
    forward_bars: int = 1
    threshold_atr_multiplier: float = 0.5  # unused (binary target ignores it)
    test_size_pct: float = 0.2
    confidence_threshold: float = 0.55
    random_state: int = 42

    # v3: triple-barrier target + AFML uniqueness weights.
    # Enable for retraining; defaults preserve legacy sign(return) target
    # so existing callers (production trend/high_vol/range models) are
    # untouched.
    use_triple_barrier: bool = False
    use_uniqueness_weights: bool = False
    barrier_pt_multiplier: float = 1.0
    barrier_sl_multiplier: float = 1.0
    barrier_max_holding: int = 6
    # Optional model-file suffix (e.g. "_v3"); written between regime
    # and ".pkl" so v3 retrains never overwrite production weights.
    model_suffix: str = ""
    # Optional feature whitelist (clustered-MDA selection output).
    # When set, _prepare_xy restricts X to the intersection of the
    # detected feature columns and this list (preserving the whitelist's
    # order); symbol_encoded is still auto-appended downstream.
    feature_whitelist: list[str] | None = None
    # H13: opt-in to LightGBM's native, text-based ``save_model`` format
    # (sidecar ``.lgb`` next to the legacy ``.pkl`` metadata bundle). The
    # default is False so production retraining keeps the byte-for-byte
    # historic artifact; new training pipelines set True to insulate
    # models from Python / LightGBM / numpy version drift.
    use_native_save: bool = False

    # LightGBM hyperparameters (defaults; Optuna can refine later)
    lgbm_params: dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_estimators": 200,
        "random_state": 42,
        "verbose": -1,
    })


@dataclass
class EvaluationResult:
    """Model evaluation with both ML and trading-relevant metrics."""

    regime: str
    accuracy: float
    precision: float
    recall: float
    f1: float

    # Trading metrics (more important than accuracy)
    win_rate: float       # % correct directional predictions
    profit_factor: float  # Σ|correct returns| / Σ|incorrect returns|
    signal_rate: float    # % of bars where confidence ≥ threshold
    avg_confidence: float  # mean max(P(up), P(down)) on signals

    # Per-symbol breakdown
    per_symbol: dict[str, dict[str, Any]]

    def passes_minimum_thresholds(self) -> bool:
        """Check against master-document go-live criteria."""
        return (
            self.win_rate >= 52.0
            and self.profit_factor >= 1.3
            and self.signal_rate >= 0.10  # at least 10% signals
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LGBMTrainer:
    """Train and evaluate a LightGBM multiclass model.

    Parameters
    ----------
    config:
        ModelConfig with regime filter, split ratio, hyperparameters.
    features_dir:
        Directory containing ``{SYMBOL}_{interval}_features.parquet``.
    models_dir:
        Directory where trained models are saved.
    """

    # H26: canonical importance type — ``"gain"`` (cumulative gain per
    # feature) is less biased toward high-cardinality features than
    # the LightGBM default ``"split"`` (raw split count). Use the
    # ``permutation_importance`` helper below for fully unbiased
    # (model-agnostic) MDA when doing feature selection — gain MDI
    # still slightly over-weights continuous features.
    IMPORTANCE_TYPE: str = "gain"

    def __init__(
        self,
        config: ModelConfig,
        features_dir: Path,
        models_dir: Path,
        use_mtf_params: bool = False,
        min_child_samples: int | None = None,
    ) -> None:
        self.config = config
        # When True, train() uses the stricter MTF_LGBM_PARAMS profile
        # instead of config.lgbm_params (1H/15m anti-overfit). 4H stays
        # on defaults (use_mtf_params=False).
        self.use_mtf_params = use_mtf_params
        # Per-timeframe leaf-size override for the MTF profile only
        # (1H=30, 15m=20). None → keep MTF_LGBM_PARAMS default. Ignored
        # when use_mtf_params=False (4H untouched).
        self.min_child_samples = min_child_samples
        self.features_dir = Path(features_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._builder = DatasetBuilder(
            data_dir=self.features_dir.parent,
            symbols=config.symbols,
        )
        self._feature_columns: list[str] = []

        # PR-G — training facts the manifest needs but that only exist
        # inside train(). Reset at the top of every train() so a reused
        # trainer instance can never stamp a previous run's values into
        # the next bundle.
        self._effective_lgbm_params: dict[str, Any] = {}
        self._embargo_rows: int | None = None
        self._train_rows_after_embargo: int | None = None

        # PR-K — the wall-clock instant train/test were cut on. Set by
        # prepare_data (and reset there), so a bundle records the actual
        # boundary rather than the first bar that happened to survive
        # after it. None when prepare_data was never called (1H/15m
        # scripts feed their frames in directly).
        self._oos_start_ms: int | None = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load, label, and split train/test on one wall-clock boundary.

        Steps
        -----
        1. **Per symbol**: load features from parquet, then create the
           target on that symbol's own contiguous series — a forward-looking
           label must never read across a symbol boundary (ML-002).
        2. Concatenate every labelled symbol into one frame.
        3. Compute **one** OOS boundary on that combined frame, *before*
           the regime filter, via ``compute_default_oos_start_ms`` with
           ``oos_fraction = config.test_size_pct``.
        4. Apply the regime filter to the combined frame.
        5. Skip symbols with nothing on one side of the boundary (warning,
           not an exception); warn about a thin test side.
        6. Split with ``temporal_split_multi``, embargoing one label
           horizon off the tail of every symbol's train part.

        Why the boundary is computed there (PR-K)
        -----------------------------------------
        It used to be a per-symbol ``head(80%) / tail(20%)`` cut **by rows**,
        taken after the regime filter. Symbols survive the filter in
        different numbers, so each symbol's cut landed on a different date
        and the concatenated train frame overlapped test by weeks — WR / PF
        were scored on rows the booster had already fitted. Cutting once,
        on the combined frame, fixes that; cutting *before* the filter also
        keeps the OOS window identical across regimes, so their metrics stay
        comparable. It is computed *after* the target step because the
        triple barrier drops the trailing ``max_holding`` bars (and, with
        ``drop_timeout``, interior rows) — the span that matters is the one
        that actually reaches the booster.

        Returns ``(train_df, test_df)``.
        """
        # Stale-state guard: a reused trainer must not stamp the previous
        # run's boundary into the next manifest.
        self._oos_start_ms = None

        labelled_parts: list[pl.DataFrame] = []

        for symbol in self.config.symbols:
            # Load single symbol
            sym_df = self._builder.load_and_combine(
                self.features_dir, symbols=[symbol],
            )
            if sym_df.is_empty():
                _log.warning(f"No data for {symbol} — skipping")
                continue

            # Target FIRST — on full contiguous data to ensure
            # consistent 1-bar return horizons (ML-002 fix).
            # v3: vol-scaled symmetric triple-barrier (drops timeouts);
            # legacy: 1-bar sign(return).
            if self.config.use_triple_barrier:
                sym_df = self._builder.create_target_triple_barrier(
                    sym_df,
                    pt_multiplier=self.config.barrier_pt_multiplier,
                    sl_multiplier=self.config.barrier_sl_multiplier,
                    max_holding=self.config.barrier_max_holding,
                )
            else:
                sym_df = self._builder.create_target(
                    sym_df,
                    forward_bars=self.config.forward_bars,
                    threshold_atr_multiplier=self.config.threshold_atr_multiplier,
                )

            labelled_parts.append(sym_df)

        if not labelled_parts:
            raise ValueError("No data loaded — check features_dir and symbols")

        labelled = pl.concat(labelled_parts, how="diagonal")

        # ---- One boundary for every symbol and every regime -------------
        oos_start_ms = compute_default_oos_start_ms(
            labelled,
            time_col="open_time",
            oos_fraction=self.config.test_size_pct,
        )
        self._oos_start_ms = oos_start_ms
        oos_start_iso = datetime.fromtimestamp(
            oos_start_ms / 1000, tz=timezone.utc
        ).isoformat()
        _log.info(
            f"OOS boundary @ {oos_start_iso} ({oos_start_ms}) — "
            f"one cut for all symbols and regimes, computed on "
            f"{len(labelled)} labelled rows before the regime filter"
        )

        # ---- Regime filter ----------------------------------------------
        if self.config.regime != "all":
            filtered = self._filter_by_regime(labelled, self.config.regime)
        else:
            filtered = labelled

        # ---- Reject unusable symbols BEFORE the splitter sees them ------
        # temporal_split_multi asserts on an empty test set; those asserts
        # stay as the last line of defence, but a symbol that simply has no
        # rows on one side of the cut must degrade to the historic
        # warning-and-skip, not to an AssertionError.
        keep: list[str] = []
        for symbol in self.config.symbols:
            sym_df = filtered.filter(pl.col("symbol") == symbol)
            if sym_df.is_empty():
                _log.warning(
                    f"No data for {symbol} after regime filter "
                    f"'{self.config.regime}' — skipping"
                )
                continue

            n_train = int((sym_df["open_time"] < oos_start_ms).sum())
            n_test = len(sym_df) - n_train
            if n_train == 0 or n_test == 0:
                empty_side = "train" if n_train == 0 else "test"
                _log.warning(
                    f"{symbol}: empty {empty_side} side at the OOS boundary "
                    f"{oos_start_iso} (regime '{self.config.regime}', "
                    f"{len(sym_df)} rows) — skipping"
                )
                continue

            if n_test < MIN_TEST_ROWS_WARN:
                _log.warning(
                    f"{symbol}: only {n_test} test rows after the OOS "
                    f"boundary (regime '{self.config.regime}') — per-symbol "
                    f"metrics for it are noise"
                )

            keep.append(symbol)
            _log.info(
                f"  {symbol}: {len(sym_df)} rows → train={n_train}, "
                f"test={n_test}"
            )

        if not keep:
            raise ValueError(
                "No symbol has data on both sides of the OOS boundary "
                f"{oos_start_iso} — check features_dir, symbols and regime"
            )

        split_input = filtered.filter(pl.col("symbol").is_in(keep))

        # One label horizon off the train tail, so no train label can look
        # across the cut into test (AFML Ch.7). Measured in wall-clock time,
        # not in rows: ``temporal_split_multi``'s ``embargo_bars`` drops N
        # *rows*, and after the triple barrier's drop_timeout (or the regime
        # filter) the surviving rows are far apart — N rows can span
        # hundreds of bars and delete most of the train side. The cut and
        # its leakage assertions still come from the splitter; only the
        # embargo is expressed as a duration here.
        embargo_bars = (
            self.config.barrier_max_holding
            if self.config.use_triple_barrier
            else max(1, self.config.forward_bars)
        )
        bar_ms = _INTERVAL_MS.get(self.config.interval)

        if bar_ms is None:
            _log.warning(
                f"Train/test embargo falling back to a row count: unknown "
                f"bar duration for interval '{self.config.interval}' — on a "
                f"sparse frame this over-embargoes"
            )
            train_df, test_df = temporal_split_multi(
                split_input,
                oos_start_ms=oos_start_ms,
                symbol_col="symbol",
                time_col="open_time",
                embargo_bars=embargo_bars,
            )
        else:
            train_df, test_df = temporal_split_multi(
                split_input,
                oos_start_ms=oos_start_ms,
                symbol_col="symbol",
                time_col="open_time",
                embargo_bars=0,
            )
            # A pure time filter needs no per-symbol loop — the boundary and
            # the horizon are the same instant for every symbol.
            embargo_cutoff = oos_start_ms - embargo_bars * bar_ms
            n_pre_embargo = len(train_df)
            train_df = train_df.filter(pl.col("open_time") < embargo_cutoff)
            _log.info(
                f"Train/test embargo: {embargo_bars} bars "
                f"({embargo_bars * bar_ms} ms) before the cut — dropped "
                f"{n_pre_embargo - len(train_df)} train rows"
            )

        if train_df.is_empty():
            raise ValueError(
                f"Train set is empty after the {embargo_bars}-bar embargo "
                f"before {oos_start_iso} — the labelled data does not reach "
                f"far enough back"
            )

        _log.info(
            f"Wall-clock split @ {oos_start_iso}: "
            f"train={len(train_df)}, test={len(test_df)} "
            f"({self.config.test_size_pct*100:.0f}% OOS target), "
            f"embargo={embargo_bars} bars"
        )
        return train_df, test_df

    def _embargo_fit_end(
        self,
        train_df: pl.DataFrame,
        val_split: int,
        horizon_bars: int,
    ) -> int:
        """Row count LightGBM may fit on before the early-stopping val set.

        The embargo is measured in **wall-clock time**, not in rows (PR-K).
        ``prepare_data`` now returns a frame sorted by time across all
        symbols, so ``horizon_bars`` *rows* of a three-symbol frame span
        only a third of the interval the label horizon actually covers —
        counting rows would silently under-embargo by that factor. Every
        row later than ``val_start - horizon_bars × bar_duration`` is
        dropped instead.

        Falls back to the legacy row-count cut, with a warning, when the
        frame carries no ``open_time`` or ``config.interval`` is not a
        known bar duration — the 1H/15m scripts and the unit fixtures feed
        frames straight into ``train()``.

        The historic guard is kept in both paths: train_fit is never
        shrunk below half of ``val_split``.
        """
        floor_fit_end = val_split - max(0, val_split // 2)
        bar_ms = _INTERVAL_MS.get(self.config.interval)

        if bar_ms is None or "open_time" not in train_df.columns:
            reason = (
                f"unknown bar duration for interval '{self.config.interval}'"
                if bar_ms is None
                else "no 'open_time' column in the training frame"
            )
            _log.warning(
                f"Train→val embargo falling back to a row count ({reason}) — "
                f"on a multi-symbol frame this under-embargoes by roughly "
                f"the number of symbols"
            )
            return max(
                floor_fit_end, val_split - min(horizon_bars, max(0, val_split // 2))
            )

        open_times = train_df["open_time"].to_numpy()
        val_start_ms = int(open_times[val_split:].min())
        cutoff_ms = val_start_ms - horizon_bars * bar_ms

        # First row whose label horizon reaches into val; everything from
        # there on is embargoed. The comparison is ``>=``: a row exactly
        # one horizon before val_start has its label land ON the first val
        # bar, which is the leak this embargo exists to stop. Taking the
        # first violation (rather than a count) keeps the prefix slice
        # honest even if a caller hands in a frame that is not perfectly
        # time-ordered.
        violations = np.flatnonzero(open_times[:val_split] >= cutoff_ms)
        fit_end = int(violations[0]) if violations.size else val_split
        return max(fit_end, floor_fit_end)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_df: pl.DataFrame,
    ) -> lgb.Booster:
        """Train a LightGBM multiclass model.

        Steps
        -----
        1. Extract feature matrix X and target y.
        2. Encode symbols as integers, labels as 0/1/2.
        3. Split last 10% of training data for validation (early stopping).
        4. Train with ``lgb.train`` + early stopping (50 rounds).
        5. Log to MLflow (parameters, losses, feature importance).

        Returns the trained Booster.

        PR-G: this method no longer writes anything to disk. The bundle
        is written by :meth:`save_bundle`, which is the first point where
        the evaluation exists — a model file with no provenance was the
        whole problem. Callers must therefore do::

            booster = trainer.train(train_df)
            result  = trainer.evaluate(booster, test_df)
            trainer.save_bundle(booster, result, train_df, test_df)
        """
        # Stale-state guard: a reused trainer must never carry the
        # previous run's params/embargo into the next manifest.
        self._effective_lgbm_params = {}
        self._embargo_rows = None
        self._train_rows_after_embargo = None

        X_train, y_train, feature_names = self._prepare_xy(train_df)
        self._feature_columns = feature_names
        _log.info(f"Feature columns ({len(feature_names)}): {feature_names[:5]}...")

        # Validation set for early stopping (temporal, from train — NOT
        # the OOS test set). MTF profile uses the last 15% (noisier
        # financial features → larger, more stable val); 4H keeps 10%.
        val_frac = 0.85 if self.use_mtf_params else 0.90
        val_split = int(len(X_train) * val_frac)

        # Phase 3 Step 3.4 — AFML Ch.7 embargo for the eval_set boundary.
        # Triple-barrier / forward-bars labels of the last train rows look
        # forward into the val window; without a gap early stopping reads
        # a leaked, falsely-optimistic val loss and halts too early. Drop
        # one label-horizon worth of train tail rows so no train label
        # can reach across into val.
        horizon_bars = (
            self.config.barrier_max_holding
            if self.config.use_triple_barrier
            else max(1, self.config.forward_bars)
        )
        fit_end = self._embargo_fit_end(train_df, val_split, horizon_bars)
        # Manifest inputs: n_train_rows counts the frame, these two count
        # what LightGBM actually fitted on.
        embargo_rows = val_split - fit_end
        self._embargo_rows = embargo_rows
        self._train_rows_after_embargo = fit_end

        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train_fit = X_train[:fit_end]
        y_train_fit = y_train[:fit_end]

        # Balanced sample weights — upweight minority classes (UP/DOWN)
        train_weights = compute_sample_weight("balanced", y_train_fit)
        val_weights = compute_sample_weight("balanced", y_val)

        # v3: multiply in AFML uniqueness weights (per-symbol concurrency
        # over the triple-barrier holding window). Aligned to train_df row
        # order, so the same val_split slice applies. Multiplied with
        # balanced weights; mean(uniqueness) ≈ 1 so the balanced scale is
        # preserved and LightGBM's effective sample count stays in range.
        if self.config.use_uniqueness_weights:
            uniq_all = self._builder.compute_uniqueness_weights_by_symbol(
                train_df, max_holding=self.config.barrier_max_holding,
            )
            # Slice uniqueness weights to match the embargoed train_fit
            # length so train_weights * uniq_fit stays element-aligned.
            uniq_fit = uniq_all[:fit_end]
            uniq_val = uniq_all[val_split:]
            train_weights = train_weights * uniq_fit
            val_weights = val_weights * uniq_val
            _log.info(
                f"Uniqueness weights applied (h={self.config.barrier_max_holding}): "
                f"fit mean={uniq_fit.mean():.3f}, "
                f"range=[{uniq_fit.min():.3f}, {uniq_fit.max():.3f}]"
            )

        _log.info(
            f"Class balance — train: "
            f"DOWN(0)={int((y_train_fit==0).sum())}, "
            f"UP(1)={int((y_train_fit==1).sum())} "
            f"→ weight range [{train_weights.min():.2f}, {train_weights.max():.2f}]"
        )

        train_data = lgb.Dataset(
            X_train_fit, label=y_train_fit,
            weight=train_weights, feature_name=feature_names,
        )
        val_data = lgb.Dataset(
            X_val, label=y_val,
            weight=val_weights, feature_name=feature_names,
            reference=train_data,
        )

        # MTF profile (1H/15m) uses stricter regularization; 4H uses
        # config.lgbm_params defaults.
        raw_params = MTF_LGBM_PARAMS if self.use_mtf_params else self.config.lgbm_params
        # n_estimators / early_stopping_rounds are not lgb.train params:
        # the former maps to num_boost_round, the latter to a callback.
        params = {
            k: v for k, v in raw_params.items()
            if k not in ("n_estimators", "early_stopping_rounds")
        }
        num_rounds = raw_params.get("n_estimators", 200)
        stopping_rounds = raw_params.get("early_stopping_rounds", 50)

        if self.use_mtf_params:
            if self.min_child_samples is not None:
                params["min_child_samples"] = self.min_child_samples
            _log.info(
                "Using MTF_LGBM_PARAMS (stricter regularization profile, "
                f"min_child_samples={params.get('min_child_samples')}, "
                f"val_frac={1 - val_frac:.0%}, early_stop={stopping_rounds})"
            )

        # Snapshot what lgb.train actually receives — after the MTF swap
        # and the min_child_samples override — so the manifest records
        # the real hyperparameters, not ModelConfig's defaults.
        self._effective_lgbm_params = dict(params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),  # suppress per-iteration output
        ]

        _log.info(
            f"Training LightGBM: regime={self.config.regime}, "
            f"train={len(X_train_fit)}, val={len(X_val)}, "
            f"features={len(feature_names)}"
        )

        booster = lgb.train(
            params,
            train_data,
            num_boost_round=num_rounds,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        # --- MLflow logging ---
        self._log_to_mlflow(booster, feature_names)

        # NOTE: no disk write here — see save_bundle().
        return booster

    # ------------------------------------------------------------------
    # Persistence — bundle + manifest (PR-G)
    # ------------------------------------------------------------------

    def save_bundle(
        self,
        booster: lgb.Booster,
        result: EvaluationResult,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        *,
        filename: str | None = None,
        allow_failing: bool = False,
    ) -> Path:
        """Write the model bundle, stamped with a provenance manifest.

        Called after ``evaluate()`` — that is the earliest moment the
        eval metrics exist, and a bundle without them is exactly the
        artifact whose lineage could not be reconstructed.

        Parameters
        ----------
        booster:
            The booster returned by :meth:`train`.
        result:
            The evaluation returned by :meth:`evaluate`. Required: the
            manifest is not worth writing without it.
        train_df / test_df:
            The exact frames used for training and evaluation. Row counts
            and the ``open_time`` data window are read from them.
        filename:
            Override the artifact name (e.g. ``"trend_model_1h.pkl"``).
            Defaults to ``"{regime}_model{model_suffix}.pkl"``.
        allow_failing:
            PR-H escape hatch. False (default) keeps the go-live gate
            armed. True writes a model that failed the thresholds and
            stamps ``written_despite_failing`` into its manifest — for
            research runs such as the barrier grid, whose cells must all
            be kept so the DSR multiple-testing correction still counts
            every trial. It does NOT waive the feature-columns check.

        Returns the path written.

        Raises
        ------
        ModelRejectedError
            ``reason="no_feature_columns"`` when the bundle would carry
            no feature list, or ``reason="thresholds"`` when the model
            failed :meth:`EvaluationResult.passes_minimum_thresholds`
            (or no result was supplied at all). Both checks run before
            any file is touched, so the bundle already on disk survives
            a rejected save untouched.
        """
        # model_suffix lets v3 retrains coexist with production weights
        # (empty string → legacy "{regime}_model.pkl"; "_v3" → "_v3.pkl").
        name = filename or (
            f"{self.config.regime}_model{self.config.model_suffix}.pkl"
        )
        model_path = self.models_dir / name

        if not self._effective_lgbm_params:
            _log.warning(
                f"save_bundle({name}): no training state on this trainer — "
                "lgbm_params / embargo fields will be empty in the manifest "
                "(was train() called on this instance?)"
            )
        # --- Gate 1: broken artifact (PR-H) ---------------------------
        # The booster still predicts, but nothing records WHICH columns
        # to feed it in which order — every consumer (ml_strategy,
        # MetaSignalGate) builds its vector from this list. That is a
        # broken artifact rather than a weak model, so allow_failing
        # deliberately does NOT waive it. Checked first: a broken bundle
        # outranks bad metrics as a reason.
        if not self._feature_columns:
            raise ModelRejectedError(
                reason="no_feature_columns",
                path=model_path,
                regime=self.config.regime,
                result=result,
                detail=(
                    "feature_columns is empty — the bundle could not be "
                    "used for inference (was train() called on this "
                    "trainer?)"
                ),
            )

        # --- Gate 2: go-live thresholds (PR-H) ------------------------
        # Both checks sit above _build_manifest and above the
        # use_native_save branch on purpose: nothing may touch the
        # filesystem before the verdict, or a rejected save would leave
        # an orphan .lgb beside an untouched .pkl.
        if result is None:
            raise ModelRejectedError(
                reason="thresholds",
                path=model_path,
                regime=self.config.regime,
                result=None,
            )
        if not allow_failing and not result.passes_minimum_thresholds():
            raise ModelRejectedError(
                reason="thresholds",
                path=model_path,
                regime=self.config.regime,
                result=result,
            )

        written_despite_failing = not result.passes_minimum_thresholds()
        if written_despite_failing:
            _log.warning(
                f"save_bundle({name}): writing a model that FAILED the "
                f"go-live thresholds (allow_failing=True) — "
                f"WR={result.win_rate}%, PF={result.profit_factor}, "
                f"sig={result.signal_rate}. Research artifact, not for "
                "production."
            )

        manifest = self._build_manifest(
            booster, result, train_df, test_df,
            written_despite_failing=written_despite_failing,
        )

        # H13: native ``.lgb`` text save is opt-in. Default behaviour
        # (use_native_save=False) keeps the historic artifact shape:
        # a single pickle bundle containing the live booster object.
        if self.config.use_native_save:
            lgb_path = model_path.with_suffix(".lgb")
            booster.save_model(str(lgb_path))
            model_bundle = {
                "booster": None,                 # legacy slot kept for shape compat
                "booster_file": lgb_path.name,   # sidecar reference for new loaders
                "feature_columns": self._feature_columns,
                "regime": self.config.regime,
                "symbols": self.config.symbols,
                "manifest": manifest,
            }
        else:
            model_bundle = {
                "booster": booster,
                "feature_columns": self._feature_columns,
                "regime": self.config.regime,
                "symbols": self.config.symbols,
                "manifest": manifest,
            }
        with open(model_path, "wb") as f:
            pickle.dump(model_bundle, f)
        _log.info(
            f"Model saved: {model_path} "
            f"({manifest['n_features']} features, "
            f"WR={result.win_rate}%, PF={result.profit_factor}, "
            f"passes={manifest['passes']})"
        )
        return model_path

    def _build_manifest(
        self,
        booster: lgb.Booster,
        result: EvaluationResult,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        *,
        written_despite_failing: bool = False,
    ) -> dict[str, Any]:
        """Assemble the provenance record embedded in the bundle."""
        cfg = self.config

        class_balance: float | None = None
        if "target" in train_df.columns and len(train_df) > 0:
            class_balance = float((train_df["target"] == 1).sum()) / len(train_df)

        # What the data actually contains — a symbol whose feature file
        # is missing is skipped with a warning upstream, so config.symbols
        # can overstate the training set.
        symbols_in_train: list[str] = []
        if "symbol" in train_df.columns:
            symbols_in_train = sorted(train_df["symbol"].unique().to_list())

        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "regime": cfg.regime,
            "interval": cfg.interval,
            "symbols": list(cfg.symbols),
            "symbols_in_train": symbols_in_train,
            "n_features": len(self._feature_columns),
            "feature_columns_hash": hashlib.sha256(
                json.dumps(sorted(self._feature_columns)).encode()
            ).hexdigest(),
            "barriers": {
                "pt": cfg.barrier_pt_multiplier,
                "sl": cfg.barrier_sl_multiplier,
                "max_holding": cfg.barrier_max_holding,
                "use_triple_barrier": cfg.use_triple_barrier,
            },
            "target_kind": (
                "triple_barrier" if cfg.use_triple_barrier else "sign_return"
            ),
            "forward_bars": cfg.forward_bars,
            # The threshold eval actually ran at — NOT any production
            # signal threshold applied later at inference.
            "confidence_threshold": cfg.confidence_threshold,
            "eval": {
                "accuracy": result.accuracy,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "signal_rate": result.signal_rate,
                "avg_confidence": result.avg_confidence,
            },
            # PR-H: the gate's verdict. A False here can only reach disk
            # via allow_failing, which is recorded on the next line — so
            # a research artifact is never mistaken for a vetted one.
            "passes": result.passes_minimum_thresholds(),
            "written_despite_failing": written_despite_failing,
            "data_range": self._data_range(train_df, test_df),
            # PR-K: the cut itself, not the first bar that survived after
            # it. data_range's test_start_ms can sit days later when the
            # regime filter removed the bars right after the boundary.
            # None when the frame did not come from prepare_data.
            "oos_start_ms": self._oos_start_ms,
            "oos_start_iso": (
                datetime.fromtimestamp(
                    self._oos_start_ms / 1000, tz=timezone.utc
                ).isoformat()
                if self._oos_start_ms is not None
                else None
            ),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "embargo_rows": self._embargo_rows,
            "train_rows_after_embargo": self._train_rows_after_embargo,
            "class_balance": class_balance,
            "lgbm_params": dict(self._effective_lgbm_params),
            "use_mtf_params": self.use_mtf_params,
            "use_uniqueness_weights": cfg.use_uniqueness_weights,
            "feature_whitelist_size": (
                len(cfg.feature_whitelist) if cfg.feature_whitelist else None
            ),
            "best_iteration": int(booster.best_iteration),
            "num_trees": int(booster.num_trees()),
            "git_commit": _git_commit(),
            "lightgbm_version": lgb.__version__,
            "python_version": sys.version.split()[0],
        }

    @staticmethod
    def _data_range(
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
    ) -> dict[str, int | str | None]:
        """Train/test window bounds from ``open_time`` (epoch ms, UTC).

        ``open_time`` is the key DataStore.get_klines filters and sorts
        on. Frames fed in by the 1H/15m scripts are not guaranteed to
        carry it, so a missing column degrades to None + a warning
        instead of taking the whole save down.
        """
        empty: dict[str, int | str | None] = {
            f"{side}_{suffix}": None
            for side in ("train_start", "train_end", "test_start", "test_end")
            for suffix in ("ms", "iso")
        }
        if "open_time" not in train_df.columns or "open_time" not in test_df.columns:
            _log.warning(
                "Manifest data_range unavailable: no 'open_time' column in "
                "the training frames — recording nulls"
            )
            return empty
        if len(train_df) == 0 or len(test_df) == 0:
            _log.warning(
                "Manifest data_range unavailable: empty train/test frame "
                "(no 'open_time' values) — recording nulls"
            )
            return empty

        bounds = {
            "train_start": int(train_df["open_time"].min()),
            "train_end": int(train_df["open_time"].max()),
            "test_start": int(test_df["open_time"].min()),
            "test_end": int(test_df["open_time"].max()),
        }
        out: dict[str, int | str | None] = {}
        for key, ms in bounds.items():
            out[f"{key}_ms"] = ms
            out[f"{key}_iso"] = datetime.fromtimestamp(
                ms / 1000, tz=timezone.utc
            ).isoformat()
        return out

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: lgb.Booster,
        test_df: pl.DataFrame,
    ) -> EvaluationResult:
        """Evaluate model on test data with ML + trading metrics.

        Returns an EvaluationResult.
        """
        X_test, y_test_raw, _ = self._prepare_xy(test_df)
        y_true_labels = np.array([CLASS_TO_LABEL[int(c)] for c in y_test_raw])

        # Binary: predict() returns 1D vector of P(class=1=UP)
        proba_up = model.predict(X_test)
        y_pred_class = (proba_up >= 0.5).astype(int)
        y_pred_labels = np.array([CLASS_TO_LABEL[int(c)] for c in y_pred_class])

        # --- ML metrics ---
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        precision = precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
        recall = recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
        f1 = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

        # --- Trading metrics ---
        future_returns = test_df["future_return"].to_numpy()

        # Binary: confidence = max(p, 1-p); fires whenever above threshold.
        # Threshold 0.55 is calibrated for binary (random baseline = 0.50, ML-017).
        confidence_threshold = self.config.confidence_threshold

        max_proba = np.maximum(proba_up, 1.0 - proba_up)
        signal_mask = max_proba >= confidence_threshold
        signal_rate = float(signal_mask.sum()) / len(signal_mask) if len(signal_mask) > 0 else 0.0

        # Win rate & profit factor on signal bars
        signal_preds = y_pred_labels[signal_mask]
        signal_returns = future_returns[signal_mask]
        avg_conf = float(max_proba[signal_mask].mean()) if signal_mask.sum() > 0 else 0.0

        win_rate, profit_factor = self._compute_trading_metrics(signal_preds, signal_returns)

        # --- Per-symbol breakdown ---
        per_symbol = self._compute_per_symbol(
            test_df, X_test, model, confidence_threshold
        )

        result = EvaluationResult(
            regime=self.config.regime,
            accuracy=round(accuracy * 100, 2),
            precision=round(precision * 100, 2),
            recall=round(recall * 100, 2),
            f1=round(f1 * 100, 2),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 4),
            signal_rate=round(signal_rate, 4),
            avg_confidence=round(avg_conf, 4),
            per_symbol=per_symbol,
        )

        _log.info(
            f"Eval [{self.config.regime}]: acc={result.accuracy}%, "
            f"WR={result.win_rate}%, PF={result.profit_factor}, "
            f"sig={result.signal_rate*100:.1f}%, "
            f"passes={result.passes_minimum_thresholds()}"
        )
        return result

    # ------------------------------------------------------------------
    # Model persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_model_bundle(pkl_path: Path | str) -> dict:
        """Load a model bundle in either legacy or H13 sidecar format.

        Returns a dict with at minimum ``booster`` (an ``lgb.Booster``)
        and ``feature_columns``. Detects the new format by the presence
        of a non-null ``booster_file`` key in the pickle metadata; loads
        the sidecar via ``lgb.Booster(model_file=...)``. Falls back to
        the legacy "booster pickled inline" shape transparently — so a
        single call site reads both formats and existing production
        ``.pkl`` files keep working.

        The sidecar path is resolved relative to the pickle file's
        directory, so models can be moved as a (``.pkl`` + ``.lgb``)
        pair without breaking the link.

        PR-G: bundles written before the manifest existed still load —
        they only get a WARNING, once per path per process. Refusing to
        load them is PR-I; VMs are running manifest-less bundles today.
        """
        pkl_path = Path(pkl_path)
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        sidecar = bundle.get("booster_file")
        booster = bundle.get("booster")
        if sidecar and booster is None:
            lgb_path = pkl_path.parent / sidecar
            bundle["booster"] = lgb.Booster(model_file=str(lgb_path))

        if "manifest" not in bundle:
            key = str(pkl_path)
            if key not in _MANIFEST_WARNED_PATHS:
                _MANIFEST_WARNED_PATHS.add(key)
                _log.warning(
                    f"Model bundle without manifest: {pkl_path} — pre-PR-G "
                    "bundle, provenance unknown (barriers, eval metrics and "
                    "data window cannot be recovered). Loading anyway."
                )
        return bundle

    # ------------------------------------------------------------------
    # Feature importance — unbiased MDA (López de Prado AFML Ch.8)
    # ------------------------------------------------------------------

    @staticmethod
    def permutation_importance(
        booster: lgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Model-agnostic permutation feature importance (MDA).

        For each feature, permute its column ``n_repeats`` times, run
        the booster, and measure the resulting accuracy drop relative
        to the unpermuted baseline. Return ``{name: mean_drop}``.

        Higher drop ⇒ more important feature. Unlike ``booster.
        feature_importance(importance_type='gain')`` this estimator is
        unbiased w.r.t. feature cardinality — a binary feature that
        perfectly determines ``y`` ranks above a continuous noise
        feature regardless of how many unique values the noise has.
        Prefer this for feature selection; reserve gain MDI for
        cheap tiebreaking inside a cluster.

        Fail-soft on degenerate inputs (empty X, n_repeats<=0, shape
        mismatch) — returns an empty dict.
        """
        try:
            X = np.asarray(X)
            y = np.asarray(y)
            if X.ndim != 2 or X.shape[0] == 0:
                return {}
            if n_repeats <= 0:
                return {}
            if X.shape[1] != len(feature_names):
                return {}
            from sklearn.metrics import accuracy_score
            rng = np.random.RandomState(random_state)
            baseline_pred = (booster.predict(X) >= 0.5).astype(int)
            baseline_acc = accuracy_score(y, baseline_pred)
            out: dict[str, float] = {}
            n = X.shape[0]
            for col, name in enumerate(feature_names):
                drops = np.empty(n_repeats, dtype=np.float64)
                for r in range(n_repeats):
                    X_perm = X.copy()
                    perm = rng.permutation(n)
                    X_perm[:, col] = X_perm[perm, col]
                    pred = (booster.predict(X_perm) >= 0.5).astype(int)
                    drops[r] = baseline_acc - accuracy_score(y, pred)
                out[name] = float(drops.mean())
            return out
        except Exception as exc:  # noqa: BLE001 — selection-side helper
            _log.warning(
                "permutation_importance failed (non-fatal): {e}", e=exc,
            )
            return {}

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    @staticmethod
    def get_signal(
        model: lgb.Booster,
        features: np.ndarray,
        confidence_threshold: float = 0.55,
    ) -> tuple[int, float]:
        """Return ``(direction, confidence)`` for a single feature vector.

        Binary model (ML-017): ``model.predict`` returns scalar P(UP).
        Direction is +1 if P(UP) > 0.5, otherwise -1.  Confidence is
        ``max(P(UP), 1-P(UP))``.  No signal fires if confidence is below
        *confidence_threshold* (random baseline = 0.50, default 0.55).

        Static (no self) — the body never needed an instance.  Previously
        callers passed ``None`` as ``self``; new callers should drop it.

        Returns
        -------
        direction:
            1 = UP, -1 = DOWN, 0 = NO_SIGNAL
        confidence:
            probability of the predicted direction
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        p_up = float(model.predict(features)[0])
        direction = 1 if p_up > 0.5 else -1
        confidence = p_up if direction == 1 else 1.0 - p_up

        # ML-017 diagnostic: log everything so a "dir=0 conf=0.57 thr=0.55"
        # discrepancy in prod can be traced to whichever value is wrong.
        _log.debug(
            f"get_signal | p_up={p_up:.4f} | direction={direction} | "
            f"confidence={confidence:.4f} | threshold={confidence_threshold}"
        )

        if confidence < confidence_threshold:
            return 0, confidence

        return direction, confidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_by_regime(df: pl.DataFrame, regime: str) -> pl.DataFrame:
        """Filter DataFrame rows by regime label."""
        if "regime" not in df.columns:
            _log.warning("No 'regime' column — returning all rows")
            return df

        if regime == "trend":
            return df.filter(pl.col("regime").is_in(["trend_up", "trend_down"]))
        elif regime == "range":
            return df.filter(pl.col("regime") == "range")
        elif regime == "high_vol":
            return df.filter(pl.col("regime") == "high_vol")
        else:
            return df

    def _prepare_xy(
        self,
        df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract feature matrix X and encoded target y from DataFrame.

        Feature discovery
        -----------------
        * First call (training): discovers numeric feature columns via
          ``get_feature_columns()``, appends ``symbol_encoded``.
        * Subsequent calls (evaluate/signal): re-uses the saved
          ``self._feature_columns`` so the exact same features and
          order are used for prediction.

        Returns ``(X, y_encoded, feature_names)``.
        """
        if self._feature_columns:
            # Re-use training feature list — extract the same columns
            # in the same order that the booster was trained on.
            df_cols: list[str] = []   # columns to pull from df
            for col in self._feature_columns:
                if col == "symbol_encoded":
                    continue  # handled separately below
                if col in df.columns:
                    df_cols.append(col)
        else:
            # First call — discover features
            df_cols = self._builder.get_feature_columns(df)
            # v3 feature selection: restrict to the whitelist if provided.
            # Preserves the whitelist's order so retrains with the same
            # JSON produce identical feature_columns ordering.
            wl = self.config.feature_whitelist
            if wl:
                present = set(df_cols)
                df_cols = [f for f in wl if f in present]
                _log.info(
                    f"Feature whitelist active: {len(df_cols)}/{len(wl)} "
                    f"selected features present in data"
                )

        X = df.select(df_cols).to_numpy().astype(np.float64)

        # Always append symbol_encoded as the last feature
        all_cols = list(df_cols)
        if "symbol" in df.columns:
            symbol_encoded = (
                df["symbol"]
                .replace(SYMBOL_ENCODING, default=-1)
                .cast(pl.Float64)
                .to_numpy()
                .reshape(-1, 1)
            )
            X = np.hstack([X, symbol_encoded])
            all_cols.append("symbol_encoded")

        # Encode target: -1→0 (DOWN), +1→1 (UP)
        # LABEL_TO_CLASS = {-1: 0, +1: 1} — binary classification
        y = df["target"].to_numpy()
        y_encoded = np.array([LABEL_TO_CLASS[int(v)] for v in y], dtype=np.int32)

        # Convert ±inf → NaN; KEEP NaN intact. LightGBM natively routes
        # NaN to the optimal branch of every tree split (and to the same
        # branch at inference — train/serve consistency). Mapping NaN→0
        # erases the distinction between "data unavailable" (e.g. a
        # rolling feature still warming up) and "value really is zero"
        # (e.g. funding_rate = 0%). ``np.isfinite`` is False for both
        # NaN and ±inf, so this replaces ±inf with NaN and is a no-op
        # for already-NaN cells. Finite values pass through unchanged.
        X = np.where(np.isfinite(X), X, np.nan)

        return X, y_encoded, all_cols

    def _log_to_mlflow(
        self,
        booster: lgb.Booster,
        feature_names: list[str],
    ) -> None:
        """Log training run to MLflow (lazy import to avoid matplotlib at startup)."""
        try:
            import mlflow  # lazy: avoids matplotlib OSError under systemd

            mlflow.set_tracking_uri("sqlite:///data/mlflow.db")
            mlflow.set_experiment("AtomiCortex_LightGBM")

            with mlflow.start_run(run_name=f"lgbm_{self.config.regime}") as run:
                # Log parameters
                mlflow.log_params({
                    "regime": self.config.regime,
                    "symbols": ",".join(self.config.symbols),
                    "forward_bars": self.config.forward_bars,
                    "threshold_atr_multiplier": self.config.threshold_atr_multiplier,
                    "num_features": len(feature_names),
                })
                # Log LightGBM params
                for k, v in self.config.lgbm_params.items():
                    mlflow.log_param(f"lgbm_{k}", v)

                # Log eval results from booster
                eval_results = booster.best_score
                for ds_name, metrics in eval_results.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{ds_name}_{metric_name}", value)

                # Feature importance (top 10)
                importance = booster.feature_importance(
                    importance_type=self.IMPORTANCE_TYPE,
                )
                if len(importance) > 0:
                    feat_imp = sorted(
                        zip(feature_names, importance),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    for i, (name, imp) in enumerate(feat_imp[:10]):
                        mlflow.log_metric(f"feat_imp_{i}_{name}", float(imp))

                _log.info(f"MLflow run logged: {run.info.run_id}")
        except Exception as exc:
            _log.warning(f"MLflow logging failed (non-fatal): {exc}")

    @staticmethod
    def _compute_trading_metrics(
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> tuple[float, float]:
        """Compute win rate and profit factor from directional predictions.

        Returns ``(win_rate_pct, profit_factor)``.
        """
        if len(predictions) == 0:
            return 0.0, 0.0

        # A "win" = prediction direction matches actual return direction
        # prediction: 1=UP, -1=DOWN, 0=FLAT
        # Only count bars where model gave a directional signal (not FLAT)
        directional = predictions != 0
        if directional.sum() == 0:
            return 0.0, 0.0

        dir_preds = predictions[directional]
        dir_returns = actual_returns[directional]

        # Win when prediction direction matches return direction
        correct = (dir_preds * dir_returns) > 0
        win_rate = float(correct.sum()) / len(dir_preds) * 100

        # Profit factor = sum of |returns| on wins / sum of |returns| on losses
        wins_abs = np.abs(dir_returns[correct]).sum()
        losses_abs = np.abs(dir_returns[~correct]).sum()

        if losses_abs == 0:
            profit_factor = float("inf") if wins_abs > 0 else 0.0
        else:
            profit_factor = float(wins_abs / losses_abs)

        # Cap inf for serialization
        if math.isinf(profit_factor):
            profit_factor = 999.0

        return win_rate, profit_factor

    def _compute_per_symbol(
        self,
        test_df: pl.DataFrame,
        X_test: np.ndarray,
        model: lgb.Booster,
        confidence_threshold: float,
    ) -> dict[str, dict[str, Any]]:
        """Compute per-symbol metrics breakdown."""
        per_symbol: dict[str, dict[str, Any]] = {}

        if "symbol" not in test_df.columns:
            return per_symbol

        symbols_in_test = test_df["symbol"].unique().to_list()

        for symbol in symbols_in_test:
            mask = (test_df["symbol"] == symbol).to_numpy()
            if mask.sum() == 0:
                continue

            X_sym = X_test[mask]
            proba_up_sym = model.predict(X_sym)  # binary: 1D P(UP)

            # Target & returns for this symbol
            y_true_sym = test_df.filter(pl.col("symbol") == symbol)["target"].to_numpy()
            returns_sym = test_df.filter(pl.col("symbol") == symbol)["future_return"].to_numpy()

            # Predictions & confidence (binary)
            y_pred_class = (proba_up_sym >= 0.5).astype(int)
            y_pred_labels = np.array([CLASS_TO_LABEL[int(c)] for c in y_pred_class])
            max_proba_sym = np.maximum(proba_up_sym, 1.0 - proba_up_sym)
            signal_mask_sym = max_proba_sym >= confidence_threshold

            signal_rate_sym = float(signal_mask_sym.sum()) / len(signal_mask_sym)

            # Win rate on signals
            signal_preds = y_pred_labels[signal_mask_sym]
            signal_returns = returns_sym[signal_mask_sym]
            win_rate_sym, pf_sym = self._compute_trading_metrics(signal_preds, signal_returns)

            per_symbol[symbol] = {
                "win_rate": round(win_rate_sym, 2),
                "profit_factor": round(pf_sym, 4),
                "signal_rate": round(signal_rate_sym, 4),
                "n_bars": int(mask.sum()),
                "n_signals": int(signal_mask_sym.sum()),
            }

        return per_symbol
