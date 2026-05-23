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

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

from src.logger import get_logger
from src.models.dataset_builder import DatasetBuilder

_log = get_logger(__name__)

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for a single LightGBM model."""

    regime: str  # "trend", "range", "high_vol", "all"
    symbols: list[str]
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

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Load, filter, create target, and per-symbol walk-forward split.

        Steps
        -----
        For **each symbol** independently:
        1. Load features from parquet.
        2. Filter by regime (if not "all").
        3. Create target via DatasetBuilder.create_target().
        4. Walk-forward split: first 80% → train, last 20% → test.

        Then concat all per-symbol train parts and test parts.
        This ensures every symbol is represented in both train and test.

        Returns ``(train_df, test_df)``.
        """
        train_parts: list[pl.DataFrame] = []
        test_parts: list[pl.DataFrame] = []

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

            # Regime filter AFTER target creation
            if self.config.regime != "all":
                sym_df = self._filter_by_regime(sym_df, self.config.regime)
                if sym_df.is_empty():
                    _log.warning(
                        f"No data for {symbol} after regime filter "
                        f"'{self.config.regime}' — skipping"
                    )
                    continue

            # Per-symbol temporal split
            n = len(sym_df)
            train_n = int(n * (1 - self.config.test_size_pct))
            train_parts.append(sym_df.head(train_n))
            test_parts.append(sym_df.tail(n - train_n))

            _log.info(
                f"  {symbol}: {n} rows → train={train_n}, "
                f"test={n - train_n}"
            )

        if not train_parts:
            raise ValueError("No data loaded — check features_dir and symbols")

        train_df = pl.concat(train_parts, how="diagonal")
        test_df = pl.concat(test_parts, how="diagonal")

        _log.info(
            f"Walk-forward split (per-symbol): "
            f"train={len(train_df)}, test={len(test_df)} "
            f"({self.config.test_size_pct*100:.0f}% test)"
        )
        return train_df, test_df

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
        6. Save model to ``models_dir/{regime}_model.pkl``.

        Returns the trained Booster.
        """
        X_train, y_train, feature_names = self._prepare_xy(train_df)
        self._feature_columns = feature_names
        _log.info(f"Feature columns ({len(feature_names)}): {feature_names[:5]}...")

        # Validation set for early stopping (temporal, from train — NOT
        # the OOS test set). MTF profile uses the last 15% (noisier
        # financial features → larger, more stable val); 4H keeps 10%.
        val_frac = 0.85 if self.use_mtf_params else 0.90
        val_split = int(len(X_train) * val_frac)
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train_fit = X_train[:val_split]
        y_train_fit = y_train[:val_split]

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
            uniq_fit = uniq_all[:val_split]
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

        # --- Save model + feature columns ---
        # model_suffix lets v3 retrains coexist with production weights
        # (empty string → legacy "{regime}_model.pkl"; "_v3" → "_v3.pkl").
        model_path = (
            self.models_dir
            / f"{self.config.regime}_model{self.config.model_suffix}.pkl"
        )
        model_bundle = {
            "booster": booster,
            "feature_columns": feature_names,
            "regime": self.config.regime,
            "symbols": self.config.symbols,
        }
        with open(model_path, "wb") as f:
            pickle.dump(model_bundle, f)
        _log.info(f"Model saved: {model_path} ({len(feature_names)} features)")

        return booster

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

        # Replace NaN/inf with 0 in features
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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
                importance = booster.feature_importance(importance_type="gain")
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
