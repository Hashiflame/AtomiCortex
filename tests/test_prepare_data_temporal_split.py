"""PR-K — one wall-clock train/test boundary for the 4H training path.

``LGBMTrainer.prepare_data`` used to cut every symbol on its own as
``head(80%) / tail(20%)`` **by rows**, after the regime filter. Symbols
survive the filter in different numbers, so each symbol's cut landed on a
different date and the concatenated train frame overlapped the test frame
by weeks. Measured on the shipped bundles: trend train ran to 2025-10-02
while test started 2025-08-11; high_vol train ran to 2025-10-14 while test
started 2025-08-17. WR / PF / signal_rate were therefore scored on rows the
booster had already fitted.

The same defect was fixed for 1H/15m as ML-018, and the fix already lives in
``src/models/temporal_split.py``. PR-K routes the 4H path through it:

* the boundary is computed **once**, on the combined post-target frame,
  **before** the regime filter — so every regime is scored on the same
  window and the numbers are comparable between regimes;
* ``temporal_split_multi`` cuts each symbol at that instant, and one label
  horizon of wall-clock time is embargoed off the train tail — a duration,
  not the splitter's row count, because after ``drop_timeout`` and the
  regime filter N surviving rows can span hundreds of bars;
* a symbol with nothing on one side of the boundary is skipped with a
  warning instead of tripping the assertions inside the splitter.

Two knock-on properties are pinned here as well, because PR-K is what makes
them true:

* the frame ``prepare_data`` returns is globally time-sorted, so the
  early-stopping validation slice inside ``train()`` is the last 10 % **by
  time across all symbols** rather than the tail of whichever symbol was
  concatenated last;
* the train→val embargo inside ``train()`` is measured in **time**, not in
  rows — on a three-symbol interleaved frame, N rows span a third of the
  wall-clock interval the label horizon actually needs.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from loguru import logger as _loguru_logger

from src.models.dataset_builder import DatasetBuilder
from src.models.lgbm_trainer import LGBMTrainer, ModelConfig
from src.models.temporal_split import compute_default_oos_start_ms

BAR_MS = 4 * 3600 * 1000
# 2024-01-01 00:00 UTC — the last bar every full-length symbol ends on is
# derived from this, so symbols with fewer rows start later, not end earlier.
LAST_OPEN_TIME = 1704067200000 + 999 * BAR_MS

TREND_LABELS = ["trend_up", "trend_down"]


# ---------------------------------------------------------------------------
# Synthetic features
# ---------------------------------------------------------------------------


def _make_feature_df(
    n: int,
    symbol: str,
    seed: int,
    last_open_time: int = LAST_OPEN_TIME,
) -> pl.DataFrame:
    """Feature frame ending on ``last_open_time``, ``n`` bars of 4H history.

    Different ``seed`` values give different regime label draws, so the
    regime filter leaves a different number of rows per symbol — that is the
    mechanism behind the real defect. Different ``n`` staggers the start
    date, which makes the per-symbol row-based cuts land on different dates
    even before the filter runs.
    """
    rng = np.random.RandomState(seed)
    close = 40000 + np.cumsum(rng.randn(n) * 100)
    open_times = [last_open_time - (n - 1 - i) * BAR_MS for i in range(n)]

    return pl.DataFrame({
        "open_time": open_times,
        "open": close + rng.randn(n) * 50,
        "high": close + rng.uniform(50, 200, n),
        "low": close - rng.uniform(50, 200, n),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
        "close_time": [t + BAR_MS - 1 for t in open_times],
        "symbol": [symbol] * n,
        "regime": rng.choice(
            ["trend_up", "trend_down", "range", "high_vol"], n
        ).tolist(),
        "cvd": rng.randn(n),
        "cvd_slope_3": rng.randn(n),
        "taker_buy_ratio": rng.uniform(0.4, 0.6, n),
        "volume_ratio": rng.uniform(0.5, 2.0, n),
        "volume_zscore": rng.randn(n),
        "returns_1": rng.randn(n) * 0.01,
        "returns_3": rng.randn(n) * 0.02,
        "returns_6": rng.randn(n) * 0.03,
        "body_ratio": rng.uniform(0.1, 0.9, n),
        "funding_rate": rng.randn(n) * 0.0001,
        "funding_zscore_7d": rng.randn(n),
        "oi_delta_4h": rng.randn(n) * 0.01,
        "oi_zscore": rng.randn(n),
        "ls_ratio": rng.uniform(0.8, 1.2, n),
        "hurst": rng.uniform(0.3, 0.7, n),
        "adx": rng.uniform(10, 50, n),
        "atr_pct": rng.uniform(0.01, 0.05, n),
        "atr_percentile": rng.uniform(0, 1, n),
        "trend_strength": rng.uniform(0, 1, n),
        "regime_confidence": rng.uniform(0, 1, n),
    })


def _write_features(base: Path, spec: dict[str, tuple[int, int]]) -> Path:
    """Write one parquet per symbol. ``spec`` maps symbol → (n_rows, seed)."""
    features_dir = base / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    for symbol, (n, seed) in spec.items():
        _make_feature_df(n=n, symbol=symbol, seed=seed).write_parquet(
            features_dir / f"{symbol}_4h_features.parquet"
        )
    return features_dir


# Different row counts AND different seeds: n guarantees the per-symbol
# row-based cuts diverge, the seed reproduces the real mechanism (unequal
# survival of the regime filter).
SKEWED_SPEC: dict[str, tuple[int, int]] = {
    "BTCUSDT": (600, 1),
    "ETHUSDT": (520, 2),
    "SOLUSDT": (460, 3),
}


def _make_trainer(
    base: Path,
    spec: dict[str, tuple[int, int]] | None = None,
    **config_kwargs,
) -> tuple[LGBMTrainer, Path]:
    spec = spec or SKEWED_SPEC
    features_dir = _write_features(base, spec)
    models_dir = base / "models"
    config = ModelConfig(
        regime=config_kwargs.pop("regime", "trend"),
        symbols=list(spec),
        **config_kwargs,
    )
    trainer = LGBMTrainer(
        config=config, features_dir=features_dir, models_dir=models_dir,
    )
    return trainer, features_dir


# ---------------------------------------------------------------------------
# Reference pipeline — replicates load + target + regime filter WITHOUT
# calling prepare_data, so the control assertions stay independent of the
# code under test.
# ---------------------------------------------------------------------------


def _labelled(
    features_dir: Path,
    symbols: list[str],
    *,
    use_triple_barrier: bool = False,
    barrier_max_holding: int = 6,
    forward_bars: int = 1,
) -> dict[str, pl.DataFrame]:
    """Per-symbol frames after the target step, before the regime filter.

    Mirrors the target step of prepare_data, so the caller must pass the
    same label parameters it configured the trainer with — both targets
    drop a different number of trailing rows.
    """
    builder = DatasetBuilder(data_dir=features_dir.parent, symbols=symbols)
    out: dict[str, pl.DataFrame] = {}
    for symbol in symbols:
        df = builder.load_and_combine(features_dir, symbols=[symbol])
        if use_triple_barrier:
            df = builder.create_target_triple_barrier(
                df,
                pt_multiplier=1.0,
                sl_multiplier=1.0,
                max_holding=barrier_max_holding,
            )
        else:
            df = builder.create_target(
                df, forward_bars=forward_bars, threshold_atr_multiplier=0.5,
            )
        out[symbol] = df
    return out


def _regime_filtered(
    frames: dict[str, pl.DataFrame], regime: str,
) -> dict[str, pl.DataFrame]:
    if regime == "all":
        return dict(frames)
    wanted = TREND_LABELS if regime == "trend" else [regime]
    return {
        symbol: df.filter(pl.col("regime").is_in(wanted))
        for symbol, df in frames.items()
    }


def _expected_boundary(
    frames: dict[str, pl.DataFrame], test_size_pct: float = 0.2,
) -> int:
    """The boundary PR-K must use: combined, post-target, pre-regime-filter."""
    combined = pl.concat(list(frames.values()), how="diagonal")
    return compute_default_oos_start_ms(
        combined, time_col="open_time", oos_fraction=test_size_pct,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def warnings_sink():
    """Collect loguru WARNING messages emitted during the test."""
    sink: list[str] = []
    sink_id = _loguru_logger.add(
        lambda msg: sink.append(str(msg)), level="WARNING", format="{message}",
    )
    try:
        yield sink
    finally:
        _loguru_logger.remove(sink_id)


class _DummyBooster:
    """Just enough Booster surface for the tail of train()."""

    best_iteration = 1
    best_score = {"val": {"binary_logloss": 0.5}}

    def feature_importance(self, importance_type="gain"):
        return np.array([])

    def feature_name(self):
        return []

    def num_trees(self):
        return 1

    def predict(self, *args, **kwargs):
        return np.zeros(1)


def _capture_train_sets():
    """Patch target for ``lgb.train`` that records the Datasets it got."""
    capture: dict = {}

    def fake_train(params, train_set, num_boost_round, valid_sets=None,
                   valid_names=None, callbacks=None, **kwargs):
        capture["train_data"] = train_set
        capture["val_data"] = valid_sets[-1] if valid_sets else None
        return _DummyBooster()

    return capture, fake_train


def _run_train(trainer: LGBMTrainer, train_df: pl.DataFrame) -> dict:
    capture, fake_train = _capture_train_sets()
    with patch.object(
        LGBMTrainer, "_log_to_mlflow", return_value=None,
    ), patch("src.models.lgbm_trainer.lgb.train", side_effect=fake_train):
        trainer.train(train_df)
    return capture


# ===========================================================================
# 1. One global boundary — the headline defect
# ===========================================================================


class TestGlobalBoundary:
    def test_train_end_strictly_before_test_start_globally(
        self, tmp_path: Path,
    ):
        """Global (not per-symbol) walk-forward: no train row may sit at or
        after the first test row, across all symbols at once."""
        trainer, features_dir = _make_trainer(tmp_path, regime="trend")
        symbols = list(SKEWED_SPEC)

        # --- CONTROL: prove the fixture reproduces the defect -------------
        # Replicate the old per-symbol head(80%)/tail(20%) row split on the
        # very same data. If this does NOT overlap, the fixture is useless
        # and the assertion below would be green for the wrong reason.
        filtered = _regime_filtered(_labelled(features_dir, symbols), "trend")
        rowwise_train_max = -1
        rowwise_test_min = 1 << 62
        for df in filtered.values():
            n = len(df)
            train_n = int(n * 0.8)
            rowwise_train_max = max(
                rowwise_train_max, int(df.head(train_n)["open_time"].max())
            )
            rowwise_test_min = min(
                rowwise_test_min, int(df.tail(n - train_n)["open_time"].min())
            )
        assert rowwise_train_max >= rowwise_test_min, (
            "CONTROL FAILED: the row-based split does not overlap on this "
            "fixture, so the test below cannot prove anything"
        )

        # --- The actual requirement --------------------------------------
        train_df, test_df = trainer.prepare_data()
        assert int(train_df["open_time"].max()) < int(
            test_df["open_time"].min()
        ), "train must end strictly before test begins, globally"

    def test_split_boundary_is_the_combined_pre_filter_percentile(
        self, tmp_path: Path,
    ):
        """The cut is compute_default_oos_start_ms over the combined,
        post-target, pre-regime-filter frame — not a per-regime one."""
        trainer, features_dir = _make_trainer(tmp_path, regime="trend")
        boundary = _expected_boundary(
            _labelled(features_dir, list(SKEWED_SPEC))
        )

        train_df, test_df = trainer.prepare_data()

        assert int(train_df["open_time"].max()) < boundary
        assert int(test_df["open_time"].min()) >= boundary


# ===========================================================================
# 2. Embargo between train and test
# ===========================================================================


class TestTrainTestEmbargo:
    def test_embargo_trims_one_label_horizon_of_wall_clock(
        self, tmp_path: Path,
    ):
        """The gap before the cut is one label horizon of *time*.

        Not a row count: after the regime filter (and, with the triple
        barrier, drop_timeout) the surviving rows are far apart, so N rows
        can span hundreds of bars and delete most of the train side.
        """
        trainer, features_dir = _make_trainer(
            tmp_path, regime="trend", use_triple_barrier=False, forward_bars=3,
        )
        symbols = list(SKEWED_SPEC)
        labelled = _labelled(features_dir, symbols, forward_bars=3)
        boundary = _expected_boundary(labelled)
        filtered = _regime_filtered(labelled, "trend")
        embargo_bars = 3  # max(1, forward_bars) with use_triple_barrier=False
        cutoff = boundary - embargo_bars * BAR_MS

        train_df, test_df = trainer.prepare_data()

        for symbol, df in filtered.items():
            left = df.filter(pl.col("open_time") < cutoff)
            right = df.filter(pl.col("open_time") >= boundary)
            train_sym = train_df.filter(pl.col("symbol") == symbol)
            test_sym = test_df.filter(pl.col("symbol") == symbol)

            assert len(train_sym) == len(left), (
                f"{symbol}: train must be exactly the rows more than "
                f"{embargo_bars} bars before the cut"
            )
            assert len(test_sym) == len(right)
            assert int(train_sym["open_time"].max()) < cutoff

        # The embargo really did remove rows on this fixture — otherwise the
        # equality above would hold for a no-op embargo too.
        dropped = sum(
            len(df.filter(
                (pl.col("open_time") >= cutoff)
                & (pl.col("open_time") < boundary)
            ))
            for df in filtered.values()
        )
        assert dropped > 0, "CONTROL FAILED: nothing fell inside the embargo"


# ===========================================================================
# 3. The boundary does not move between regimes
# ===========================================================================


class TestBoundaryStableAcrossRegimes:
    def test_same_boundary_for_trend_high_vol_and_range(self, tmp_path: Path):
        """Comparable OOS metrics require the identical wall-clock window
        for every regime — the whole point of cutting before the filter."""
        features_dir = _write_features(tmp_path, SKEWED_SPEC)
        boundary = _expected_boundary(
            _labelled(features_dir, list(SKEWED_SPEC))
        )

        observed: dict[str, tuple[int, int]] = {}
        for regime in ("trend", "high_vol", "range"):
            config = ModelConfig(regime=regime, symbols=list(SKEWED_SPEC))
            trainer = LGBMTrainer(
                config=config,
                features_dir=features_dir,
                models_dir=tmp_path / f"models_{regime}",
            )
            train_df, test_df = trainer.prepare_data()
            observed[regime] = (
                int(train_df["open_time"].max()),
                int(test_df["open_time"].min()),
            )

        for regime, (train_end, test_start) in observed.items():
            assert train_end < boundary, f"{regime}: train crossed the cut"
            assert test_start >= boundary, f"{regime}: test started too early"


# ===========================================================================
# 4. The boundary is computed after the triple barrier
# ===========================================================================


class TestBoundaryAfterTripleBarrier:
    def test_boundary_uses_post_barrier_frame_not_raw_parquet(
        self, tmp_path: Path,
    ):
        """The barrier drops the trailing max_holding bars, so the raw
        parquet's time span is longer than the one actually trained on. The
        cut has to come from the frame that reaches the booster."""
        max_holding = 48
        trainer, features_dir = _make_trainer(
            tmp_path,
            regime="all",
            use_triple_barrier=True,
            barrier_max_holding=max_holding,
        )
        symbols = list(SKEWED_SPEC)

        builder = DatasetBuilder(data_dir=features_dir.parent, symbols=symbols)
        raw = builder.load_and_combine(features_dir, symbols=symbols)
        boundary_raw = compute_default_oos_start_ms(
            raw, time_col="open_time", oos_fraction=0.2,
        )
        boundary_post = _expected_boundary(
            _labelled(
                features_dir,
                symbols,
                use_triple_barrier=True,
                barrier_max_holding=max_holding,
            )
        )
        assert boundary_post < boundary_raw, (
            "CONTROL FAILED: the barrier did not shorten the span, so the "
            "two boundaries are indistinguishable on this fixture"
        )

        train_df, test_df = trainer.prepare_data()

        assert int(train_df["open_time"].max()) < boundary_post
        assert int(test_df["open_time"].min()) >= boundary_post
        assert int(test_df["open_time"].min()) < boundary_raw, (
            "test must include the bars between the post-barrier cut and "
            "the raw-parquet cut — otherwise the raw span was used"
        )


# ===========================================================================
# 5. Thin and empty test sides
# ===========================================================================


class TestThinAndEmptyTestSides:
    def test_symbol_with_few_test_rows_warns_but_is_kept(
        self, tmp_path: Path, warnings_sink: list[str],
    ):
        """Under 50 test rows the per-symbol metrics are noise — say so, but
        keep the symbol: dropping it silently would be worse."""
        # BTCUSDT spans bars 0..998 and sets the boundary at bar ~798;
        # ETHUSDT stops at bar 828, leaving it 30 rows on the test side.
        eth = _make_feature_df(
            n=830, symbol="ETHUSDT", seed=2,
            last_open_time=LAST_OPEN_TIME - (1000 - 830) * BAR_MS,
        )
        features_dir = _write_features(tmp_path, {"BTCUSDT": (1000, 1)})
        eth.write_parquet(features_dir / "ETHUSDT_4h_features.parquet")

        config = ModelConfig(regime="all", symbols=["BTCUSDT", "ETHUSDT"])
        trainer = LGBMTrainer(
            config=config,
            features_dir=features_dir,
            models_dir=tmp_path / "models",
        )

        train_df, test_df = trainer.prepare_data()

        eth_test = test_df.filter(pl.col("symbol") == "ETHUSDT")
        assert 0 < len(eth_test) < 50
        assert "ETHUSDT" in train_df["symbol"].unique().to_list()
        assert any(
            "ETHUSDT" in msg and str(len(eth_test)) in msg
            for msg in warnings_sink
        ), f"expected a thin-test warning naming ETHUSDT; got {warnings_sink}"

    def test_symbol_with_empty_test_side_is_skipped_not_asserted(
        self, tmp_path: Path, warnings_sink: list[str],
    ):
        """A symbol whose history ends before the cut is skipped with a
        warning — it must not reach temporal_split_multi's assertions."""
        eth = _make_feature_df(
            n=400, symbol="ETHUSDT", seed=2,
            last_open_time=LAST_OPEN_TIME - (1000 - 400) * BAR_MS,
        )
        features_dir = _write_features(tmp_path, {"BTCUSDT": (1000, 1)})
        eth.write_parquet(features_dir / "ETHUSDT_4h_features.parquet")

        config = ModelConfig(regime="all", symbols=["BTCUSDT", "ETHUSDT"])
        trainer = LGBMTrainer(
            config=config,
            features_dir=features_dir,
            models_dir=tmp_path / "models",
        )

        train_df, test_df = trainer.prepare_data()

        assert "ETHUSDT" not in test_df["symbol"].unique().to_list()
        assert "ETHUSDT" not in train_df["symbol"].unique().to_list()
        assert "BTCUSDT" in test_df["symbol"].unique().to_list()
        assert any("ETHUSDT" in msg for msg in warnings_sink), (
            f"expected a skip warning naming ETHUSDT; got {warnings_sink}"
        )


# ===========================================================================
# 6. Knock-on: the early-stopping val slice is temporal across symbols
# ===========================================================================


class TestValSliceIsTemporal:
    def test_val_slice_spans_every_symbol(self, tmp_path: Path):
        """With a globally time-sorted train frame the last 10 % of rows is
        the last 10 % of *time*. Before PR-K it was the tail of whichever
        symbol happened to be concatenated last — a single-symbol val set."""
        trainer, _ = _make_trainer(tmp_path, regime="all")
        train_df, _ = trainer.prepare_data()

        capture = _run_train(trainer, train_df)

        # symbol_encoded is appended by _prepare_xy as the last feature.
        val_symbols = np.unique(capture["val_data"].data[:, -1])
        fit_symbols = np.unique(capture["train_data"].data[:, -1])
        assert len(val_symbols) == len(SKEWED_SPEC), (
            f"val slice must cover every symbol, saw {val_symbols}"
        )
        assert len(fit_symbols) == len(SKEWED_SPEC)


# ===========================================================================
# 7. Knock-on: the train→val embargo is measured in time, not rows
# ===========================================================================


class TestValEmbargoIsTemporal:
    def test_embargo_gap_is_one_label_horizon_of_wall_clock(
        self, tmp_path: Path,
    ):
        """On a three-symbol interleaved frame, ``barrier_max_holding`` rows
        span a third of the label horizon. The gap must be measured in
        milliseconds so early stopping never reads a leaked val loss.

        The frame carries the dense legacy target — the embargo horizon is
        driven by ``config``, and train() takes whatever frame it is handed.
        """
        max_holding = 12
        features_dir = _write_features(tmp_path, SKEWED_SPEC)
        symbols = list(SKEWED_SPEC)
        labelled = _labelled(features_dir, symbols)
        # Globally time-sorted, exactly the shape prepare_data now returns.
        train_df = pl.concat(
            list(labelled.values()), how="diagonal",
        ).sort("open_time")

        config = ModelConfig(
            regime="all",
            symbols=symbols,
            interval="4h",
            use_triple_barrier=True,
            barrier_max_holding=max_holding,
        )
        trainer = LGBMTrainer(
            config=config,
            features_dir=features_dir,
            models_dir=tmp_path / "models",
        )

        capture = _run_train(trainer, train_df)

        open_times = train_df["open_time"].to_numpy()
        val_split = int(len(train_df) * 0.90)
        t_val_start = int(open_times[val_split])
        horizon_ms = max_holding * BAR_MS
        n_fit = capture["train_data"].data.shape[0]

        # CONTROL: rows are denser than bars here, so a row-count embargo
        # cannot reach back one horizon — that is what makes this test bite.
        assert int(open_times[val_split - max_holding]) > (
            t_val_start - horizon_ms
        ), (
            "CONTROL FAILED: on this frame a row-based embargo already "
            "covers the horizon, so the assertion below proves nothing"
        )

        assert n_fit > 0
        assert int(open_times[n_fit - 1]) <= t_val_start - horizon_ms, (
            "the last fitted row's label horizon still reaches into val"
        )
