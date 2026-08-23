"""Tests — PR-Э1.3: the 4H loader refuses to start on an unattested bundle.

Before this PR ``MLTradingStrategy._load_models`` logged a warning for a
missing bundle and carried on with ``self._trend_model is None``, so a
deployment with no models started cleanly, subscribed to bars and then
skipped every one of them for as long as it was left running.  Nothing
in the process exit code, the systemd unit state or the journal said the
bot was not trading.  Э1.3 turns that into a refusal: three conditions
(no file, no registry entry, hash mismatch) each raise
``ModelLoadError``, which travels out of ``on_start`` untouched and ends
as exit code 78 -- the code the unit's ``RestartPreventExitStatus=``
already knows means "do not retry".

Nothing here builds a TradingNode, opens a socket or runs the launcher
in a real process: the strategy is constructed directly (the actor is
never registered, which is fine because the refusal happens before
``subscribe_bars``), and ``run_live.main()`` runs in-process with every
side effect replaced, exactly as ``tests/test_run_live_guard.py`` does.

The real production bundles are copied into a tmp root and described by
a tmp registry built with ``build_entry``, so no expected hash is ever
spelled out here and the repository's own artifacts are never touched.
"""

from __future__ import annotations

import importlib
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.execution.strategies.ml_strategy as ml_strategy
from src.execution.strategies.ml_strategy import MLStrategyConfig, MLTradingStrategy
from src.models.model_paths import MODELS_ROOT_4H, PROD_STEMS_4H, prod_path
from src.models.model_registry import (
    REGISTRY_SCHEMA_VERSION,
    RegistryError,
    build_entry,
    load_registry,
    save_registry,
    sha256_file,
)

_ROOT = Path(__file__).resolve().parent.parent
_REAL_PROD = _ROOT / MODELS_ROOT_4H

# Unpacked rather than spelled out: the loader takes its stems from this
# constant, and a test that hard-coded the names would keep passing after
# the two drifted apart.
_TREND, _HIGH_VOL = PROD_STEMS_4H


# ---------------------------------------------------------------------------
# Lazy access to the names this PR introduces
# ---------------------------------------------------------------------------
#
# Imported inside functions, not at module scope: before the PR lands the
# module does not carry them, and a top-level import would fail collection
# for the whole file instead of failing the tests that actually need them.


def _load_error() -> type[BaseException]:
    from src.models.lgbm_trainer import ModelLoadError

    return ModelLoadError


def _reasons() -> Any:
    from src.models import lgbm_trainer as lt

    return SimpleNamespace(
        missing=lt.LOAD_MISSING,
        no_entry=lt.LOAD_NO_ENTRY,
        hash_mismatch=lt.LOAD_HASH_MISMATCH,
        unreadable=lt.LOAD_UNREADABLE,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _forget_manifest_warnings() -> Any:
    """Д-8: keep the process-wide "no manifest" cache out of the results.

    ``LGBMTrainer.load_model_bundle`` remembers every path it has warned
    about in a module-level set that nothing clears.  Several tests here
    load bundles from tmp paths, and without this the warning a test sees
    would depend on which test ran first.  Deliberately local to this
    file rather than in ``conftest.py`` -- the rest of the suite has no
    such need.
    """
    from src.models import lgbm_trainer

    lgbm_trainer._MANIFEST_WARNED_PATHS.clear()
    yield
    lgbm_trainer._MANIFEST_WARNED_PATHS.clear()


@dataclass
class _Tree:
    """A production root and the registry that describes it."""

    prod: Path
    registry: Path

    def bundle(self, stem: str) -> Path:
        return prod_path(self.prod, stem)

    def entry(self, stem: str) -> dict[str, Any]:
        return load_registry(self.registry)["models"][stem]

    def rewrite(self, registry: dict[str, Any]) -> None:
        save_registry(registry, self.registry)


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Tree:
    """The real bundles copied into a tmp root, plus a matching registry.

    Skipping when the artifacts are absent follows
    ``tests/test_model_registry.py``: a checkout that has not pulled them
    is not a regression, and there is nothing to verify.
    """
    sources = {stem: prod_path(_REAL_PROD, stem) for stem in PROD_STEMS_4H}
    absent = sorted(str(p) for p in sources.values() if not p.is_file())
    if absent:
        pytest.skip(f"production bundles are not present: {absent}")

    prod = tmp_path / "prod"
    prod.mkdir()
    registry_path = tmp_path / "deploy" / "model_registry.json"

    models: dict[str, Any] = {}
    for stem, source in sources.items():
        target = prod / source.name
        shutil.copy2(source, target)
        with open(target, "rb") as fh:
            bundle = pickle.load(fh)
        models[stem] = build_entry(
            bundle_path=target,
            manifest=bundle.get("manifest", {}),
            promoted_at_utc="2026-08-23T00:00:00+00:00",
            source_path=source,
            prod_root=prod,
        )

    save_registry(
        {"schema_version": REGISTRY_SCHEMA_VERSION, "models": models},
        registry_path,
    )
    # The registry path is a module constant, not a strategy config field
    # (Э1.3 keeps it that way to avoid dragging LiveTraderConfig and a CLI
    # flag along), so a test redirects it here.
    monkeypatch.setattr(ml_strategy, "REGISTRY_PATH", registry_path, raising=False)
    return _Tree(prod=prod, registry=registry_path)


class _Recorder:
    """Stand-in for the Nautilus logger the strategy writes through."""

    _LEVELS = ("debug", "info", "warning", "error", "critical")

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        if name not in type(self)._LEVELS:
            raise AttributeError(name)

        def _emit(message: object = "", *args: object, **kwargs: object) -> None:
            self.records.append((name, str(message)))

        return _emit

    def messages(self, level: str) -> list[str]:
        return [message for lvl, message in self.records if lvl == level]


@pytest.fixture
def captured_log() -> Any:
    """Route ``self.log`` into a recorder.

    ``Actor.log`` is a read-only getset on the Cython base, so the
    instance cannot be patched; a property on the Python subclass shadows
    it for the duration of one test and is removed afterwards.
    """
    recorder = _Recorder()
    missing = object()
    previous = MLTradingStrategy.__dict__.get("log", missing)
    MLTradingStrategy.log = property(lambda self: recorder)  # type: ignore[assignment]
    try:
        yield recorder
    finally:
        if previous is missing:
            del MLTradingStrategy.log
        else:
            MLTradingStrategy.log = previous  # type: ignore[assignment]


def _strategy(models_dir: Path) -> MLTradingStrategy:
    return MLTradingStrategy(
        config=MLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
            models_dir=str(models_dir),
            initial_equity=10_000.0,
            dry_run=True,
            preload_enabled=False,
        )
    )


# ===========================================================================
# 1. The three refusals of Д-2, plus the two ways a read can fail
# ===========================================================================


def test_missing_bundle_refuses(tree: _Tree) -> None:
    """No file is the case the old loader answered with a warning."""
    tree.bundle(_TREND).unlink()
    strategy = _strategy(tree.prod)

    with pytest.raises(_load_error()) as excinfo:
        strategy._load_models()

    exc = excinfo.value
    assert exc.reason == _reasons().missing
    assert exc.stem == _TREND
    assert exc.path == tree.bundle(_TREND)
    assert strategy._trend_model is None


def test_stem_without_registry_entry_refuses(tree: _Tree) -> None:
    """An unattested file is refused even though it is readable."""
    registry = load_registry(tree.registry)
    del registry["models"][_TREND]
    tree.rewrite(registry)

    with pytest.raises(_load_error()) as excinfo:
        _strategy(tree.prod)._load_models()

    assert excinfo.value.reason == _reasons().no_entry
    assert excinfo.value.stem == _TREND


def test_entry_without_sha256_refuses(tree: _Tree) -> None:
    """An entry that cannot authenticate its file is no entry at all."""
    registry = load_registry(tree.registry)
    registry["models"][_HIGH_VOL].pop("sha256")
    tree.rewrite(registry)

    with pytest.raises(_load_error()) as excinfo:
        _strategy(tree.prod)._load_models()

    assert excinfo.value.reason == _reasons().no_entry
    assert excinfo.value.stem == _HIGH_VOL


def test_hash_mismatch_refuses_and_names_both_hashes(tree: _Tree) -> None:
    """The tamper case: the operator must be able to compare, not guess.

    Both digests go into the message in full -- truncated hashes cannot
    be pasted into a command, and this is the one line of journal output
    the refusal leaves behind.
    """
    expected = tree.entry(_TREND)["sha256"]
    target = tree.bundle(_TREND)
    with open(target, "ab") as fh:
        fh.write(b"\x00")
    actual = sha256_file(target)

    with pytest.raises(_load_error()) as excinfo:
        _strategy(tree.prod)._load_models()

    exc = excinfo.value
    assert exc.reason == _reasons().hash_mismatch
    assert exc.stem == _TREND
    assert expected in str(exc)
    assert actual in str(exc)


def test_corrupt_registry_refuses(tree: _Tree) -> None:
    """A registry that cannot be parsed is not an empty registry."""
    tree.registry.write_text("{ not json", encoding="utf-8")

    with pytest.raises(_load_error()) as excinfo:
        _strategy(tree.prod)._load_models()

    exc = excinfo.value
    assert exc.reason == _reasons().unreadable
    assert exc.path == tree.registry
    assert isinstance(exc.__cause__, RegistryError)


def test_unreadable_bundle_refuses(tree: _Tree) -> None:
    """A file that hashes correctly but will not unpickle.

    The registry is updated to the garbage so the hash check passes and
    the failure lands in the load phase -- otherwise this test would be
    another spelling of the mismatch case.
    """
    target = tree.bundle(_HIGH_VOL)
    target.write_bytes(b"not a pickle at all")
    registry = load_registry(tree.registry)
    registry["models"][_HIGH_VOL]["sha256"] = sha256_file(target)
    registry["models"][_HIGH_VOL]["size_bytes"] = target.stat().st_size
    tree.rewrite(registry)

    with pytest.raises(_load_error()) as excinfo:
        _strategy(tree.prod)._load_models()

    assert excinfo.value.reason == _reasons().unreadable
    assert excinfo.value.stem == _HIGH_VOL


def test_every_broken_stem_is_named(tree: _Tree) -> None:
    """One refusal, both problems.

    ``RestartPreventExitStatus=78`` means there is no second start on
    which the operator would discover the second broken bundle, so the
    first refusal has to carry the whole picture.
    """
    for stem in PROD_STEMS_4H:
        tree.bundle(stem).unlink()

    with pytest.raises(_load_error()) as excinfo:
        _strategy(tree.prod)._load_models()

    message = str(excinfo.value)
    for stem in PROD_STEMS_4H:
        assert stem in message, message


# ===========================================================================
# 2. The path that must stay open
# ===========================================================================


def test_successful_load_populates_models(tree: _Tree) -> None:
    """The attested tree loads exactly as before."""
    strategy = _strategy(tree.prod)
    strategy._load_models()

    assert strategy._trend_model is not None
    assert strategy._highvol_model is not None
    assert strategy._trend_features
    assert strategy._highvol_features


def test_successful_load_logs_path_hash_and_commit(
    tree: _Tree, captured_log: _Recorder
) -> None:
    """Д-5: what was loaded must be identifiable from the journal alone.

    Without the hash and the training commit, "Loaded trend model" is
    compatible with every bundle that has ever existed.
    """
    _strategy(tree.prod)._load_models()
    logged = "\n".join(captured_log.messages("info"))

    for stem in PROD_STEMS_4H:
        entry = tree.entry(stem)
        assert str(tree.bundle(stem)) in logged
        assert entry["sha256"][:16] in logged
        assert str(entry["trained_git_commit"]) in logged


def test_registry_is_read_once(tree: _Tree, monkeypatch: pytest.MonkeyPatch) -> None:
    """Д-6: one read for the whole call, not one per stem."""
    calls: list[Path] = []
    real = ml_strategy.load_registry

    def _counting(path: Any = None, *args: Any, **kwargs: Any) -> Any:
        calls.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ml_strategy, "load_registry", _counting, raising=False)
    _strategy(tree.prod)._load_models()

    assert len(calls) == 1, calls


def test_registry_stem_outside_prod_stems_is_ignored(tree: _Tree) -> None:
    """The registry describes the whole tree; the loader owns two stems.

    A meta or 15m entry appearing here later must not be able to stop the
    4H bot, and its absence on disk must not be read as a refusal.
    """
    registry = load_registry(tree.registry)
    registry["models"]["orb"] = {
        "path": "data/models/15m/orb_model_15m.pkl",
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    tree.rewrite(registry)

    strategy = _strategy(tree.prod)
    strategy._load_models()

    assert strategy._trend_model is not None
    assert strategy._highvol_model is not None


# ===========================================================================
# 3. The exception itself
# ===========================================================================


def test_model_load_error_shape() -> None:
    """Д-1: a RuntimeError, and one that describes a single artifact.

    The base class is load-bearing.  ``TradingNode.run()`` has a
    ``except RuntimeError`` branch that logs through the kernel logger;
    anything else escapes that branch unlogged.
    """
    err = _load_error()
    assert issubclass(err, RuntimeError)
    assert not issubclass(err, OSError)

    exc = err(_reasons().missing, Path("models/prod/x.pkl"), _TREND, "nothing there")
    assert exc.reason == _reasons().missing
    assert exc.path == Path("models/prod/x.pkl")
    assert exc.stem == _TREND
    assert exc.detail == "nothing there"

    message = str(exc)
    assert _TREND in message
    assert _reasons().missing in message
    assert "models/prod/x.pkl" in message
    assert "nothing there" in message


def test_model_load_and_rejected_are_unrelated() -> None:
    """"Will not write this" and "will not read that" are two verdicts.

    A single ``except`` catching both would let a training-time refusal
    be mistaken for a deployment-time one, and they have opposite fixes.
    """
    from src.models.lgbm_trainer import ModelRejectedError

    load = _load_error()
    assert not issubclass(load, ModelRejectedError)
    assert not issubclass(ModelRejectedError, load)


# ===========================================================================
# 4. The subclasses
# ===========================================================================


def test_meta_strategy_does_not_swallow_refusal(tmp_path: Path) -> None:
    """``MetaMLTradingStrategy`` calls ``super().on_start()`` outside its
    own fail-soft ``try``, so the refusal passes straight through.

    One line moved into that ``try`` would turn fail-closed back into
    fail-soft without a word in the diff about it.
    """
    from src.execution.strategies.meta_strategy import (
        MetaMLStrategyConfig,
        MetaMLTradingStrategy,
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    strategy = MetaMLTradingStrategy(
        config=MetaMLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
            models_dir=str(empty),
            dry_run=True,
            preload_enabled=False,
        )
    )

    with pytest.raises(_load_error()):
        strategy.on_start()


def test_paper_strategy_inherits_refusal(tmp_path: Path) -> None:
    """``PaperTradingStrategy`` overrides neither hook, so it refuses too."""
    from src.execution.strategies.paper_strategy import PaperTradingStrategy

    assert "on_start" not in PaperTradingStrategy.__dict__
    assert "_load_models" not in PaperTradingStrategy.__dict__

    empty = tmp_path / "empty"
    empty.mkdir()
    strategy = PaperTradingStrategy(
        config=MLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            bar_type="BTCUSDT-PERP.BINANCE-4-HOUR-LAST-EXTERNAL",
            models_dir=str(empty),
            dry_run=True,
            preload_enabled=False,
        ),
        metrics_db=str(tmp_path / "metrics.db"),
    )

    with pytest.raises(_load_error()):
        strategy._load_models()


# ===========================================================================
# 5. The chain to exit code 78
# ===========================================================================


class _FakeNode:
    """Only what ``LiveTrader.run()`` and ``_dispose()`` touch."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.run_kwargs: dict[str, Any] | None = None
        self.disposed = False

    def run(self, **kwargs: Any) -> None:
        self.run_kwargs = kwargs
        if self.error is not None:
            raise self.error

    def dispose(self) -> None:
        self.disposed = True


def _trader_with(node: _FakeNode, monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.execution import live_trader as lt

    monkeypatch.setattr(lt.LiveTrader, "_start_connection_checker", lambda self: None)
    monkeypatch.setattr(lt.time, "sleep", lambda seconds: None)
    trader = lt.LiveTrader(
        lt.LiveTraderConfig(
            trading_mode="testnet", symbols=["BTCUSDT-PERP"], dry_run=True
        )
    )
    trader._node = node
    return trader


def test_live_trader_records_model_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal must survive the launcher's blanket ``except Exception``.

    ``run()`` still returns normally -- the ``finally`` block owns the
    dispose -- so the only thing that can carry the verdict to the exit
    code is this flag.
    """
    error = _load_error()(_reasons().hash_mismatch, Path("models/prod/x.pkl"), _TREND)
    node = _FakeNode(error)
    trader = _trader_with(node, monkeypatch)

    trader.run()

    assert trader.model_load_error is error
    assert node.disposed


def test_live_trader_asks_node_to_reraise(monkeypatch: pytest.MonkeyPatch) -> None:
    """``TradingNode.run()`` swallows RuntimeError unless asked not to.

    Its ``except RuntimeError`` branch re-raises only under
    ``raise_exception=True``; with the default the refusal would be
    logged by Nautilus and then lost, the launcher would see a clean
    return and the process would exit 0.  This one kwarg is the whole
    chain.
    """
    node = _FakeNode()
    trader = _trader_with(node, monkeypatch)

    trader.run()

    assert node.run_kwargs == {"raise_exception": True}


class _RunLiveLog:
    _LEVELS = ("debug", "info", "warning", "error", "critical", "success")

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        if name not in type(self)._LEVELS:
            raise AttributeError(name)

        def _emit(message: object = "", *args: object, **kwargs: object) -> None:
            self.records.append((name, str(message)))

        return _emit

    def messages(self, level: str) -> list[str]:
        return [message for lvl, message in self.records if lvl == level]


def _exit_config_error(module: Any) -> int:
    """The launcher's own constant -- never a literal here.

    ``tests/test_run_live_guard.py`` pins it against the unit's
    ``RestartPreventExitStatus=``; repeating the number would create a
    third place for it to drift.
    """
    code = getattr(module, "_EXIT_CONFIG_ERROR", None)
    assert code is not None, "scripts/run_live.py must define _EXIT_CONFIG_ERROR"
    return int(code)


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_load_error: BaseException | None,
    startup_failed: bool = False,
) -> tuple[Any, SystemExit | None, _RunLiveLog]:
    run_live = importlib.import_module("scripts.run_live")
    log = _RunLiveLog()

    class _SpyTrader:
        def __init__(self, config: Any) -> None:
            self.startup_failed = startup_failed
            self.model_load_error = model_load_error

        def run(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class _Stdin:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_live.py", "--mode", "testnet", "--symbols", "BTCUSDT-PERP", "--dry-run"],
    )
    monkeypatch.setattr(sys, "stdin", _Stdin())
    monkeypatch.setattr(run_live, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(run_live, "get_logger", lambda name: log)
    monkeypatch.setattr(
        run_live, "get_settings", lambda: SimpleNamespace(trading_mode="testnet")
    )
    monkeypatch.setattr(run_live, "LiveTrader", _SpyTrader)
    monkeypatch.setattr(run_live.signal, "signal", lambda *a, **k: None)

    raised: SystemExit | None = None
    try:
        run_live.main()
    except SystemExit as exc:
        raised = exc
    return run_live, raised, log


def test_run_live_exits_78_on_model_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not 0 (silent ``inactive (dead)``) and not 1 (restart loop).

    78 is the only code the unit answers by failing once and staying
    failed, which is the correct answer to a bundle no restart can fix.
    """
    error = _load_error()(_reasons().missing, Path("models/prod/x.pkl"), _TREND)
    run_live, raised, log = _run_main(monkeypatch, model_load_error=error)

    assert raised is not None, "main() returned instead of exiting"
    assert raised.code == _exit_config_error(run_live)
    assert raised.code not in (0, 1)
    assert any(str(error) in message for message in log.messages("critical"))


def test_run_live_checks_model_error_before_startup_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfixable refusal outranks a retryable one.

    The engine checker can also have fired by the time the node is
    disposed; if that branch won, a bundle nobody can repair would be
    retried until ``StartLimitBurst`` is spent.
    """
    error = _load_error()(_reasons().missing, Path("models/prod/x.pkl"), _TREND)
    run_live, raised, _ = _run_main(
        monkeypatch, model_load_error=error, startup_failed=True
    )

    assert raised is not None
    assert raised.code == _exit_config_error(run_live)
