"""Tests -- the promotion gate and the CLI around it.

Everything here runs against ``tmp_path``.  The bundles are plain dicts
pickled by hand rather than trained models: the gate refuses before it
ever looks at the booster, so a real LightGBM object would only make the
tests slow and the failures harder to read.

The eight refusals are one test each, and every one of them asserts that
the production directory stayed empty -- a gate that reports a refusal
and writes anyway would otherwise pass.
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.models.model_paths import (
    PROD_STEMS_4H,
    candidate_path,
    prod_path,
)
from src.models.model_registry import (
    REGISTRY_SCHEMA_VERSION,
    RegistryError,
    load_registry,
    sha256_file,
    verify_registry,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import promote_model as pm  # noqa: E402

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "promote_model.py"


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------

def _manifest(stem: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 1,
        "created_at": "2026-08-14T17:09:29.289016+00:00",
        "regime": stem,
        "interval": "4h",
        "n_features": 46,
        "feature_columns_hash": "f" * 64,
        "barriers": {
            "pt": 1.0, "sl": 0.8, "max_holding": 6, "use_triple_barrier": True,
        },
        "target_kind": "triple_barrier",
        "eval": {
            "accuracy": 53.85, "win_rate": 54.83, "profit_factor": 1.3249,
            "signal_rate": 0.629, "avg_confidence": 0.6132,
        },
        "passes": True,
        "written_despite_failing": False,
        "num_trees": 196,
        "best_iteration": 196,
        "git_commit": "e67a451",
    }
    base.update(overrides)
    return base


def _write_candidate(
    candidates_dir: Path,
    stem: str,
    *,
    manifest: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
    bundle_regime: str | None = None,
    drop_manifest: bool = False,
    raw: bytes | None = None,
) -> Path:
    """Write one synthetic candidate bundle and return its path."""
    path = candidate_path(candidates_dir, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_bytes(raw)
        return path
    bundle: dict[str, Any] = {
        "booster": None,
        "feature_columns": (
            ["f1", "f2"] if feature_columns is None else feature_columns
        ),
        "regime": stem if bundle_regime is None else bundle_regime,
        "symbols": ["BTCUSDT"],
    }
    if not drop_manifest:
        bundle["manifest"] = manifest if manifest is not None else _manifest(stem)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    return path


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, Path]:
    """Candidate root, production root and registry path, all under tmp."""
    return {
        "candidates": tmp_path / "candidates",
        "prod": tmp_path / "prod",
        "registry": tmp_path / "deploy" / "registry.json",
    }


def _gate(env: dict[str, Path], stem: str, *, force: bool = False):
    return pm.gate_candidate(
        stem, candidates_dir=env["candidates"], prod_dir=env["prod"], force=force,
    )


def _promote(env: dict[str, Path], stems: list[str], **kwargs: Any):
    return pm.promote(
        stems=stems,
        candidates_dir=env["candidates"],
        prod_dir=env["prod"],
        registry_path=env["registry"],
        **kwargs,
    )


def _prod_contents(env: dict[str, Path]) -> list[str]:
    prod = env["prod"]
    return sorted(p.name for p in prod.iterdir()) if prod.exists() else []


# ---------------------------------------------------------------------------
# 1. The control -- without it every refusal test could pass vacuously
# ---------------------------------------------------------------------------

def test_gate_accepts_valid_candidate(env: dict[str, Path]) -> None:
    """A good bundle must pass, or the eight refusals below prove nothing."""
    source = _write_candidate(env["candidates"], "trend")
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.PromotionPlan), outcome
    assert outcome.stem == "trend"
    assert outcome.source == source
    assert outcome.dest == prod_path(env["prod"], "trend")
    assert outcome.sha256 == sha256_file(source)
    assert outcome.size_bytes == source.stat().st_size
    assert outcome.dest_exists is False
    # The gate itself must not have created anything.
    assert _prod_contents(env) == []


# ---------------------------------------------------------------------------
# 2. The eight refusals
# ---------------------------------------------------------------------------

def test_gate_refuses_source_missing(env: dict[str, Path]) -> None:
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.SOURCE_MISSING
    assert _prod_contents(env) == []


def test_gate_refuses_unreadable(env: dict[str, Path]) -> None:
    _write_candidate(env["candidates"], "trend", raw=b"this is not a pickle")
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.UNREADABLE
    assert _prod_contents(env) == []


def test_gate_refuses_no_manifest(env: dict[str, Path]) -> None:
    """A bundle with no provenance is refused, not promoted with a warning.

    The loader tolerates these -- there are pre-manifest bundles in
    production today -- but tolerating one on the way *in* would create
    a new artifact nobody can trace.
    """
    _write_candidate(env["candidates"], "trend", drop_manifest=True)
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NO_MANIFEST
    assert _prod_contents(env) == []


def test_gate_refuses_no_feature_columns(env: dict[str, Path]) -> None:
    _write_candidate(env["candidates"], "trend", feature_columns=[])
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NO_FEATURE_COLUMNS
    assert _prod_contents(env) == []


def test_gate_refuses_regime_mismatch(env: dict[str, Path]) -> None:
    """Both the manifest's regime and the bundle's are checked.

    They are written independently, so a file that disagrees with itself
    must be refused as surely as one promoted under the wrong name.
    """
    _write_candidate(
        env["candidates"], "trend", manifest=_manifest("high_vol"),
    )
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.REGIME_MISMATCH
    assert _prod_contents(env) == []

    _write_candidate(env["candidates"], "trend", bundle_regime="high_vol")
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.REGIME_MISMATCH


def test_gate_refuses_not_passing(env: dict[str, Path]) -> None:
    """False and absent are both refusals.

    An older bundle with no verdict must not default to one: "we never
    checked" is not "it passed".
    """
    _write_candidate(
        env["candidates"], "trend", manifest=_manifest("trend", passes=False),
    )
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NOT_PASSING
    assert _prod_contents(env) == []

    absent = _manifest("trend")
    del absent["passes"]
    _write_candidate(env["candidates"], "trend", manifest=absent)
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NOT_PASSING


def test_gate_refuses_research_artifact(env: dict[str, Path]) -> None:
    """Refused even when the thresholds pass now.

    allow_failing stamps both flags, so this check is redundant for the
    common case -- but a research cell that would clear the bar today is
    still a research cell, and only this flag says so.
    """
    _write_candidate(
        env["candidates"], "trend",
        manifest=_manifest("trend", passes=True, written_despite_failing=True),
    )
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.RESEARCH_ARTIFACT
    assert _prod_contents(env) == []


def test_gate_refuses_dest_exists_without_force(env: dict[str, Path]) -> None:
    _write_candidate(env["candidates"], "trend")
    dest = prod_path(env["prod"], "trend")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"the incumbent")
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.DEST_EXISTS
    assert dest.read_bytes() == b"the incumbent"


# ---------------------------------------------------------------------------
# 3. Ordering and --force
# ---------------------------------------------------------------------------

def test_gate_order_broken_beats_bad_metrics(env: dict[str, Path]) -> None:
    """A bundle that is both broken and weak is reported as broken.

    Same precedence as save_bundle: "unusable for inference" is the more
    useful answer, and it is the one --force can never waive.
    """
    _write_candidate(
        env["candidates"], "trend",
        manifest=_manifest("trend", passes=False),
        feature_columns=[],
    )
    outcome = _gate(env, "trend")
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NO_FEATURE_COLUMNS


def test_force_does_not_waive_gate(env: dict[str, Path]) -> None:
    """--force reaches the last check only."""
    _write_candidate(
        env["candidates"], "trend", manifest=_manifest("trend", passes=False),
    )
    outcome = _gate(env, "trend", force=True)
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NOT_PASSING

    _write_candidate(env["candidates"], "high_vol", feature_columns=[],
                     manifest=_manifest("high_vol"))
    outcome = _gate(env, "high_vol", force=True)
    assert isinstance(outcome, pm.Refusal)
    assert outcome.code == pm.NO_FEATURE_COLUMNS
    assert _prod_contents(env) == []


def test_force_overwrites_existing_dest(env: dict[str, Path]) -> None:
    """The old bytes are replaced and no backup is left beside them.

    Backups are deliberately not written: git is the history, and a
    stray copy would be one more untracked file in the production root.
    """
    source = _write_candidate(env["candidates"], "trend")
    dest = prod_path(env["prod"], "trend")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"the incumbent")

    result = _promote(env, ["trend"], force=True)
    assert result.ok, result.refusals
    assert dest.read_bytes() == source.read_bytes()
    assert _prod_contents(env) == ["trend_model.pkl"]


# ---------------------------------------------------------------------------
# 4. --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing(env: dict[str, Path]) -> None:
    _write_candidate(env["candidates"], "trend")
    result = _promote(env, ["trend"], dry_run=True)
    assert result.ok
    assert result.promoted == []
    assert len(result.planned) == 1
    assert _prod_contents(env) == []
    assert not env["registry"].exists()


def test_dry_run_runs_full_gate(env: dict[str, Path]) -> None:
    """A dry run that skipped the gate would be worth nothing.

    It reports the same refusal a real run would, which is what makes it
    usable as a check before promoting.
    """
    _write_candidate(
        env["candidates"], "trend", manifest=_manifest("trend", passes=False),
    )
    result = _promote(env, ["trend"], dry_run=True)
    assert not result.ok
    assert [r.code for r in result.refusals] == [pm.NOT_PASSING]
    assert _prod_contents(env) == []


# ---------------------------------------------------------------------------
# 5. --regime all
# ---------------------------------------------------------------------------

def test_regime_all_is_atomic(env: dict[str, Path]) -> None:
    """One bad bundle stops both.

    The two models are one traded configuration -- they share a feature
    list and the strategy routes a third regime onto the first of them.
    A half-applied promotion is a production nobody designed.
    """
    _write_candidate(env["candidates"], "trend")
    _write_candidate(
        env["candidates"], "high_vol",
        manifest=_manifest("high_vol", passes=False),
    )
    result = _promote(env, list(PROD_STEMS_4H))
    assert not result.ok
    assert [r.code for r in result.refusals] == [pm.NOT_PASSING]
    assert len(result.planned) == 1  # trend gated fine, and was still not written
    assert _prod_contents(env) == []
    assert not env["registry"].exists()


def test_regime_all_promotes_both(env: dict[str, Path]) -> None:
    for stem in PROD_STEMS_4H:
        _write_candidate(env["candidates"], stem, manifest=_manifest(stem))
    result = _promote(env, list(PROD_STEMS_4H))
    assert result.ok, result.refusals
    assert _prod_contents(env) == ["high_vol_model.pkl", "trend_model.pkl"]
    registry = load_registry(env["registry"])
    assert set(registry["models"]) == set(PROD_STEMS_4H)
    assert verify_registry(registry, env["prod"]) == []


# ---------------------------------------------------------------------------
# 6. The registry a run leaves behind
# ---------------------------------------------------------------------------

def test_registry_written_after_pkl(env: dict[str, Path]) -> None:
    """Every path the registry names exists by the time it is written."""
    for stem in PROD_STEMS_4H:
        _write_candidate(env["candidates"], stem, manifest=_manifest(stem))
    _promote(env, list(PROD_STEMS_4H))
    registry = load_registry(env["registry"])
    for stem, entry in registry["models"].items():
        assert (env["prod"] / Path(entry["path"]).name).is_file(), stem


def test_promote_writes_registry_entry_fields(env: dict[str, Path]) -> None:
    """The entry carries the provenance a binary diff cannot show."""
    source = _write_candidate(env["candidates"], "trend")
    _promote(env, ["trend"])
    entry = load_registry(env["registry"])["models"]["trend"]

    assert entry["sha256"] == sha256_file(prod_path(env["prod"], "trend"))
    assert entry["size_bytes"] == source.stat().st_size
    assert entry["regime"] == "trend"
    assert entry["interval"] == "4h"
    assert entry["target_kind"] == "triple_barrier"
    assert entry["trained_at_utc"] == "2026-08-14T17:09:29.289016+00:00"
    assert entry["trained_git_commit"] == "e67a451"
    assert entry["n_features"] == 46
    assert entry["barriers"]["max_holding"] == 6
    assert entry["eval"]["win_rate"] == 54.83
    assert entry["passes"] is True
    assert entry["written_despite_failing"] is False
    assert entry["promoted_at_utc"]
    assert entry["path"].endswith("trend_model.pkl")


def test_promote_is_idempotent_with_force(env: dict[str, Path]) -> None:
    """Re-promoting the same candidate changes only the timestamp."""
    _write_candidate(env["candidates"], "trend")
    _promote(env, ["trend"])
    first = load_registry(env["registry"])["models"]["trend"]
    _promote(env, ["trend"], force=True)
    second = load_registry(env["registry"])["models"]["trend"]

    assert first["sha256"] == second["sha256"]
    assert first["size_bytes"] == second["size_bytes"]
    assert {k: v for k, v in first.items() if k != "promoted_at_utc"} == {
        k: v for k, v in second.items() if k != "promoted_at_utc"
    }


def test_promote_refuses_corrupt_registry(env: dict[str, Path]) -> None:
    """A registry that cannot be read stops the run before any write.

    Reading it first is the whole point: treating it as empty would let
    this run silently drop entries it never understood.
    """
    _write_candidate(env["candidates"], "trend")
    env["registry"].parent.mkdir(parents=True, exist_ok=True)
    env["registry"].write_text("{ not json", encoding="utf-8")
    with pytest.raises(RegistryError):
        _promote(env, ["trend"])
    assert _prod_contents(env) == []


# ---------------------------------------------------------------------------
# 7. The CLI itself
# ---------------------------------------------------------------------------

def _run_cli(env: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, str(_SCRIPT),
            "--candidates-dir", str(env["candidates"]),
            "--prod-dir", str(env["prod"]),
            "--registry", str(env["registry"]),
            *extra,
        ],
        capture_output=True, text=True, check=False,
    )


def test_exit_code_zero_on_success(env: dict[str, Path]) -> None:
    for stem in PROD_STEMS_4H:
        _write_candidate(env["candidates"], stem, manifest=_manifest(stem))
    proc = _run_cli(env, "--regime", "all")
    assert proc.returncode == 0, proc.stderr
    assert "Registry updated" in proc.stdout
    assert _prod_contents(env) == ["high_vol_model.pkl", "trend_model.pkl"]


def test_exit_code_one_on_refusal(env: dict[str, Path]) -> None:
    """A refusal must be visible in the exit code, not only in the text.

    The dry run is checked the same way: a --dry-run that returned 0 on a
    bad bundle could not be used as a pre-flight check anywhere.
    """
    _write_candidate(
        env["candidates"], "trend", manifest=_manifest("trend", passes=False),
    )
    proc = _run_cli(env, "--regime", "trend")
    assert proc.returncode == 1
    assert pm.NOT_PASSING in proc.stderr
    assert _prod_contents(env) == []

    dry = _run_cli(env, "--regime", "trend", "--dry-run")
    assert dry.returncode == 1
    assert pm.NOT_PASSING in dry.stderr


def test_cli_dry_run_is_silent_on_disk(env: dict[str, Path]) -> None:
    """The reported plan must match what a real run would do."""
    for stem in PROD_STEMS_4H:
        _write_candidate(env["candidates"], stem, manifest=_manifest(stem))
    proc = _run_cli(env, "--regime", "all", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert "DRY-RUN" in proc.stdout
    assert "Nothing written" in proc.stdout
    assert _prod_contents(env) == []
    assert not env["registry"].exists()


def test_cli_force_announces_what_it_replaces(env: dict[str, Path]) -> None:
    """An overwrite that printed nothing would be unreviewable."""
    _write_candidate(env["candidates"], "trend")
    _run_cli(env, "--regime", "trend")
    old_sha = sha256_file(prod_path(env["prod"], "trend"))

    _write_candidate(
        env["candidates"], "trend",
        manifest=_manifest("trend", created_at="2026-09-01T00:00:00+00:00"),
        feature_columns=["f1", "f2", "f3"],
    )
    proc = _run_cli(env, "--regime", "trend", "--force")
    assert proc.returncode == 0, proc.stderr
    assert "overwriting" in proc.stdout
    assert old_sha in proc.stdout
    assert "2026-09-01T00:00:00+00:00" in proc.stdout


def test_cli_rejects_unknown_regime(env: dict[str, Path]) -> None:
    """--regime is constrained to the production stems plus 'all'."""
    proc = _run_cli(env, "--regime", "orb")
    assert proc.returncode == 2
    assert "invalid choice" in proc.stderr


def test_cli_registry_is_valid_json(env: dict[str, Path]) -> None:
    """What the script writes is readable by something that is not it."""
    _write_candidate(env["candidates"], "trend")
    _run_cli(env, "--regime", "trend")
    raw = json.loads(env["registry"].read_text(encoding="utf-8"))
    assert raw["schema_version"] == REGISTRY_SCHEMA_VERSION
    assert list(raw["models"]) == ["trend"]
