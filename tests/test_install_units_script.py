"""Tests — linter for ``deploy/install_units.sh``.

The installer is bash, so it cannot be imported and exercised the way a
Python module can.  What *can* be asserted is the shape of its source:
that the set of units it touches comes from the manifest
(``deploy/units.enabled``) instead of a hard-coded array, that classification
is driven by ``Type=`` rather than by unit names, that the health-check
covers timers as well as services, and that the directories named in
``ReadWritePaths=`` are created before anything is started.

Same approach as ``tests/test_systemd_units.py``: parse the text, assert
structural properties, and keep one control assertion so a renamed or
emptied input cannot make the whole file go vacuously green.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_DEPLOY_DIR = _ROOT / "deploy"
_SCRIPT = _DEPLOY_DIR / "install_units.sh"
_MANIFEST = _DEPLOY_DIR / "units.enabled"

# The four units PR-0.6 actually deploys.  Everything else in deploy/ is
# kept in the repository but deliberately not installed.
_EXPECTED_MANIFEST_ENTRIES = {
    "atomicortex-bot.service",
    "atomicortex-telegram.service",
    "atomicortex-signal-check.service",
    "atomicortex-signal-check.timer",
}

# A literal unit name anywhere in executable script text means the script
# carries its own copy of the deployment list.
_UNIT_NAME_RE = re.compile(r"atomicortex-[A-Za-z0-9_-]+\.(?:service|timer)")

# The pre-PR-0.6 array elements were bare names without an extension.
_BARE_UNIT_NAME_RE = re.compile(
    r"atomicortex-(?:api|bot|bot-15m|telegram|watchdog|watchdog-15m|"
    r"reconciler|signal-check)\b"
)


def _script_text() -> str:
    """Full source of the installer."""
    return _SCRIPT.read_text(encoding="utf-8")


def _code_text() -> str:
    """Installer source with the shebang and whole-line comments removed.

    Unit names are allowed to appear in the header comment (it documents
    the sudoers entry); only executable lines are linted.
    """
    kept: list[str] = []
    for line in _script_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def _manifest_entries() -> list[str]:
    """Non-empty, non-comment lines of ``deploy/units.enabled``."""
    entries: list[str] = []
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(stripped)
    return entries


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

def test_script_text_is_not_empty() -> None:
    """The linters below must actually be reading the installer.

    Without this, a renamed or emptied script would make every regex
    assertion pass on an empty string and the whole file would go green
    while checking nothing.
    """
    code = _code_text()
    assert code.strip(), f"{_SCRIPT} is empty or all comments"
    assert "systemctl" in code, (
        f"{_SCRIPT.name}: no 'systemctl' in executable lines — this is not "
        "the installer, or the comment stripper ate the body"
    )
    assert "daemon-reload" in code, (
        f"{_SCRIPT.name}: no 'daemon-reload' — installer body looks wrong"
    )


# ---------------------------------------------------------------------------
# P1 / P2 — the unit list is derived, never hard-coded
# ---------------------------------------------------------------------------

def test_no_hardcoded_unit_names() -> None:
    """No literal unit name may appear in executable script lines.

    The deployment list lives in deploy/units.enabled.  A second copy
    inside the script is a second source of truth that silently drifts.
    """
    code = _code_text()
    found = sorted(set(_UNIT_NAME_RE.findall(code)) | set(_BARE_UNIT_NAME_RE.findall(code)))
    assert not found, (
        f"{_SCRIPT.name}: hard-coded unit names in executable lines: {found}. "
        "The list must come from the manifest."
    )


def test_no_literal_timer_names_in_enable() -> None:
    """`systemctl enable --now` must not name a timer literally (P2)."""
    code = _code_text()
    literal_enables = [
        line.strip()
        for line in code.splitlines()
        if "enable" in line and ".timer" in line
    ]
    assert not literal_enables, (
        f"{_SCRIPT.name}: 'enable' with a literal .timer name: "
        f"{literal_enables}. Timers must be enabled from the copied list."
    )


def test_classifier_uses_type_simple() -> None:
    """Long-running units are identified by ``Type=simple`` (R2)."""
    code = _code_text()
    assert "Type=" in code, (
        f"{_SCRIPT.name}: no 'Type=' in the script — units are not being "
        "classified by their own Type= directive"
    )
    assert "simple" in code, (
        f"{_SCRIPT.name}: 'Type=' present but 'simple' absent — the "
        "long-running predicate is not Type=simple"
    )


def test_classifier_does_not_use_restart_directive() -> None:
    """Classification must not fall back to ``Restart=`` or to timer names (R2)."""
    code = _code_text()
    assert "Restart=" not in code, (
        f"{_SCRIPT.name}: classification reads 'Restart=' — R2 fixes the "
        "predicate as Type=simple only"
    )


def test_health_check_covers_timers() -> None:
    """The health-check verifies timers too, not just services (R4)."""
    code = _code_text()
    assert code.count("is-active") >= 2, (
        f"{_SCRIPT.name}: only {code.count('is-active')} 'is-active' call(s) — "
        "services and timers must both be verified"
    )
    assert "is-enabled" in code, (
        f"{_SCRIPT.name}: no 'is-enabled' — P2 requires the result of "
        "'enable --now' to be checked"
    )
    assert re.search(r"\bTIMERS?\b", code), (
        f"{_SCRIPT.name}: no timer list variable in the script"
    )


def test_health_check_collects_all_failures() -> None:
    """All failures are collected and printed, not just the first one.

    The pre-PR-0.6 script exits inside the loop, so an operator learns
    about one broken unit per deploy.
    """
    code = _code_text()
    assert re.search(r"\bFAILED\b", code), (
        f"{_SCRIPT.name}: no FAILED accumulator — the health-check still "
        "exits on the first bad unit"
    )


# ---------------------------------------------------------------------------
# P5 — manifest of target units
# ---------------------------------------------------------------------------

def test_script_reads_the_manifest() -> None:
    """The installer sources its unit list from deploy/units.enabled (P5)."""
    code = _code_text()
    assert "units.enabled" in code, (
        f"{_SCRIPT.name}: does not reference units.enabled — the manifest "
        "is not being read"
    )


def test_script_does_not_glob_the_deploy_dir() -> None:
    """Copying must be manifest-driven, not ``cp deploy/*.service`` (P5)."""
    code = _code_text()
    globs = [
        line.strip()
        for line in code.splitlines()
        if "*.service" in line or "*.timer" in line
    ]
    assert not globs, (
        f"{_SCRIPT.name}: still copies by glob: {globs}. P5 installs only "
        "what the manifest lists."
    )


def test_missing_manifest_is_fatal() -> None:
    """A missing deploy/units.enabled aborts the install (P5)."""
    code = _code_text()
    assert re.search(r'-f\s+"?\$\{?MANIFEST', code), (
        f"{_SCRIPT.name}: no existence test on the manifest path — a missing "
        "manifest must be a hard error, not an empty install"
    )


def test_manifest_entry_without_file_is_fatal() -> None:
    """A manifest name with no matching file in deploy/ aborts the install (P5)."""
    code = _code_text()
    assert re.search(r'-f\s+"?\$\{?REPO_DIR', code), (
        f"{_SCRIPT.name}: no existence test on REPO_DIR/<unit> — a typo in "
        "the manifest would silently install nothing for that entry"
    )


def test_manifest_lists_exactly_the_four_target_units() -> None:
    """deploy/units.enabled holds exactly the four deployed units (P5)."""
    assert _MANIFEST.is_file(), f"{_MANIFEST} does not exist"
    entries = _manifest_entries()
    assert set(entries) == _EXPECTED_MANIFEST_ENTRIES, (
        f"{_MANIFEST.name}: got {sorted(entries)}, "
        f"expected {sorted(_EXPECTED_MANIFEST_ENTRIES)}"
    )
    assert len(entries) == len(set(entries)), (
        f"{_MANIFEST.name}: duplicate entries in {entries}"
    )


def test_manifest_entries_all_exist_in_deploy() -> None:
    """Every manifest name resolves to a real file in deploy/ (P5)."""
    assert _MANIFEST.is_file(), f"{_MANIFEST} does not exist"
    missing = [name for name in _manifest_entries() if not (_DEPLOY_DIR / name).is_file()]
    assert not missing, (
        f"{_MANIFEST.name}: names with no file in deploy/: {missing}"
    )


# ---------------------------------------------------------------------------
# P6 — ReadWritePaths directories are created before start
# ---------------------------------------------------------------------------

def test_script_creates_readwritepaths_directories() -> None:
    """Directories named in ReadWritePaths= are created by the installer (P6).

    Under ProtectSystem=strict a ReadWritePaths= entry that does not exist
    makes systemd abort the unit with 226/NAMESPACE before exec — no
    application output at all.  logs/ is gitignored, so a fresh checkout
    hits exactly that.
    """
    code = _code_text()
    assert "ReadWritePaths" in code, (
        f"{_SCRIPT.name}: never reads ReadWritePaths= — the directories it "
        "must create are unknown to it"
    )
    assert "install -d" in code, (
        f"{_SCRIPT.name}: no 'install -d' — directories are not created "
        "with the unit's own User/Group and a fixed mode"
    )


def test_script_skips_optional_readwritepaths() -> None:
    """A ``-`` prefixed ReadWritePaths entry is optional and must be skipped (P6)."""
    code = _code_text()
    assert "#-" in code, (
        f"{_SCRIPT.name}: no '-' prefix stripping/skip for ReadWritePaths — "
        "an optional path (e.g. -/…/mlruns) would be created anyway"
    )


# ---------------------------------------------------------------------------
# V3 — the installer is executable in the git index
# ---------------------------------------------------------------------------

def test_script_is_executable_in_git_index() -> None:
    """deploy/install_units.sh is mode 100755 in git (V3).

    .github/workflows/deploy.yml gates the install on ``[ -x … ]``.  A
    non-executable file in the index is one careless copy away from a
    deploy that silently skips systemd entirely.
    """
    proc = subprocess.run(
        ["git", "ls-files", "-s", "deploy/install_units.sh"],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        pytest.skip("not a git checkout, or file not tracked")
    mode = proc.stdout.split()[0]
    assert mode == "100755", (
        f"deploy/install_units.sh has git mode {mode}, expected 100755. "
        "Fix with: git update-index --chmod=+x deploy/install_units.sh"
    )
