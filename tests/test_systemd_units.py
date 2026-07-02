"""Tests — systemd unit-file linter.

Verifies that StartLimitIntervalSec / StartLimitBurst live in [Unit]
and NOT in [Service].  Uses a line-by-line section parser (NOT
configparser) because systemd units contain repeated keys like
Environment= which configparser cannot handle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy"
_START_LIMIT_KEYS = {"StartLimitIntervalSec", "StartLimitBurst"}


def _parse_sections(path: Path) -> dict[str, list[str]]:
    """Return ``{section_name: [lines]}`` for a systemd unit file.

    Section names are stored WITH brackets, e.g. ``"[Unit]"``.
    Lines before any section header are stored under ``""``.
    """
    sections: dict[str, list[str]] = {"": []}
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            current = line
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return sections


def _service_files() -> list[Path]:
    """Collect all ``deploy/*.service`` files."""
    files = sorted(_DEPLOY_DIR.glob("*.service"))
    assert files, "No .service files found in deploy/"
    return files


@pytest.mark.parametrize("service_file", _service_files(), ids=lambda p: p.name)
def test_start_limit_keys_in_unit_section(service_file: Path) -> None:
    """StartLimitIntervalSec and StartLimitBurst, when present, MUST be in [Unit]."""
    sections = _parse_sections(service_file)
    full_text = service_file.read_text(encoding="utf-8")
    unit_lines = "\n".join(sections.get("[Unit]", []))
    for key in _START_LIMIT_KEYS:
        if key not in full_text:
            continue  # key not used in this unit (e.g. oneshot timer services)
        assert key in unit_lines, (
            f"{service_file.name}: {key} not found in [Unit] section"
        )


@pytest.mark.parametrize("service_file", _service_files(), ids=lambda p: p.name)
def test_no_start_limit_in_service_section(service_file: Path) -> None:
    """StartLimitIntervalSec and StartLimitBurst MUST NOT be in [Service]."""
    sections = _parse_sections(service_file)
    service_lines = "\n".join(sections.get("[Service]", []))
    for key in _START_LIMIT_KEYS:
        assert key not in service_lines, (
            f"{service_file.name}: {key} found in [Service] section — "
            "must be in [Unit] (systemd 255+ ignores it in [Service])"
        )
