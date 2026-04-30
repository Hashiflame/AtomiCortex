#!/usr/bin/env python
"""
AtomiCortex — environment verification script.

Run from the project root:
    python scripts/verify_setup.py

Exit code 0 on full success, 1 if any check fails.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

_results: list[tuple[bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    icon = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{suffix}")
    _results.append((ok, label))
    return ok


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------

def check_python() -> None:
    section("Python")
    major, minor, micro = sys.version_info[:3]
    ver_str = f"{major}.{minor}.{micro}"
    ok = (major, minor) == (3, 11)
    check(f"Python {ver_str}", ok, "requires 3.11.x" if not ok else "")


# ---------------------------------------------------------------------------
# 2. Library imports
# ---------------------------------------------------------------------------

_LIBRARIES: list[tuple[str, str]] = [
    # (import_name, display_name)
    ("polars", "polars"),
    ("numpy", "numpy"),
    ("duckdb", "duckdb"),
    ("pyarrow", "pyarrow"),
    ("zstandard", "zstandard"),
    ("lightgbm", "lightgbm"),
    ("optuna", "optuna"),
    ("mlflow", "mlflow"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("hurst", "hurst"),
    ("ta", "ta"),
    ("nautilus_trader", "nautilus_trader"),
    ("cryptofeed", "cryptofeed"),
    ("redis", "redis"),
    ("telegram", "python-telegram-bot"),
    ("loguru", "loguru"),
    ("pydantic", "pydantic"),
    ("dotenv", "python-dotenv"),
]

_OPTIONAL_LIBRARIES: list[tuple[str, str]] = [
    ("questdb", "questdb"),
]


def _import_version(module_name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", getattr(mod, "VERSION", "?"))
        return True, str(version)
    except ImportError as exc:
        return False, str(exc)


def check_libraries() -> None:
    section("Required Libraries")
    for import_name, display_name in _LIBRARIES:
        ok, detail = _import_version(import_name)
        check(f"{display_name}", ok, detail)

    section("Optional Libraries")
    for import_name, display_name in _OPTIONAL_LIBRARIES:
        ok, detail = _import_version(import_name)
        icon = PASS if ok else WARN
        suffix = f"  ({detail})" if detail else ""
        print(f"  {icon} {display_name} [optional]{suffix}")
        # optional — do not add to _results so it doesn't count as failure


# ---------------------------------------------------------------------------
# 3. Directory structure
# ---------------------------------------------------------------------------

_REQUIRED_DIRS: list[str] = [
    "data/raw",
    "data/features",
    "logs",
    "scripts",
    "src",
    "src/ingestion",
    "src/features",
    "src/models",
    "src/risk",
    "src/execution",
    "src/telegram_bot",
    "tests",
    "notebooks",
]


def check_directories() -> None:
    section("Directory Structure")
    root = Path(__file__).resolve().parent.parent
    for rel in _REQUIRED_DIRS:
        p = root / rel
        check(rel, p.is_dir())


# ---------------------------------------------------------------------------
# 4. .env file
# ---------------------------------------------------------------------------

def check_env_file() -> None:
    section(".env File")
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    env_example_path = root / ".env.example"

    check(".env exists", env_path.is_file(), "run scripts/create_env.sh" if not env_path.is_file() else "")
    check(".env.example exists", env_example_path.is_file())


# ---------------------------------------------------------------------------
# 5. src/config.py import
# ---------------------------------------------------------------------------

def check_config_import() -> None:
    section("src/config.py")
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from src.config import get_settings  # noqa: PLC0415
        settings = get_settings()
        check("src/config imports cleanly", True)
        check(
            f"trading_mode = {settings.trading_mode}",
            settings.trading_mode in ("testnet", "live", "paper"),
        )
        check(
            f"symbols parsed ({len(settings.symbols)} symbols)",
            len(settings.symbols) > 0,
        )
    except Exception as exc:  # noqa: BLE001
        check("src/config imports cleanly", False, str(exc))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> int:
    total = len(_results)
    passed = sum(1 for ok, _ in _results if ok)
    failed_labels = [label for ok, label in _results if not ok]

    section("Summary")
    if failed_labels:
        print(f"\n  {FAIL} FAILED: {passed}/{total} checks passed\n")
        for label in failed_labels:
            print(f"     • {label}")
        print()
        return 1
    else:
        print(f"\n  {PASS} {passed}/{total} CHECKS PASSED\n")
        return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════╗")
    print("║       AtomiCortex — Environment Verification      ║")
    print("╚══════════════════════════════════════════════════╝")

    check_python()
    check_libraries()
    check_directories()
    check_env_file()
    check_config_import()

    sys.exit(print_summary())
