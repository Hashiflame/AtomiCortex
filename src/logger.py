"""
AtomiCortex — centralized logging module.

Built on loguru.  Deliberately does NOT import src.config at module load
time to avoid circular imports — call setup_logging() after config is ready.

Usage
-----
    from src.logger import get_logger, setup_logging, set_correlation_id

    setup_logging()                 # call once at startup
    log = get_logger(__name__)
    log.info("hello")

    set_correlation_id("req-abc")   # attach to current async context
    log.info("with correlation id")
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Correlation-ID context variable
# ---------------------------------------------------------------------------

_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def set_correlation_id(correlation_id: str) -> None:
    """Attach *correlation_id* to the current async/thread context."""
    _correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    """Return the correlation ID for the current context, or empty string."""
    return _correlation_id_var.get()


# ---------------------------------------------------------------------------
# Custom sink: JSON formatter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PnL filter
# ---------------------------------------------------------------------------

def _pnl_filter(record: "loguru.Record") -> bool:
    """Only pass records that carry the PNL tag."""
    return "PNL" in record.get("extra", {})


# ---------------------------------------------------------------------------
# Correlation-ID patcher
# ---------------------------------------------------------------------------

def _correlation_patcher(record: "loguru.Record") -> None:
    """Inject the current correlation_id into every log record."""
    record["extra"]["correlation_id"] = get_correlation_id()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_logging_configured: bool = False


def setup_logging(
    *,
    logs_dir: Path | str = Path("./logs"),
    trading_mode: str = "testnet",
    level_console: str | None = None,
) -> None:
    """Configure loguru handlers.  Call exactly once at application startup.

    Parameters
    ----------
    logs_dir:
        Directory where log files are written.
    trading_mode:
        ``"live"`` → console shows WARNING+; anything else → INFO+.
    level_console:
        Override the automatic console level selection.
    """
    global _logging_configured
    if _logging_configured:
        return

    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Remove loguru's default handler
    logger.remove()

    # ----- console level -----
    if level_console is None:
        level_console = "WARNING" if trading_mode.lower() == "live" else "INFO"

    console_fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )

    logger.add(
        sys.stderr,
        level=level_console,
        format=console_fmt,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ----- main trading log (JSON, daily rotation) -----
    # loguru's serialize=True produces structured JSON with all record fields.
    # Correlation-ID is injected by the patcher below and appears in "extra".
    trading_log_path = logs_dir / "trading_{time:YYYY-MM-DD}.log"
    logger.add(
        str(trading_log_path),
        level="DEBUG",
        serialize=True,
        rotation="00:00",
        retention="30 days",
        compression="gz",
        enqueue=True,
        catch=True,
    )

    # ----- PnL log -----
    pnl_log_path = logs_dir / "pnl_{time:YYYY-MM-DD}.log"
    logger.add(
        str(pnl_log_path),
        level="DEBUG",
        filter=_pnl_filter,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {extra}",
        rotation="00:00",
        retention="30 days",
        compression="gz",
        enqueue=True,
        catch=True,
    )

    # Patch every record with correlation_id
    logger.configure(patcher=_correlation_patcher)

    _logging_configured = True


def get_logger(name: str) -> "loguru.Logger":
    """Return a loguru logger bound to *name*.

    If :func:`setup_logging` has not been called yet the logger still works —
    loguru's default handler (stderr) remains active until removed.
    """
    return logger.bind(name=name)


# ---------------------------------------------------------------------------
# Convenience re-export so callers can do: from src.logger import logger
# ---------------------------------------------------------------------------
__all__ = [
    "logger",
    "get_logger",
    "setup_logging",
    "set_correlation_id",
    "get_correlation_id",
]
