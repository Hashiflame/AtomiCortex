"""
Unified timeframe registry for the AtomiCortex Telegram bot.

Single source of truth: adding a new timeframe = one line in
``TIMEFRAMES``; keyboards, formatter and filters adapt automatically.
"""

from __future__ import annotations

import os
from pathlib import Path

TIMEFRAMES: dict[str, dict] = {
    "1m":  {"label": "1M",  "emoji": "⚡", "color": "red",    "tier": "elite"},
    "5m":  {"label": "5M",  "emoji": "🔴", "color": "red",    "tier": "elite"},
    "15m": {"label": "15M", "emoji": "🔵", "color": "blue",   "tier": "elite"},
    "1h":  {"label": "1H",  "emoji": "🟡", "color": "yellow", "tier": "pro"},
    "4h":  {"label": "4H",  "emoji": "🟢", "color": "green",  "tier": "basic"},
    "1d":  {"label": "1D",  "emoji": "⚪", "color": "white",  "tier": "basic"},
}

MARKET_TYPES: dict[str, dict] = {
    "futures_perp":  {"label": "Перп", "emoji": "🔄"},
    "futures_dated": {"label": "Фьюч", "emoji": "📅"},
    "spot":          {"label": "Спот", "emoji": "💱"},
}

# 4H is the canonical DB; 15m/1h are appended when their isolated DBs
# exist (mirrors TelegramBot._get_shared_db_paths discovery order).
_DB_MAP: dict[str, str] = {
    "4h": "data/atomicortex.db",
    "15m": "data/atomicortex_15m.db",
    "1h": "data/atomicortex_1h.db",
}

# Repo root so discovery works regardless of CWD.
_ROOT = Path(__file__).resolve().parent.parent.parent


def get_tf_emoji(timeframe: str) -> str:
    return TIMEFRAMES.get(timeframe, {}).get("emoji", "⚪")


def get_tf_label(timeframe: str) -> str:
    return TIMEFRAMES.get(timeframe, {}).get("label", str(timeframe).upper())


def db_path_for(timeframe: str) -> str | None:
    """Absolute DB path for a timeframe, or None if unknown."""
    rel = _DB_MAP.get(timeframe)
    return str(_ROOT / rel) if rel else None


def tf_for_db_path(path: str) -> str:
    """Infer timeframe label from a DB filename."""
    name = str(path)
    if "_15m" in name:
        return "15m"
    if "_1h" in name:
        return "1h"
    return "4h"


def active_timeframes() -> list[str]:
    """Timeframes whose isolated DB exists (data present)."""
    result: list[str] = []
    for tf, rel in _DB_MAP.items():
        if os.path.exists(rel) or (_ROOT / rel).exists():
            result.append(tf)
    return result or ["4h"]
