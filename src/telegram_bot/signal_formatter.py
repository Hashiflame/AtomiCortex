"""
Signal formatting for Telegram messages.

Single source of truth for how a signal is rendered (full / compact /
open). Pure functions — no DB, no Telegram objects — so it is trivially
testable and reusable by handlers, broadcaster and callbacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.telegram_bot.timeframes import get_tf_emoji, get_tf_label


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO timestamp (tz-aware or naive) → aware UTC, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None

_REGIME_LABELS = {
    "trend_up": "Тренд ↗",
    "trend_down": "Тренд ↘",
    "high_vol": "Высокая волатильность ⚡",
    "range": "Боковик ↔",
    "orb_breakout": "ORB пробой 🎯",
}
_RESULT_EMOJI = {"win": "✅", "loss": "❌", "open": "🔄", "breakeven": "➖"}


def _norm_direction(value: Any) -> str:
    """Accept 'long'/'short' or +1/-1 → 'long'/'short'."""
    if isinstance(value, (int, float)):
        return "long" if value >= 0 else "short"
    return "short" if str(value).lower() == "short" else "long"


def _clean_symbol(symbol: str) -> str:
    s = str(symbol or "BTC")
    s = s.replace("-PERP.BINANCE", "").replace(".BINANCE", "")
    s = s.replace("-PERP", "")
    if s.endswith("USDT"):
        s = s[:-4] + "/USDT"
    return s


def format_signal_card(signal: dict, mode: str = "full") -> str:
    """Render a signal. ``mode`` = ``full`` | ``compact`` | ``open``."""
    tf = signal.get("timeframe") or "4h"
    direction = _norm_direction(signal.get("direction", "long"))
    symbol = _clean_symbol(signal.get("symbol", "BTC"))

    tf_emoji = get_tf_emoji(tf)
    dir_emoji = "🟢" if direction == "long" else "🔴"
    dir_label = "LONG" if direction == "long" else "SHORT"

    entry = float(signal.get("entry_price") or 0)
    tp = float(signal.get("take_profit") or 0)
    sl = float(signal.get("stop_loss") or 0)
    conf = float(signal.get("confidence") or 0)
    regime = signal.get("regime") or "unknown"
    result = signal.get("result") or "open"
    pnl = signal.get("pnl_pct")
    pnl = float(pnl) if pnl is not None else 0.0

    if entry and tp and sl:
        reward = abs(tp - entry)
        risk = abs(sl - entry)
        rr = f"1 : {reward / risk:.1f}" if risk > 0 else "—"
    else:
        rr = "—"
    tp_pct = ((tp - entry) / entry * 100) if entry else 0.0
    sl_pct = ((sl - entry) / entry * 100) if entry else 0.0

    bars = max(0, min(10, int(conf * 10)))
    conf_bar = "█" * bars + "░" * (10 - bars)

    # ORB regime is encoded as "orb:trend_up" by the 15m strategy.
    base_regime = str(regime).split(":")[0]
    if base_regime == "orb":
        regime_label = "ORB пробой 🎯"
    else:
        regime_label = _REGIME_LABELS.get(str(regime), str(regime))

    created_dt = _parse_dt(signal.get("created_at"))
    closed_dt = _parse_dt(signal.get("closed_at"))

    if mode == "compact":
        r_emoji = _RESULT_EMOJI.get(result, "❓")
        line = (
            f"{r_emoji} {dir_emoji} {dir_label} {symbol} "
            f"{tf_emoji}{get_tf_label(tf)}"
        )
        if result != "open":
            line += f"  {pnl:+.1f}%"
        if created_dt:
            line += f"  ·  {created_dt.strftime('%d.%m %H:%M')}"
        return line

    header_status = ""
    if mode == "open":
        header_status = "  •  🔄 ОТКРЫТА"
    elif result in ("win", "loss", "breakeven"):
        header_status = f"  •  {_RESULT_EMOJI.get(result, '')} {result.upper()}"

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        f"{dir_emoji} {dir_label}  •  {symbol}  •  "
        f"{tf_emoji} {get_tf_label(tf)}{header_status}",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    if created_dt:
        lines.append(f"🕐 {created_dt.strftime('%d.%m.%Y %H:%M UTC')}")
    lines += [
        f"💰 Вход:    ${entry:,.0f}",
        f"🎯 Тейк:    ${tp:,.0f}  ({tp_pct:+.1f}%)",
        f"🛑 Стоп:    ${sl:,.0f}  ({sl_pct:+.1f}%)",
        f"⚖️ R:R:     {rr}",
        f"📊 Конф:    {conf_bar}  {conf * 100:.0f}%",
        f"🤖 Режим:   {regime_label}",
    ]
    if result in ("win", "loss", "breakeven"):
        if created_dt and closed_dt:
            secs = max(0, int((closed_dt - created_dt).total_seconds()))
            lines.append(f"⏱ Длит.:   {secs // 3600}ч {secs % 3600 // 60}мин")
        lines.append(
            f"💹 P&L:     {pnl:+.2f}%  {_RESULT_EMOJI.get(result, '')}"
        )
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)
