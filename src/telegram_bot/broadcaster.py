"""
AtomiCortex — Telegram Bot Broadcaster.

Sends automatic notifications to users based on their role.
Handles signal alerts, regime changes, circuit breakers, and daily reports.

Phase 7 — Telegram Bot.
"""

from __future__ import annotations

import asyncio
from typing import Any

from telegram import Bot

from src.logger import get_logger
from src.telegram_bot.database import Database
from src.telegram_bot.roles import OWNER_ID

_log = get_logger(__name__)

_ROLE_LEVELS = {"free": 0, "premium": 1, "owner": 2}

# TG-006: concurrency limit for broadcasts
_BROADCAST_SEMAPHORE = asyncio.Semaphore(25)

# TG-013: retry settings
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5


class Broadcaster:
    """Sends role-aware automatic notifications via the Telegram Bot API."""

    def __init__(self, bot: Bot, db: Database) -> None:
        self._bot = bot
        self._db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def broadcast_signal(
        self,
        signal_data: dict[str, Any],
        fill_data: dict[str, Any] | None = None,
    ) -> None:
        """Send trading signal to users based on role."""
        direction = signal_data.get("direction", "")
        d_upper = direction.upper() if isinstance(direction, str) else (
            "LONG" if direction == 1 else "SHORT"
        )
        emoji = "🟢" if d_upper == "LONG" else "🔴"
        symbol = signal_data.get("symbol", "N/A")
        entry = signal_data.get("entry_price", 0)
        sl = signal_data.get("stop_loss", 0)
        tp = signal_data.get("take_profit", 0)
        conf = signal_data.get("confidence", 0)
        regime = signal_data.get("regime", "N/A")

        sl_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
        tp_pct = abs(tp - entry) / entry * 100 if entry > 0 else 0
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0

        full_msg = (
            f"{'═' * 30}\n"
            f"{emoji} {d_upper} {symbol} PERP\n"
            f"{'═' * 30}\n"
            f"Цена входа: ${entry:,.2f}\n"
            f"Стоп: ${sl:,.2f} (-{sl_pct:.1f}%)\n"
            f"Тейк: ${tp:,.2f} (+{tp_pct:.1f}%)\n"
            f"R:R: 1:{rr:.1f}\n"
            f"Режим: {regime.upper()}\n"
            f"Уверенность: {conf:.0%}\n"
            f"{'═' * 30}"
        )

        teaser_msg = (
            f"🔔 Новый сигнал: {emoji} {d_upper} {symbol}\n\n"
            f"Подпишитесь на Premium для полных деталей.\n"
            f"/subscribe"
        )

        signal_id = self._db.log_signal(signal_data)
        self._db.log_event("signal", f"{d_upper} {symbol} @ ${entry:,.2f}")

        await self._send_role_filtered(
            full_msg=full_msg,
            teaser_msg=teaser_msg,
            min_full_role="premium",
        )

        _log.info(
            "Signal broadcast | {sym} {dir} | id={sid}",
            sym=symbol, dir=d_upper, sid=signal_id,
        )

    async def broadcast_regime_change(
        self, old_regime: str, new_regime: str,
    ) -> None:
        """Notify about market regime change (premium+ only)."""
        msg = (
            f"📊 Смена режима рынка\n"
            f"{'═' * 30}\n"
            f"{old_regime.upper()} → {new_regime.upper()}\n"
            f"{'═' * 30}"
        )
        self._db.log_event("regime_change", f"{old_regime} -> {new_regime}")
        await self._send_to_min_role(msg, "premium")

    async def broadcast_circuit_breaker(self, reason: str) -> None:
        """Notify about circuit breaker activation (premium+ only)."""
        msg = (
            f"🚨 CIRCUIT BREAKER\n"
            f"{'═' * 30}\n"
            f"{reason}\n"
            f"{'═' * 30}"
        )
        self._db.log_event("circuit_breaker", reason)
        await self._send_to_min_role(msg, "premium")

    async def broadcast_daily_report(self, metrics: Any) -> None:
        """Send daily report. Premium+ get full, free gets summary."""
        full_msg = (
            f"📊 AtomiCortex — Daily Report\n"
            f"{'═' * 30}\n"
            f"Equity:   ${metrics.equity:,.2f}\n"
            f"P&L:      {metrics.daily_pnl_pct:+.2%}\n"
            f"Сделок:   {metrics.total_trades}\n"
            f"Win rate: {metrics.win_rate:.1%}\n"
            f"PF:       {metrics.profit_factor:.2f}\n"
            f"DD:       {metrics.current_drawdown:.2%}\n"
            f"Sharpe:   {metrics.sharpe_ratio:.2f}\n"
            f"Режим:    {metrics.regime}\n"
            f"{'═' * 30}"
        )

        summary_msg = (
            f"📊 AtomiCortex — Daily Summary\n"
            f"{'═' * 30}\n"
            f"P&L: {metrics.daily_pnl_pct:+.2%}\n"
            f"Win rate: {metrics.win_rate:.1%}\n"
            f"Режим: {metrics.regime}\n\n"
            f"Полный отчёт доступен в Premium /subscribe\n"
            f"{'═' * 30}"
        )

        self._db.log_event("daily_report", f"pnl={metrics.daily_pnl_pct:+.2%}")
        await self._send_role_filtered(
            full_msg=full_msg,
            teaser_msg=summary_msg,
            min_full_role="premium",
        )

    async def send_to_owner(self, message: str) -> None:
        """Send critical alert to owner only.  TG-015: guard None OWNER_ID."""
        if OWNER_ID is None:
            _log.warning("OWNER_ID not configured — cannot send owner alert")
            return
        await self._send_with_retry(OWNER_ID, message)

    # ------------------------------------------------------------------
    # Internal helpers  (TG-006: concurrent, TG-013: retry)
    # ------------------------------------------------------------------

    async def _send_with_retry(
        self, chat_id: int, text: str, max_retries: int = _MAX_RETRIES,
    ) -> bool:
        """Send message with exponential backoff retry.  TG-013."""
        for attempt in range(max_retries):
            try:
                await self._bot.send_message(chat_id=chat_id, text=text)
                return True
            except Exception as exc:
                if attempt == max_retries - 1:
                    _log.warning(
                        "Send failed uid={uid} after {n} retries: {err}",
                        uid=chat_id, n=max_retries, err=str(exc),
                    )
                    return False
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
        return False

    async def _send_to_min_role(self, message: str, min_role: str) -> None:
        """Send message to all non-banned users at or above min_role.

        TG-006: uses asyncio.gather with semaphore for concurrency.
        """
        min_level = _ROLE_LEVELS.get(min_role, 0)
        users = self._db.get_non_banned_users()
        targets = [
            u for u in users
            if _ROLE_LEVELS.get(u.get("role", "free"), 0) >= min_level
        ]

        if not targets:
            return

        async def _send_one(uid: int) -> None:
            async with _BROADCAST_SEMAPHORE:
                await self._send_with_retry(uid, message)

        await asyncio.gather(
            *[_send_one(u["user_id"]) for u in targets],
            return_exceptions=True,
        )

    async def _send_role_filtered(
        self, full_msg: str, teaser_msg: str, min_full_role: str,
    ) -> None:
        """Send full message to premium+, teaser to free users.

        TG-006: concurrent sends via asyncio.gather.
        """
        min_full_level = _ROLE_LEVELS.get(min_full_role, 1)
        users = self._db.get_non_banned_users()

        if not users:
            return

        async def _send_one(u: dict) -> None:
            user_level = _ROLE_LEVELS.get(u.get("role", "free"), 0)
            msg = full_msg if user_level >= min_full_level else teaser_msg
            async with _BROADCAST_SEMAPHORE:
                await self._send_with_retry(u["user_id"], msg)

        await asyncio.gather(
            *[_send_one(u) for u in users],
            return_exceptions=True,
        )
