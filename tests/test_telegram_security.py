"""Tests for Steps H24+H25 — Telegram bot security fixes.

H24: ``AIORateLimiter`` is now plugged into the PTB Application so
broadcast traffic respects Telegram's flood control.

H25: Owner-only inline-keyboard callbacks (health/refresh, health/logs,
health/restart, users pagination, stats period) require the caller to
have the owner role — previously only the matching ``/command``
handlers enforced this, so an owner who forwarded a /health card into
a group let any member press the destructive buttons.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from telegram.ext import AIORateLimiter

from src.telegram_bot.bot import TelegramBot
from src.telegram_bot.database import Database


# ---------------------------------------------------------------------------
# H24 — AIORateLimiter wired into Application builder
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_application_built_with_aio_rate_limiter(self, tmp_path):
        bot = TelegramBot(
            token="123:dummy",
            admin_id=1,
            db_path=str(tmp_path / "bot.db"),
        )
        app = bot.build()
        # PTB exposes the limiter via the underlying Bot.
        assert app.bot.rate_limiter is not None
        assert isinstance(app.bot.rate_limiter, AIORateLimiter)


# ---------------------------------------------------------------------------
# H25 — _is_owner_callback prefix table
# ---------------------------------------------------------------------------


class TestOwnerCallbackPrefixes:
    @pytest.mark.parametrize("data", [
        "health_refresh",
        "health_logs_20",
        "health_logs_100",
        "health_restart",
        "users_page_0",
        "users_page_5",
        "stats_period_7d",
        "stats_period_30d",
    ])
    def test_owner_callbacks_recognised(self, data):
        assert TelegramBot._is_owner_callback(data) is True

    @pytest.mark.parametrize("data", [
        "pay_stars_30",
        "pay_usdt_90",
        "show_subscribe",
        "signals_back",
        "signals_tf:4h",
        "signal_detail:42",
        "history_page:0:all",
        "noop",
        "users_noop",
        "",
    ])
    def test_public_callbacks_not_gated(self, data):
        assert TelegramBot._is_owner_callback(data) is False


# ---------------------------------------------------------------------------
# H25 — runtime role gating in _handle_callback
# ---------------------------------------------------------------------------


def _bot(tmp_path) -> TelegramBot:
    return TelegramBot(token="t", admin_id=1, db_path=str(tmp_path / "bot.db"))


def _ctx() -> SimpleNamespace:
    """Minimal ContextTypes stand-in carrying the bot_data dict."""
    return SimpleNamespace(bot_data={"prices": {}})


def _query(data: str, user_id: int):
    """Build a fake callback_query with answer/edit_message_text as AsyncMocks."""
    q = MagicMock()
    q.data = data
    q.from_user = SimpleNamespace(id=user_id)
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    return q


def _update(query, user_id: int):
    upd = MagicMock()
    upd.callback_query = query
    upd.effective_user = SimpleNamespace(id=user_id)
    upd.effective_chat = MagicMock()
    return upd


class TestHandleCallbackRoleGate:
    @pytest.mark.asyncio
    async def test_non_owner_blocked_with_alert(self, tmp_path):
        bot = _bot(tmp_path)
        bot._db.create_user(42, "intruder", "Mallory")
        # Default role is "free".

        query = _query("health_restart", user_id=42)
        upd = _update(query, user_id=42)

        # Should NOT call _restart_bot_inline.
        bot._restart_bot_inline = AsyncMock()

        await bot._handle_callback(upd, _ctx())

        # First .answer() is the implicit acknowledgement; the second
        # carries the show_alert rejection.
        alert_calls = [
            c for c in query.answer.await_args_list
            if c.kwargs.get("show_alert")
            or (c.args and "прав" in str(c.args[0]))
        ]
        assert alert_calls, "expected an explicit rejection alert"
        bot._restart_bot_inline.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_allowed_through(self, tmp_path):
        bot = _bot(tmp_path)
        bot._db.create_user(7, "boss", "Owner")
        bot._db.set_role(7, "owner")

        query = _query("health_restart", user_id=7)
        upd = _update(query, user_id=7)

        bot._restart_bot_inline = AsyncMock()
        await bot._handle_callback(upd, _ctx())
        bot._restart_bot_inline.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_premium_user_blocked(self, tmp_path):
        bot = _bot(tmp_path)
        bot._db.create_user(99, "vip", "Premium")
        bot._db.set_role(99, "premium")

        query = _query("health_logs_20", user_id=99)
        upd = _update(query, user_id=99)
        bot._send_logs_inline = AsyncMock()

        await bot._handle_callback(upd, _ctx())
        bot._send_logs_inline.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_owner_can_use_public_callback(self, tmp_path, monkeypatch):
        """A free user must still be able to press public buttons
        (e.g. subscribe view) — gate is owner-only, not blanket."""
        bot = _bot(tmp_path)
        bot._db.create_user(123, "joe", "Free")

        # Stub the subscribe view so we don't pull in handlers_free.
        from src.telegram_bot import bot as bot_mod
        sub = AsyncMock()
        monkeypatch.setattr(bot_mod, "cmd_subscribe", sub)

        query = _query("show_subscribe", user_id=123)
        upd = _update(query, user_id=123)
        await bot._handle_callback(upd, _ctx())
        sub.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_db_error_falls_back_to_block(self, tmp_path):
        """If the role lookup throws, the gate fails CLOSED — better
        to reject the owner once than authorise everyone."""
        bot = _bot(tmp_path)
        bot._db.get_user = MagicMock(side_effect=RuntimeError("synthetic"))

        query = _query("users_page_3", user_id=1)
        upd = _update(query, user_id=1)
        bot._paginate_users = AsyncMock()

        await bot._handle_callback(upd, _ctx())
        bot._paginate_users.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_id_fallback_when_db_record_missing(
        self, tmp_path, monkeypatch,
    ):
        """If the DB doesn't carry the owner's role yet but the env
        OWNER_ID matches, the gate must still let them through."""
        from src.telegram_bot import bot as bot_mod
        monkeypatch.setattr(bot_mod, "OWNER_ID", 555)

        bot = _bot(tmp_path)
        # No record in DB for user 555.

        query = _query("health_refresh", user_id=555)
        upd = _update(query, user_id=555)
        bot._refresh_health = AsyncMock()

        await bot._handle_callback(upd, _ctx())
        bot._refresh_health.assert_awaited_once()
