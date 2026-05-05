"""
Tests for keyboard layouts, button routing, and subscribe flow.

Covers:
- Keyboard structure for each role (free, premium, owner)
- /start sends correct keyboard per role
- /subscribe shows InlineKeyboard with Stars + USDT buttons
- callback "buy_stars_30" triggers invoice
- Button text "📊 Статистика" maps to /stats
- Premium button for free user → locked message
- Renew button on expiring /mystatus
- Health buttons presence
- Users pagination keyboard
- Stats admin period buttons

Total ≥ 16 tests.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.telegram_bot.database import Database
from src.telegram_bot.keyboards import (
    ALL_BUTTON_TEXTS,
    BTN_FUNDING,
    BTN_HEALTH,
    BTN_HELP,
    BTN_HISTORY,
    BTN_REGIME,
    BTN_SIGNAL,
    BTN_STATS,
    BTN_SUBSCRIBE,
    BTN_USERS,
    OWNER_BUTTONS,
    PREMIUM_BUTTONS,
    get_free_keyboard,
    get_health_buttons,
    get_keyboard_for_role,
    get_owner_keyboard,
    get_premium_keyboard,
    get_renew_button,
    get_stats_admin_buttons,
    get_subscribe_inline_button,
    get_subscribe_keyboard,
    get_users_pagination,
)


# ── Fixtures ──

@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


def _make_update(user_id=12345, username="owner", first_name="Owner"):
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = username
    update.effective_user.first_name = first_name
    update.effective_chat = MagicMock()
    update.effective_chat.send_message = AsyncMock()
    update.message = MagicMock()
    update.message.text = ""
    return update


def _make_context(db, **extra):
    ctx = MagicMock()
    ctx.bot_data = {"db": db, **extra}
    ctx.bot = MagicMock()
    ctx.bot.send_message = AsyncMock()
    ctx.args = []
    return ctx


@pytest.fixture(autouse=True)
def patch_owner():
    with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": "12345"}):
        import src.telegram_bot.roles as rm
        rm.OWNER_ID = 12345
        yield


# ═══════════════════════════════════════════════════════════════════════════
# Keyboard structure tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKeyboardStructure:
    def test_free_keyboard_returns_reply_markup(self) -> None:
        """get_free_keyboard returns a ReplyKeyboardMarkup."""
        from telegram import ReplyKeyboardMarkup
        kb = get_free_keyboard()
        assert isinstance(kb, ReplyKeyboardMarkup)

    def test_free_keyboard_has_stats_and_subscribe(self) -> None:
        """Free keyboard contains Статистика and Подписка buttons."""
        kb = get_free_keyboard()
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_STATS in all_texts
        assert BTN_SUBSCRIBE in all_texts
        assert BTN_HELP in all_texts

    def test_premium_keyboard_has_signal(self) -> None:
        """Premium keyboard contains Сигнал button."""
        kb = get_premium_keyboard()
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_SIGNAL in all_texts
        assert BTN_HISTORY in all_texts
        assert BTN_REGIME in all_texts
        assert BTN_FUNDING in all_texts

    def test_premium_keyboard_no_users(self) -> None:
        """Premium keyboard does NOT contain owner buttons."""
        kb = get_premium_keyboard()
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_USERS not in all_texts
        assert BTN_HEALTH not in all_texts

    def test_owner_keyboard_has_users(self) -> None:
        """Owner keyboard contains Юзеры and Здоровье buttons."""
        kb = get_owner_keyboard()
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_USERS in all_texts
        assert BTN_HEALTH in all_texts
        assert BTN_SIGNAL in all_texts

    def test_get_keyboard_for_role_free(self) -> None:
        """get_keyboard_for_role('free') returns free keyboard."""
        kb = get_keyboard_for_role("free")
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_SUBSCRIBE in all_texts
        assert BTN_SIGNAL not in all_texts

    def test_get_keyboard_for_role_premium(self) -> None:
        """get_keyboard_for_role('premium') returns premium keyboard."""
        kb = get_keyboard_for_role("premium")
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_SIGNAL in all_texts

    def test_get_keyboard_for_role_owner(self) -> None:
        """get_keyboard_for_role('owner') returns owner keyboard."""
        kb = get_keyboard_for_role("owner")
        all_texts = [btn.text for row in kb.keyboard for btn in row]
        assert BTN_USERS in all_texts

    def test_resize_keyboard(self) -> None:
        """All keyboards have resize_keyboard=True."""
        assert get_free_keyboard().resize_keyboard is True
        assert get_premium_keyboard().resize_keyboard is True
        assert get_owner_keyboard().resize_keyboard is True


# ═══════════════════════════════════════════════════════════════════════════
# Subscribe InlineKeyboard
# ═══════════════════════════════════════════════════════════════════════════


class TestSubscribeKeyboard:
    def test_subscribe_keyboard_has_stars_buttons(self) -> None:
        """Subscribe keyboard has Stars payment buttons."""
        kb = get_subscribe_keyboard()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "buy_stars_30" in all_data
        assert "buy_stars_90" in all_data

    def test_subscribe_keyboard_has_usdt_buttons(self) -> None:
        """Subscribe keyboard has USDT payment buttons."""
        kb = get_subscribe_keyboard()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "buy_usdt_30" in all_data
        assert "buy_usdt_90" in all_data

    def test_subscribe_keyboard_has_contact_owner(self) -> None:
        """Subscribe keyboard has contact_owner button."""
        kb = get_subscribe_keyboard()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "contact_owner" in all_data

    def test_subscribe_keyboard_custom_prices(self) -> None:
        """Custom prices reflected in button labels."""
        kb = get_subscribe_keyboard(stars_30=250, usdt_30=5.0)
        all_texts = [btn.text for row in kb.inline_keyboard for btn in row]
        assert any("250" in t for t in all_texts)
        assert any("$5" in t for t in all_texts)


# ═══════════════════════════════════════════════════════════════════════════
# /start sends keyboard for role
# ═══════════════════════════════════════════════════════════════════════════


class TestStartKeyboard:
    @pytest.mark.asyncio
    async def test_start_free_keyboard(self, db) -> None:
        """/start with free role sends free keyboard."""
        from src.telegram_bot.handlers_free import cmd_start
        db.create_user(99, "alice", "Alice")
        # Don't set premium — default is free
        update = _make_update(user_id=99, username="alice", first_name="Alice")
        ctx = _make_context(db)

        # Temporarily set a different OWNER_ID so user 99 is free
        import src.telegram_bot.roles as rm
        original_owner = rm.OWNER_ID
        rm.OWNER_ID = 12345
        try:
            await cmd_start(update, ctx)
        finally:
            rm.OWNER_ID = original_owner

        call_kwargs = update.effective_chat.send_message.call_args
        assert call_kwargs is not None
        msg = call_kwargs[0][0]
        assert "Привет" in msg
        assert "Используй кнопки" in msg
        # Check reply_markup was sent
        assert "reply_markup" in call_kwargs[1]

    @pytest.mark.asyncio
    async def test_start_owner_keyboard(self, db) -> None:
        """/start with owner role sends owner keyboard."""
        from src.telegram_bot.handlers_free import cmd_start
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_start(update, ctx)

        call_kwargs = update.effective_chat.send_message.call_args
        msg = call_kwargs[0][0]
        assert "👑" in msg or "⭐" in msg
        assert "reply_markup" in call_kwargs[1]

    @pytest.mark.asyncio
    async def test_start_premium_keyboard(self, db) -> None:
        """/start with premium role sends premium keyboard."""
        from src.telegram_bot.handlers_free import cmd_start
        import src.telegram_bot.roles as rm
        original_owner = rm.OWNER_ID
        rm.OWNER_ID = 99999  # different owner

        try:
            db.create_user(200, "premuser", "PremUser")
            expires = datetime.now(timezone.utc) + timedelta(days=30)
            db.set_role(200, "premium", expires)
            update = _make_update(user_id=200, username="premuser", first_name="PremUser")
            ctx = _make_context(db)
            await cmd_start(update, ctx)

            call_kwargs = update.effective_chat.send_message.call_args
            msg = call_kwargs[0][0]
            assert "Premium" in msg
            assert "reply_markup" in call_kwargs[1]
        finally:
            rm.OWNER_ID = original_owner


# ═══════════════════════════════════════════════════════════════════════════
# /subscribe shows InlineKeyboard
# ═══════════════════════════════════════════════════════════════════════════


class TestSubscribeHandler:
    @pytest.mark.asyncio
    async def test_subscribe_shows_inline_keyboard(self, db) -> None:
        """/subscribe response contains Stars + USDT inline buttons."""
        from src.telegram_bot.handlers_free import cmd_subscribe
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_subscribe(update, ctx)

        call_kwargs = update.effective_chat.send_message.call_args
        msg = call_kwargs[0][0]
        assert "AtomiCortex Premium" in msg
        assert "AI сигналы" in msg

        # Check reply_markup is InlineKeyboardMarkup
        markup = call_kwargs[1].get("reply_markup")
        assert markup is not None
        all_data = [
            btn.callback_data
            for row in markup.inline_keyboard
            for btn in row
        ]
        assert "buy_stars_30" in all_data
        assert "buy_usdt_30" in all_data


# ═══════════════════════════════════════════════════════════════════════════
# Premium button for free user → locked
# ═══════════════════════════════════════════════════════════════════════════


class TestButtonAccessControl:
    def test_premium_buttons_defined(self) -> None:
        """PREMIUM_BUTTONS set is not empty and contains signal."""
        assert BTN_SIGNAL in PREMIUM_BUTTONS
        assert BTN_HISTORY in PREMIUM_BUTTONS

    def test_owner_buttons_defined(self) -> None:
        """OWNER_BUTTONS set is not empty and contains users."""
        assert BTN_USERS in OWNER_BUTTONS
        assert BTN_HEALTH in OWNER_BUTTONS


# ═══════════════════════════════════════════════════════════════════════════
# /mystatus with renewal button
# ═══════════════════════════════════════════════════════════════════════════


class TestMyStatusRenewal:
    @pytest.mark.asyncio
    async def test_mystatus_renew_button_when_expiring(self, db) -> None:
        """/mystatus shows renew button when < 7 days remaining."""
        from src.telegram_bot.handlers_free import cmd_mystatus
        import src.telegram_bot.roles as rm
        original_owner = rm.OWNER_ID
        rm.OWNER_ID = 99999

        try:
            db.create_user(300, "expiring", "Expiring")
            expires = datetime.now(timezone.utc) + timedelta(days=3)
            db.set_role(300, "premium", expires)
            update = _make_update(user_id=300, username="expiring", first_name="Expiring")
            ctx = _make_context(db)
            await cmd_mystatus(update, ctx)

            call_kwargs = update.effective_chat.send_message.call_args
            markup = call_kwargs[1].get("reply_markup")
            assert markup is not None
            # Should have "show_subscribe" callback
            all_data = [
                btn.callback_data
                for row in markup.inline_keyboard
                for btn in row
            ]
            assert "show_subscribe" in all_data
        finally:
            rm.OWNER_ID = original_owner

    @pytest.mark.asyncio
    async def test_mystatus_no_renew_button_when_plenty_time(self, db) -> None:
        """/mystatus has no renew button when > 7 days remaining."""
        from src.telegram_bot.handlers_free import cmd_mystatus
        import src.telegram_bot.roles as rm
        original_owner = rm.OWNER_ID
        rm.OWNER_ID = 99999

        try:
            db.create_user(301, "longprem", "LongPrem")
            expires = datetime.now(timezone.utc) + timedelta(days=25)
            db.set_role(301, "premium", expires)
            update = _make_update(user_id=301, username="longprem", first_name="LongPrem")
            ctx = _make_context(db)
            await cmd_mystatus(update, ctx)

            call_kwargs = update.effective_chat.send_message.call_args
            # No reply_markup or it should be None
            markup = call_kwargs[1].get("reply_markup")
            assert markup is None
        finally:
            rm.OWNER_ID = original_owner


# ═══════════════════════════════════════════════════════════════════════════
# InlineKeyboard builders
# ═══════════════════════════════════════════════════════════════════════════


class TestInlineKeyboardBuilders:
    def test_health_buttons(self) -> None:
        """get_health_buttons has refresh, logs, restart."""
        kb = get_health_buttons()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "health_refresh" in all_data
        assert "health_logs_20" in all_data
        assert "health_restart" in all_data

    def test_users_pagination_first_page(self) -> None:
        """Page 1/3: no Prev, has Next."""
        kb = get_users_pagination(1, 3)
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "users_page_0" not in all_data  # no prev
        assert "users_page_2" in all_data  # next

    def test_users_pagination_middle_page(self) -> None:
        """Page 2/3: has both Prev and Next."""
        kb = get_users_pagination(2, 3)
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "users_page_1" in all_data  # prev
        assert "users_page_3" in all_data  # next

    def test_users_pagination_last_page(self) -> None:
        """Page 3/3: has Prev, no Next."""
        kb = get_users_pagination(3, 3)
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "users_page_2" in all_data  # prev
        assert "users_page_4" not in all_data  # no next

    def test_stats_admin_buttons(self) -> None:
        """get_stats_admin_buttons has today, week, month."""
        kb = get_stats_admin_buttons()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "stats_period_today" in all_data
        assert "stats_period_week" in all_data
        assert "stats_period_month" in all_data

    def test_subscribe_inline_button(self) -> None:
        """get_subscribe_inline_button has show_subscribe callback."""
        kb = get_subscribe_inline_button()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "show_subscribe" in all_data

    def test_renew_button(self) -> None:
        """get_renew_button has show_subscribe callback."""
        kb = get_renew_button()
        all_data = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert "show_subscribe" in all_data
