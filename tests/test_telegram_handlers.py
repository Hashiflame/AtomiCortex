"""Tests for telegram_bot handlers and broadcaster modules."""
from __future__ import annotations
import os, pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.telegram_bot.database import Database
from src.telegram_bot.broadcaster import Broadcaster


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


class TestFreeHandlers:
    @pytest.mark.asyncio
    async def test_start(self, db):
        from src.telegram_bot.handlers_free import cmd_start
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_start(update, ctx)
        update.effective_chat.send_message.assert_awaited_once()
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Привет" in msg

    @pytest.mark.asyncio
    async def test_help(self, db):
        from src.telegram_bot.handlers_free import cmd_help
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_help(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "OWNER" in msg

    @pytest.mark.asyncio
    async def test_stats(self, db):
        from src.telegram_bot.handlers_free import cmd_stats
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_stats(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Win Rate" in msg

    @pytest.mark.asyncio
    async def test_subscribe(self, db):
        from src.telegram_bot.handlers_free import cmd_subscribe
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_subscribe(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Premium" in msg


class TestPremiumHandlers:
    @pytest.mark.asyncio
    async def test_signal_empty(self, db):
        from src.telegram_bot.handlers_premium import cmd_signal
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_signal(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Нет активных" in msg

    @pytest.mark.asyncio
    async def test_signal_with_data(self, db):
        from src.telegram_bot.handlers_premium import cmd_signal
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.log_signal({"symbol": "BTCUSDT", "direction": "long",
            "entry_price": 94000, "stop_loss": 92000, "take_profit": 97000,
            "confidence": 0.73, "regime": "trend"})
        update = _make_update()
        ctx = _make_context(db)
        await cmd_signal(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "LONG" in msg

    @pytest.mark.asyncio
    async def test_history_empty(self, db):
        from src.telegram_bot.handlers_premium import cmd_history
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_history(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "пуста" in msg

    @pytest.mark.asyncio
    async def test_regime(self, db):
        from src.telegram_bot.handlers_premium import cmd_regime
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.log_signal({"symbol": "BTC", "direction": "long", "regime": "trend"})
        update = _make_update()
        ctx = _make_context(db)
        await cmd_regime(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "TREND" in msg

    @pytest.mark.asyncio
    async def test_risk_no_args(self, db):
        from src.telegram_bot.handlers_premium import cmd_risk
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_risk(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Использование" in msg

    @pytest.mark.asyncio
    async def test_risk_with_capital(self, db):
        from src.telegram_bot.handlers_premium import cmd_risk
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["1000"]
        await cmd_risk(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Капитал" in msg

    # TG-010: test negative capital
    @pytest.mark.asyncio
    async def test_risk_negative_capital(self, db):
        from src.telegram_bot.handlers_premium import cmd_risk
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["-1000"]
        await cmd_risk(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "положительным" in msg

    @pytest.mark.asyncio
    async def test_risk_too_small(self, db):
        from src.telegram_bot.handlers_premium import cmd_risk
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["5"]
        await cmd_risk(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Минимальный" in msg


class TestOwnerHandlers:
    @pytest.mark.asyncio
    async def test_users(self, db):
        from src.telegram_bot.handlers_owner import cmd_users
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(222, "bob", "Bob")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_users(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "bob" in msg

    @pytest.mark.asyncio
    async def test_grant(self, db):
        from src.telegram_bot.handlers_owner import cmd_grant
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(222, "bob", "Bob")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["222", "premium", "30d"]
        await cmd_grant(update, ctx)
        assert db.get_user(222)["role"] == "premium"

    # TG-011: test grant owner blocked
    @pytest.mark.asyncio
    async def test_grant_owner_blocked(self, db):
        from src.telegram_bot.handlers_owner import cmd_grant
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(222, "bob", "Bob")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["222", "owner"]
        await cmd_grant(update, ctx)
        assert db.get_user(222)["role"] == "free"  # not changed

    @pytest.mark.asyncio
    async def test_revoke(self, db):
        from src.telegram_bot.handlers_owner import cmd_revoke
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(222, "bob", "Bob")
        db.set_role(222, "premium")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["222"]
        await cmd_revoke(update, ctx)
        assert db.get_user(222)["role"] == "free"

    @pytest.mark.asyncio
    async def test_ban(self, db):
        from src.telegram_bot.handlers_owner import cmd_ban
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(222, "bob", "Bob")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["222"]
        await cmd_ban(update, ctx)
        assert db.get_user(222)["is_banned"] == 1

    @pytest.mark.asyncio
    async def test_ban_owner_blocked(self, db):
        from src.telegram_bot.handlers_owner import cmd_ban
        import src.telegram_bot.handlers_owner as ho
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["12345"]
        with patch.object(ho, "OWNER_ID", 12345):
            await cmd_ban(update, ctx)
        assert db.get_user(12345)["is_banned"] == 0

    @pytest.mark.asyncio
    async def test_stats_admin(self, db):
        from src.telegram_bot.handlers_owner import cmd_stats_admin
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_stats_admin(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Admin" in msg

    # TG-003: test log redaction
    def test_redact_sensitive(self):
        from src.telegram_bot.handlers_owner import _redact_sensitive
        text = 'api_key="abc123" and secret="xyz789"'
        result = _redact_sensitive(text)
        assert "abc123" not in result
        assert "REDACTED" in result

    # TG-007: test broadcast empty users
    @pytest.mark.asyncio
    async def test_broadcast_empty(self, db):
        from src.telegram_bot.handlers_owner import cmd_broadcast
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        # Ban all non-owner users (only owner left, but owner IS non-banned)
        update = _make_update()
        ctx = _make_context(db)
        ctx.args = ["test message"]
        await cmd_broadcast(update, ctx)
        # Should complete without error


class TestBroadcaster:
    @pytest.mark.asyncio
    async def test_broadcast_signal(self, db):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        db.create_user(1, "premium", "P")
        db.set_role(1, "premium")
        db.create_user(2, "free", "F")
        bc = Broadcaster(bot, db)
        await bc.broadcast_signal({"symbol": "BTCUSDT", "direction": "long",
            "entry_price": 94000, "stop_loss": 92000, "take_profit": 97000,
            "confidence": 0.73, "regime": "trend"})
        assert bot.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_signal_role_filtering(self, db):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        db.create_user(1, "premium", "P")
        db.set_role(1, "premium")
        db.create_user(2, "free", "F")
        bc = Broadcaster(bot, db)
        await bc.broadcast_signal({"symbol": "BTC", "direction": "long",
            "entry_price": 94000, "confidence": 0.7, "regime": "trend"})
        calls = bot.send_message.call_args_list
        texts = [c.kwargs.get("text", "") for c in calls]
        assert any("subscribe" in t.lower() for t in texts)

    @pytest.mark.asyncio
    async def test_broadcast_regime_change(self, db):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        db.create_user(1, "premium", "P")
        db.set_role(1, "premium")
        db.create_user(2, "free", "F")
        bc = Broadcaster(bot, db)
        await bc.broadcast_regime_change("range", "trend")
        assert bot.send_message.await_count == 1  # only premium

    @pytest.mark.asyncio
    async def test_broadcast_circuit_breaker(self, db):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        db.create_user(1, "premium", "P")
        db.set_role(1, "premium")
        bc = Broadcaster(bot, db)
        await bc.broadcast_circuit_breaker("Daily loss -3%")
        assert bot.send_message.await_count == 1

    @pytest.mark.asyncio
    async def test_send_to_owner(self, db):
        import src.telegram_bot.broadcaster as bm
        bot = MagicMock()
        bot.send_message = AsyncMock()
        bc = Broadcaster(bot, db)
        with patch.object(bm, "OWNER_ID", 12345):
            await bc.send_to_owner("Critical alert!")
        bot.send_message.assert_awaited_once()

    # TG-015: test send_to_owner with None OWNER_ID
    @pytest.mark.asyncio
    async def test_send_to_owner_none(self, db):
        import src.telegram_bot.broadcaster as bm
        bot = MagicMock()
        bot.send_message = AsyncMock()
        bc = Broadcaster(bot, db)
        with patch.object(bm, "OWNER_ID", None):
            await bc.send_to_owner("Critical alert!")
        bot.send_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_broadcast_skips_banned(self, db):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        db.create_user(1, "premium", "P")
        db.set_role(1, "premium")
        db.ban_user(1)
        bc = Broadcaster(bot, db)
        await bc.broadcast_regime_change("range", "trend")
        assert bot.send_message.await_count == 0

    @pytest.mark.asyncio
    async def test_daily_report(self, db):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        db.create_user(1, "premium", "P")
        db.set_role(1, "premium")
        db.create_user(2, "free", "F")
        metrics = MagicMock()
        metrics.equity = 10500
        metrics.daily_pnl_pct = 0.05
        metrics.total_trades = 10
        metrics.win_rate = 0.7
        metrics.profit_factor = 1.8
        metrics.current_drawdown = 0.02
        metrics.sharpe_ratio = 1.5
        metrics.regime = "TREND"
        bc = Broadcaster(bot, db)
        await bc.broadcast_daily_report(metrics)
        assert bot.send_message.await_count == 2


class TestBotWiring:
    def test_build(self, db, tmp_path):
        with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": "12345"}):
            from src.telegram_bot.bot import TelegramBot
            bot = TelegramBot(
                token="fake:token",
                admin_id=12345,
                db_path=tmp_path / "bot.db",
            )
            app = bot.build()
            assert app is not None
            assert bot.broadcaster is not None
            assert bot.database is not None
