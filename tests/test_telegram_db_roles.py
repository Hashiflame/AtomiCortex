"""Tests for telegram_bot database and roles modules."""
from __future__ import annotations
import os, pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.telegram_bot.database import Database


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


class TestDatabaseUsers:
    def test_create_and_get_user(self, db):
        db.create_user(111, "alice", "Alice")
        u = db.get_user(111)
        assert u is not None
        assert u["username"] == "alice"
        assert u["role"] == "free"
        assert u["is_banned"] == 0

    def test_get_nonexistent_user(self, db):
        assert db.get_user(999) is None

    def test_create_user_upsert(self, db):
        db.create_user(111, "alice", "Alice")
        db.create_user(111, "alice2", "Alice2")
        assert db.get_user(111)["username"] == "alice2"

    def test_set_role(self, db):
        db.create_user(111, "alice", "Alice")
        db.set_role(111, "premium")
        assert db.get_user(111)["role"] == "premium"

    def test_set_role_with_expiry(self, db):
        db.create_user(111, "a", "A")
        exp = datetime.now(timezone.utc) + timedelta(days=30)
        db.set_role(111, "premium", exp)
        u = db.get_user(111)
        assert u["role"] == "premium"
        assert u["expires_at"] is not None

    def test_auto_downgrade_expired(self, db):
        db.create_user(111, "a", "A")
        past = datetime.now(timezone.utc) - timedelta(days=1)
        db.set_role(111, "premium", past)
        u = db.get_user(111)
        assert u["role"] == "free"
        assert u["expires_at"] is None

    def test_ban_user(self, db):
        db.create_user(111, "a", "A")
        db.ban_user(111)
        assert db.get_user(111)["is_banned"] == 1

    def test_unban_user(self, db):
        db.create_user(111, "a", "A")
        db.ban_user(111)
        db.unban_user(111)
        assert db.get_user(111)["is_banned"] == 0

    def test_get_all_users(self, db):
        db.create_user(1, "a", "A")
        db.create_user(2, "b", "B")
        assert len(db.get_all_users()) == 2

    def test_get_users_by_role(self, db):
        db.create_user(1, "a", "A")
        db.create_user(2, "b", "B")
        db.set_role(2, "premium")
        assert len(db.get_users_by_role("premium")) == 1
        assert len(db.get_users_by_role("free")) == 1

    def test_get_non_banned_users(self, db):
        db.create_user(1, "a", "A")
        db.create_user(2, "b", "B")
        db.ban_user(2)
        assert len(db.get_non_banned_users()) == 1

    def test_set_notes(self, db):
        db.create_user(111, "a", "A")
        db.set_notes(111, "VIP client")
        assert db.get_user(111)["notes"] == "VIP client"

    # TG-014: test SQL-level username lookup
    def test_get_user_by_username(self, db):
        db.create_user(111, "alice", "Alice")
        u = db.get_user_by_username("alice")
        assert u is not None
        assert u["user_id"] == 111

    def test_get_user_by_username_case_insensitive(self, db):
        db.create_user(111, "Alice", "Alice")
        u = db.get_user_by_username("alice")
        assert u is not None

    def test_get_user_by_username_not_found(self, db):
        assert db.get_user_by_username("nobody") is None


class TestDatabaseSignals:
    def test_log_signal(self, db):
        sid = db.log_signal({"symbol": "BTCUSDT", "direction": "long",
            "entry_price": 94000, "stop_loss": 92000, "take_profit": 97000,
            "confidence": 0.73, "regime": "trend"})
        assert sid > 0

    def test_close_signal(self, db):
        sid = db.log_signal({"symbol": "BTCUSDT", "direction": "long",
            "entry_price": 94000, "confidence": 0.7, "regime": "trend"})
        db.close_signal(sid, 2.5, "win")
        signals = db.get_signals_history(1)
        assert signals[0]["result"] == "win"
        assert signals[0]["pnl_pct"] == 2.5

    def test_get_open_signals(self, db):
        db.log_signal({"symbol": "BTC", "direction": "long"})
        db.log_signal({"symbol": "ETH", "direction": "short"})
        sid = db.log_signal({"symbol": "SOL", "direction": "long"})
        db.close_signal(sid, 1.0, "win")
        assert len(db.get_open_signals()) == 2

    def test_get_signals_history_limit(self, db):
        for i in range(5):
            db.log_signal({"symbol": f"SYM{i}", "direction": "long"})
        assert len(db.get_signals_history(3)) == 3

    def test_get_stats_empty(self, db):
        s = db.get_stats()
        assert s["total_trades"] == 0
        assert s["win_rate"] == 0.0

    def test_get_stats_with_data(self, db):
        s1 = db.log_signal({"symbol": "BTC", "direction": "long"})
        s2 = db.log_signal({"symbol": "ETH", "direction": "short"})
        s3 = db.log_signal({"symbol": "SOL", "direction": "long"})
        db.close_signal(s1, 2.0, "win")
        db.close_signal(s2, -1.0, "loss")
        db.close_signal(s3, 3.0, "win")
        s = db.get_stats()
        assert s["wins"] == 2
        assert s["losses"] == 1
        assert s["win_rate"] == pytest.approx(2/3)
        assert s["total_pnl_pct"] == pytest.approx(4.0)

    def test_get_signals_today_count(self, db):
        db.log_signal({"symbol": "BTC", "direction": "long"})
        assert db.get_signals_today_count() >= 0


class TestDatabaseEvents:
    def test_log_and_get_events(self, db):
        db.log_event("signal", "LONG BTC")
        db.log_event("error", "connection timeout")
        events = db.get_events(10)
        assert len(events) == 2
        event_types = {e["event_type"] for e in events}
        assert "signal" in event_types
        assert "error" in event_types


class TestRoles:
    @pytest.fixture(autouse=True)
    def setup_owner(self):
        with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": "12345"}):
            import importlib
            import src.telegram_bot.roles as roles_mod
            roles_mod.OWNER_ID = 12345
            self.roles = roles_mod
            yield

    def _make_update(self, user_id=111, username="test", first_name="Test"):
        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = user_id
        update.effective_user.username = username
        update.effective_user.first_name = first_name
        update.effective_chat = MagicMock()
        update.effective_chat.send_message = AsyncMock()
        return update

    def _make_context(self, db):
        ctx = MagicMock()
        ctx.bot_data = {"db": db}
        return ctx

    @pytest.mark.asyncio
    async def test_require_role_free_passes(self, db):
        db.create_user(111, "test", "Test")
        handler = self.roles.require_role("free")(AsyncMock())
        update = self._make_update()
        ctx = self._make_context(db)
        await handler(update, ctx)
        handler.__wrapped__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_require_role_premium_blocks_free(self, db):
        db.create_user(111, "test", "Test")
        inner = AsyncMock()
        handler = self.roles.require_role("premium")(inner)
        update = self._make_update()
        ctx = self._make_context(db)
        await handler(update, ctx)
        inner.assert_not_awaited()
        update.effective_chat.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_owner_bypasses_all(self, db):
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        inner = AsyncMock()
        handler = self.roles.require_role("owner")(inner)
        update = self._make_update(user_id=12345, username="owner")
        ctx = self._make_context(db)
        await handler(update, ctx)
        inner.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_banned_user_blocked(self, db):
        db.create_user(111, "test", "Test")
        db.ban_user(111)
        inner = AsyncMock()
        handler = self.roles.require_role("free")(inner)
        update = self._make_update()
        ctx = self._make_context(db)
        await handler(update, ctx)
        inner.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_auto_registration(self, db):
        inner = AsyncMock()
        handler = self.roles.require_role("free")(inner)
        update = self._make_update(user_id=999, username="newuser")
        ctx = self._make_context(db)
        await handler(update, ctx)
        assert db.get_user(999) is not None
        inner.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_owner_auto_role(self, db):
        inner = AsyncMock()
        handler = self.roles.require_role("free")(inner)
        update = self._make_update(user_id=12345, username="owner")
        ctx = self._make_context(db)
        await handler(update, ctx)
        u = db.get_user(12345)
        assert u["role"] == "owner"

    @pytest.mark.asyncio
    async def test_require_not_banned_passes(self, db):
        db.create_user(111, "test", "Test")
        inner = AsyncMock()
        handler = self.roles.require_not_banned(inner)
        update = self._make_update()
        ctx = self._make_context(db)
        await handler(update, ctx)
        inner.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_require_not_banned_blocks(self, db):
        db.create_user(111, "test", "Test")
        db.ban_user(111)
        inner = AsyncMock()
        handler = self.roles.require_not_banned(inner)
        update = self._make_update()
        ctx = self._make_context(db)
        await handler(update, ctx)
        inner.assert_not_awaited()

    # TG-001: test OWNER_ID=None behavior
    @pytest.mark.asyncio
    async def test_owner_id_none_no_auto_owner(self, db):
        self.roles.OWNER_ID = None
        inner = AsyncMock()
        handler = self.roles.require_role("free")(inner)
        update = self._make_update(user_id=12345, username="owner")
        ctx = self._make_context(db)
        await handler(update, ctx)
        u = db.get_user(12345)
        assert u["role"] == "free"  # not auto-promoted
