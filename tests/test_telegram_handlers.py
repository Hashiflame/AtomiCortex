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


def _seed_signal(database, **kw):
    """Insert a signal row with sane defaults into a telegram Database."""
    import sqlite3
    from datetime import datetime, timezone
    d = {
        "symbol": "BTCUSDT-PERP.BINANCE", "direction": "long",
        "entry_price": 94000.0, "stop_loss": 92000.0,
        "take_profit": 97000.0, "confidence": 0.73, "regime": "trend_up",
        "timeframe": "4h", "result": "open", "pnl_pct": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    d.update(kw)
    conn = sqlite3.connect(str(database._db_path))
    try:
        conn.execute(
            "INSERT INTO signals_log (symbol,direction,entry_price,"
            "stop_loss,take_profit,confidence,regime,timeframe,result,"
            "pnl_pct,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (d["symbol"], d["direction"], d["entry_price"], d["stop_loss"],
             d["take_profit"], d["confidence"], d["regime"], d["timeframe"],
             d["result"], d["pnl_pct"], d["created_at"]),
        )
        conn.commit()
    finally:
        conn.close()


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
        assert "Нет" in msg and "сигнал" in msg.lower()

    @pytest.mark.asyncio
    async def test_signal_with_data(self, db):
        from src.telegram_bot.handlers_premium import cmd_signal
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        _seed_signal(db, direction="long", regime="trend_up")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_signal(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        kw = update.effective_chat.send_message.call_args.kwargs
        assert "LONG" in msg
        assert kw.get("reply_markup") is not None  # filter keyboard

    @pytest.mark.asyncio
    async def test_signal_shows_last_signal_when_none_open(self, db):
        """All signals closed → still show the most recent (closed) one,
        not '📭 нет сигналов'."""
        from src.telegram_bot.handlers_premium import cmd_signal
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        _seed_signal(db, result="loss", pnl_pct=-1.5, direction="short",
                     regime="trend_down")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_signal(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "SHORT" in msg
        assert "Нет" not in msg

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
    async def test_history_returns_paginated_results(self, db):
        from src.telegram_bot.handlers_premium import cmd_history
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        for i in range(13):
            _seed_signal(db, result="win", pnl_pct=1.0,
                         entry_price=90000 + i)
        update = _make_update()
        ctx = _make_context(db)
        await cmd_history(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        kw = update.effective_chat.send_message.call_args.kwargs
        # 13 signals, per_page=10 → page 1/2, 10 rows shown.
        assert "1/2" in msg and "всего 13" in msg
        assert msg.count("BTC/USDT") == 10
        assert kw.get("reply_markup") is not None

    @pytest.mark.asyncio
    async def test_regime_fallback_to_last_signal(self, db):
        """Regime comes from the most recent signal (bot_metrics is
        unreliable / often UNKNOWN)."""
        from src.telegram_bot.handlers_premium import cmd_regime
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        _seed_signal(db, regime="trend_down", direction="short")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_regime(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Тренд вниз" in msg

    @pytest.mark.asyncio
    async def test_regime_no_signal_is_unknown(self, db):
        """No signals anywhere → '❓ Неизвестно', never raises."""
        from src.telegram_bot.handlers_premium import cmd_regime
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db)
        await cmd_regime(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Неизвестно" in msg

    @pytest.mark.asyncio
    async def test_signals_filter_by_timeframe(self, db, tmp_path):
        """signals_tf filter scopes to one timeframe across the
        isolated DBs."""
        from src.telegram_bot.database import Database
        from src.telegram_bot.handlers_premium import render_signals_view
        p4 = str(tmp_path / "atomicortex.db")
        p15 = str(tmp_path / "atomicortex_15m.db")
        d4, d15 = Database(p4), Database(p15)
        _seed_signal(d4, timeframe="4h", direction="long", regime="trend_up")
        _seed_signal(d15, timeframe="15m", direction="short",
                     regime="orb:trend_down")
        ctx = _make_context(db, shared_db_paths=[p4, p15])
        t_all, _ = render_signals_view(ctx, "all")
        t_15, _ = render_signals_view(ctx, "15m")
        # 15m filter must surface the SHORT/15m signal specifically.
        assert "SHORT" in t_15 and "15M" in t_15
        assert "Нет" not in t_all

    @pytest.mark.asyncio
    async def test_callback_signals_tf_handler(self, db, tmp_path):
        """The inline 'signals_tf:' callback edits the message via the
        bot dispatcher without raising."""
        from src.telegram_bot.database import Database
        from src.telegram_bot.bot import TelegramBot
        p4 = str(tmp_path / "atomicortex.db")
        d4 = Database(p4)
        _seed_signal(d4, direction="long", regime="trend_up")
        with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": "12345"}):
            tg = TelegramBot(token="fake:token", admin_id=12345,
                             db_path=tmp_path / "bot.db")
        update = MagicMock()
        update.callback_query = MagicMock()
        update.callback_query.data = "signals_tf:4h"
        update.callback_query.answer = AsyncMock()
        update.callback_query.edit_message_text = AsyncMock()
        ctx = _make_context(db, shared_db_paths=[p4])
        await tg._handle_callback(update, ctx)
        update.callback_query.edit_message_text.assert_awaited()
        sent = update.callback_query.edit_message_text.call_args[0][0]
        assert "LONG" in sent

    @pytest.mark.asyncio
    async def test_funding_live_success(self, db):
        """/funding fetches Binance premiumIndex and renders top-3 by |rate|."""
        from src.telegram_bot import handlers_premium
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")

        fake_payload = [
            {"symbol": "BTCUSDT", "lastFundingRate": "0.0001"},
            {"symbol": "AAAUSDT", "lastFundingRate": "-0.009"},  # |rate|=0.9%
            {"symbol": "BBBUSDT", "lastFundingRate": "0.0070"},  # 0.70%
            {"symbol": "CCCUSDT", "lastFundingRate": "-0.005"},  # 0.50%
            {"symbol": "DDDUSDT", "lastFundingRate": "0.0001"},
        ]

        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return fake_payload

        class FakeClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url): return FakeResp()

        with patch.object(handlers_premium.httpx, "AsyncClient", FakeClient):
            update = _make_update()
            ctx = _make_context(db)
            await handlers_premium.cmd_funding(update, ctx)

        msg = update.effective_chat.send_message.call_args[0][0]
        # Top-3 by absolute rate: AAA (0.9%), BBB (0.7%), CCC (0.5%)
        assert "AAAUSDT" in msg and "BBBUSDT" in msg and "CCCUSDT" in msg
        assert "BTCUSDT" not in msg and "DDDUSDT" not in msg

    @pytest.mark.asyncio
    async def test_funding_network_error(self, db):
        """Binance fetch failure must not crash the handler."""
        from src.telegram_bot import handlers_premium
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")

        class BoomClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def get(self, url): raise RuntimeError("network down")

        with patch.object(handlers_premium.httpx, "AsyncClient", BoomClient):
            update = _make_update()
            ctx = _make_context(db)
            await handlers_premium.cmd_funding(update, ctx)

        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Не удалось" in msg

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


class TestStatsNoneSafety:
    """Regression: /stats must not crash on NULL cached metrics
    (root cause of the production outage — f"{None:.1f}")."""

    @pytest.mark.asyncio
    async def test_cmd_stats_survives_null_cache_metrics(self, db, tmp_path):
        import sqlite3
        from datetime import datetime, timedelta, timezone
        from scripts.migrate_db_v3 import migrate
        from src.execution.signal_bridge import SignalBridge
        from src.telegram_bot.database import Database
        from src.telegram_bot.handlers_free import cmd_stats

        sdb = str(tmp_path / "atomicortex.db")
        SignalBridge(sdb)
        migrate(sdb)
        now = datetime.now(timezone.utc)
        conn = sqlite3.connect(sdb)
        conn.execute(
            "INSERT INTO signals_log (symbol,direction,entry_price,"
            "stop_loss,take_profit,confidence,regime,timeframe,"
            "created_at,closed_at,pnl_pct,result) VALUES "
            "('BTCUSDT-PERP.BINANCE','short',100,101,98,0.7,'trend_up',"
            "'4h',?,?,1.5,'win')",
            ((now - timedelta(days=3)).isoformat(),
             (now - timedelta(days=3)).isoformat()),
        )
        # Fresh cache rows with NULL numeric metrics → forces the
        # None-formatting path that previously crashed.
        for tf in ("all", "4h"):
            conn.execute(
                "INSERT INTO performance_cache "
                "(timeframe,period_days,symbol,win_rate,profit_factor,"
                "expected_value,total_pnl_pct,max_drawdown,sharpe_ratio,"
                "sortino_ratio,calmar_ratio,updated_at) "
                "VALUES (?,30,'all',NULL,NULL,NULL,NULL,NULL,NULL,"
                "NULL,NULL,?)",
                (tf, now.isoformat()),
            )
        conn.commit()
        conn.close()

        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        update = _make_update()
        ctx = _make_context(db, shared_db_paths=[sdb])
        await cmd_stats(update, ctx)

        update.effective_chat.send_message.assert_awaited_once()
        msg = update.effective_chat.send_message.call_args[0][0]
        # Rendered the real report (not the failure fallback) and
        # placeholders appear instead of a TypeError.
        assert "Статистика" in msg
        assert "временно недоступна" not in msg
        assert "— (мало данных)" in msg  # NULL sharpe
