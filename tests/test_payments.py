"""Tests for AtomiCortex payment system — Stars + CryptoBot."""
from __future__ import annotations

import os
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.telegram_bot.database import Database
from src.telegram_bot.payments_stars import (
    _parse_payload,
    send_invoice_stars,
    pre_checkout_handler,
    successful_payment_handler,
    _STARS_TO_USD,
)
from src.telegram_bot.payments_crypto import CryptoBotPayment


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test_pay.db")


@pytest.fixture(autouse=True)
def patch_owner():
    with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": "12345"}):
        import src.telegram_bot.roles as rm
        rm.OWNER_ID = 12345
        import src.telegram_bot.payments_stars as ps
        ps.OWNER_ID = 12345
        import src.telegram_bot.payments_crypto as pc
        pc.OWNER_ID = 12345
        yield


def _make_update(user_id=111, username="alice", first_name="Alice"):
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = username
    update.effective_user.first_name = first_name
    update.effective_chat = MagicMock()
    update.effective_chat.id = user_id
    update.effective_chat.send_message = AsyncMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.pre_checkout_query = None
    update.callback_query = None
    return update


def _make_context(db, **extra):
    ctx = MagicMock()
    ctx.bot_data = {"db": db, **extra}
    ctx.bot = MagicMock()
    ctx.bot.send_invoice = AsyncMock()
    ctx.bot.send_message = AsyncMock()
    ctx.args = []
    return ctx


# ═══════════════════════════════════════
# 1. Stars invoice: currency=XTR, provider_token=""
# ═══════════════════════════════════════

class TestStarsInvoice:
    @pytest.mark.asyncio
    async def test_stars_invoice_currency_xtr(self, db):
        """send_invoice must use currency=XTR and provider_token=''."""
        db.create_user(111, "alice", "Alice")
        update = _make_update()
        ctx = _make_context(db)

        await send_invoice_stars(update, ctx, days=30, price_stars=500)

        ctx.bot.send_invoice.assert_awaited_once()
        kw = ctx.bot.send_invoice.call_args[1]
        assert kw["currency"] == "XTR"
        assert kw["provider_token"] == ""
        assert kw["chat_id"] == 111

    @pytest.mark.asyncio
    async def test_stars_invoice_creates_db_record(self, db):
        """send_invoice_stars creates a pending payment in DB."""
        db.create_user(111, "alice", "Alice")
        update = _make_update()
        ctx = _make_context(db)

        await send_invoice_stars(update, ctx, days=30, price_stars=500)

        payments = db.get_payments()
        assert len(payments) == 1
        p = payments[0]
        assert p["user_id"] == 111
        assert p["method"] == "stars"
        assert p["days"] == 30
        assert p["stars_amount"] == 500
        assert p["status"] == "pending"
        assert "premium_30d_111" in p["payload"]

    @pytest.mark.asyncio
    async def test_owner_cannot_pay_self(self, db):
        """Owner paying via Stars gets a warning, no invoice sent."""
        db.create_user(12345, "owner", "Owner")
        update = _make_update(user_id=12345, username="owner")
        ctx = _make_context(db)

        await send_invoice_stars(update, ctx, days=30, price_stars=500)

        ctx.bot.send_invoice.assert_not_awaited()
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Владелец" in msg


# ═══════════════════════════════════════
# 2. Stars payload parsing
# ═══════════════════════════════════════

class TestPayloadParsing:
    def test_valid_30d(self):
        assert _parse_payload("premium_30d_111") == (30, 111)

    def test_valid_90d(self):
        assert _parse_payload("premium_90d_222") == (90, 222)

    def test_invalid_prefix(self):
        assert _parse_payload("free_30d_111") is None

    def test_invalid_format(self):
        assert _parse_payload("premium_bad") is None

    def test_empty(self):
        assert _parse_payload("") is None

    def test_negative_days(self):
        assert _parse_payload("premium_-5d_111") is None

    def test_zero_user(self):
        assert _parse_payload("premium_30d_0") is None


# ═══════════════════════════════════════
# 3. successful_payment activates premium
# ═══════════════════════════════════════

class TestSuccessfulPayment:
    @pytest.mark.asyncio
    async def test_payment_activates_premium(self, db):
        """Successful Stars payment sets user to premium."""
        db.create_user(111, "alice", "Alice")
        db.create_payment(111, "stars", 6.50, 30,
                         payload="premium_30d_111", stars_amount=500)

        update = _make_update()
        update.message.successful_payment = MagicMock()
        update.message.successful_payment.invoice_payload = "premium_30d_111"
        update.message.successful_payment.total_amount = 500

        ctx = _make_context(db)
        await successful_payment_handler(update, ctx)

        user = db.get_user(111)
        assert user["role"] == "premium"
        assert user["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_payment_logs_in_db(self, db):
        """Successful payment updates payment status to 'paid'."""
        db.create_user(111, "alice", "Alice")
        db.create_payment(111, "stars", 6.50, 30,
                         payload="premium_30d_111", stars_amount=500)

        update = _make_update()
        update.message.successful_payment = MagicMock()
        update.message.successful_payment.invoice_payload = "premium_30d_111"
        update.message.successful_payment.total_amount = 500

        ctx = _make_context(db)
        await successful_payment_handler(update, ctx)

        p = db.get_payment_by_payload("premium_30d_111")
        assert p["status"] == "paid"
        assert p["paid_at"] is not None

    @pytest.mark.asyncio
    async def test_duplicate_payment_ignored(self, db):
        """Already-paid payment should not extend again."""
        db.create_user(111, "alice", "Alice")
        now = datetime.now(timezone.utc)
        pid = db.create_payment(111, "stars", 6.50, 30,
                               payload="premium_30d_111", stars_amount=500)
        db.update_payment_status(pid, "paid", now.isoformat())

        update = _make_update()
        update.message.successful_payment = MagicMock()
        update.message.successful_payment.invoice_payload = "premium_30d_111"
        update.message.successful_payment.total_amount = 500

        ctx = _make_context(db)
        await successful_payment_handler(update, ctx)

        msg = update.message.reply_text.call_args[0][0]
        assert "уже был обработан" in msg

    @pytest.mark.asyncio
    async def test_payment_with_bad_payload_no_crash(self, db):
        """Bad payload in successful payment doesn't crash."""
        update = _make_update()
        update.message.successful_payment = MagicMock()
        update.message.successful_payment.invoice_payload = "garbage_payload"
        update.message.successful_payment.total_amount = 500

        ctx = _make_context(db)
        # Should not raise
        await successful_payment_handler(update, ctx)
        update.message.reply_text.assert_not_awaited()


# ═══════════════════════════════════════
# 4. Pre-checkout handler
# ═══════════════════════════════════════

class TestPreCheckout:
    @pytest.mark.asyncio
    async def test_valid_payload_approved(self, db):
        update = _make_update()
        update.pre_checkout_query = MagicMock()
        update.pre_checkout_query.invoice_payload = "premium_30d_111"
        update.pre_checkout_query.answer = AsyncMock()

        ctx = _make_context(db)
        await pre_checkout_handler(update, ctx)
        update.pre_checkout_query.answer.assert_awaited_once_with(ok=True)

    @pytest.mark.asyncio
    async def test_invalid_payload_rejected(self, db):
        update = _make_update()
        update.pre_checkout_query = MagicMock()
        update.pre_checkout_query.invoice_payload = "garbage"
        update.pre_checkout_query.answer = AsyncMock()

        ctx = _make_context(db)
        await pre_checkout_handler(update, ctx)
        call_kwargs = update.pre_checkout_query.answer.call_args[1]
        assert call_kwargs["ok"] is False


# ═══════════════════════════════════════
# 5. CryptoBot create_invoice — correct endpoint
# ═══════════════════════════════════════

class TestCryptoBotInvoice:
    @pytest.mark.asyncio
    async def test_create_invoice_calls_correct_endpoint(self, db):
        """create_invoice POSTs to /createInvoice with string amount."""
        crypto = CryptoBotPayment("fake_token", db)
        mock_resp = {
            "ok": True,
            "result": {
                "invoice_id": "INV123",
                "bot_invoice_url": "https://t.me/CryptoBot?start=INV123",
            },
        }

        with patch("aiohttp.ClientSession") as mock_session:
            session_cm = AsyncMock()
            resp_cm = AsyncMock()
            resp_cm.__aenter__.return_value.json = AsyncMock(return_value=mock_resp)
            session_cm.__aenter__.return_value.post = MagicMock(return_value=resp_cm)
            mock_session.return_value = session_cm

            url = await crypto.create_invoice(111, 30, 7.00)

        assert url == "https://t.me/CryptoBot?start=INV123"

        # Verify the POST body used string amount
        post_call = session_cm.__aenter__.return_value.post
        post_call.assert_called_once()
        call_kwargs = post_call.call_args[1]
        assert call_kwargs["json"]["amount"] == "7.00"
        assert call_kwargs["json"]["asset"] == "USDT"

    @pytest.mark.asyncio
    async def test_create_invoice_records_in_db(self, db):
        """create_invoice creates a pending payment in DB."""
        crypto = CryptoBotPayment("fake_token", db)
        mock_resp = {
            "ok": True,
            "result": {
                "invoice_id": "INV123",
                "bot_invoice_url": "https://t.me/CryptoBot?start=INV123",
            },
        }

        with patch("aiohttp.ClientSession") as mock_session:
            session_cm = AsyncMock()
            resp_cm = AsyncMock()
            resp_cm.__aenter__.return_value.json = AsyncMock(return_value=mock_resp)
            session_cm.__aenter__.return_value.post = MagicMock(return_value=resp_cm)
            mock_session.return_value = session_cm

            await crypto.create_invoice(111, 30, 7.00)

        payments = db.get_payments()
        assert len(payments) == 1
        assert payments[0]["method"] == "usdt"
        assert payments[0]["invoice_id"] == "INV123"

    @pytest.mark.asyncio
    async def test_create_invoice_api_failure(self, db):
        """CryptoBot API failure returns None."""
        crypto = CryptoBotPayment("fake_token", db)

        with patch("aiohttp.ClientSession") as mock_session:
            session_cm = AsyncMock()
            resp_cm = AsyncMock()
            resp_cm.__aenter__.return_value.json = AsyncMock(
                return_value={"ok": False, "error": {"code": 401}}
            )
            session_cm.__aenter__.return_value.post = MagicMock(return_value=resp_cm)
            mock_session.return_value = session_cm

            url = await crypto.create_invoice(111, 30, 7.00)

        assert url is None


# ═══════════════════════════════════════
# 6. poll_payments finds paid invoices
# ═══════════════════════════════════════

class TestCryptoBotPolling:
    @pytest.mark.asyncio
    async def test_poll_finds_paid_invoices(self, db):
        """poll_payments activates premium for paid invoices."""
        db.create_user(222, "bob", "Bob")
        db.create_payment(222, "usdt", 7.00, 30,
                         payload="premium_30d_222", invoice_id="INV456")

        crypto = CryptoBotPayment("fake_token", db, bot=MagicMock())
        crypto._bot.send_message = AsyncMock()

        with patch.object(crypto, "get_paid_invoices", return_value=[{
            "invoice_id": "INV456",
            "status": "paid",
            "payload": "premium_30d_222",
            "amount": "7.00",
        }]):
            activated = await crypto.poll_payments()

        assert activated == 1
        user = db.get_user(222)
        assert user["role"] == "premium"

    @pytest.mark.asyncio
    async def test_poll_does_not_process_twice(self, db):
        """Already-processed invoice IDs are skipped."""
        db.create_user(222, "bob", "Bob")
        db.create_payment(222, "usdt", 7.00, 30,
                         payload="premium_30d_222", invoice_id="INV456")

        crypto = CryptoBotPayment("fake_token", db, bot=MagicMock())
        crypto._bot.send_message = AsyncMock()

        paid_invoices = [{
            "invoice_id": "INV456",
            "status": "paid",
            "payload": "premium_30d_222",
            "amount": "7.00",
        }]

        with patch.object(crypto, "get_paid_invoices", return_value=paid_invoices):
            activated1 = await crypto.poll_payments()
            activated2 = await crypto.poll_payments()

        assert activated1 == 1
        assert activated2 == 0  # second poll skips (in _processed_ids)


# ═══════════════════════════════════════
# 7. /mystatus
# ═══════════════════════════════════════

class TestMyStatus:
    @pytest.mark.asyncio
    async def test_mystatus_premium_shows_remaining(self, db):
        from src.telegram_bot.handlers_free import cmd_mystatus
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(111, "alice", "Alice")
        future = datetime.now(timezone.utc) + timedelta(days=25)
        db.set_role(111, "premium", future)

        update = _make_update()
        ctx = _make_context(db)
        await cmd_mystatus(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Premium" in msg
        assert "дней" in msg

    @pytest.mark.asyncio
    async def test_mystatus_free_shows_subscribe(self, db):
        from src.telegram_bot.handlers_free import cmd_mystatus
        db.create_user(111, "alice", "Alice")

        update = _make_update()
        ctx = _make_context(db)
        await cmd_mystatus(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Free" in msg
        assert "/subscribe" in msg

    @pytest.mark.asyncio
    async def test_mystatus_expired_downgrades(self, db):
        """Expired premium auto-downgrades to free."""
        db.create_user(111, "alice", "Alice")
        past = datetime.now(timezone.utc) - timedelta(days=1)
        db.set_role(111, "premium", past)

        user = db.get_user(111)  # triggers auto-downgrade
        assert user["role"] == "free"


# ═══════════════════════════════════════
# 8. /payments only for owner
# ═══════════════════════════════════════

class TestOwnerPaymentCommands:
    @pytest.mark.asyncio
    async def test_payments_only_for_owner(self, db):
        """Free user blocked from /payments."""
        from src.telegram_bot.handlers_owner import cmd_payments
        db.create_user(111, "alice", "Alice")
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")

        update = _make_update(user_id=111)
        ctx = _make_context(db)
        await cmd_payments(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        # Decorator sends access denied message
        assert "подписки" in msg.lower() or "команда" in msg.lower() or "доступ" in msg.lower()

    @pytest.mark.asyncio
    async def test_payments_shows_history(self, db):
        from src.telegram_bot.handlers_owner import cmd_payments
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_payment(111, "stars", 6.50, 30,
                         payload="pay_test", stars_amount=500)

        update = _make_update(user_id=12345, username="owner")
        ctx = _make_context(db)
        await cmd_payments(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Платежи" in msg


# ═══════════════════════════════════════
# 9. /revenue stats correct
# ═══════════════════════════════════════

class TestRevenue:
    @pytest.mark.asyncio
    async def test_revenue_stats_correct(self, db):
        from src.telegram_bot.handlers_owner import cmd_revenue
        db.create_user(12345, "owner", "Owner")
        db.set_role(12345, "owner")
        db.create_user(1, "a", "A")
        db.set_role(1, "premium")
        now = datetime.now(timezone.utc).isoformat()

        p1 = db.create_payment(1, "stars", 6.50, 30,
                              payload="r1", stars_amount=500)
        db.update_payment_status(p1, "paid", now)
        p2 = db.create_payment(2, "usdt", 7.00, 30, payload="r2")
        db.update_payment_status(p2, "paid", now)

        rev = db.get_revenue_stats()
        assert rev["total_usd"] == pytest.approx(13.50)
        assert rev["stars_total"] == 500
        assert rev["usdt_total"] == pytest.approx(7.00)
        assert rev["active_premiums"] == 1
        assert rev["total_payments"] == 2

        update = _make_update(user_id=12345, username="owner")
        ctx = _make_context(db)
        await cmd_revenue(update, ctx)
        msg = update.effective_chat.send_message.call_args[0][0]
        assert "Revenue" in msg


# ═══════════════════════════════════════
# 10. /subscribe shows buttons
# ═══════════════════════════════════════

class TestSubscribeButtons:
    @pytest.mark.asyncio
    async def test_subscribe_shows_inline_keyboard(self, db):
        from src.telegram_bot.handlers_free import cmd_subscribe
        db.create_user(111, "alice", "Alice")

        update = _make_update()
        ctx = _make_context(db, prices={
            "stars_30d": 500, "stars_90d": 1200,
            "usdt_30d": 7.00, "usdt_90d": 18.00,
        })

        await cmd_subscribe(update, ctx)
        call_kwargs = update.effective_chat.send_message.call_args[1]
        keyboard = call_kwargs["reply_markup"]
        # Should have 3 rows: [Stars30, Stars90], [USDT30, USDT90], [Manual]
        assert len(keyboard.inline_keyboard) == 3
        assert "Stars" in keyboard.inline_keyboard[0][0].text
        assert "USDT" in keyboard.inline_keyboard[1][0].text

    @pytest.mark.asyncio
    async def test_subscribe_callback_stars_triggers_invoice(self, db):
        """Clicking Stars button should trigger send_invoice."""
        # This tests the callback data format matches expectations
        from src.telegram_bot.handlers_free import cmd_subscribe
        db.create_user(111, "alice", "Alice")

        update = _make_update()
        ctx = _make_context(db, prices={
            "stars_30d": 500, "stars_90d": 1200,
            "usdt_30d": 7.00, "usdt_90d": 18.00,
        })

        await cmd_subscribe(update, ctx)
        keyboard = update.effective_chat.send_message.call_args[1]["reply_markup"]
        # Verify callback_data for Stars buttons
        assert keyboard.inline_keyboard[0][0].callback_data == "buy_stars_30"
        assert keyboard.inline_keyboard[0][1].callback_data == "buy_stars_90"


# ═══════════════════════════════════════
# 11. Database payment CRUD
# ═══════════════════════════════════════

class TestDatabasePayments:
    def test_create_payment_with_status(self, db):
        """create_payment supports status param for direct 'paid'."""
        pid = db.create_payment(
            111, "stars", 6.50, 30,
            payload="direct", stars_amount=500, status="paid",
        )
        p = db.get_payment_by_payload("direct")
        assert p["status"] == "paid"

    def test_get_pending_payments(self, db):
        db.create_payment(1, "stars", 6.50, 30, payload="p1")
        db.create_payment(2, "usdt", 7.00, 30, payload="p2")
        pid3 = db.create_payment(3, "stars", 6.50, 30, payload="p3")
        db.update_payment_status(pid3, "paid")

        pending = db.get_pending_payments()
        assert len(pending) == 2
        pending_usdt = db.get_pending_payments(method="usdt")
        assert len(pending_usdt) == 1

    def test_get_payments_limit(self, db):
        for i in range(5):
            db.create_payment(i, "stars", 6.50, 30, payload=f"pay_{i}")
        assert len(db.get_payments(3)) == 3

    def test_amount_usd_column(self, db):
        """amount_usd stored correctly."""
        db.create_payment(111, "usdt", 7.00, 30, payload="col_test")
        p = db.get_payment_by_payload("col_test")
        assert p["amount_usd"] == pytest.approx(7.00)

    def test_manual_payment(self, db):
        """Manual payment can be recorded and retrieved."""
        db.create_user(111, "alice", "Alice")
        pid = db.create_payment(111, "manual", 10.00, 30,
                               payload="manual_30d_111", status="paid")
        p = db.get_payment_by_payload("manual_30d_111")
        assert p["method"] == "manual"
        assert p["status"] == "paid"
