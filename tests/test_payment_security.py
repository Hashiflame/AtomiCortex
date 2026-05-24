"""
Phase 5.3 — payment security tests.

Covers:
  Stars
    * pre_checkout: amount mismatch → rejected
    * pre_checkout: correct amount → approved
    * successful_payment: duplicate charge_id → no double-activation
    * successful_payment: distinct charge_ids → both activate (renewal)
    * /refund handler revokes premium & calls refundStarPayment
  CryptoBot
    * replay-attack: same invoice across "restarts" → only one activation
    * amount mismatch (forged payload) → activation refused
    * persistent dedup: a brand-new CryptoBotPayment instance still skips
      already-paid invoices (state lives in DB, not memory).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.telegram_bot.database import Database
from src.telegram_bot.payments_crypto import CryptoBotPayment
from src.telegram_bot.payments_stars import (
    pre_checkout_handler,
    refund_stars_handler,
    successful_payment_handler,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "sec.db")


@pytest.fixture(autouse=True)
def _patch_owner():
    with patch.dict(os.environ, {"TELEGRAM_ADMIN_ID": "99999"}):
        import src.telegram_bot.roles as rm
        import src.telegram_bot.payments_stars as ps
        rm.OWNER_ID = 99999
        ps.OWNER_ID = 99999
        yield


def _ctx(db, prices=None, args=None):
    ctx = MagicMock()
    ctx.bot_data = {"db": db}
    if prices is not None:
        ctx.bot_data["prices"] = prices
    ctx.bot = MagicMock()
    ctx.bot.send_message = AsyncMock()
    ctx.bot.refund_star_payment = AsyncMock()
    ctx.args = args or []
    return ctx


def _pre_checkout_update(user_id: int, payload: str, total_amount: int):
    update = MagicMock()
    update.pre_checkout_query = MagicMock()
    update.pre_checkout_query.invoice_payload = payload
    update.pre_checkout_query.total_amount = total_amount
    update.pre_checkout_query.from_user = MagicMock()
    update.pre_checkout_query.from_user.id = user_id
    update.pre_checkout_query.answer = AsyncMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.message = None
    return update


def _payment_update(
    user_id: int, payload: str, total_amount: int, charge_id: str,
):
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = user_id
    update.effective_user.username = "u"
    update.effective_chat = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.successful_payment = MagicMock()
    update.message.successful_payment.invoice_payload = payload
    update.message.successful_payment.total_amount = total_amount
    update.message.successful_payment.telegram_payment_charge_id = charge_id
    update.pre_checkout_query = None
    return update


_PRICES = {"stars_30d": 500, "stars_90d": 1200}


# ═══════════════════════════════════════════════════════════════════════
# 1. Telegram Stars — pre_checkout amount verification
# ═══════════════════════════════════════════════════════════════════════
class TestStarsAmountVerification:
    @pytest.mark.asyncio
    async def test_wrong_amount_rejected(self, db):
        """Underpaying (1 star instead of 500) must be rejected pre-charge."""
        upd = _pre_checkout_update(111, "premium_30d_111", total_amount=1)
        ctx = _ctx(db, prices=_PRICES)

        await pre_checkout_handler(upd, ctx)

        upd.pre_checkout_query.answer.assert_awaited_once()
        kwargs = upd.pre_checkout_query.answer.call_args.kwargs
        assert kwargs["ok"] is False
        assert "сумма" in kwargs["error_message"].lower()

    @pytest.mark.asyncio
    async def test_correct_amount_approved(self, db):
        """Exact price match passes pre-checkout."""
        upd = _pre_checkout_update(111, "premium_30d_111", total_amount=500)
        ctx = _ctx(db, prices=_PRICES)

        await pre_checkout_handler(upd, ctx)

        upd.pre_checkout_query.answer.assert_awaited_once_with(ok=True)

    @pytest.mark.asyncio
    async def test_overpaying_also_rejected(self, db):
        """Even overpayment is rejected — strict equality with config."""
        upd = _pre_checkout_update(111, "premium_30d_111", total_amount=10_000)
        ctx = _ctx(db, prices=_PRICES)

        await pre_checkout_handler(upd, ctx)
        assert upd.pre_checkout_query.answer.call_args.kwargs["ok"] is False

    @pytest.mark.asyncio
    async def test_payload_user_spoofing_rejected(self, db):
        """Payer ID must match the user_id encoded in the payload."""
        # Payer is 999 but payload claims 111.
        upd = _pre_checkout_update(999, "premium_30d_111", total_amount=500)
        ctx = _ctx(db, prices=_PRICES)

        await pre_checkout_handler(upd, ctx)
        assert upd.pre_checkout_query.answer.call_args.kwargs["ok"] is False

    @pytest.mark.asyncio
    async def test_fail_soft_when_no_price_config(self, db):
        """Missing price config: log + approve (don't block real users)."""
        upd = _pre_checkout_update(111, "premium_30d_111", total_amount=500)
        ctx = _ctx(db, prices=None)  # no prices in bot_data

        await pre_checkout_handler(upd, ctx)
        upd.pre_checkout_query.answer.assert_awaited_once_with(ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 2. Telegram Stars — charge_id deduplication
# ═══════════════════════════════════════════════════════════════════════
class TestStarsChargeIdDedup:
    @pytest.mark.asyncio
    async def test_duplicate_charge_id_rejected(self, db):
        """Replaying the same charge_id must not re-activate / re-extend."""
        db.create_user(111, "alice", "Alice")
        # Seed an already-paid row keyed by charge_id.
        pid = db.create_payment(
            111, "stars", 6.50, 30,
            payload="premium_30d_111", invoice_id="ch_X",
            stars_amount=500,
        )
        db.update_payment_status(pid, "paid", datetime.now(timezone.utc).isoformat())

        upd = _payment_update(111, "premium_30d_111", 500, charge_id="ch_X")
        ctx = _ctx(db)

        await successful_payment_handler(upd, ctx)

        # User exists but should not have been promoted by this handler.
        user = db.get_user(111)
        assert user is None or user.get("role") != "premium"
        # User saw the dedup message.
        msg = upd.message.reply_text.call_args[0][0]
        assert "уже был обработан" in msg
        # Only one paid Stars row in DB.
        paid = [p for p in db.get_payments() if p.get("status") == "paid"]
        assert len(paid) == 1

    @pytest.mark.asyncio
    async def test_distinct_charge_ids_both_processed(self, db):
        """A genuine renewal (same payload, new charge_id) must activate."""
        db.create_user(111, "alice", "Alice")

        upd1 = _payment_update(111, "premium_30d_111", 500, charge_id="ch_1")
        upd2 = _payment_update(111, "premium_30d_111", 500, charge_id="ch_2")
        ctx = _ctx(db)

        await successful_payment_handler(upd1, ctx)
        first_user = db.get_user(111)
        assert first_user["role"] == "premium"

        await successful_payment_handler(upd2, ctx)
        second_user = db.get_user(111)
        assert second_user["role"] == "premium"

        # Both charges recorded as paid.
        paid_rows = [
            p for p in db.get_payments()
            if p.get("status") == "paid" and p.get("method") == "stars"
        ]
        charge_ids = {p["invoice_id"] for p in paid_rows}
        assert charge_ids == {"ch_1", "ch_2"}

        # Renewal extended the expiry beyond the first activation.
        assert second_user["expires_at"] > first_user["expires_at"]


# ═══════════════════════════════════════════════════════════════════════
# 3. Telegram Stars — refund handler
# ═══════════════════════════════════════════════════════════════════════
class TestRefundHandler:
    @pytest.mark.asyncio
    async def test_owner_refund_revokes_premium(self, db):
        """/refund calls refundStarPayment, marks row refunded, downgrades user."""
        db.create_user(111, "alice", "Alice")
        # Simulate prior activation.
        pid = db.create_payment(
            111, "stars", 6.50, 30,
            payload="premium_30d_111", invoice_id="ch_R",
            stars_amount=500,
        )
        db.update_payment_status(pid, "paid", datetime.now(timezone.utc).isoformat())
        from datetime import timedelta
        db.set_role(
            111, "premium",
            datetime.now(timezone.utc) + timedelta(days=30),
        )

        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 99999          # owner
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()

        ctx = _ctx(db, args=["ch_R"])

        await refund_stars_handler(update, ctx)

        ctx.bot.refund_star_payment.assert_awaited_once()
        kw = ctx.bot.refund_star_payment.call_args.kwargs
        assert kw["user_id"] == 111
        assert kw["telegram_payment_charge_id"] == "ch_R"

        row = db.get_payment_by_invoice_id("ch_R")
        assert row["status"] == "refunded"
        user = db.get_user(111)
        assert user["role"] == "free"

    @pytest.mark.asyncio
    async def test_non_owner_refund_blocked(self, db):
        """Anyone other than the owner can't run /refund."""
        update = MagicMock()
        update.effective_user = MagicMock()
        update.effective_user.id = 111            # not owner
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()

        ctx = _ctx(db, args=["ch_anything"])

        await refund_stars_handler(update, ctx)

        ctx.bot.refund_star_payment.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════════
# 4. CryptoBot — replay & amount verification
# ═══════════════════════════════════════════════════════════════════════
class TestCryptoBotSecurity:
    @pytest.mark.asyncio
    async def test_replay_same_invoice_only_processed_once(self, db):
        db.create_user(222, "bob", "Bob")
        db.create_payment(
            222, "usdt", 7.00, 30,
            payload="premium_30d_222", invoice_id="",
        )
        crypto = CryptoBotPayment("tok", db, bot=MagicMock())
        crypto._bot.send_message = AsyncMock()

        paid_invoices = [{
            "invoice_id": "INV_R",
            "status": "paid",
            "payload": "premium_30d_222",
            "amount": "7.00",
        }]

        with patch.object(crypto, "get_paid_invoices", return_value=paid_invoices):
            first = await crypto.poll_payments()
            second = await crypto.poll_payments()

        assert first == 1
        assert second == 0

    @pytest.mark.asyncio
    async def test_amount_mismatch_refused(self, db):
        """Forged payload activation with wrong paid amount is rejected."""
        db.create_user(222, "bob", "Bob")
        db.create_payment(
            222, "usdt", 7.00, 30,
            payload="premium_30d_222", invoice_id="",
        )
        crypto = CryptoBotPayment("tok", db, bot=MagicMock())
        crypto._bot.send_message = AsyncMock()

        # Attacker pays 0.01 USDT but reuses our payload.
        paid_invoices = [{
            "invoice_id": "INV_FORGED",
            "status": "paid",
            "payload": "premium_30d_222",
            "amount": "0.01",
        }]

        with patch.object(crypto, "get_paid_invoices", return_value=paid_invoices):
            activated = await crypto.poll_payments()

        assert activated == 0
        user = db.get_user(222)
        # User existed but must NOT have been promoted to premium.
        assert user is None or user.get("role") != "premium"
        # The pending row stays pending.
        row = db.get_payment_by_payload("premium_30d_222")
        assert row["status"] == "pending"

    @pytest.mark.asyncio
    async def test_invoice_without_matching_pending_refused(self, db):
        """Paid invoice whose payload was never issued by us → skipped."""
        # Note: no create_payment call → no pending row.
        db.create_user(333, "carol", "Carol")
        crypto = CryptoBotPayment("tok", db, bot=MagicMock())
        crypto._bot.send_message = AsyncMock()

        paid_invoices = [{
            "invoice_id": "INV_UNKNOWN",
            "status": "paid",
            "payload": "premium_30d_333",
            "amount": "7.00",
        }]

        with patch.object(crypto, "get_paid_invoices", return_value=paid_invoices):
            activated = await crypto.poll_payments()

        assert activated == 0
        assert db.get_user(333) is None or db.get_user(333).get("role") != "premium"

    @pytest.mark.asyncio
    async def test_dedup_survives_restart(self, db):
        """A fresh CryptoBotPayment (empty memory) still skips paid invoices.

        Persistent dedup must come from the DB, not from ``_processed_ids``.
        """
        db.create_user(222, "bob", "Bob")
        db.create_payment(
            222, "usdt", 7.00, 30,
            payload="premium_30d_222", invoice_id="",
        )
        paid_invoices = [{
            "invoice_id": "INV_PERSIST",
            "status": "paid",
            "payload": "premium_30d_222",
            "amount": "7.00",
        }]

        # First "process": legitimate activation.
        crypto1 = CryptoBotPayment("tok", db, bot=MagicMock())
        crypto1._bot.send_message = AsyncMock()
        with patch.object(crypto1, "get_paid_invoices", return_value=paid_invoices):
            assert await crypto1.poll_payments() == 1

        # Simulate restart: fresh instance, empty in-memory state.
        crypto2 = CryptoBotPayment("tok", db, bot=MagicMock())
        crypto2._bot.send_message = AsyncMock()
        assert crypto2._processed_ids == set()

        with patch.object(crypto2, "get_paid_invoices", return_value=paid_invoices):
            replay_count = await crypto2.poll_payments()

        # Must NOT replay: persistent dedup via DB.
        assert replay_count == 0
        # Premium activated once, not twice.
        user = db.get_user(222)
        assert user["role"] == "premium"
        # Exactly one paid USDT row for this payload.
        rows = [
            p for p in db.get_payments()
            if p.get("payload") == "premium_30d_222"
            and p.get("status") == "paid"
        ]
        assert len(rows) == 1
