"""
AtomiCortex — Telegram Stars Payment Handler.

Implements native Telegram Stars (XTR) payments for premium subscriptions
using the python-telegram-bot v21 invoice API.

Security (Phase 5.3):
  * Pre-checkout verifies the amount against the configured price
    (``context.bot_data["prices"]``) before approving the charge.
  * Pre-checkout verifies that the payer matches the user_id encoded in
    the payload (no payload spoofing).
  * Successful-payment deduplication uses ``telegram_payment_charge_id``
    (unique per transaction) — stored in the existing ``invoice_id``
    column — instead of the payload, which repeats across renewals.
  * Owner-only ``/refund <charge_id|user_id>`` refunds via Telegram's
    ``refundStarPayment`` API and revokes premium.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from telegram import LabeledPrice, Update
from telegram.ext import ContextTypes

from src.logger import get_logger
from src.telegram_bot.database import Database
from src.telegram_bot.roles import OWNER_ID

_log = get_logger(__name__)

# Stars → USD approximate conversion: 1 Star ≈ $0.013
_STARS_TO_USD = 0.013


def _parse_payload(payload: str) -> tuple[int, int] | None:
    """Parse payload ``premium_30d_123456`` → (days, user_id)."""
    try:
        parts = payload.split("_")
        if len(parts) != 3 or parts[0] != "premium":
            return None
        days = int(parts[1].rstrip("d"))
        user_id = int(parts[2])
        if days <= 0 or user_id <= 0:
            return None
        return days, user_id
    except (ValueError, IndexError):
        return None


def _expected_stars_price(
    context: ContextTypes.DEFAULT_TYPE, days: int,
) -> int | None:
    """Look up the configured Stars price for ``days``.

    Returns None when no price config is available (fail-soft).
    """
    prices = context.bot_data.get("prices") if context.bot_data else None
    if not isinstance(prices, dict):
        return None
    val = prices.get(f"stars_{days}d")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


async def send_invoice_stars(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    days: int,
    price_stars: int,
) -> None:
    """Send a Telegram Stars invoice to the user."""
    if update.effective_user is None or update.effective_chat is None:
        return

    user_id = update.effective_user.id

    if OWNER_ID is not None and user_id == OWNER_ID:
        await update.effective_chat.send_message(
            "⚠️ Владелец бота не может оплатить подписку самому себе.\n"
            "Используйте /grant для ручной активации."
        )
        return

    payload = f"premium_{days}d_{user_id}"

    db: Database = context.bot_data["db"]
    db.create_payment(
        user_id=user_id,
        method="stars",
        amount_usd=round(price_stars * _STARS_TO_USD, 2),
        days=days,
        payload=payload,
        stars_amount=price_stars,
    )

    await context.bot.send_invoice(
        chat_id=update.effective_chat.id,
        title=f"AtomiCortex Premium {days} дней",
        description=f"AI торговые сигналы BTC/ETH/SOL на {days} дней",
        payload=payload,
        provider_token="",   # empty string required for XTR
        currency="XTR",
        prices=[LabeledPrice(f"Premium {days}d", price_stars)],
    )

    _log.info(
        "Stars invoice sent | user={uid} days={d} stars={s}",
        uid=user_id, d=days, s=price_stars,
    )


async def pre_checkout_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Approve or reject a pre-checkout query for Stars payment.

    Validates:
      1. Payload structure.
      2. Payer matches the user_id encoded in the payload.
      3. ``total_amount`` matches the configured price for ``days``.
         If price config is unavailable, fail-soft (log + approve)
         so a misconfigured deployment does not block real users.
    """
    query = update.pre_checkout_query
    if query is None:
        return

    payload = query.invoice_payload
    parsed = _parse_payload(payload)

    if parsed is None:
        await query.answer(ok=False, error_message="Неверный payload платежа.")
        _log.warning("Pre-checkout REJECTED: bad payload {p}", p=payload)
        return

    days, user_id = parsed

    # Reject payload spoofing — the payer must match the encoded user_id.
    payer_id = getattr(query.from_user, "id", None)
    if isinstance(payer_id, int) and payer_id != user_id:
        await query.answer(
            ok=False,
            error_message="Несоответствие пользователя в платеже.",
        )
        _log.warning(
            "Pre-checkout REJECTED: payer={fu} payload_user={pu}",
            fu=payer_id, pu=user_id,
        )
        return

    expected = _expected_stars_price(context, days)
    if expected is None:
        # Fail-soft: log loudly but do not block the user.
        _log.warning(
            "Pre-checkout: no price config for {d}d — skipping amount check "
            "(payload={p}, amount={a})",
            d=days, p=payload, a=query.total_amount,
        )
    elif query.total_amount != expected:
        await query.answer(
            ok=False,
            error_message="Неверная сумма платежа.",
        )
        _log.warning(
            "Pre-checkout REJECTED: amount mismatch payload={p} "
            "got={g} expected={e}",
            p=payload, g=query.total_amount, e=expected,
        )
        return

    await query.answer(ok=True)
    _log.info(
        "Pre-checkout approved | payload={p} amount={a}",
        p=payload, a=query.total_amount,
    )


async def successful_payment_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle successful Stars payment — activate premium subscription.

    Deduplication is keyed on ``telegram_payment_charge_id`` (unique per
    transaction), stored in the existing ``payments.invoice_id`` column.
    The same payload can legitimately appear across renewals.
    """
    if update.message is None or update.message.successful_payment is None:
        return

    payment = update.message.successful_payment
    payload = payment.invoice_payload
    parsed = _parse_payload(payload)

    if parsed is None:
        _log.error("Successful payment with bad payload: {p}", p=payload)
        return

    days, user_id = parsed
    raw_charge = getattr(payment, "telegram_payment_charge_id", "")
    charge_id = raw_charge if isinstance(raw_charge, str) else ""
    db: Database = context.bot_data["db"]

    # Primary dedup: telegram_payment_charge_id.
    if charge_id:
        prior = db.get_payment_by_invoice_id(charge_id)
        if prior and prior.get("status") == "paid":
            _log.warning(
                "Duplicate Stars charge ignored | charge_id={c}", c=charge_id,
            )
            await update.message.reply_text(
                "⚠️ Этот платёж уже был обработан ранее."
            )
            return

    now = datetime.now(timezone.utc)

    # Extend subscription if already premium.
    user = db.get_user(user_id)
    if user and user.get("role") == "premium" and user.get("expires_at"):
        try:
            current_expiry = datetime.fromisoformat(user["expires_at"])
            if current_expiry.tzinfo is None:
                current_expiry = current_expiry.replace(tzinfo=timezone.utc)
            if current_expiry > now:
                expires_at = current_expiry + timedelta(days=days)
            else:
                expires_at = now + timedelta(days=days)
        except (ValueError, TypeError):
            expires_at = now + timedelta(days=days)
    else:
        expires_at = now + timedelta(days=days)

    db.set_role(user_id, "premium", expires_at)

    # Persist: prefer attaching this charge to the pending row created by
    # ``send_invoice_stars`` (so the invoice → activation lifecycle stays
    # in a single DB row). If the pending row already has a different
    # charge_id, this is a renewal — create a new row instead so charge
    # history is preserved.
    pending = db.get_payment_by_payload(payload)
    reuse = (
        pending is not None
        and pending.get("status") != "paid"
        and not pending.get("invoice_id")
    )
    if reuse:
        if charge_id:
            db.set_payment_invoice_id(int(pending["id"]), charge_id)
        db.update_payment_status(int(pending["id"]), "paid", now.isoformat())
    else:
        pid = db.create_payment(
            user_id=user_id,
            method="stars",
            amount_usd=round(payment.total_amount * _STARS_TO_USD, 2),
            days=days,
            payload=payload,
            invoice_id=charge_id,
            stars_amount=payment.total_amount,
        )
        db.update_payment_status(pid, "paid", now.isoformat())

    db.log_event(
        "payment",
        f"Stars payment: {payment.total_amount} XTR "
        f"for {days}d by user {user_id} charge={charge_id}",
    )

    exp_str = expires_at.strftime("%Y-%m-%d %H:%M UTC")
    await update.message.reply_text(
        f"✅ Оплата получена!\n\n"
        f"{'═' * 30}\n"
        f"💎 Premium активирован\n"
        f"Срок: {days} дней\n"
        f"Истекает: {exp_str}\n"
        f"{'═' * 30}\n\n"
        f"Используйте /signal для получения сигналов."
    )

    _log.info(
        "Stars payment success | user={uid} days={d} charge={c} expires={e}",
        uid=user_id, d=days, c=charge_id, e=exp_str,
    )

    if OWNER_ID is not None:
        try:
            username = (
                update.effective_user.username if update.effective_user else "?"
            )
            await context.bot.send_message(
                chat_id=OWNER_ID,
                text=(
                    f"💰 Новая оплата Stars!\n"
                    f"User: @{username} ({user_id})\n"
                    f"Сумма: {payment.total_amount} ⭐\n"
                    f"Срок: {days} дней\n"
                    f"Expires: {exp_str}"
                ),
            )
        except Exception:
            pass


async def refund_stars_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Owner-only: refund a Stars payment and revoke premium.

    Usage::

        /refund <telegram_payment_charge_id>
        /refund <user_id>       # refunds the user's latest paid Stars charge

    Calls Telegram's ``refundStarPayment`` API, marks the payment row as
    ``refunded``, and downgrades the user back to ``free``.
    """
    if update.effective_user is None or update.message is None:
        return
    if OWNER_ID is None or update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⚠️ Команда доступна только владельцу.")
        return

    args = context.args or []
    if not args:
        await update.message.reply_text(
            "Использование: /refund <charge_id или user_id>"
        )
        return

    db: Database = context.bot_data["db"]
    arg = args[0].strip()

    # 1. Try as charge_id (stored in invoice_id).
    payment = db.get_payment_by_invoice_id(arg)

    # 2. Fall back to user_id → latest paid Stars payment for that user.
    if payment is None and arg.isdigit():
        candidates = [
            p for p in db.get_payments(limit=200)
            if p.get("user_id") == int(arg)
            and p.get("method") == "stars"
            and p.get("status") == "paid"
            and p.get("invoice_id")
        ]
        if candidates:
            payment = candidates[0]   # newest first per get_payments ordering

    if payment is None or payment.get("status") != "paid":
        await update.message.reply_text("Платёж не найден или уже возвращён.")
        return

    charge_id = str(payment.get("invoice_id") or "")
    target_user = int(payment.get("user_id") or 0)
    if not charge_id or target_user <= 0:
        await update.message.reply_text("Платёж не содержит charge_id.")
        return

    try:
        await context.bot.refund_star_payment(
            user_id=target_user,
            telegram_payment_charge_id=charge_id,
        )
    except Exception as exc:
        _log.error(
            "refund_star_payment failed | user={u} charge={c} err={e}",
            u=target_user, c=charge_id, e=str(exc),
        )
        await update.message.reply_text(f"❌ Ошибка refund: {exc}")
        return

    db.update_payment_status(int(payment["id"]), "refunded", None)
    db.set_role(target_user, "free", None)

    db.log_event(
        "refund",
        f"Stars refund issued | user={target_user} charge={charge_id}",
    )
    _log.info(
        "Stars refund | user={u} charge={c}", u=target_user, c=charge_id,
    )

    await update.message.reply_text(
        f"✅ Refund выполнен.\nUser: {target_user}\nCharge: {charge_id}"
    )
