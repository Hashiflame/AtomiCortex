"""
AtomiCortex — Telegram Stars Payment Handler.

Implements native Telegram Stars (XTR) payments for premium subscriptions
using the python-telegram-bot v21 invoice API.

Phase 7.1 — Payments.
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


async def send_invoice_stars(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    days: int,
    price_stars: int,
) -> None:
    """Send a Telegram Stars invoice to the user.

    Parameters
    ----------
    days:
        Subscription duration in days (30 or 90).
    price_stars:
        Price in Telegram Stars (integer, e.g. 500).
    """
    if update.effective_user is None or update.effective_chat is None:
        return

    user_id = update.effective_user.id

    # Owner cannot pay their own bot (Telegram blocks this)
    if OWNER_ID is not None and user_id == OWNER_ID:
        await update.effective_chat.send_message(
            "⚠️ Владелец бота не может оплатить подписку самому себе.\n"
            "Используйте /grant для ручной активации."
        )
        return

    payload = f"premium_{days}d_{user_id}"

    # Log the pending payment in DB
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
    """Approve or reject a pre-checkout query for Stars payment."""
    query = update.pre_checkout_query
    if query is None:
        return

    payload = query.invoice_payload
    parsed = _parse_payload(payload)

    if parsed is None:
        await query.answer(ok=False, error_message="Неверный payload платежа.")
        _log.warning("Pre-checkout rejected: bad payload {p}", p=payload)
        return

    # Accept the payment
    await query.answer(ok=True)
    _log.info("Pre-checkout approved | payload={p}", p=payload)


async def successful_payment_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle successful Stars payment — activate premium subscription."""
    if update.message is None or update.message.successful_payment is None:
        return

    payment = update.message.successful_payment
    payload = payment.invoice_payload
    parsed = _parse_payload(payload)

    if parsed is None:
        _log.error("Successful payment with bad payload: {p}", p=payload)
        return

    days, user_id = parsed
    db: Database = context.bot_data["db"]

    # Prevent duplicate activation
    existing = db.get_payment_by_payload(payload)
    if existing and existing["status"] == "paid":
        _log.warning("Duplicate payment ignored | payload={p}", p=payload)
        await update.message.reply_text(
            "⚠️ Этот платёж уже был обработан ранее."
        )
        return

    # Calculate expiry — extend if already premium
    user = db.get_user(user_id)
    now = datetime.now(timezone.utc)

    if user and user.get("role") == "premium" and user.get("expires_at"):
        try:
            current_expiry = datetime.fromisoformat(user["expires_at"])
            if current_expiry.tzinfo is None:
                current_expiry = current_expiry.replace(tzinfo=timezone.utc)
            if current_expiry > now:
                # Extend from current expiry
                expires_at = current_expiry + timedelta(days=days)
            else:
                expires_at = now + timedelta(days=days)
        except (ValueError, TypeError):
            expires_at = now + timedelta(days=days)
    else:
        expires_at = now + timedelta(days=days)

    # Activate subscription
    db.set_role(user_id, "premium", expires_at)

    # Update payment record
    if existing:
        db.update_payment_status(
            existing["id"], "paid", now.isoformat(),
        )
    else:
        pid = db.create_payment(
            user_id=user_id,
            method="stars",
            amount_usd=round(payment.total_amount * _STARS_TO_USD, 2),
            days=days,
            payload=payload,
            stars_amount=payment.total_amount,
        )
        db.update_payment_status(pid, "paid", now.isoformat())

    db.log_event(
        "payment",
        f"Stars payment: {payment.total_amount} XTR "
        f"for {days}d by user {user_id}",
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
        "Stars payment success | user={uid} days={d} expires={e}",
        uid=user_id, d=days, e=exp_str,
    )

    # Notify owner
    if OWNER_ID is not None:
        try:
            username = update.effective_user.username if update.effective_user else "?"
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
