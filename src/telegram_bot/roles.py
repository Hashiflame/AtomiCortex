"""
AtomiCortex — Telegram Bot Roles & Middleware.

Decorators for role-based access control in PTB v21 command handlers.
Reads OWNER_ID from TELEGRAM_ADMIN_ID environment variable.

Role hierarchy: owner > premium > free

Phase 7 — Telegram Bot.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Coroutine

from telegram import Update
from telegram.ext import ContextTypes

from src.logger import get_logger
from src.telegram_bot.database import Database

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Owner ID  (TG-001: strict validation, None if missing)
# ---------------------------------------------------------------------------

def get_owner_id() -> int | None:
    """Read the owner Telegram user ID from environment.

    Returns None if not configured — callers must handle this.
    """
    raw = os.getenv("TELEGRAM_ADMIN_ID", "").strip()
    if not raw:
        _log.error("TELEGRAM_ADMIN_ID not set in .env — owner features DISABLED")
        return None
    if not raw.isdigit():
        _log.error(
            "TELEGRAM_ADMIN_ID is not a valid positive integer: {v} "
            "— owner features DISABLED",
            v=raw,
        )
        return None
    owner_id = int(raw)
    if owner_id == 0:
        _log.error("TELEGRAM_ADMIN_ID cannot be 0 — owner features DISABLED")
        return None
    return owner_id


OWNER_ID: int | None = get_owner_id()

# ---------------------------------------------------------------------------
# Role hierarchy
# ---------------------------------------------------------------------------

_ROLE_LEVELS: dict[str, int] = {
    "free": 0,
    "premium": 1,
    "owner": 2,
}


def _role_level(role: str) -> int:
    """Return the numeric level for a role string."""
    return _ROLE_LEVELS.get(role.lower(), 0)


# ---------------------------------------------------------------------------
# Internal: get or create user
# ---------------------------------------------------------------------------

def _ensure_user(db: Database, update: Update) -> dict[str, Any] | None:
    """Get or auto-register the user from the update.

    Returns the user dict, or None if the update has no user info.
    """
    if update.effective_user is None:
        return None

    tg_user = update.effective_user
    user_id = tg_user.id

    # Check if this is the owner (TG-001: guard against None)
    is_owner = OWNER_ID is not None and user_id == OWNER_ID

    user = db.get_user(user_id)
    if user is None:
        # Auto-register
        db.create_user(
            user_id=user_id,
            username=tg_user.username,
            first_name=tg_user.first_name,
        )
        if is_owner:
            db.set_role(user_id, "owner")
        user = db.get_user(user_id)
        _log.info(
            "Auto-registered user {uid} (@{un}) as {role}",
            uid=user_id,
            un=tg_user.username,
            role="owner" if is_owner else "free",
        )
    else:
        # Update username/first_name in case they changed
        db.create_user(
            user_id=user_id,
            username=tg_user.username,
            first_name=tg_user.first_name,
        )
        # Ensure owner role is always set
        if is_owner and user.get("role") != "owner":
            db.set_role(user_id, "owner")
            user["role"] = "owner"

    return user


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

HandlerFunc = Callable[..., Coroutine[Any, Any, Any]]


def require_role(
    min_role: str,
) -> Callable[[HandlerFunc], HandlerFunc]:
    """Decorator that restricts a handler to users with at least ``min_role``.

    The database instance must be stored in ``context.bot_data["db"]``.

    Role hierarchy: owner > premium > free.
    Owner always has access to everything.
    Banned users are always blocked.

    Parameters
    ----------
    min_role:
        Minimum role required (``"free"``, ``"premium"``, ``"owner"``).
    """
    required_level = _role_level(min_role)

    def decorator(func: HandlerFunc) -> HandlerFunc:
        @functools.wraps(func)
        async def wrapper(
            update: Update,
            context: ContextTypes.DEFAULT_TYPE,
        ) -> Any:
            if update.effective_user is None or update.effective_chat is None:
                return

            db: Database = context.bot_data["db"]
            user = _ensure_user(db, update)

            if user is None:
                return

            # Check ban
            if user.get("is_banned"):
                await update.effective_chat.send_message(
                    "🚫 Ваш аккаунт заблокирован. "
                    "Свяжитесь с администратором для разблокировки."
                )
                return

            # Check role
            user_level = _role_level(user.get("role", "free"))

            if user_level < required_level:
                role_names = {
                    "free": "бесплатной",
                    "premium": "Premium",
                    "owner": "администратора",
                }
                role_display = role_names.get(min_role, min_role)
                await update.effective_chat.send_message(
                    f"🔒 Эта команда доступна только для подписки "
                    f"{role_display}.\n\n"
                    f"Используйте /subscribe для информации о подписке."
                )
                return

            return await func(update, context)

        return wrapper
    return decorator


def require_not_banned(func: HandlerFunc) -> HandlerFunc:
    """Decorator that blocks banned users.

    The database instance must be stored in ``context.bot_data["db"]``.
    """
    @functools.wraps(func)
    async def wrapper(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> Any:
        if update.effective_user is None or update.effective_chat is None:
            return

        db: Database = context.bot_data["db"]
        user = _ensure_user(db, update)

        if user is None:
            return

        if user.get("is_banned"):
            await update.effective_chat.send_message(
                "🚫 Ваш аккаунт заблокирован. "
                "Свяжитесь с администратором для разблокировки."
            )
            return

        return await func(update, context)

    return wrapper
