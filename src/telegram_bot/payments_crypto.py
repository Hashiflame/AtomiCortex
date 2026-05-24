"""
AtomiCortex — CryptoBot Payment Handler.

Implements USDT/TON payments via @CryptoBot (https://pay.crypt.bot/).
Uses polling instead of webhooks for simplicity.

Phase 7.1 — Payments.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from src.logger import get_logger
from src.telegram_bot.database import Database
from src.telegram_bot.roles import OWNER_ID

_log = get_logger(__name__)

_CRYPTOBOT_API = "https://pay.crypt.bot/api"


class CryptoBotPayment:
    """CryptoBot payment integration for USDT subscriptions.

    Parameters
    ----------
    token:
        CryptoBot API token (from @CryptoBot /pay command).
    db:
        Telegram bot database instance.
    bot:
        python-telegram-bot Bot instance (for sending notifications).
    """

    def __init__(
        self,
        token: str,
        db: Database,
        bot: Any = None,
    ) -> None:
        self._token = token
        self._db = db
        self._bot = bot
        self._headers = {
            "Crypto-Pay-API-Token": token,
            "Content-Type": "application/json",
        }
        self._polling_task: asyncio.Task | None = None
        # Track processed invoice IDs in memory to avoid double-processing
        self._processed_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_invoice(
        self,
        user_id: int,
        days: int,
        amount_usdt: float,
        bot_username: str = "AtomiCortexBot",
    ) -> str | None:
        """Create a CryptoBot invoice and return the payment URL.

        Returns None on API failure.
        """
        payload = f"premium_{days}d_{user_id}"

        body = {
            "asset": "USDT",
            "amount": f"{amount_usdt:.2f}",   # CryptoBot requires string!
            "description": f"AtomiCortex Premium {days} days",
            "payload": payload,
            "paid_btn_name": "callback",
            "paid_btn_url": f"https://t.me/{bot_username}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{_CRYPTOBOT_API}/createInvoice",
                    json=body,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()

            if not data.get("ok"):
                _log.error(
                    "CryptoBot createInvoice failed: {err}",
                    err=data.get("error", data),
                )
                return None

            result = data["result"]
            invoice_id = str(result.get("invoice_id", ""))
            pay_url = result.get("bot_invoice_url", "")

            # Record in DB
            self._db.create_payment(
                user_id=user_id,
                method="usdt",
                amount_usd=amount_usdt,
                days=days,
                payload=payload,
                invoice_id=invoice_id,
            )

            _log.info(
                "CryptoBot invoice created | user={uid} days={d} "
                "amount={a} invoice={inv}",
                uid=user_id, d=days, a=amount_usdt, inv=invoice_id,
            )
            return pay_url

        except Exception as exc:
            _log.error("CryptoBot API error: {err}", err=str(exc))
            return None

    async def get_paid_invoices(self) -> list[dict[str, Any]]:
        """GET /getInvoices?status=paid — returns paid invoices."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{_CRYPTOBOT_API}/getInvoices",
                    params={"status": "paid"},
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()

            if not data.get("ok"):
                return []

            return data.get("result", {}).get("items", [])

        except Exception as exc:
            _log.debug("CryptoBot get_paid_invoices error: {err}", err=str(exc))
            return []

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def start_polling(self, interval: int = 60) -> None:
        """Start background polling for paid invoices."""
        if self._polling_task is not None:
            return
        self._polling_task = asyncio.create_task(
            self._poll_loop(interval),
        )
        _log.info("CryptoBot payment polling started (interval={i}s)", i=interval)

    def stop_polling(self) -> None:
        """Stop background polling."""
        if self._polling_task is not None:
            self._polling_task.cancel()
            self._polling_task = None
            _log.info("CryptoBot payment polling stopped")

    async def _poll_loop(self, interval: int) -> None:
        """Background loop: check paid invoices via CryptoBot API."""
        while True:
            try:
                await self.poll_payments()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                _log.warning("CryptoBot poll error: {err}", err=str(exc))
            await asyncio.sleep(interval)

    async def poll_payments(self) -> int:
        """Check paid invoices and activate new ones.

        Hardened (Phase 5.3):
          * Deduplication is keyed on ``invoice_id`` and **persisted in
            the DB** via ``get_payment_by_invoice_id``. The in-memory
            ``_processed_ids`` set is now just a per-process cache; it is
            re-hydratable from DB so replays survive restart.
          * Each paid invoice must match a **pending row we created**
            (``get_payment_by_payload``). An incoming invoice with no
            matching pending row is treated as untrusted (we did not
            issue it) and skipped with a WARNING.
          * The reported ``amount`` is verified against the amount we
            recorded when the invoice was created (tolerance 0.01 USD).
            Amount mismatch is logged as ERROR and the activation is
            skipped — defends against forged-payload replay.

        Returns the number of newly activated payments.
        """
        paid_invoices = await self.get_paid_invoices()
        if not paid_invoices:
            return 0

        activated = 0
        for inv in paid_invoices:
            invoice_id = str(inv.get("invoice_id", ""))
            if not invoice_id:
                continue

            # 1. In-memory short-circuit (avoids hitting DB on every poll).
            if invoice_id in self._processed_ids:
                continue

            # 2. Persistent dedup — survives restart.
            prior = self._db.get_payment_by_invoice_id(invoice_id)
            if prior and prior.get("status") in ("paid", "refunded"):
                self._processed_ids.add(invoice_id)
                continue

            payload = str(inv.get("payload", ""))
            if not payload:
                _log.warning(
                    "CryptoBot: paid invoice without payload "
                    "| invoice_id={i} — skipping",
                    i=invoice_id,
                )
                self._processed_ids.add(invoice_id)
                continue

            # Parse payload.
            try:
                parts = payload.split("_")
                if len(parts) != 3 or parts[0] != "premium":
                    raise ValueError("bad shape")
                days = int(parts[1].rstrip("d"))
                user_id = int(parts[2])
                if days <= 0 or user_id <= 0:
                    raise ValueError("bad values")
            except (ValueError, IndexError):
                _log.warning(
                    "CryptoBot: unparseable payload {p} | invoice_id={i}",
                    p=payload, i=invoice_id,
                )
                self._processed_ids.add(invoice_id)
                continue

            # 3. Anti-replay: the invoice must match a pending row we
            #    created via create_invoice(). If no such row exists,
            #    the invoice was not issued by us — refuse activation.
            pending = self._db.get_payment_by_payload(payload)
            if pending is None:
                _log.warning(
                    "CryptoBot: paid invoice with no matching pending row "
                    "| payload={p} invoice_id={i} — refusing activation",
                    p=payload, i=invoice_id,
                )
                self._processed_ids.add(invoice_id)
                continue
            if pending.get("status") == "paid":
                # Same payload already activated (e.g. a stale duplicate
                # from CryptoBot's side after we processed via charge_id).
                self._processed_ids.add(invoice_id)
                continue

            # 4. Amount verification.
            try:
                paid_amount = float(inv.get("amount", "0"))
            except (TypeError, ValueError):
                paid_amount = 0.0
            expected_amount = float(pending.get("amount_usd") or 0.0)
            if expected_amount > 0 and abs(paid_amount - expected_amount) > 0.01:
                _log.error(
                    "CryptoBot: AMOUNT MISMATCH | payload={p} invoice={i} "
                    "paid={pa} expected={ea} — refusing activation",
                    p=payload, i=invoice_id, pa=paid_amount, ea=expected_amount,
                )
                self._processed_ids.add(invoice_id)
                continue

            # 5. Activate.
            await self._activate_payment(
                user_id=user_id,
                days=days,
                amount_usd=paid_amount,
                payload=payload,
                invoice_id=invoice_id,
                pending=pending,
            )
            self._processed_ids.add(invoice_id)
            activated += 1

        if activated:
            _log.info("CryptoBot poll: activated {n} payments", n=activated)

        return activated

    async def _activate_payment(
        self,
        user_id: int,
        days: int,
        amount_usd: float,
        payload: str,
        invoice_id: str,
        pending: dict[str, Any] | None = None,
    ) -> None:
        """Activate a paid CryptoBot payment.

        ``pending`` is the pre-existing pending DB row (matched on
        payload) — its row is updated in place so the same charge yields
        exactly one DB record.
        """
        now = datetime.now(timezone.utc)

        # Calculate expiry — extend if already premium
        user = self._db.get_user(user_id)
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

        # Activate subscription
        self._db.set_role(user_id, "premium", expires_at)

        # Update the pre-existing pending row in place (preferred) so we
        # keep a 1:1 mapping between create_invoice() and activation.
        row = pending or self._db.get_payment_by_payload(payload)
        if row:
            self._db.set_payment_invoice_id(int(row["id"]), invoice_id)
            self._db.update_payment_status(
                int(row["id"]), "paid", now.isoformat(),
            )
        else:
            # Defensive fallback — should not happen given poll_payments
            # checks; kept so callers invoking _activate_payment directly
            # (e.g. ad-hoc admin tooling) still get a record.
            pid = self._db.create_payment(
                user_id=user_id,
                method="usdt",
                amount_usd=amount_usd,
                days=days,
                payload=payload,
                invoice_id=invoice_id,
            )
            self._db.update_payment_status(pid, "paid", now.isoformat())

        self._db.log_event(
            "payment",
            f"USDT payment: ${amount_usd:.2f} "
            f"for {days}d by user {user_id}",
        )

        _log.info(
            "USDT payment activated | user={uid} days={d} amount=${a}",
            uid=user_id, d=days, a=amount_usd,
        )

        # Notify user and owner
        if self._bot is not None:
            try:
                exp_str = expires_at.strftime("%Y-%m-%d %H:%M UTC")
                await self._bot.send_message(
                    chat_id=user_id,
                    text=(
                        f"✅ Оплата USDT получена!\n\n"
                        f"{'═' * 30}\n"
                        f"💎 Premium активирован\n"
                        f"Срок: {days} дней\n"
                        f"Истекает: {exp_str}\n"
                        f"{'═' * 30}\n\n"
                        f"Используйте /signal для получения сигналов."
                    ),
                )
            except Exception:
                pass

            if OWNER_ID is not None:
                try:
                    user_data = self._db.get_user(user_id)
                    username = user_data.get("username", "?") if user_data else "?"
                    await self._bot.send_message(
                        chat_id=OWNER_ID,
                        text=(
                            f"💰 Новая оплата USDT!\n"
                            f"User: @{username} ({user_id})\n"
                            f"Сумма: ${amount_usd:.2f}\n"
                            f"Срок: {days} дней"
                        ),
                    )
                except Exception:
                    pass
