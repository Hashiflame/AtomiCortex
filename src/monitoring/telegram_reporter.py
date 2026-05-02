"""
AtomiCortex — Telegram Reporter.

Sends trade alerts, daily/weekly reports, and critical alerts to a
Telegram chat via the Bot API.  Uses ``aiohttp`` with no heavy
framework dependencies.

Phase 5 — Paper Trading.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.logger import get_logger

_log = get_logger(__name__)


class TelegramReporter:
    """Sends formatted messages to Telegram.

    Parameters
    ----------
    bot_token:
        Telegram Bot API token.
    admin_id:
        Chat ID of the admin to send messages to.
    """

    def __init__(self, bot_token: str, admin_id: str) -> None:
        self._bot_token = bot_token
        self._admin_id = admin_id
        self._base_url = f"https://api.telegram.org/bot{bot_token}"
        _log.info("TelegramReporter initialised")

    # ------------------------------------------------------------------
    # Trade alert
    # ------------------------------------------------------------------

    async def send_trade_alert(
        self,
        signal: Any,
        decision: Any,
        fill: Any | None = None,
    ) -> bool:
        """Send a detailed trade alert.

        Parameters
        ----------
        signal:
            TradeSignal instance.
        decision:
            RiskDecision instance.
        fill:
            PaperFill or None.
        """
        direction_emoji = "🟢" if signal.direction == 1 else "🔴"
        direction_str = "LONG" if signal.direction == 1 else "SHORT"
        sym = signal.symbol.replace("-PERP.BINANCE", "").replace("USDT", "/USDT")

        sl_pct = abs(signal.entry_price - decision.stop_loss) / signal.entry_price * 100
        tp_pct = abs(decision.take_profit - signal.entry_price) / signal.entry_price * 100

        fill_price = fill.fill_price if fill else signal.entry_price
        fill_fee = fill.fee if fill else 0.0

        msg = (
            f"{direction_emoji} {direction_str} {sym} PERP\n"
            f"{'═' * 30}\n"
            f"Время:    {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"Цена:     ${fill_price:,.2f}\n"
            f"Размер:   {decision.position_size:.4f} (${decision.notional:,.2f})\n"
            f"Стоп:     ${decision.stop_loss:,.2f} (-{sl_pct:.1f}%)\n"
            f"Тейк:     ${decision.take_profit:,.2f} (+{tp_pct:.1f}%)\n"
            f"R:R:      1:{decision.risk_reward_ratio:.1f}\n"
            f"Режим:    {signal.regime.upper()}\n"
            f"Conf:     {signal.confidence:.0%}\n"
            f"Funding:  {signal.funding_rate:+.4%}\n"
            f"Fee:      ${fill_fee:.4f}\n"
            f"{'═' * 30}"
        )

        return await self.send_alert(msg)

    # ------------------------------------------------------------------
    # Daily report
    # ------------------------------------------------------------------

    async def send_daily_report(self, report_text: str) -> bool:
        """Send a pre-formatted daily report string."""
        return await self.send_alert(report_text)

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    async def send_weekly_report(self, report_text: str) -> bool:
        """Send a pre-formatted weekly report string."""
        return await self.send_alert(report_text)

    # ------------------------------------------------------------------
    # Generic alert
    # ------------------------------------------------------------------

    async def send_alert(self, message: str) -> bool:
        """Send a plain text message to the admin chat.

        Returns True on success, False on failure.
        """
        if not self._bot_token or not self._admin_id:
            _log.warning("Telegram not configured — alert skipped")
            return False

        import aiohttp

        url = f"{self._base_url}/sendMessage"
        payload = {
            "chat_id": self._admin_id,
            "text": message,
            "parse_mode": "HTML",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        _log.info("Telegram alert sent")
                        return True
                    else:
                        body = await resp.text()
                        _log.warning(
                            "Telegram API error: {status} {body}",
                            status=resp.status,
                            body=body[:200],
                        )
                        return False
        except Exception as exc:
            _log.warning("Telegram send failed: {err}", err=str(exc))
            return False

    # ------------------------------------------------------------------
    # Format helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_trade_alert(
        direction: int,
        symbol: str,
        entry_price: float,
        quantity: float,
        notional: float,
        stop_loss: float,
        take_profit: float,
        rr_ratio: float,
        regime: str,
        confidence: float,
        funding_rate: float,
    ) -> str:
        """Build a trade alert string without needing full signal/decision objects."""
        emoji = "🟢" if direction == 1 else "🔴"
        dir_str = "LONG" if direction == 1 else "SHORT"
        sl_pct = abs(entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
        tp_pct = abs(take_profit - entry_price) / entry_price * 100 if entry_price > 0 else 0

        return (
            f"{emoji} {dir_str} {symbol}\n"
            f"{'═' * 30}\n"
            f"Цена:     ${entry_price:,.2f}\n"
            f"Размер:   {quantity:.4f} (${notional:,.2f})\n"
            f"Стоп:     ${stop_loss:,.2f} (-{sl_pct:.1f}%)\n"
            f"Тейк:     ${take_profit:,.2f} (+{tp_pct:.1f}%)\n"
            f"R:R:      1:{rr_ratio:.1f}\n"
            f"Режим:    {regime.upper()}\n"
            f"Conf:     {confidence:.0%}\n"
            f"Funding:  {funding_rate:+.4%}\n"
            f"{'═' * 30}"
        )
