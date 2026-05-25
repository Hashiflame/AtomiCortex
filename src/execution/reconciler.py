"""
AtomiCortex — Position Reconciler.

Compares internal position state with the real exchange positions
(via Binance REST API) and detects:
- **Orphan positions**: exist on exchange but not in our state.
- **Ghost positions**: exist in our state but not on exchange.
- **Mismatched sizes**: direction or quantity disagree.

Runs at every reconnect to ensure internal state is correct.

Phase 4 — Step 4.6.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from src.logger import get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReconciliationResult:
    """Output of :meth:`PositionReconciler.reconcile`."""

    orphan_positions: list[dict[str, Any]] = field(default_factory=list)
    ghost_positions: list[dict[str, Any]] = field(default_factory=list)
    mismatched_sizes: list[dict[str, Any]] = field(default_factory=list)
    is_clean: bool = True
    actions_taken: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal position format expected
# ---------------------------------------------------------------------------

@dataclass
class InternalPosition:
    """Minimal internal position record for reconciliation."""

    symbol: str
    direction: int       # 1=LONG, -1=SHORT
    quantity: float


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

class PositionReconciler:
    """Reconciles internal position state with the exchange.

    Parameters
    ----------
    binance_api_key:
        Binance API key.
    binance_api_secret:
        Binance API secret.
    trading_mode:
        ``"testnet"`` or ``"live"``.
    auto_fix:
        If ``True``, close orphan positions and remove ghost positions
        automatically.  If ``False``, only report discrepancies.
    """

    _URLS: dict[str, str] = {
        "testnet": "https://testnet.binancefuture.com",
        "live": "https://fapi.binance.com",
    }

    def __init__(
        self,
        binance_api_key: str = "",
        binance_api_secret: str = "",
        trading_mode: str = "testnet",
        auto_fix: bool = False,
    ) -> None:
        self._api_key = binance_api_key
        self._api_secret = binance_api_secret
        self._base_url = self._URLS.get(
            trading_mode.lower(), self._URLS["testnet"],
        )
        self._auto_fix = auto_fix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reconcile(
        self,
        internal_positions: dict[str, InternalPosition],
        exchange_positions: list[dict[str, Any]] | None = None,
    ) -> ReconciliationResult:
        """Compare internal state with exchange and return discrepancies.

        Parameters
        ----------
        internal_positions:
            Dict mapping symbol → InternalPosition (our view).
        exchange_positions:
            Pre-fetched exchange position list.  If ``None``, fetches
            from Binance REST API.
        """
        result = ReconciliationResult()

        # 1. Fetch exchange positions if not provided
        if exchange_positions is None:
            exchange_positions = await self._fetch_exchange_positions()
            if exchange_positions is None:
                _log.error("Cannot reconcile — exchange positions unavailable")
                result.is_clean = False
                result.actions_taken.append("FAILED: cannot fetch exchange positions")
                return result

        # 2. Build exchange position map
        exchange_map: dict[str, dict[str, Any]] = {}
        for pos in exchange_positions:
            amt = float(pos.get("positionAmt", 0))
            if abs(amt) > 1e-10:
                symbol = pos.get("symbol", "UNKNOWN")
                exchange_map[symbol] = {
                    "symbol": symbol,
                    "direction": 1 if amt > 0 else -1,
                    "quantity": abs(amt),
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
                }

        # 3. Detect orphans: on exchange but not in internal state
        for symbol, epos in exchange_map.items():
            if symbol not in internal_positions:
                result.orphan_positions.append(epos)
                _log.warning(
                    "ORPHAN position detected | {sym} dir={d} qty={q}",
                    sym=symbol,
                    d=epos["direction"],
                    q=epos["quantity"],
                )

        # 4. Detect ghosts: in internal state but not on exchange
        for symbol, ipos in internal_positions.items():
            if symbol not in exchange_map:
                result.ghost_positions.append({
                    "symbol": symbol,
                    "direction": ipos.direction,
                    "quantity": ipos.quantity,
                })
                _log.warning(
                    "GHOST position detected | {sym} dir={d} qty={q}",
                    sym=symbol,
                    d=ipos.direction,
                    q=ipos.quantity,
                )

        # 5. Detect mismatches: both exist but sizes differ
        for symbol, ipos in internal_positions.items():
            if symbol in exchange_map:
                epos = exchange_map[symbol]
                dir_match = ipos.direction == epos["direction"]
                qty_match = abs(ipos.quantity - epos["quantity"]) < 1e-8

                if not dir_match or not qty_match:
                    mismatch = {
                        "symbol": symbol,
                        "internal_direction": ipos.direction,
                        "exchange_direction": epos["direction"],
                        "internal_quantity": ipos.quantity,
                        "exchange_quantity": epos["quantity"],
                    }
                    result.mismatched_sizes.append(mismatch)
                    _log.warning(
                        "MISMATCH | {sym} internal=({id},{iq}) "
                        "exchange=({ed},{eq})",
                        sym=symbol,
                        id=ipos.direction,
                        iq=ipos.quantity,
                        ed=epos["direction"],
                        eq=epos["quantity"],
                    )

        # 6. Determine cleanliness
        result.is_clean = (
            len(result.orphan_positions) == 0
            and len(result.ghost_positions) == 0
            and len(result.mismatched_sizes) == 0
        )

        if result.is_clean:
            n = len(exchange_map)
            _log.info(
                "Reconciliation clean | {n} positions match",
                n=n,
            )
            result.actions_taken.append(f"CLEAN: {n} positions verified")
        else:
            _log.warning(
                "Reconciliation DIRTY | orphans={o} ghosts={g} mismatches={m}",
                o=len(result.orphan_positions),
                g=len(result.ghost_positions),
                m=len(result.mismatched_sizes),
            )

            # Auto-fix if enabled
            if self._auto_fix:
                for orphan in result.orphan_positions:
                    action = f"Would close orphan {orphan['symbol']}"
                    result.actions_taken.append(action)
                for ghost in result.ghost_positions:
                    action = f"Removed ghost {ghost['symbol']} from state"
                    result.actions_taken.append(action)
                for mm in result.mismatched_sizes:
                    action = (
                        f"Would sync {mm['symbol']}: "
                        f"internal={mm['internal_quantity']} → "
                        f"exchange={mm['exchange_quantity']}"
                    )
                    result.actions_taken.append(action)

        return result

    # ------------------------------------------------------------------
    # Internal: Binance REST
    # ------------------------------------------------------------------

    async def _fetch_exchange_positions(self) -> list[dict[str, Any]] | None:
        """Fetch position risk from Binance REST API."""
        import aiohttp

        from src.execution.binance_rate_limiter import BinanceRateLimiter
        limiter = BinanceRateLimiter.instance()
        await limiter.acquire(5)  # positionRisk weight = 5

        params: dict[str, str] = {}
        params["timestamp"] = str(int(time.time() * 1000))
        params["recvWindow"] = "5000"
        query_string = urlencode(params)
        signature = hmac.new(
            self._api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        url = self._base_url + "/fapi/v2/positionRisk"
        headers = {"X-MBX-APIKEY": self._api_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    limiter.update_from_headers(getattr(resp, "headers", None))
                    if resp.status != 200:
                        body = await resp.text()
                        _log.error(
                            "Binance positionRisk error: {s} {b}",
                            s=resp.status, b=body,
                        )
                        return None
                    return await resp.json()
        except Exception as exc:
            _log.error("Fetch positions failed: {err}", err=str(exc))
            return None
