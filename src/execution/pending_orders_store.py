"""Crash-safe on-disk store for pending stop-loss parameters.

When ``MLTradingStrategy`` submits a market entry it stashes the associated
``RiskDecision`` + ``TradeSignal`` in an in-memory dict keyed by
``client_order_id``; ``on_order_filled`` reads them back to place the stop
once the entry is confirmed. If the bot dies between ``submit_order`` and
the fill event, that dict is gone — the next fill event is then mis-
classified as an exit and the position is left without a stop.

This module mirrors that dict to a small JSON file. Writes go via
``tmp + fsync + os.replace`` so a crash mid-write cannot corrupt the
file (POSIX guarantees rename atomicity). The store is fail-soft on every
path: a missing / corrupted / unwritable file logs a warning and returns
an empty state — the bot stays alive with in-memory accounting only.

Recovery semantics
------------------
``load_all`` returns previously-persisted entries. The strategy populates
``_pending_sl_params`` from them, so a fill event arriving *after* restart
is still recognised as an entry and the SL is placed. Fills that occurred
*during* the crash window (and that Binance does not re-deliver) leave an
unprotected position on the exchange; that scenario is covered by the
``PositionReconciler`` which surfaces it as ORPHAN.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logger import get_logger
from src.risk.risk_engine import RiskDecision, TradeSignal

_log = get_logger(__name__)


_DEFAULT_TTL_SECONDS: float = 24 * 3600  # entries older than this are dropped


# ---------------------------------------------------------------------------
# Serialisation helpers — kept tiny and explicit so a future field rename
# fails loudly instead of silently dropping data.
# ---------------------------------------------------------------------------

def _signal_to_dict(sig: TradeSignal) -> dict[str, Any]:
    d = asdict(sig)
    # datetime is not JSON-serialisable; store as ISO 8601.
    d["timestamp"] = sig.timestamp.isoformat()
    return d


def _signal_from_dict(d: dict[str, Any]) -> TradeSignal:
    d = dict(d)  # copy so we don't mutate caller
    d["timestamp"] = datetime.fromisoformat(d["timestamp"])
    return TradeSignal(**d)


def _decision_to_dict(dec: RiskDecision) -> dict[str, Any]:
    return asdict(dec)


def _decision_from_dict(d: dict[str, Any]) -> RiskDecision:
    return RiskDecision(**d)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class PendingOrdersStore:
    """JSON-backed persistence for pending SL parameters."""

    def __init__(
        self,
        path: Path | str,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._path = Path(path)
        self._ttl_seconds = ttl_seconds
        self._data: dict[str, dict[str, Any]] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        client_order_id: str,
        decision: RiskDecision,
        signal: TradeSignal,
    ) -> None:
        """Mirror an in-memory pending entry to disk."""
        self._data[client_order_id] = {
            "created_at": time.time(),
            "decision": _decision_to_dict(decision),
            "signal": _signal_to_dict(signal),
        }
        self._flush()

    def pop(self, client_order_id: str) -> dict[str, Any] | None:
        """Remove an entry and persist the deletion. Returns the deserialised
        ``{"decision": RiskDecision, "signal": TradeSignal}`` or None."""
        entry = self._data.pop(client_order_id, None)
        if entry is None:
            return None
        self._flush()
        try:
            return {
                "decision": _decision_from_dict(entry["decision"]),
                "signal": _signal_from_dict(entry["signal"]),
            }
        except Exception as exc:
            _log.warning(
                "pending_orders_store: failed to deserialise popped "
                "entry {oid}: {err}",
                oid=client_order_id, err=str(exc),
            )
            return None

    def load_all(self) -> dict[str, dict[str, Any]]:
        """Read the persisted entries, dropping any past their TTL.

        Returns ``{client_order_id: {"decision": RiskDecision,
        "signal": TradeSignal}}``. Corrupted / missing file → ``{}``.
        """
        self._load_from_disk()
        now = time.time()
        result: dict[str, dict[str, Any]] = {}
        expired: list[str] = []
        for oid, entry in self._data.items():
            age = now - float(entry.get("created_at", 0.0))
            if age > self._ttl_seconds:
                expired.append(oid)
                continue
            try:
                result[oid] = {
                    "decision": _decision_from_dict(entry["decision"]),
                    "signal": _signal_from_dict(entry["signal"]),
                }
            except Exception as exc:
                _log.warning(
                    "pending_orders_store: dropping unparseable entry "
                    "{oid}: {err}",
                    oid=oid, err=str(exc),
                )
                expired.append(oid)

        if expired:
            for oid in expired:
                self._data.pop(oid, None)
            self._flush()
            _log.info(
                "pending_orders_store: dropped {n} expired/invalid entries",
                n=len(expired),
            )
        return result

    def __contains__(self, client_order_id: str) -> bool:
        return client_order_id in self._data

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Populate ``self._data`` from disk. Fail-soft on every error."""
        if not self._path.exists():
            self._data = {}
            self._loaded = True
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("top-level JSON is not an object")
            self._data = raw
        except Exception as exc:
            _log.warning(
                "pending_orders_store: corrupted/unreadable file {p} "
                "({err}) — starting with empty state",
                p=str(self._path), err=str(exc),
            )
            self._data = {}
        self._loaded = True

    def _flush(self) -> None:
        """Atomically persist ``self._data`` to ``self._path``.

        Writes to a sibling temp file (so the rename is on the same
        filesystem), fsync's it, then ``os.replace`` — POSIX guarantees
        the rename is atomic, so the reader never sees a half-written file.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Sibling tmp file so os.replace stays on the same filesystem.
            fd, tmp_path = tempfile.mkstemp(
                prefix=self._path.name + ".",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._path)
            except Exception:
                # Make sure we don't leak the tmp file on failure.
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as exc:
            _log.warning(
                "pending_orders_store: flush failed for {p}: {err}",
                p=str(self._path), err=str(exc),
            )
