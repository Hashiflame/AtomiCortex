"""
Monkey-patch for Nautilus Trader BinanceFuturesContractStatus enum.

Adds 'TRADING_HALT' to BinanceFuturesContractStatus enum to unblock connection
due to missing enum value in msgspec decoder (upstream PR #4320).

Remove this patch after upgrading nautilus-trader to >= 1.222.0.
"""

try:
    from src.logger import get_logger
    _log = get_logger(__name__)
except ImportError:
    # Fallback if src.logger causes circular import or isn't available in some contexts
    import logging as _log

def apply():
    try:
        from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesContractStatus
        
        # Idempotency check: if already patched or upstream fixed it
        if "TRADING_HALT" in BinanceFuturesContractStatus._member_names_:
            return
            
        # Extend the Enum dynamically
        new_member = object.__new__(BinanceFuturesContractStatus)
        new_member._name_ = "TRADING_HALT"
        new_member._value_ = "TRADING_HALT"
        
        BinanceFuturesContractStatus._member_map_["TRADING_HALT"] = new_member
        BinanceFuturesContractStatus._member_names_.append("TRADING_HALT")
        BinanceFuturesContractStatus._value2member_map_["TRADING_HALT"] = new_member
        
        _log.warning("Nautilus enum patch active: TRADING_HALT added (remove after upgrade to >=1.222.0)")
        
    except Exception as e:
        _log.error(f"Failed to patch BinanceFuturesContractStatus: {e}")

# Auto-apply on import
apply()
