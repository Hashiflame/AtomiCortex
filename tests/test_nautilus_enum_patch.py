import pytest
import msgspec
import sys

# Remove patch module from sys.modules if it's there to ensure tests are isolated
def _reset_patch():
    if "src.patches.nautilus_enums" in sys.modules:
        del sys.modules["src.patches.nautilus_enums"]

@pytest.fixture(autouse=True)
def setup_teardown():
    _reset_patch()
    yield
    _reset_patch()

def test_trading_halt_in_enum_after_patch():
    import src.patches.nautilus_enums
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesContractStatus
    
    # This should not raise ValueError
    assert BinanceFuturesContractStatus("TRADING_HALT") == BinanceFuturesContractStatus.TRADING_HALT

def test_patch_idempotent():
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesContractStatus
    import src.patches.nautilus_enums
    
    # First application was on import. Second call:
    src.patches.nautilus_enums.apply()
    
    # Verify exact length (8 original + 1 added = 9)
    assert len(BinanceFuturesContractStatus._member_names_) == 9
    
    # Verify exactly one TRADING_HALT
    assert BinanceFuturesContractStatus._member_names_.count("TRADING_HALT") == 1
    
    # Verify it resolves correctly
    assert BinanceFuturesContractStatus("TRADING_HALT") == BinanceFuturesContractStatus.TRADING_HALT

def test_existing_values_preserved():
    import src.patches.nautilus_enums
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesContractStatus
    
    expected_values = [
        "PENDING_TRADING", "TRADING", "PRE_DELIVERING", "DELIVERING",
        "DELIVERED", "PRE_SETTLE", "SETTLING", "CLOSE"
    ]
    for val in expected_values:
        assert BinanceFuturesContractStatus(val).value == val

def test_msgspec_decoder_accepts_trading_halt():
    import src.patches.nautilus_enums
    from nautilus_trader.adapters.binance.futures.enums import BinanceFuturesContractStatus
    from nautilus_trader.adapters.binance.futures.schemas.market import BinanceFuturesSymbolInfo
    
    class MiniStruct(msgspec.Struct):
        status: BinanceFuturesContractStatus
        
    decoder = msgspec.json.Decoder(MiniStruct)
    
    # Test 1: mini struct
    result = decoder.decode(b'{"status": "TRADING_HALT"}')
    assert result.status == BinanceFuturesContractStatus.TRADING_HALT
    
    # Test 2: Real BinanceFuturesSymbolInfo
    symbol_decoder = msgspec.json.Decoder(BinanceFuturesSymbolInfo)
    
    json_payload = b'''{
        "symbol": "BTCUSDT",
        "pair": "BTCUSDT",
        "contractType": "PERPETUAL",
        "deliveryDate": 0,
        "onboardDate": 0,
        "status": "TRADING_HALT",
        "maintMarginPercent": "0.4000",
        "requiredMarginPercent": "5.0000",
        "baseAsset": "BTC",
        "quoteAsset": "USDT",
        "marginAsset": "USDT",
        "pricePrecision": 2,
        "quantityPrecision": 3,
        "baseAssetPrecision": 8,
        "quotePrecision": 8,
        "underlyingType": "COIN",
        "underlyingSubType": [],
        "settlePlan": 0,
        "triggerProtect": "0.05",
        "liquidationFee": "0.01",
        "marketTakeBound": "0.05",
        "filters": [],
        "orderTypes": [],
        "timeInForce": []
    }'''
    
    symbol_info = symbol_decoder.decode(json_payload)
    assert symbol_info.status == BinanceFuturesContractStatus.TRADING_HALT
