from pathlib import Path
from src.execution.strategies.ml_strategy import MLTradingStrategy
from scripts.run_reconciler import resolve_trading_mode
import argparse

def test_no_reconciler_methods_on_strategy():
    for name in ("_schedule_reconciliation", "_run_reconciliation", "_reconcile_async"):
        assert not hasattr(MLTradingStrategy, name)

def test_no_position_reconciler_import():
    content = Path("src/execution/strategies/ml_strategy.py").read_text()
    assert "PositionReconciler" not in content

def test_reconciler_argparse_default_is_testnet():
    from scripts.run_reconciler import get_parser
    args = get_parser().parse_args([])
    assert args.trading_mode == "testnet"

def test_empty_trading_mode_falls_back_to_testnet():
    assert resolve_trading_mode("") == "testnet"
    assert resolve_trading_mode("LIVE") == "live"
    assert resolve_trading_mode("garbage") == "testnet"
    assert resolve_trading_mode(None) == "testnet"
    assert resolve_trading_mode("testnet") == "testnet"

def test_reconciler_service_no_hardcoded_live():
    content = Path("deploy/atomicortex-reconciler.service").read_text()
    assert "--trading-mode live" not in content
    assert "--trading-mode ${TRADING_MODE}" in content
    
    lines = content.splitlines()
    env_testnet_idx = next(i for i, line in enumerate(lines) if "Environment=TRADING_MODE=testnet" in line)
    env_file_idx = next(i for i, line in enumerate(lines) if line.startswith("EnvironmentFile="))
    assert env_testnet_idx < env_file_idx

def test_paper_mode_maps_to_testnet():
    from scripts.run_reconciler import resolve_trading_mode
    assert resolve_trading_mode("paper") == "testnet"

def test_argparse_accepts_paper():
    from scripts.run_reconciler import get_parser
    args = get_parser().parse_args(["--trading-mode", "paper"])
    assert args.trading_mode == "paper"
