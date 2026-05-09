#!/bin/bash
# =============================================================================
# Phase 1 data download script for AtomiCortex v2.0
#
# Downloads MTF klines (1h, 15m, 5m, 1m) from Binance Data Portal,
# converts to Parquet, and runs quality checks.
#
# Usage:
#   screen -S mtf_download -dm bash scripts/run_phase1_download.sh
#
# Monitor:
#   tail -f logs/phase1_download_*.log
#   screen -r mtf_download
# =============================================================================

set -e  # exit on error

cd ~/AtomiCortex
source .venv/bin/activate

LOG="logs/phase1_download_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Redirect all output to log AND terminal
exec > >(tee -a "$LOG") 2>&1

echo "=== Phase 1 Download Started: $(date) ==="
echo "=== Log file: $LOG ==="
echo ""

# Verify 4H bot is running
echo "--- Checking 4H bot status ---"
sudo systemctl status atomicortex-bot.service --no-pager | head -5 || true
echo ""

for INTERVAL in 1h 15m 5m 1m; do
    echo "========================================"
    echo "--- Downloading $INTERVAL: $(date) ---"
    echo "========================================"
    python scripts/download_mtf_data.py --interval $INTERVAL --symbol BTCUSDT \
        --start 2023-01 --end 2025-12

    echo ""
    echo "--- Converting $INTERVAL to Parquet: $(date) ---"
    python scripts/convert_mtf_to_parquet.py --interval $INTERVAL

    echo ""
    echo "--- $INTERVAL complete: $(date) ---"
    echo ""
done

echo "========================================"
echo "--- Running quality check: $(date) ---"
echo "========================================"
python scripts/check_mtf_data_quality.py --all

echo ""
echo "========================================"
echo "--- Running verification: $(date) ---"
echo "========================================"
python scripts/verify_phase1.py

echo ""
echo "--- Disk usage ---"
du -sh data/raw/exchange=BINANCE_UM/symbol=BTCUSDT/interval=*/

echo ""
echo "=== Phase 1 Download COMPLETE: $(date) ==="
