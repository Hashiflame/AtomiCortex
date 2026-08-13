#!/usr/bin/env bash
#
# AtomiCortex Systemd Unit Installation script.
#
# DO NOT RUN FROM THE REPOSITORY DIRECTORY DURING DEPLOYMENT.
# This script should be copied to /usr/local/sbin/atomicortex-install-units
# and owned by root. The deployment pipeline will call it via sudo -n.
#
# REQUIRED sudoers configuration:
# hashiflame ALL=(ALL) NOPASSWD: /usr/local/sbin/atomicortex-install-units

set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (or with sudo)"
  exit 1
fi

REPO_DIR="/home/hashiflame/AtomiCortex/deploy"
TARGET_DIR="/etc/systemd/system/"

echo "==> Copying systemd units..."
cp "$REPO_DIR"/*.service "$TARGET_DIR"
cp "$REPO_DIR"/*.timer "$TARGET_DIR"

echo "==> Reloading daemon..."
systemctl daemon-reload

LONG_RUNNING_SERVICES=(
    atomicortex-api
    atomicortex-bot
    atomicortex-bot-15m
    atomicortex-telegram
    atomicortex-watchdog
    atomicortex-watchdog-15m
)

echo "==> Restarting long-running services..."
for service in "${LONG_RUNNING_SERVICES[@]}"; do
    echo "Restarting ${service}..."
    systemctl restart "${service}" || echo "WARNING: Failed to restart ${service}"
done

echo "==> Enabling timers..."
systemctl enable --now atomicortex-reconciler.timer
systemctl enable --now atomicortex-signal-check.timer

echo "==> Performing health-check..."
for service in "${LONG_RUNNING_SERVICES[@]}"; do
    if ! systemctl is-active --quiet "${service}"; then
        echo "ERROR: ${service} is not active!"
        exit 1
    fi
done

echo "==> All units installed and active successfully."
