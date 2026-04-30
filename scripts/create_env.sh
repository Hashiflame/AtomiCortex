#!/usr/bin/env bash
# AtomiCortex — bootstrap .env from .env.example
# Usage: bash scripts/create_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_FILE="$ROOT_DIR/.env"
EXAMPLE_FILE="$ROOT_DIR/.env.example"

if [ ! -f "$EXAMPLE_FILE" ]; then
    echo "❌ .env.example not found at $EXAMPLE_FILE"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    echo "✅ .env создан. Заполни API ключи в файле: $ENV_FILE"
else
    echo "⚠️  .env уже существует, не перезаписываем: $ENV_FILE"
fi
