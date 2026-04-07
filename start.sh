#!/usr/bin/env bash
set -euo pipefail

# Glacier Vision API startup script
# Usage: ./start.sh [--port PORT] [--host HOST] [--reload]

PORT=8000
HOST=0.0.0.0
RELOAD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --reload) RELOAD="--reload"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Ensure uv is available
if ! command -v uv &>/dev/null; then
  echo "uv is not installed. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# Install / sync dependencies (fast no-op if already in sync)
echo "Syncing dependencies..."
uv sync --quiet

# Suppress TF verbose logging unless user already set it
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

echo "Starting Glacier Vision API on http://${HOST}:${PORT}"
echo "  Docs: http://localhost:${PORT}/docs"
echo ""

uv run uvicorn api.app:app \
  --host "$HOST" \
  --port "$PORT" \
  $RELOAD
