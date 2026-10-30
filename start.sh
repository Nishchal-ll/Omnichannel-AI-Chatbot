#!/bin/bash
set -e
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PORT="${PORT:-8080}"

echo "================================================"
echo "  Naasa Securities (Broker 58) WhatsApp Support"
echo "  Dashboard & Webhook → http://0.0.0.0:${PORT}"
echo "  LLM                 → ${LLM_URL:-http://localhost:11434}"
echo "  Model               → ${LLM_MODEL:-gemma4:e2b}"
echo "================================================"

echo "=== Ingesting knowledge base ==="
python scripts/ingest_kb.py

exec python src/webhook_server.py
