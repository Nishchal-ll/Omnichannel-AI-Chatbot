#!/bin/bash
set -e
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

PORT="${PORT:-8080}"

echo "================================================"
echo "  Apex Auto Recondition House AI Assistant"
echo "  Web Console & Webhook → http://0.0.0.0:${PORT}"
echo "  LLM                   → ${LLM_URL:-http://localhost:11434}"
echo "  Model                 → ${LLM_MODEL:-llama3.2}"
echo "================================================"

echo "=== Ingesting car inventory knowledge base ==="
python scripts/ingest_data.py

exec python src/webhook_server.py
