#!/bin/bash
set -e
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "================================================"
echo "  CrewAI WhatsApp + Email Bot"
echo "  Webhook   → :5001"
echo "  Dashboard → :8501"
echo "  LLM       → ${LLM_URL:-http://localhost:11434}"
echo "  Model     → ${LLM_MODEL:-gemma4:e2b}"
echo "================================================"

echo "=== Ingesting knowledge base ==="
python scripts/ingest_kb.py

python src/webhook_server.py &
python src/email_poller.py &
sleep 2

exec streamlit run src/streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
