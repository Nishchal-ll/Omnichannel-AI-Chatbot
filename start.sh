#!/bin/bash
set -e
echo "================================================"
echo "  CrewAI WhatsApp + Email Bot"
echo "  Webhook  → :5001"
echo "  Dashboard → :8501"
echo "  LLM      → $LLM_URL"
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
