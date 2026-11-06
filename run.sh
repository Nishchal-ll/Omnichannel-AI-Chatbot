#!/usr/bin/env bash
# ==============================================================================
# Apex Auto Recondition Nepal — Master One-Command Launcher (run.sh)
# ==============================================================================
# Usage:
#   bash run.sh             -> Starts Ollama (if offline) + Interactive Terminal Chatbot
#   bash run.sh --server    -> Starts Ollama (if offline) + Webhook Server & Reports UI (:8080)
#   bash run.sh --bench     -> Runs automated 5-turn benchmark suite with latency report
#   bash run.sh --query "q" -> Runs a single query with latency breakdown and exits
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

OLLAMA_PORT=11434
OLLAMA_URL="http://127.0.0.1:$OLLAMA_PORT"

# 1. Check if Ollama Daemon is running
echo "----------------------------------------------------------------------"
echo "🔍 Checking Ollama Service at $OLLAMA_URL..."

if curl -s -f "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "✓ Ollama is already running."
else
    echo "⚙️ Ollama is not running. Starting 'ollama serve' in background..."
    nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
    
    # Wait up to 10 seconds for Ollama to become active
    READY=0
    for i in {1..20}; do
        sleep 0.5
        if curl -s -f "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
            READY=1
            break
        fi
    done

    if [ "$READY" -eq 1 ]; then
        echo "✓ Ollama successfully started on port $OLLAMA_PORT."
    else
        echo "⚠️ Warning: Could not confirm Ollama startup within 10s. Logs at /tmp/ollama_serve.log."
    fi
fi

# 2. Check Models (llama3.2 & nomic-embed-text)
echo "🔍 Checking resident models..."
TAGS=$(curl -s "$OLLAMA_URL/api/tags" || echo "")
if [[ "$TAGS" != *"llama3.2"* ]]; then
    echo "⚠️ 'llama3.2' not detected. Pulling llama3.2 (2.0 GB)..."
    ollama pull llama3.2
fi
if [[ "$TAGS" != *"nomic-embed-text"* ]]; then
    echo "⚠️ 'nomic-embed-text' not detected. Pulling nomic-embed-text (274 MB)..."
    ollama pull nomic-embed-text
fi
echo "✓ Required models (llama3.2, nomic-embed-text) ready."
echo "----------------------------------------------------------------------"

# 3. Dispatch to requested mode
if [ "$1" == "--server" ]; then
    shift
    exec "$PYTHON_BIN" server.py "$@"
elif [ "$1" == "--bench" ]; then
    exec "$PYTHON_BIN" scripts/terminal_chat.py --bench
elif [ "$1" == "--query" ]; then
    shift
    exec "$PYTHON_BIN" scripts/terminal_chat.py --test-query "$*"
else
    exec "$PYTHON_BIN" scripts/terminal_chat.py "$@"
fi

