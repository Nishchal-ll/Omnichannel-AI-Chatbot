"""
server.py
─────────
Unified Root-Level Server Launcher for Apex Auto Recondition Nepal.
Pre-flight health checks:
1. Validates local Ollama connection (http://127.0.0.1:11434)
2. Confirms llama3.2:latest and nomic-embed-text models
3. Validates 31-chunk Vector Database and SQLite Memory MESH
4. Launches Uvicorn hosting the Webhook endpoints & Dealership Reports Console
"""

import os
import sys
import argparse
import requests
import uvicorn
from pathlib import Path

# Ensure 'src' is in python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from dotenv import load_dotenv
load_dotenv(override=True)


def check_preflight(ollama_url: str = "http://127.0.0.1:11434") -> bool:
    """Performs pre-flight checks before server boot."""
    print("\n" + "=" * 65)
    print("🚀 APEX AUTO RECONDITION NEPAL — SERVER PRE-FLIGHT CHECK")
    print("=" * 65)

    # 1. Check Ollama server
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            print(f"✓ Ollama Daemon    : ONLINE at {ollama_url}")
            
            # Check generation model
            has_llama = any("llama3.2" in m for m in models)
            if has_llama:
                print(f"✓ Generation Model : READY (llama3.2 resident)")
            else:
                print(f"⚠️ Generation Model: 'llama3.2' not found. Available: {models}")

            # Check embedding model
            has_embed = any("nomic-embed-text" in m for m in models)
            if has_embed:
                print(f"✓ Embedding Model  : READY (nomic-embed-text resident)")
            else:
                print(f"⚠️ Embedding Model : 'nomic-embed-text' not found. Will use fallback.")
        else:
            print(f"⚠️ Ollama returned HTTP {r.status_code}")
    except Exception as e:
        print(f"⚠️ Ollama Daemon    : OFFLINE or unreachable ({e})")
        print(f"  ↳ Note: Run 'ollama serve' in background if not running.")

    # 2. Check Vector Database
    vdb_dir = ROOT_DIR / "data" / "vector_db"
    vec_file = vdb_dir / "vectors.npy"
    chunk_file = vdb_dir / "chunks.json"
    if vec_file.exists() and chunk_file.exists():
        import json
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✓ Offline Vector DB: LOADED ({len(chunks)} certified Nepali car & policy chunks)")
    else:
        print(f"⚠️ Offline Vector DB: Not found in {vdb_dir}. Run: python scripts/ingest_data.py")

    # 3. Check SQLite Memory Store
    db_file = ROOT_DIR / "chatbot.db"
    if db_file.exists():
        print(f"✓ SQLite Memory    : READY ({db_file.name})")
    else:
        print(f"✓ SQLite Memory    : Will be initialized on first request ({db_file.name})")

    print("=" * 65)
    return True


def main():
    parser = argparse.ArgumentParser(description="Apex Auto Recondition Nepal Server Launcher")
    parser.add_argument("--host", default=os.getenv("SERVER_HOST", "0.0.0.0"), help="Host IP to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8080")), help="Port to listen on (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for local development")
    args = parser.parse_args()

    check_preflight(os.getenv("LLM_URL", "http://127.0.0.1:11434"))

    print(f"\n🌐 Starting Webhook Server & Analytics Dashboard:")
    print(f"   • Dealership Reports Console : http://localhost:{args.port}")
    print(f"   • WhatsApp Cloud API Webhook  : http://localhost:{args.port}/webhook")
    print(f"   • Messenger API Webhook       : http://localhost:{args.port}/messenger-webhook")
    print(f"   • System Health Endpoint      : http://localhost:{args.port}/api/health")
    print("\nPress Ctrl+C to stop.\n")

    uvicorn.run(
        "src.webhook_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
