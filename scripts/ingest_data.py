"""
scripts/ingest_data.py
──────────────────────
Universal Ingestion Script for Lionel Messi Knowledge Base.
Supports both Markdown (.md) and CSV (.csv) datasets using DocumentLoader.
"""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from stage2_memory.document_loader import document_loader

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DB_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "car_inventory_kb")


def ingest_all(force: bool = True):
    print(f"\n=======================================================")
    print(f"📦 INGESTING KNOWLEDGE BASE (Markdown + CSV)")
    print(f"Target DB: {DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"=======================================================\n")

    ef = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    if force:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"• Reset existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []

    # 1. Load Markdown files
    for md_file in KB_DIR.glob("*.md"):
        print(f"• Loading Markdown: {md_file.name}")
        chunks = document_loader.load_markdown(str(md_file))
        all_chunks.extend(chunks)
        print(f"  ↳ Loaded {len(chunks)} section chunks")

    # 2. Load CSV files
    for csv_file in KB_DIR.glob("*.csv"):
        print(f"• Loading CSV Dataset: {csv_file.name}")
        chunks = document_loader.load_csv(str(csv_file))
        all_chunks.extend(chunks)
        print(f"  ↳ Loaded {len(chunks)} record rows with metadata")

    if not all_chunks:
        print("⚠️ No documents found in", KB_DIR)
        return 0

    ids = [c.chunk_id for c in all_chunks]
    texts = [c.text for c in all_chunks]
    metas = [c.metadata for c in all_chunks]

    col.add(ids=ids, documents=texts, metadatas=metas)
    print(f"\n✓ Successfully indexed {len(all_chunks)} chunks in ChromaDB ('{COLLECTION_NAME}').")
    return len(all_chunks)


if __name__ == "__main__":
    force_reload = "--force" in sys.argv or True
    ingest_all(force=force_reload)

