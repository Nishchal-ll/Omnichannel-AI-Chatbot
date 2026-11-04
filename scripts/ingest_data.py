"""
scripts/ingest_data.py
──────────────────────
Offline Ingestion Script for Apex Auto Recondition Nepal Knowledge Base.
Generates:
1. FAISS & NumPy Cosine Vector Database (data/vector_db/) with nomic-embed-text
2. ChromaDB persistent collection (chroma_db/) for backward-compatibility
"""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stage2_memory.document_loader import document_loader
from stage2_memory.text_chunker import TextChunker, Chunk
from stage2_memory.ollama_embedder import ollama_embedder
from stage2_memory.vector_db import vector_db

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
VECTOR_DB_DIR = Path(__file__).resolve().parent.parent / "data" / "vector_db"
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "car_inventory_kb")


def ingest_all(force: bool = True):
    print(f"\n=======================================================")
    print(f"📦 INGESTING APEX AUTO RECONDITION NEPAL KNOWLEDGE BASE")
    print(f"Source Directory: {KB_DIR}")
    print(f"Target Vector DB: {VECTOR_DB_DIR}")
    print(f"Embedding Model : nomic-embed-text (768-dim)")
    print(f"=======================================================\n")

    all_doc_chunks = []

    # 1. Load Markdown files (Showroom policies, warranties, Satasat exchange, Bluebook Namsari)
    for md_file in KB_DIR.glob("*.md"):
        print(f"• Loading Markdown: {md_file.name}")
        chunks = document_loader.load_markdown(str(md_file))
        all_doc_chunks.extend(chunks)
        print(f"  ↳ Loaded {len(chunks)} policy section chunks")

    # 2. Load CSV files (Certified cars inventory: Santro, i10, Magnite, Swift, Alto, Brezza, Creta)
    for csv_file in KB_DIR.glob("*.csv"):
        print(f"• Loading CSV Inventory: {csv_file.name}")
        chunks = document_loader.load_csv(str(csv_file))
        all_doc_chunks.extend(chunks)
        print(f"  ↳ Loaded {len(chunks)} verified car record rows")

    if not all_doc_chunks:
        print("⚠️ No documents found in", KB_DIR)
        return 0

    print(f"\nGenerating sentence-boundary chunks and nomic-embed-text vectors for {len(all_doc_chunks)} items...")

    faiss_chunks = []
    texts_to_embed = []

    for i, c in enumerate(all_doc_chunks):
        chunk_obj = Chunk(
            id=c.chunk_id,
            text=c.text,
            metadata=c.metadata.copy()
        )
        chunk_obj.metadata["title"] = c.title
        chunk_obj.metadata["source"] = c.source
        faiss_chunks.append(chunk_obj)
        texts_to_embed.append(c.text)

    # Generate embeddings via Ollama nomic-embed-text
    print(f"Calling Ollama nomic-embed-text (/api/embeddings)...")
    vectors_list = []
    for idx, text in enumerate(texts_to_embed):
        vec = ollama_embedder.embed_text(text)
        vectors_list.append(vec)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(texts_to_embed):
            print(f"  ↳ Embedded {idx + 1}/{len(texts_to_embed)} chunks")

    vectors_arr = np.array(vectors_list, dtype=np.float32)

    # 3. Save into VectorDatabase (FAISS / NumPy)
    vector_db.chunks = []
    vector_db.vectors = None
    if vector_db.faiss_index is not None:
        vector_db.faiss_index.reset()

    vector_db.add_chunks(faiss_chunks, vectors_arr)
    vector_db.save(str(VECTOR_DB_DIR))
    print(f"✓ Successfully persisted {len(faiss_chunks)} chunks to {VECTOR_DB_DIR}")

    # 4. Also update ChromaDB collection for backward compatibility
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        if force:
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
        ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cpu")
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )
        ids = [c.chunk_id for c in all_doc_chunks]
        texts = [c.text for c in all_doc_chunks]
        metas = [c.metadata for c in all_doc_chunks]
        col.add(ids=ids, documents=texts, metadatas=metas)
        print(f"✓ Synchronized ChromaDB collection '{COLLECTION_NAME}' ({len(all_doc_chunks)} chunks)")
    except Exception as e:
        print(f"⚠️ ChromaDB sync skipped (optional): {e}")

    print(f"\n🎉 Ingestion complete! Knowledge base is fully offline and operational.")
    return len(faiss_chunks)


if __name__ == "__main__":
    ingest_all(force=True)
