"""
knowledge.py
────────────
Chunks KB files → embeds with all-MiniLM-L6-v2 → stores in ChromaDB.
At query time: embeds query → returns top-k chunks as one context string.

Design constraints:
- Each chunk is ≤150 words → always under all-MiniLM-L6-v2's 512-token limit
- top_k=3 → ~300-350 words of context → ~420 tokens → fits 1.5B context window
"""

import re
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

log = logging.getLogger(__name__)

KB_DIR   = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DB_PATH  = str(Path(__file__).resolve().parent.parent / "chroma_db")

# Embedding model: 384-dim vectors, runs on CPU, ~50ms per query
_ef      = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
)
_client  = chromadb.PersistentClient(path=DB_PATH)
_col     = _client.get_or_create_collection(
    name="coconut_kb",
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},
)


# ── Chunker ───────────────────────────────────────────────────────
def _chunk(text: str, size: int = 150, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-level chunks.
    size=150 words stays safely under the 512-token embedding model limit.
    overlap=20 prevents losing context at chunk boundaries.
    """
    text  = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if not words:
        return []
    chunks, step = [], size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        if i + size >= len(words):
            break
    return chunks


# ── Ingest ────────────────────────────────────────────────────────
def ingest_kb(force: bool = False) -> int:
    """
    Load all .md and .txt files from data/knowledge/ into ChromaDB.
    Skip if already loaded unless force=True.
    Returns number of chunks stored.
    """
    if _col.count() > 0 and not force:
        log.info("KB already loaded (%d chunks). Pass force=True to reload.", _col.count())
        return _col.count()

    # Clear before force re-ingestion
    if force:
        # Get all IDs and delete them to reset the collection cleanly
        existing_count = _col.count()
        if existing_count > 0:
            # Delete by query or all ids
            all_ids = _col.get()["ids"]
            if all_ids:
                _col.delete(ids=all_ids)

    if not KB_DIR.exists():
        log.warning("data/knowledge/ not found. Create it and add .md files.")
        return 0

    all_files = list(KB_DIR.glob("*.md")) + list(KB_DIR.glob("*.txt"))
    if not all_files:
        log.warning("No .md or .txt files found in data/knowledge/")
        return 0

    all_chunks, all_ids, all_metas = [], [], []

    for path in sorted(all_files):
        text   = path.read_text(encoding="utf-8")
        chunks = _chunk(text)
        for i, chunk in enumerate(chunks):
            cid = f"{path.stem}_c{i}"
            all_chunks.append(chunk)
            all_ids.append(cid)
            all_metas.append({"source": path.name, "chunk_index": i})

    if not all_chunks:
        return 0

    # Upsert in batches of 100 to avoid memory spikes
    batch = 100
    for i in range(0, len(all_chunks), batch):
        _col.upsert(
            ids       = all_ids[i : i + batch],
            documents = all_chunks[i : i + batch],
            metadatas = all_metas[i : i + batch],
        )

    log.info("KB ingested: %d chunks from %d files.", len(all_chunks), len(all_files))
    return len(all_chunks)


# ── Retrieve ──────────────────────────────────────────────────────
def get_relevant_knowledge(query: str, top_k: int = 3) -> str:
    """
    Embed query → cosine search → return top_k chunks joined as one string.
    Total output is ~300 words / ~420 tokens — fits inside 1.5B context window.
    Auto-ingests KB on first call if ChromaDB is empty.
    """
    if _col.count() == 0:
        loaded = ingest_kb()
        if loaded == 0:
            return ""

    k       = min(top_k, _col.count())
    if k == 0:
        return ""
    results = _col.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    docs    = results.get("documents", [[]])[0]
    metas   = results.get("metadatas", [[]])[0]

    # Join with source label so agent knows where each fact came from
    parts = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "kb")
        parts.append(f"[{source}]\n{doc}")

    return "\n\n".join(parts)


def kb_stats() -> dict:
    return {"total_chunks": _col.count(), "db_path": DB_PATH}
