"""
knowledge.py
────────────
RAG Knowledge Base Engine for Naasa Securities Co. Ltd. (Broker No. 58).
- Chunks markdown knowledge files section-by-section (preserving headers & context).
- Embeds using all-MiniLM-L6-v2 (384-dimensional dense vectors).
- Stores in ChromaDB persistent collection 'naasa_broker58_kb'.
- Performs semantic retrieval with cosine distance thresholding.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

log = logging.getLogger(__name__)

KB_DIR   = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DB_PATH  = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "naasa_broker58_kb"

# Embedding model: 384-dim vectors, CPU-optimized, ~30-50ms latency
_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
)
_client = chromadb.PersistentClient(path=DB_PATH)
_col = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},
)


# ── Markdown Section-Aware Chunker ─────────────────────────────────
def chunk_markdown_file(text: str, filename: str) -> List[Dict[str, str]]:
    """
    Splits markdown by '## ' section headers so each product, policy,
    or procedure remains an atomic, complete unit with its title preserved.
    """
    sections = re.split(r'\n(?=##\s+)', text.strip())
    chunks = []

    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue

        # Extract title from first line if available
        lines = section.split('\n')
        first_line = lines[0].strip()
        if first_line.startswith('##'):
            title = first_line.lstrip('#').strip()
            body = '\n'.join(lines[1:]).strip()
        else:
            title = "Overview"
            body = section

        # Format chunk with clear context header
        chunk_text = f"[{title}]\n{body if body else title}"

        chunks.append({
            "id": f"{Path(filename).stem}_sec{i}",
            "text": chunk_text,
            "title": title,
            "source": filename,
            "section_index": i
        })

    return chunks


# ── Ingest ────────────────────────────────────────────────────────
def ingest_kb(force: bool = False) -> int:
    """
    Load all .md and .txt files from data/knowledge/ into ChromaDB.
    Skip if already loaded unless force=True.
    Returns number of chunks stored.
    """
    global _col

    if _col.count() > 0 and not force:
        log.info("Naasa KB already loaded (%d chunks). Pass force=True to reload.", _col.count())
        return _col.count()

    if force:
        try:
            _client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        _col = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef,
            metadata={"hnsw:space": "cosine"},
        )

    if not KB_DIR.exists():
        log.warning("data/knowledge/ not found at %s", KB_DIR)
        return 0

    all_files = sorted(list(KB_DIR.glob("*.md")) + list(KB_DIR.glob("*.txt")))
    if not all_files:
        log.warning("No knowledge files found in %s", KB_DIR)
        return 0

    all_chunks = []
    for path in all_files:
        text = path.read_text(encoding="utf-8")
        file_chunks = chunk_markdown_file(text, path.name)
        all_chunks.extend(file_chunks)

    if not all_chunks:
        return 0

    # Batch upsert
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        _col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "title": c["title"], "section_index": c["section_index"]} for c in batch],
        )

    log.info("Naasa KB ingested: %d chunks from %d files.", len(all_chunks), len(all_files))
    return len(all_chunks)


# ── Semantic Retrieval ────────────────────────────────────────────
def retrieve_knowledge(query: str, top_k: int = 3, max_distance: float = 0.85) -> List[Dict]:
    """
    Searches ChromaDB using cosine similarity.
    Filters out chunks with distance > max_distance (cosine distance: 0=identical, 1=orthogonal).
    Returns list of dicts with document, metadata, and distance.
    """
    if _col.count() == 0:
        loaded = ingest_kb()
        if loaded == 0:
            return []

    k = min(top_k, _col.count())
    if k == 0:
        return []

    results = _col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matched = []
    for doc, meta, dist in zip(docs, metas, distances):
        if dist <= max_distance:
            matched.append({
                "text": doc,
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
                "distance": dist
            })

    return matched


def get_relevant_knowledge(query: str, top_k: int = 3, max_distance: float = 0.85) -> str:
    """
    Returns formatted knowledge string suitable for injection into system prompt.
    Returns empty string if no relevant knowledge passes the distance threshold.
    """
    chunks = retrieve_knowledge(query, top_k=top_k, max_distance=max_distance)
    if not chunks:
        return ""

    parts = []
    for c in chunks:
        parts.append(f"[{c['title']} | Source: {c['source']}]\n{c['text']}")

    return "\n\n".join(parts)


def kb_stats() -> dict:
    return {"total_chunks": _col.count(), "db_path": DB_PATH, "collection": COLLECTION_NAME}
