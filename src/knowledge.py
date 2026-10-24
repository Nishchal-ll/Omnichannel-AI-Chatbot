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
COLLECTION_NAME = "messi_kb"

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
    Splits markdown by '## ' section headers.
    If a section exceeds 800 characters, it sub-chunks cleanly by bullet points
    or paragraphs, always prepending the section title for rich semantic grounding.
    """
    sections = re.split(r'\n(?=##\s+)', text.strip())
    chunks = []
    chunk_idx = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.split('\n')
        first_line = lines[0].strip()
        if first_line.startswith('##'):
            title = first_line.lstrip('#').strip()
            body = '\n'.join(lines[1:]).strip()
        else:
            title = "Overview"
            body = section

        # If body is under 850 chars, keep as one cohesive chunk
        if len(body) <= 850:
            chunks.append({
                "id": f"{Path(filename).stem}_c{chunk_idx}",
                "text": f"[{title}]\n{body}",
                "title": title,
                "source": Path(filename).name,
                "section_index": chunk_idx
            })
            chunk_idx += 1
        else:
            # Sub-chunk by bullet points or paragraphs (~500-750 chars)
            paragraphs = [p.strip() for p in re.split(r'\n(?=[-\*]\s+|\n+)', body) if p.strip()]
            current_sub = []
            current_len = 0

            for p in paragraphs:
                if current_len + len(p) > 750 and current_sub:
                    sub_text = "\n".join(current_sub)
                    chunks.append({
                        "id": f"{Path(filename).stem}_c{chunk_idx}",
                        "text": f"[{title}]\n{sub_text}",
                        "title": title,
                        "source": Path(filename).name,
                        "section_index": chunk_idx
                    })
                    chunk_idx += 1
                    current_sub = [p]
                    current_len = len(p)
                else:
                    current_sub.append(p)
                    current_len += len(p)

            if current_sub:
                sub_text = "\n".join(current_sub)
                chunks.append({
                    "id": f"{Path(filename).stem}_c{chunk_idx}",
                    "text": f"[{title}]\n{sub_text}",
                    "title": title,
                    "source": Path(filename).name,
                    "section_index": chunk_idx
                })
                chunk_idx += 1

    return chunks


# ── Ingest ────────────────────────────────────────────────────────
def ingest_kb(force: bool = False) -> int:
    """
    Load knowledge files into ChromaDB.
    Prioritizes lionel_messi_kb.md for targeted Lionel Messi QA.
    """
    global _col

    if _col.count() > 0 and not force:
        log.info("KB already loaded (%d chunks). Pass force=True to reload.", _col.count())
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

    # Specifically use Lionel Messi KB for accurate QA
    messi_file = KB_DIR / "lionel_messi_kb.md"
    if messi_file.exists():
        all_files = [messi_file]
    else:
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


# In-memory LRU-style query cache to eliminate SentenceTransformers CPU re-computation
_QUERY_CACHE: Dict[str, Tuple[str, List[Dict]]] = {}
_MAX_CACHE_SIZE = 256

# Non-domain / conversational words that should never trigger vector embedding search
TRIVIAL_WORDS = {"ok", "okay", "yes", "no", "sure", "fine", "cool", "alright", "got it", "thanks", "thank you", "hello", "hi", "hey"}


# ── Semantic Retrieval ────────────────────────────────────────────
def retrieve_knowledge(query: str, top_k: int = 3, max_distance: float = 0.78) -> List[Dict]:
    """
    Searches ChromaDB using cosine similarity.
    Filters out chunks with distance > max_distance (cosine distance: 0=identical, 1=orthogonal).
    Returns list of dicts with document, metadata, and distance.
    """
    clean_q = query.strip()
    if len(clean_q) < 4 or clean_q.lower() in TRIVIAL_WORDS:
        return []

    if _col.count() == 0:
        loaded = ingest_kb()
        if loaded == 0:
            return []

    k = min(top_k, _col.count())
    if k == 0:
        return []

    results = _col.query(
        query_texts=[clean_q],
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


def get_relevant_knowledge_with_metadata(query: str, top_k: int = 3, max_distance: float = 0.78) -> Tuple[str, List[Dict]]:
    """
    Returns (formatted_context, chunks_list).
    Cached in memory to drop repeated/common queries from ~1.8s down to 0.1ms.
    """
    clean_key = query.strip().lower()
    if clean_key in _QUERY_CACHE:
        return _QUERY_CACHE[clean_key]

    chunks = retrieve_knowledge(query, top_k=top_k, max_distance=max_distance)
    if not chunks:
        res = ("", [])
        if len(_QUERY_CACHE) < _MAX_CACHE_SIZE:
            _QUERY_CACHE[clean_key] = res
        return res

    parts = []
    for c in chunks:
        parts.append(f"[{c['title']} | Source: {c['source']}]\n{c['text']}")

    res = ("\n\n".join(parts), chunks)
    if len(_QUERY_CACHE) < _MAX_CACHE_SIZE:
        _QUERY_CACHE[clean_key] = res
    return res


def get_relevant_knowledge(query: str, top_k: int = 3, max_distance: float = 0.85) -> str:
    """
    Returns formatted knowledge string suitable for injection into system prompt.
    Returns empty string if no relevant knowledge passes the distance threshold.
    """
    context, _ = get_relevant_knowledge_with_metadata(query, top_k=top_k, max_distance=max_distance)
    return context


def kb_stats() -> dict:
    return {"total_chunks": _col.count(), "db_path": DB_PATH, "collection": COLLECTION_NAME}
