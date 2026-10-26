"""
src/stage2_memory/retriever.py
──────────────────────────────
RAGRetriever: ChromaDB semantic vector search with in-memory caching.
Encodes queries, retrieves top-k chunks, and returns grounded context.
"""

import os
from typing import List, Dict, Tuple
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
COLLECTION_NAME = "messi_kb"
OOD_DISTANCE = float(os.getenv("OOD_DISTANCE", "0.55"))

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

# In-memory LRU cache to skip embedding computation for repeated/common questions
_QUERY_CACHE: Dict[str, Tuple[str, List[Dict]]] = {}
_MAX_CACHE = 256
TRIVIAL_WORDS = {"ok", "okay", "yes", "no", "sure", "fine", "alright", "thanks", "hello", "hi", "hey"}


class RAGRetriever:
    """Domain knowledge retriever for Stage 2."""

    @staticmethod
    def retrieve_with_metadata(query: str, top_k: int = 4, max_distance: float = 0.82) -> Tuple[str, List[Dict]]:
        """
        Retrieves top-k knowledge chunks from ChromaDB.
        Returns (formatted_context_string, chunks_list).
        """
        clean = query.strip()
        if len(clean) < 4 or clean.lower() in TRIVIAL_WORDS:
            return "", []

        clean_key = clean.lower()
        if clean_key in _QUERY_CACHE:
            return _QUERY_CACHE[clean_key]

        if _col.count() == 0:
            return "", []

        k = min(top_k, _col.count())
        results = _col.query(
            query_texts=[clean],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Section-level diversity deduplication & summary boosting (Item 8)
        is_aggregate_query = any(w in clean.lower() for w in ["all", "list", "every", "total", "how many", "msn", "trio", "records", "trophies"])

        raw_chunks = []
        for doc, meta, dist in zip(docs, metas, distances):
            title = meta.get("title", "")
            is_summary = "summary" in title.lower()
            eff_dist = dist - 0.08 if (is_aggregate_query and is_summary) else dist
            if dist <= max_distance:
                raw_chunks.append({
                    "text": doc,
                    "title": title,
                    "source": meta.get("source", ""),
                    "distance": round(dist, 4),
                    "eff_dist": eff_dist,
                    "snippet": doc[:250],
                })

        if is_aggregate_query:
            raw_chunks.sort(key=lambda x: x["eff_dist"])

        # Diversity filter: avoid duplicate section headers crowding out top_k
        chunks = []
        seen_titles = set()
        for c in raw_chunks:
            if c["title"] not in seen_titles or len(chunks) < 2:
                chunks.append(c)
                seen_titles.add(c["title"])
            if len(chunks) >= top_k:
                break

        if not chunks:
            res = ("", [])
        else:
            parts = [f"[{c['title']} | Source: {c['source']}]\n{c['text']}" for c in chunks]
            res = ("\n\n".join(parts), chunks)

        if len(_QUERY_CACHE) < _MAX_CACHE:
            _QUERY_CACHE[clean_key] = res

        return res


retriever = RAGRetriever()

