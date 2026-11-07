"""
src/stage2_memory/retriever.py
──────────────────────────────
High-performance Offline RAG Retriever with Strict Cosine Distance Gating.
Implements Article 1's architecture:
1. Pure Offline Vector Retrieval with Cosine Similarity (FAISS & NumPy IndexFlatIP)
2. Strict Distance Threshold Gating (distance <= 0.60; rejects out-of-domain noise)
3. Granular Confidence Scoring ('high' | 'medium' | 'low')
4. Structured Citation Formatting ([Source X: File, Lot/Page])
5. ChromaDB fallback compatibility
"""

import os
import logging
from typing import List, Dict, Tuple, Any
from pathlib import Path

from stage2_memory.vector_db import vector_db, VectorDatabase
from stage2_memory.ollama_embedder import ollama_embedder

log = logging.getLogger(__name__)

# Cache for sub-millisecond responses to frequent repeated queries
_QUERY_CACHE: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}
_MAX_CACHE = 256
TRIVIAL_WORDS = {"ok", "okay", "yes", "no", "sure", "fine", "alright", "thanks", "hello", "hi", "hey", "namaste"}

# Distance threshold from Article 1 (0.60 max distance in cosine space)
DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.60"))
OOD_DISTANCE = DISTANCE_THRESHOLD


class RAGRetriever:
    """Domain knowledge retriever with strict cosine threshold filtering and citation metadata."""

    def __init__(self):
        self.vector_db = vector_db
        # Attempt to load precomputed vector index on startup
        self._ensure_loaded()

    @property
    def total_chunks(self) -> int:
        """Returns total active chunks in offline vector database."""
        return len(self.vector_db.chunks)

    @property
    def collection_name(self) -> str:
        """Returns active collection identifier."""
        return "car_inventory_kb"

    @property
    def collection(self):
        """Backward-compatibility shim for code calling retriever.collection.count()."""
        class _CollectionProxy:
            def __init__(self, n: int):
                self._n = n
            def count(self) -> int:
                return self._n
        return _CollectionProxy(len(self.vector_db.chunks))

    def _ensure_loaded(self):
        """Loads cached FAISS/NumPy index from disk if available."""
        if len(self.vector_db.chunks) == 0:
            target_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_db")
            if os.path.exists(target_dir):
                self.vector_db.load(target_dir)

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 4,
        max_distance: float = DISTANCE_THRESHOLD
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieves top-k knowledge chunks with cosine distance <= max_distance.
        Returns:
            (formatted_context_string, chunks_list)
        If no chunks pass the distance threshold, returns ("", []) to trigger polite deflection.
        """
        clean = query.strip()
        if len(clean) < 3 or clean.lower() in TRIVIAL_WORDS:
            return "", []

        clean_key = clean.lower()
        if clean_key in _QUERY_CACHE:
            return _QUERY_CACHE[clean_key]

        self._ensure_loaded()

        # 1. Primary: Article 1 FAISS / NumPy Cosine Vector Database
        if len(self.vector_db.chunks) > 0:
            q_vec = ollama_embedder.embed_text(clean)
            search_res = self.vector_db.search(
                query_vector=q_vec,
                top_k=top_k,
                distance_threshold=max_distance
            )
            filtered_chunks = search_res["chunks"]

            if not filtered_chunks:
                # Strictly reject: question does not match showroom knowledge
                log.info("RAGRetriever: All chunks exceeded distance threshold %.2f for query '%s'", max_distance, clean[:40])
                res = ("", [])
            else:
                formatted_parts = []
                chunk_dicts = []
                for idx, c in enumerate(filtered_chunks, start=1):
                    src = c.metadata.get("source", "Showroom Guide")
                    title = c.metadata.get("title", f"Document Chunk {idx}")
                    lot = c.metadata.get("lot", c.metadata.get("page", "N/A"))
                    formatted_parts.append(
                        f"[Source {idx}: {src} (Lot/Ref: {lot}) - {title}]\n{c.text}"
                    )
                    chunk_dicts.append({
                        "id": c.id,
                        "text": c.text,
                        "title": title,
                        "source": src,
                        "distance": c.distance,
                        "confidence": search_res["confidence"],
                        "snippet": c.text[:220]
                    })

                res = ("\n\n".join(formatted_parts), chunk_dicts)

            if len(_QUERY_CACHE) < _MAX_CACHE:
                _QUERY_CACHE[clean_key] = res
            return res

        # 2. Fallback to ChromaDB if vector_db is not yet generated
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            db_path = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")
            if os.path.exists(db_path):
                client = chromadb.PersistentClient(path=db_path)
                col = client.get_collection(
                    name="car_inventory_kb",
                    embedding_function=SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cpu")
                )
                if col and col.count() > 0:
                    k = min(top_k, col.count())
                    results = col.query(query_texts=[clean], n_results=k, include=["documents", "metadatas", "distances"])
                    docs = results.get("documents", [[]])[0]
                    metas = results.get("metadatas", [[]])[0]
                    distances = results.get("distances", [[]])[0]

                    parts = []
                    c_list = []
                    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
                        if dist <= max_distance + 0.15:
                            parts.append(f"[Source {i}: {meta.get('source', '')} - {meta.get('title', '')}]\n{doc}")
                            c_list.append({
                                "text": doc,
                                "title": meta.get("title", ""),
                                "source": meta.get("source", ""),
                                "distance": round(dist, 4),
                                "snippet": doc[:220]
                            })
                    if parts:
                        return ("\n\n".join(parts), c_list)
        except Exception as e:
            log.debug("Chroma fallback error: %s", e)

        return "", []


retriever = RAGRetriever()
