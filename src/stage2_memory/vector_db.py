"""
src/stage2_memory/vector_db.py
──────────────────────────────
High-performance Cosine Vector Database (FAISS & NumPy L2 Normalized IndexFlatIP).
Follows Article 1's architecture with:
1. Strict Cosine Distance Cutoff (distance < 0.60)
2. Granular Confidence Scoring ('high' | 'medium' | 'low')
3. Fast Binary & JSON Persistence (data/vector_db/)
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from stage2_memory.text_chunker import Chunk
from stage2_memory.ollama_embedder import ollama_embedder, OllamaEmbedder

log = logging.getLogger(__name__)

# Try optional faiss import, fallback to highly-optimized NumPy BLAS
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class VectorDatabase:
    """Cosine Similarity Vector Database with distance threshold filtering."""

    def __init__(self, dimension: int = 768, storage_dir: str = "data/vector_db"):
        self.dimension = dimension
        self.storage_dir = storage_dir
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # shape (N, dimension)
        self.faiss_index = None

        if HAS_FAISS:
            self.faiss_index = faiss.IndexFlatIP(dimension)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """L2 normalizes vectors along axis 1 (or axis 0 for 1D)."""
        if arr.ndim == 1:
            norm = np.linalg.norm(arr)
            return arr / (norm + 1e-10)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-10)

    def add_chunks(self, chunks: List[Chunk], vectors: np.ndarray):
        """Adds chunks and their embedding vectors to the index."""
        if len(chunks) == 0 or len(vectors) == 0:
            return

        norm_vectors = self._normalize(vectors.astype(np.float32))

        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = norm_vectors
        else:
            self.vectors = np.vstack([self.vectors, norm_vectors])

        self.chunks.extend(chunks)

        if HAS_FAISS and self.faiss_index is not None:
            self.faiss_index.add(norm_vectors)

        log.info("VectorDatabase: added %d chunks (total: %d)", len(chunks), len(self.chunks))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        distance_threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Searches for top-k nearest chunks using Cosine Similarity.
        Distance = 1.0 - cosine_similarity (lower = closer match).
        
        Returns:
            {
                "chunks": List[Chunk],
                "confidence": "high" | "medium" | "low",
                "total_matched": int
            }
        """
        if self.vectors is None or len(self.chunks) == 0:
            return {"chunks": [], "confidence": "low", "total_matched": 0}

        q_norm = self._normalize(query_vector.reshape(1, -1).astype(np.float32))

        results: List[Tuple[Chunk, float]] = []

        if HAS_FAISS and self.faiss_index is not None and self.faiss_index.ntotal > 0:
            similarities, indices = self.faiss_index.search(q_norm, min(top_k, len(self.chunks)))
            for sim, idx in zip(similarities[0], indices[0]):
                if 0 <= idx < len(self.chunks):
                    dist = float(1.0 - sim)
                    results.append((self.chunks[idx], dist))
        else:
            # Pure NumPy BLAS cosine matrix multiplication (sub-0.1ms for <50k items)
            similarities = np.dot(self.vectors, q_norm.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            for idx in top_indices:
                dist = float(1.0 - similarities[idx])
                results.append((self.chunks[idx], dist))

        # Filter by strict distance threshold
        filtered_chunks: List[Chunk] = []
        for ch, dist in results:
            if dist <= distance_threshold:
                chunk_copy = Chunk(
                    id=ch.id,
                    text=ch.text,
                    metadata=ch.metadata.copy(),
                    distance=round(dist, 4)
                )
                filtered_chunks.append(chunk_copy)

        # Granular Confidence Scoring (Article 1)
        if len(filtered_chunks) >= 3 and any(c.distance < 0.35 for c in filtered_chunks):
            confidence = "high"
        elif len(filtered_chunks) >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "chunks": filtered_chunks,
            "confidence": confidence,
            "total_matched": len(filtered_chunks)
        }

    def save(self, directory: Optional[str] = None):
        """Persists database to disk (vectors.npy, chunks.json, faiss.index)."""
        target_dir = directory or self.storage_dir
        os.makedirs(target_dir, exist_ok=True)

        if self.vectors is not None:
            np.save(os.path.join(target_dir, "vectors.npy"), self.vectors)

        chunks_data = [
            {"id": c.id, "text": c.text, "metadata": c.metadata}
            for c in self.chunks
        ]
        with open(os.path.join(target_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        if HAS_FAISS and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, os.path.join(target_dir, "faiss.index"))

        log.info("VectorDatabase: persisted %d chunks to %s", len(self.chunks), target_dir)

    def load(self, directory: Optional[str] = None) -> bool:
        """Loads database from disk without needing network re-downloads."""
        target_dir = directory or self.storage_dir
        vec_path = os.path.join(target_dir, "vectors.npy")
        chunks_path = os.path.join(target_dir, "chunks.json")
        faiss_path = os.path.join(target_dir, "faiss.index")

        if not os.path.exists(chunks_path) or not os.path.exists(vec_path):
            log.warning("VectorDatabase: cache files not found in %s", target_dir)
            return False

        try:
            self.vectors = np.load(vec_path)
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            self.chunks = [
                Chunk(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))
                for d in chunks_data
            ]

            if HAS_FAISS and os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            elif HAS_FAISS:
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                if len(self.vectors) > 0:
                    self.faiss_index.add(self.vectors)

            log.info("VectorDatabase: loaded %d chunks from %s", len(self.chunks), target_dir)
            return True
        except Exception as e:
            log.error("Failed to load VectorDatabase from %s: %s", target_dir, e)
            return False


vector_db = VectorDatabase()
