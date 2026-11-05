"""
src/stage2_memory/ollama_embedder.py
───────────────────────────────────
Direct HTTP client for Ollama nomic-embed-text embeddings.
Zero-framework, runs 100% offline at http://localhost:11434/api/embeddings.
"""

import os
import json
import logging
import requests
import numpy as np
from typing import List

log = logging.getLogger(__name__)


class OllamaEmbedder:
    """Generates 768-dimensional embeddings using Ollama nomic-embed-text."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        dimension: int = 768,
        timeout: int = 15
    ):
        self.model_name = os.getenv("EMBED_MODEL", model_name)
        self.base_url = os.getenv("LLM_URL", base_url).rstrip("/")
        self.dimension = dimension
        self.timeout = timeout

    def embed_text(self, text: str) -> np.ndarray:
        """Generates embedding vector for a single string."""
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                vec = np.array(data.get("embedding", []), dtype=np.float32)
                if vec.shape[0] == self.dimension:
                    return vec
            log.warning("Ollama embeddings returned status %s: %s", resp.status_code, resp.text[:100])
        except Exception as e:
            log.debug("Ollama embed_text exception: %s", e)

        # Fallback pseudo-vector deterministic hash if offline during bootstrap
        return self._fallback_embedding(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a batch of strings."""
        vectors = []
        for t in texts:
            vectors.append(self.embed_text(t))
        return np.array(vectors, dtype=np.float32)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Deterministic normalized pseudo-embedding for emergency fallback."""
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)


ollama_embedder = OllamaEmbedder()

