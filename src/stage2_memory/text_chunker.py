"""
src/stage2_memory/text_chunker.py
─────────────────────────────────
Sentence-boundary preserving text chunker with overlap and metadata tracking.
Based on the high-precision offline RAG chunking algorithm.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Text chunk with metadata and embedding."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    distance: float = 0.0


class TextChunker:
    """Text chunking with overlap and sentence boundaries."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace while preserving punctuation needed for sentence detection."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\/\"\'\%\₹]', '', text)
        return text.strip()

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 750,
        overlap: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        source_id: str = "doc"
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks at sentence boundaries.
        Args:
            text: Raw input text
            chunk_size: Target size in characters (approx 120-150 tokens)
            overlap: Overlap size to preserve boundary context
            metadata: Source info for citations
            source_id: Identifier prefix for chunk IDs
        """
        cleaned = TextChunker.clean_text(text)
        chunks: List[Chunk] = []
        if not cleaned:
            return chunks

        start = 0
        chunk_index = 0
        meta = metadata.copy() if metadata else {}

        while start < len(cleaned):
            end = start + chunk_size

            # Break at sentence boundary (look at last 25% of chunk)
            if end < len(cleaned):
                search_start = end - int(chunk_size * 0.25)
                sentence_end = max(
                    cleaned.rfind('. ', search_start, end),
                    cleaned.rfind('! ', search_start, end),
                    cleaned.rfind('? ', search_start, end),
                    cleaned.rfind('.\n', search_start, end),
                    cleaned.rfind('\n\n', search_start, end)
                )
                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1

            chunk_text = cleaned[start:end].strip()
            if chunk_text:
                chunk_meta = meta.copy()
                chunk_meta['chunk_index'] = chunk_index
                chunk_meta['source'] = chunk_meta.get('source', source_id)
                chunk_id = f"{source_id}_{chunk_index}"

                chunks.append(Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_meta
                ))
                chunk_index += 1

            start = end - overlap
            if start >= len(cleaned) - overlap:
                break

        return chunks
