"""
src/stage2_memory/document_loader.py
───────────────────────────────────
Multi-format Document & Dataset Loader.
Supports:
1. Markdown files (.md) with section-level chunking and title preservation.
2. Structured CSV files (.csv) with rich metadata (inspired by LocalAIAgentWithRAG).
"""

import os
import re
import csv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    """Unified document chunk representation across Markdown and CSV."""
    chunk_id: str
    text: str
    title: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Loads and standardizes documents from Markdown and CSV datasets."""

    @staticmethod
    def load_markdown(file_path: str) -> List[DocumentChunk]:
        """Parses markdown file section-by-section using '## ' headers."""
        if not os.path.exists(file_path):
            return []

        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r'\n(?=##\s+)', content.strip())
        chunks: List[DocumentChunk] = []

        for idx, section in enumerate(sections):
            clean = section.strip()
            if not clean:
                continue

            lines = clean.splitlines()
            title = lines[0].replace("#", "").strip() if lines else "Overview"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else clean

            # Prepend title for dense semantic grounding
            full_text = f"[{title}]\n{body}" if body else f"[{title}]"
            chunk_id = f"{filename}_{idx:03d}"

            is_summary = bool("summary" in title.lower() or "records" in title.lower())
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=full_text,
                title=title,
                source=filename,
                metadata={
                    "type": "summary" if is_summary else "section",
                    "title": title,
                    "source": filename,
                    "format": "markdown",
                }
            ))

        return chunks

    @staticmethod
    def load_csv(
        file_path: str,
        text_cols: Optional[List[str]] = None,
        meta_cols: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """
        Parses CSV files into DocumentChunks with structured metadata.
        Inspired by LocalAIAgentWithRAG's row-to-Document mapping.
        """
        if not os.path.exists(file_path):
            return []

        filename = os.path.basename(file_path)
        chunks: List[DocumentChunk] = []

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                # Build rich textual content
                title = row.get("Title") or row.get("title") or f"Record #{idx+1}"
                details = row.get("Details") or row.get("details") or ""
                category = row.get("Category") or row.get("category") or "General"
                
                content_parts = [f"[{category}: {title}]"]
                for k, v in row.items():
                    if v and v.strip() and k.lower() not in ("title", "id"):
                        content_parts.append(f"{k}: {v.strip()}")

                full_text = "\n".join(content_parts)
                chunk_id = f"{filename}_{idx:03d}"

                metadata = {k: v for k, v in row.items()}
                metadata["source"] = filename
                metadata["format"] = "csv"
                metadata["title"] = title
                metadata["category"] = category

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=full_text,
                    title=title,
                    source=filename,
                    metadata=metadata
                ))

        return chunks


document_loader = DocumentLoader()
