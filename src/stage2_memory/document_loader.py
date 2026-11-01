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
    def recursive_split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Recursively splits text into chunks under chunk_size with chunk_overlap,
        matching LangChain's RecursiveCharacterTextSplitter behavior.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        text = text.strip()
        if len(text) <= chunk_size:
            return [text] if text else []

        chosen_sep = ""
        for sep in separators:
            if sep == "":
                chosen_sep = ""
                break
            if sep in text:
                chosen_sep = sep
                break

        splits = text.split(chosen_sep) if chosen_sep else list(text)

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for s in splits:
            s_len = len(s) + (len(chosen_sep) if chosen_sep else 0)
            if current_length + s_len > chunk_size and current_chunk:
                merged = chosen_sep.join(current_chunk).strip()
                if merged:
                    chunks.append(merged)
                # Maintain overlap
                overlap_chunk: List[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) + len(chosen_sep) <= chunk_overlap:
                        overlap_chunk.insert(0, item)
                        overlap_len += len(item) + len(chosen_sep)
                    else:
                        break
                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(s)
            current_length += s_len

        if current_chunk:
            merged = chosen_sep.join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks

    @staticmethod
    def load_markdown(
        file_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> List[DocumentChunk]:
        """Parses markdown file section-by-section with RecursiveCharacterTextSplitter for long sections."""
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
            is_summary = bool("summary" in title.lower() or "records" in title.lower())

            # If section exceeds chunk_size, split recursively
            sub_texts = DocumentLoader.recursive_split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not sub_texts:
                sub_texts = [full_text]

            for s_idx, sub_text in enumerate(sub_texts):
                chunk_id = f"{filename}_{idx:03d}_{s_idx:02d}" if len(sub_texts) > 1 else f"{filename}_{idx:03d}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=sub_text,
                    title=title,
                    source=filename,
                    metadata={
                        "type": "summary" if is_summary else "section",
                        "title": title,
                        "source": filename,
                        "format": "markdown",
                        "part": s_idx + 1,
                        "total_parts": len(sub_texts),
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
                brand = row.get("Brand") or row.get("brand") or ""
                model = row.get("Model") or row.get("model") or ""
                year = row.get("Make_Year") or row.get("make_year") or ""
                variant = row.get("Variant") or row.get("variant") or ""
                
                if brand and model:
                    title = f"{year} {brand} {model} {variant}".strip()
                    category = brand
                else:
                    title = row.get("Title") or row.get("title") or f"Record #{idx+1}"
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

