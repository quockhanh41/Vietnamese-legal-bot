"""
S4: Semantic chunking — split at embedding-similarity drop points.

Strategy: Embed consecutive sentences with BGE-M3, compute cosine similarity
between adjacent sentence pairs, and cut where similarity drops below a threshold.
This creates chunks at natural semantic boundaries where the topic shifts.

Based on the approach described in Greg Kamradt's "5 Levels of Chunking"
and adapted for Vietnamese legal text.
"""

import re
import numpy as np
from typing import List, Optional

from src.data.chunker import LegalChunk, BaseChunker


_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?:])\s+(?=[A-ZĐƯƠĂÂÊÔƠ])',
)
_PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')


class SemanticChunker(BaseChunker):
    """
    Semantic chunker using embedding similarity for boundary detection.

    Embeds each sentence, then cuts where cosine similarity between
    consecutive sentences drops below threshold. This produces chunks
    that respect natural topic shifts in the document.

    Parameters:
        threshold: Cosine similarity threshold for cut points (0.0-1.0).
                   Lower = fewer cuts, higher = more cuts.
        min_size: Minimum characters per chunk (prevents tiny fragments).
    """

    def __init__(
        self,
        threshold: float = 0.6,
        min_size: int = 300,
        embedder=None,  # Optional: pass shared embedder to avoid double-loading BGE-M3
    ):
        self.threshold = threshold
        self.min_size = min_size
        self._embedder = embedder  # If None, lazy-loads own instance

    @property
    def embedder(self):
        """Lazy-load BGE-M3 embedder (heavy — ~2GB)."""
        if self._embedder is None:
            from src.data.embedder import BGEEmbedder
            self._embedder = BGEEmbedder(device="cpu")
        return self._embedder

    def chunk_document(self, row: dict) -> List[LegalChunk]:
        content = row.get("content", "")
        if not content or len(content) < self.min_size:
            return []

        # Split into sentences
        sentences = self._split_sentences(content)
        if len(sentences) <= 1:
            return [
                LegalChunk(
                    chunk_text=content.strip(),
                    document_id=row["id"],
                    document_title=row["title"],
                    document_number=str(row.get("document_number", "")),
                    legal_type=row.get("legal_type", ""),
                    issuing_authority=row.get("issuing_authority", ""),
                    issuance_date=str(row.get("issuance_date", "")),
                    url=str(row.get("url", "")),
                    chapter=None,
                    article=None,
                    clause=None,
                    chunk_index=0,
                    parent_chunk_text=None,
                )
            ]

        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(sentences)

        # Group sentences into chunks at breakpoints
        chunks = self._build_chunks(sentences, breakpoints)

        # Build LegalChunks
        legal_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = LegalChunk(
                chunk_text=chunk_text,
                document_id=row["id"],
                document_title=row["title"],
                document_number=str(row.get("document_number", "")),
                legal_type=row.get("legal_type", ""),
                issuing_authority=row.get("issuing_authority", ""),
                issuance_date=str(row.get("issuance_date", "")),
                url=str(row.get("url", "")),
                chapter=None,
                article=None,
                clause=None,
                chunk_index=i,
                parent_chunk_text=None,
            )
            legal_chunks.append(chunk)

        return legal_chunks

    def _split_sentences(self, content: str) -> List[str]:
        """Split content into sentences."""
        paragraphs = _PARAGRAPH_BOUNDARY.split(content)
        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            parts = _SENTENCE_BOUNDARY.split(para)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 20:
                    sentences.append(part)
        return sentences

    def _find_breakpoints(self, sentences: List[str]) -> List[int]:
        """
        Find breakpoint indices where similarity drops below threshold.

        Returns list of indices BEFORE which to cut.
        e.g., [3, 7] means cut before sentence 3 and before sentence 7.
        """
        if len(sentences) <= 1:
            return []

        # Batch-embed all sentences (embedder handles its own batching + memory cleanup)
        try:
            embeddings = self.embedder.encode(
                sentences,
                batch_size=16,  # Conservative batch size for MPS stability
                return_dense=True,
                return_sparse=False,
            )["dense_vecs"]
        except Exception as e:
            # If embedding fails (e.g., OOM on MPS), fall back to paragraph boundaries
            print(f"    ⚠ Semantic embedding failed ({e}), falling back to paragraph splits")
            return self._fallback_breakpoints(sentences)

        # Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find breakpoints where similarity drops below threshold + local minima
        breakpoints = []
        for i, sim in enumerate(similarities):
            is_drop = sim < self.threshold
            # Also check if this is a local minimum (compared to neighbors)
            is_local_min = True
            if i > 0 and sim > similarities[i - 1]:
                is_local_min = False
            if i < len(similarities) - 1 and sim > similarities[i + 1]:
                is_local_min = False

            if is_drop and is_local_min:
                breakpoints.append(i + 1)  # Cut BEFORE sentence i+1

        return breakpoints

    def _build_chunks(self, sentences: List[str], breakpoints: List[int]) -> List[str]:
        """Build chunks from sentences, cutting at breakpoints."""
        chunks = []
        start = 0

        for bp in breakpoints:
            chunk = "\n\n".join(sentences[start:bp])
            if len(chunk) >= self.min_size:
                chunks.append(chunk)
                start = bp

        # Final chunk
        if start < len(sentences):
            remaining = "\n\n".join(sentences[start:])
            if len(remaining) >= self.min_size or chunks:
                chunks.append(remaining)

        return chunks

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _fallback_breakpoints(self, sentences: List[str]) -> List[int]:
        """
        Fallback breakpoint detection when embedding fails (e.g., OOM).
        Uses paragraph boundaries and sentence length heuristics.
        """
        breakpoints = []
        for i, sent in enumerate(sentences[1:], 1):
            # Cut at paragraph-like breaks (sentences starting with Chương, Điều, Mục)
            if re.match(r'^(CHƯƠNG|Điều|Mục|Phần)\s+[IVXLCDM\d]+', sent.strip()):
                breakpoints.append(i)
        return breakpoints
