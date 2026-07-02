"""
S2: Sentence-boundary chunking for Vietnamese legal text.

Strategy: Split content into sentences using Vietnamese-aware regex,
then group sentences into chunks up to max_chars.

No structural awareness (Chương/Điều) — purely sentence-based.
Falls between fixed-size (no awareness) and hierarchical (full structure).
"""

import re
from typing import List

from src.data.chunker import LegalChunk, BaseChunker


# Vietnamese sentence boundaries: . ! ? : followed by uppercase or Vietnamese diacritics
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?:])\s+(?=[A-ZĐƯƠĂÂÊÔƠ])',
)

# Secondary boundaries for legal text: paragraph breaks
_PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')


class SentenceChunker(BaseChunker):
    """
    Sentence-boundary chunker for Vietnamese legal documents.

    Splits content into sentences using Vietnamese-aware regex, then
    groups sentences into chunks respecting max_chars limit.

    Parameters:
        max_chars: Maximum characters per chunk
        min_sentences: Minimum sentences per chunk (prevents overly small chunks)
    """

    def __init__(
        self,
        max_chars: int = 1500,
        min_sentences: int = 2,
    ):
        self.max_chars = max_chars
        self.min_sentences = min_sentences

    def chunk_document(self, row: dict) -> List[LegalChunk]:
        content = row.get("content", "")
        if not content or len(content) < 200:
            return []

        # Split into sentences
        sentences = self._split_sentences(content)
        if not sentences:
            return []

        # Group sentences into chunks
        chunks = self._group_sentences(sentences)

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
        """Split content into sentences using Vietnamese-aware boundaries."""
        # First split by paragraph breaks
        paragraphs = _PARAGRAPH_BOUNDARY.split(content)

        sentences = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Split paragraph by sentence boundaries
            parts = _SENTENCE_BOUNDARY.split(para)
            for part in parts:
                part = part.strip()
                if part and len(part) >= 20:  # Skip very short fragments
                    sentences.append(part)

        return sentences

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks respecting max_chars."""
        chunks = []
        buffer = []
        buffer_len = 0

        for sent in sentences:
            sent_len = len(sent)

            # If adding this sentence exceeds limit and we have enough,
            # flush the buffer
            if buffer_len + sent_len > self.max_chars and len(buffer) >= self.min_sentences:
                chunks.append("\n\n".join(buffer))
                buffer = []
                buffer_len = 0

            buffer.append(sent)
            buffer_len += sent_len

        # Flush remaining
        if buffer:
            chunks.append("\n\n".join(buffer))

        return chunks
