"""
S1: Fixed-size chunking with overlap — no structure detection.

Strategy: Use RecursiveCharacterTextSplitter directly on document content.
No Vietnamese legal structure awareness. Serves as the simplest baseline.
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data.chunker import LegalChunk, BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size chunker using RecursiveCharacterTextSplitter.

    No legal structure detection — purely size-based splits.
    Good baseline to compare against structure-aware strategies.

    Parameters:
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between consecutive chunks
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, row: dict) -> List[LegalChunk]:
        content = row.get("content", "")
        if not content or len(content) < 200:
            return []

        texts = self.text_splitter.split_text(content)

        legal_chunks = []
        for i, text in enumerate(texts):
            chunk = LegalChunk(
                chunk_text=text,
                document_id=row["id"],
                document_title=row["title"],
                document_number=str(row.get("document_number", "")),
                legal_type=row.get("legal_type", ""),
                issuing_authority=row.get("issuing_authority", ""),
                issuance_date=str(row.get("issuance_date", "")),
                url=str(row.get("url", "")),
                chapter=None,  # No structure detection
                article=None,
                clause=None,
                chunk_index=i,
                parent_chunk_text=content[:500] if i == 0 else None,
            )
            legal_chunks.append(chunk)

        return legal_chunks
