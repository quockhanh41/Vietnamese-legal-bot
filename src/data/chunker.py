"""
Chunking strategies for Vietnamese legal documents.

Provides:
- LegalChunk: unified chunk dataclass used by all strategies
- BaseChunker: abstract base class for chunking strategies
- create_chunker(): factory function to instantiate strategies by name
- LegalDocumentChunker: backward-compat alias for HierarchicalChunker
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class LegalChunk:
    """A single chunk of a legal document with hierarchical metadata."""
    chunk_text: str
    document_id: int
    document_title: str
    document_number: str
    legal_type: str
    issuing_authority: str
    issuance_date: str
    url: str
    chapter: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    chunk_index: int = 0
    parent_chunk_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_text": self.chunk_text,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_number": self.document_number,
            "legal_type": self.legal_type,
            "issuing_authority": self.issuing_authority,
            "issuance_date": str(self.issuance_date),
            "url": self.url,
            "chapter": self.chapter,
            "article": self.article,
            "clause": self.clause,
            "chunk_index": self.chunk_index,
            "parent_chunk_text": self.parent_chunk_text,
        }


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk_document(self, row: dict) -> List[LegalChunk]:
        """
        Chunk a single legal document.

        Args:
            row: Dictionary with keys: id, title, content, document_number,
                 legal_type, issuing_authority, issuance_date, url

        Returns:
            List of LegalChunk objects
        """
        ...


def create_chunker(strategy: str, **params) -> BaseChunker:
    """
    Factory function to create a chunker by strategy name.

    Args:
        strategy: One of "fixed_size", "sentence", "hierarchical", "semantic", "article_level"
        **params: Strategy-specific parameters

    Returns:
        BaseChunker instance
    """
    if strategy == "fixed_size":
        from src.data.chunking.fixed_size import FixedSizeChunker
        return FixedSizeChunker(**params)
    elif strategy == "sentence":
        from src.data.chunking.sentence import SentenceChunker
        return SentenceChunker(**params)
    elif strategy == "hierarchical":
        from src.data.chunking.hierarchical import HierarchicalChunker
        return HierarchicalChunker(**params)
    elif strategy == "semantic":
        from src.data.chunking.semantic import SemanticChunker
        return SemanticChunker(**params)
    elif strategy == "article_level":
        from src.data.chunking.article_level import ArticleLevelChunker
        return ArticleLevelChunker(**params)
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Available: fixed_size, sentence, hierarchical, semantic, article_level"
        )


# Backward-compatibility alias
from src.data.chunking.hierarchical import HierarchicalChunker as LegalDocumentChunker  # noqa: E402
