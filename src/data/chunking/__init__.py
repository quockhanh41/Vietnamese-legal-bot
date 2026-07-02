"""
Chunking strategies for Vietnamese legal documents.

Each strategy implements the BaseChunker interface and produces
List[LegalChunk] from a document row dict.
"""

from src.data.chunking.hierarchical import HierarchicalChunker
from src.data.chunking.fixed_size import FixedSizeChunker
from src.data.chunking.article_level import ArticleLevelChunker
from src.data.chunking.sentence import SentenceChunker
from src.data.chunking.semantic import SemanticChunker

__all__ = [
    "HierarchicalChunker",
    "FixedSizeChunker",
    "ArticleLevelChunker",
    "SentenceChunker",
    "SemanticChunker",
]
