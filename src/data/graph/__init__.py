"""
Legal Knowledge Graph — Cross-Referencing Engine for Vietnamese Laws.

Module 1 of Phase 2: Build a directed graph from legal cross-references,
enabling query-time context expansion through reference chains.

Usage:
    from src.data.graph import ReferenceExtractor, LegalKnowledgeGraph

    extractor = ReferenceExtractor()
    graph = LegalKnowledgeGraph()

    refs = extractor.extract(text=content, doc_id=doc_id, law_name=title)
    graph.add_references(refs)

    # Query-time: expand retrieved articles
    expanded = graph.expand_context(["doc_123_dieu_5"], hops=1)
"""

from src.data.graph.ref_extractor import (
    ReferenceExtractor,
    Reference,
    RefType,
)
from src.data.graph.graph_builder import (
    LegalKnowledgeGraph,
    ArticleNode,
)

__all__ = [
    "ReferenceExtractor",
    "Reference",
    "RefType",
    "LegalKnowledgeGraph",
    "ArticleNode",
]
