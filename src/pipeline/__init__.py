"""
Phase 2 Integration Pipeline — Unified enrichment and retrieval.

Ties together cross-referencing, temporal versioning, and amendments tracking
into a single pipeline that enriches documents and enhances retrieval.

Usage:
    from src.pipeline import EnrichmentPipeline, EnhancedRetriever

    # Enrich documents
    pipeline = EnrichmentPipeline(vector_store, metadata_store)
    pipeline.run()

    # Enhanced retrieval
    retriever = EnhancedRetriever(vector_store, metadata_store, embedder)
    results = retriever.retrieve("Điều 123 Bộ luật Hình sự", top_k=5)
"""

from src.pipeline.enricher import EnrichmentPipeline
from src.pipeline.retriever import EnhancedRetriever

__all__ = [
    "EnrichmentPipeline",
    "EnhancedRetriever",
]
