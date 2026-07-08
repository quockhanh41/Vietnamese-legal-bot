"""
Enhanced retrieval with Phase 2 context expansion and time filtering.

Wraps the base vector store with:
- Cross-reference expansion: Follow graph edges to include referenced articles
- Time filtering: Filter results to laws effective at a given date
- Amendment status: Indicate whether retrieved articles have been amended
- Consolidation hints: Show amendment history for retrieved articles

Usage:
    retriever = EnhancedRetriever(vector_store, metadata_store, embedder)
    results = retriever.retrieve("Điều 123 xử lý tội phạm", top_k=5)
    # Results include expanded context from cross-references
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import date

from src.db.qdrant_client import LegalVectorStore, SearchResult
from src.data.metadata_store import MetadataStore

if TYPE_CHECKING:
    from src.data.embedder import BGEEmbedder

logger = logging.getLogger(__name__)


def _extract_article_number(article_str: str) -> Optional[str]:
    """Extract article number from string like 'Điều 123'."""
    m = re.search(r'(\d+[a-z]?)', article_str)
    return m.group(1) if m else None


def get_phase2_status(
    document_id: Optional[int],
    article: Optional[str],
    metadata_store: MetadataStore,
) -> Dict[str, Any]:
    """
    Look up Phase 2 (temporal + amendment) status for one retrieved chunk.

    Shared by EnhancedRetriever (search API) and the chat orchestrator
    (Plan §3.6), so amendment/temporal logic lives in exactly one place.

    Returns:
        {
            "effective_date": Optional[str],
            "is_in_effect": Optional[bool],  # None = no metadata for this doc
                                              # (fail-open — NOT the same as False)
            "ref_count": int,
            "is_amended": bool,
            "amending_laws": List[str],
        }
    """
    status: Dict[str, Any] = {
        "effective_date": None,
        "is_in_effect": None,
        "ref_count": 0,
        "is_amended": False,
        "amending_laws": [],
    }
    if not document_id:
        return status

    doc_meta = metadata_store.get_document(document_id)
    if doc_meta:
        status["effective_date"] = doc_meta.effective_date
        status["is_in_effect"] = doc_meta.is_in_effect
        status["ref_count"] = doc_meta.ref_count

        if article and doc_meta.is_amendment:
            # This chunk IS from an amendment law — it doesn't get amended itself
            status["is_amended"] = True
            status["amending_laws"] = list(doc_meta.target_laws)
        elif article:
            # Check if any amendment law targets this article
            article_num = _extract_article_number(article)
            if article_num:
                amending_laws = []
                for _, meta in metadata_store.get_all_documents().items():
                    if meta.is_amendment and article_num in meta.amended_articles:
                        amending_laws.append(meta.title[:60])
                if amending_laws:
                    status["is_amended"] = True
                    status["amending_laws"] = amending_laws

    timeline = metadata_store.get_timeline(document_id)
    if timeline:
        status["effective_date"] = status["effective_date"] or timeline.get("effective_date")

    return status


@dataclass
class EnhancedSearchResult:
    """Search result enriched with Phase 2 metadata."""

    # Original search result
    point_id: str
    score: float
    chunk_text: str
    document_title: str
    document_id: Optional[int] = None
    article: Optional[str] = None
    chapter: Optional[str] = None
    legal_type: Optional[str] = None
    url: Optional[str] = None
    parent_chunk_text: Optional[str] = None

    # Phase 2 enrichments
    effective_date: Optional[str] = None
    is_in_effect: bool = False
    is_amended: bool = False
    amending_laws: List[str] = field(default_factory=list)
    ref_count: int = 0

    # Expanded context (from cross-reference graph)
    expanded_chunks: List[Dict[str, Any]] = field(default_factory=list)
    expansion_hops: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for API responses."""
        return {
            "point_id": self.point_id,
            "score": round(self.score, 4),
            "chunk_text": self.chunk_text[:500],
            "document_title": self.document_title,
            "document_id": self.document_id,
            "article": self.article,
            "chapter": self.chapter,
            "legal_type": self.legal_type,
            "url": self.url,
            "effective_date": self.effective_date,
            "is_in_effect": self.is_in_effect,
            "is_amended": self.is_amended,
            "amending_laws": self.amending_laws,
            "ref_count": self.ref_count,
            "expanded_chunks": len(self.expanded_chunks),
        }


class EnhancedRetriever:
    """
    Enhanced retrieval with Phase 2 capabilities.

    Usage:
        retriever = EnhancedRetriever(vector_store, metadata_store, embedder)
        results = retriever.retrieve("câu hỏi về luật", top_k=5)

        # With time filtering
        results = retriever.retrieve("...", at_date=date(2018, 1, 1))

        # With context expansion
        results = retriever.retrieve("...", expand_context=True, expansion_hops=1)
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        metadata_store: MetadataStore,
        embedder: 'BGEEmbedder' = None,
    ):
        self.vector_store = vector_store
        self.metadata = metadata_store
        self._embedder = embedder

    @property
    def embedder(self) -> 'BGEEmbedder':
        """Lazy-load embedder on first use."""
        if self._embedder is None:
            from src.data.embedder import BGEEmbedder
            self._embedder = BGEEmbedder()
        return self._embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        at_date: Optional[date] = None,
        expand_context: bool = False,
        expansion_hops: int = 1,
        legal_type_filter: Optional[str] = None,
    ) -> List[EnhancedSearchResult]:
        """
        Enhanced retrieval with Phase 2 features.

        Args:
            query: User query text
            top_k: Number of results to return
            at_date: Optional date for time-aware search (None = today)
            expand_context: Whether to include cross-referenced articles
            expansion_hops: Number of hops for context expansion (1-2)
            legal_type_filter: Optional filter by legal_type

        Returns:
            List of enhanced search results
        """
        # 1. Embed and search
        output = self.embedder.encode(
            [query], return_dense=True, return_sparse=True
        )
        dense_vec = output["dense_vecs"][0]
        sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}

        # Build filter
        filter_conditions = {}
        if legal_type_filter:
            filter_conditions["legal_type"] = legal_type_filter

        # Search with more results if we're going to expand/filter
        search_k = top_k * 3 if (at_date or expand_context) else top_k

        results = self.vector_store.search_hybrid(
            dense_vec.tolist(),
            sparse_vec,
            top_k=search_k,
            filter_conditions=filter_conditions or None,
        )

        # 2. Enrich results with Phase 2 metadata
        enriched = []
        for r in results:
            er = self._enrich_result(r)

            # Time filter
            if at_date and er.effective_date:
                try:
                    eff = date.fromisoformat(er.effective_date)
                    if eff > at_date:
                        continue  # Not yet effective
                except (ValueError, TypeError):
                    pass

            enriched.append(er)

        # 3. Context expansion via cross-reference graph
        if expand_context and self.metadata.get_graph():
            enriched = self._expand_context(enriched, expansion_hops)

        # 4. Return top-k after enrichment
        return enriched[:top_k]

    def retrieve_with_expansion(
        self,
        query: str,
        top_k: int = 5,
        expansion_hops: int = 1,
    ) -> List[EnhancedSearchResult]:
        """Convenience method: retrieve with context expansion enabled."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            expand_context=True,
            expansion_hops=expansion_hops,
        )

    # ── Private Methods ─────────────────────────────────────────

    def _enrich_result(self, result: SearchResult) -> EnhancedSearchResult:
        """Enrich a single search result with Phase 2 metadata."""
        er = EnhancedSearchResult(
            point_id=result.point_id,
            score=result.score,
            chunk_text=result.chunk_text,
            document_title=result.document_title,
            document_id=result.document_id,
            article=result.article,
            chapter=result.chapter,
            legal_type=result.legal_type,
            url=result.url,
            parent_chunk_text=result.parent_chunk_text,
        )

        status = get_phase2_status(result.document_id, result.article, self.metadata)
        er.effective_date = status["effective_date"]
        er.is_in_effect = bool(status["is_in_effect"])  # None (no metadata) -> False, same as before
        er.ref_count = status["ref_count"]
        er.is_amended = status["is_amended"]
        er.amending_laws = status["amending_laws"]

        return er

    def _expand_context(
        self,
        results: List[EnhancedSearchResult],
        hops: int = 1,
    ) -> List[EnhancedSearchResult]:
        """
        Expand results with cross-referenced article chunks.

        For each retrieved chunk that has an article, follow the graph edges
        to find referenced articles and fetch their chunk texts.
        """
        graph_data = self.metadata.get_graph()
        edges = graph_data.get("edges", [])
        nodes = graph_data.get("nodes", [])

        if not edges:
            return results

        # Build adjacency index from graph edges
        adjacency: Dict[str, List[str]] = {}
        for e in edges:
            src = e.get("source", "")
            tgt = e.get("target", "")
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(tgt)

        # Build node lookup
        node_map: Dict[str, Dict] = {n["id"]: n for n in nodes}

        for er in results:
            if not er.document_id or not er.article:
                continue

            # Find matching graph node
            node_id = f"doc_{er.document_id}_{er.article.lower().replace(' ', '_')}"
            neighbors = self._get_neighbors_bfs(node_id, adjacency, hops)

            er.expansion_hops = hops
            for neighbor_id in neighbors:
                if neighbor_id in node_map:
                    node = node_map[neighbor_id]
                    er.expanded_chunks.append({
                        "node_id": neighbor_id,
                        "article": node.get("article", ""),
                        "law_name": node.get("law_name", ""),
                        "node_type": node.get("node_type", "article"),
                    })

        return results

    @staticmethod
    def _get_neighbors_bfs(
        start_node: str,
        adjacency: Dict[str, List[str]],
        hops: int,
    ) -> List[str]:
        """BFS traversal from start node, limited to N hops."""
        seen = {start_node}
        frontier = {start_node}
        neighbors = []

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for neighbor in adjacency.get(node, []):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        next_frontier.add(neighbor)
                        neighbors.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return neighbors

    @staticmethod
    def _extract_article_number(article_str: str) -> Optional[str]:
        """Extract article number from string like 'Điều 123'."""
        return _extract_article_number(article_str)

    # ── Article History ────────────────────────────────────────

    def get_article_history(
        self, doc_id: int, article: str
    ) -> Dict[str, Any]:
        """
        Get the amendment history for a specific article.

        Returns timeline of modifications, effective dates, and
        references to amending laws.
        """
        doc_meta = self.metadata.get_document(doc_id)
        if not doc_meta:
            return {"error": "Document not found", "doc_id": doc_id}

        timeline = self.metadata.get_timeline(doc_id)

        # Find amendments targeting this article
        article_num = self._extract_article_number(article) or article
        amending_laws = []
        for did, meta in self.metadata.get_all_documents().items():
            if meta.is_amendment and article_num in meta.amended_articles:
                amending_laws.append({
                    "doc_id": did,
                    "title": meta.title,
                    "effective_date": meta.effective_date,
                })

        return {
            "doc_id": doc_id,
            "document_title": doc_meta.title,
            "article": article,
            "is_in_effect": doc_meta.is_in_effect,
            "effective_date": doc_meta.effective_date or (
                timeline.get("effective_date") if timeline else None
            ),
            "is_amended": len(amending_laws) > 0,
            "amending_laws": sorted(amending_laws, key=lambda x: x.get("effective_date") or ""),
            "version_chain": doc_meta.version_chain,
            "ref_count": doc_meta.ref_count,
        }

    # ── Stats ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        """System statistics."""
        index = self.metadata.get_index()
        return {
            "collection": self.vector_store.collection_name,
            "documents": index.get("total_documents", 0),
            "in_effect": index.get("currently_in_effect", 0),
            "amendment_laws": index.get("amendment_laws", 0),
            "graph_nodes": index.get("total_graph_nodes", 0),
            "graph_edges": index.get("total_graph_edges", 0),
            "last_enriched": index.get("last_enriched"),
        }
