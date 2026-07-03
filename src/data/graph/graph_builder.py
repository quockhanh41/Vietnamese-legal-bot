"""
Build a NetworkX knowledge graph from extracted legal references.

Nodes: articles/chapters within laws
Edges: cross-references with metadata

The graph enables:
- Query-time context expansion (follow references)
- Impact analysis (what articles reference this one?)
- Dead-end detection (articles with no outgoing refs)
"""

from __future__ import annotations

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

import networkx as nx

from src.data.graph.ref_extractor import Reference, RefType


@dataclass
class ArticleNode:
    """A node in the legal knowledge graph representing one article."""
    node_id: str          # "doc_123_article_5"
    doc_id: int
    law_name: str
    article_number: str   # "Điều 5"
    chapter: Optional[str] = None
    metadata: Dict = None

    @staticmethod
    def make_id(doc_id: int, article_number: str) -> str:
        return f"doc_{doc_id}_{article_number.lower().replace(' ', '_')}"


class LegalKnowledgeGraph:
    """
    NetworkX-based knowledge graph of legal references.

    Usage:
        graph = LegalKnowledgeGraph()
        graph.add_references(refs)
        expanded = graph.expand_context(article_ids, hops=2)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._article_index: Dict[str, str] = {}  # article_id → node_id
        self._stats = {"nodes": 0, "edges": 0, "internal": 0, "external": 0}

    def _ensure_node(self, doc_id: int, article: str, law_name: str = "",
                     node_type: str = "article") -> str:
        """Get or create a node, return its ID."""
        node_id = ArticleNode.make_id(doc_id, article)
        if node_id not in self.graph:
            self.graph.add_node(node_id,
                                doc_id=doc_id, article=article,
                                law_name=law_name, node_type=node_type)
            self._stats["nodes"] += 1
        return node_id

    def add_references(self, refs: List[Reference]) -> int:
        """
        Add a batch of references to the graph.
        Creates nodes and edges as needed.

        When source_article is None, uses the document itself as source node.
        Returns number of new edges added.
        """
        added = 0
        for ref in refs:
            # Determine source node
            if ref.source_article:
                src_id = self._ensure_node(
                    ref.source_doc_id, ref.source_article,
                    ref.target_law_name or ""
                )
            else:
                # Fallback: use doc-level node
                src_id = self._ensure_node(
                    ref.source_doc_id, f"doc_{ref.source_doc_id}",
                    ref.target_law_name or "", "document"
                )

            # Internal/clause/multi refs: target is in same doc
            if ref.target_article and ref.ref_type in (
                RefType.INTERNAL, RefType.CLAUSE, RefType.MULTI
            ):
                tgt_id = self._ensure_node(
                    ref.source_doc_id, ref.target_article,
                    ref.target_law_name or ""
                )
                if not self.graph.has_edge(src_id, tgt_id):
                    self.graph.add_edge(
                        src_id, tgt_id,
                        ref_type=ref.ref_type.value,
                        clause=ref.target_clause,
                        confidence=ref.confidence,
                    )
                    self._stats["edges"] += 1
                    if ref.ref_type == RefType.INTERNAL:
                        self._stats["internal"] += 1
                    added += 1

            # External refs: create virtual node
            if ref.ref_type == RefType.EXTERNAL and ref.target_law_name and ref.target_article:
                ext_id = f"ext_{ref.target_law_name[:40]}_{ref.target_article}".lower().replace(' ', '_')
                if ext_id not in self.graph:
                    self.graph.add_node(ext_id,
                                        law_name=ref.target_law_name,
                                        article=ref.target_article,
                                        node_type="external_ref")
                    self._stats["nodes"] += 1

                if not self.graph.has_edge(src_id, ext_id):
                    self.graph.add_edge(src_id, ext_id,
                                        ref_type="external",
                                        confidence=ref.confidence)
                    self._stats["edges"] += 1
                    self._stats["external"] += 1
                    added += 1

        return added

    def get_neighbors(self, article_id: str, hops: int = 1,
                      direction: str = "out") -> List[str]:
        """
        Get neighboring article IDs within N hops.

        Args:
            article_id: Source article node ID
            hops: Number of hops (1 = direct refs only)
            direction: "out" (articles this one cites),
                       "in" (articles that cite this one),
                       "both"

        Returns:
            List of node IDs reachable within N hops
        """
        if article_id not in self.graph:
            return []

        nodes: Set[str] = set()
        if direction in ("out", "both"):
            # BFS from source following outgoing edges
            seen = {article_id}
            frontier = {article_id}
            for _ in range(hops):
                next_frontier = set()
                for node in frontier:
                    for succ in self.graph.successors(node):
                        if succ not in seen:
                            seen.add(succ)
                            next_frontier.add(succ)
                            nodes.add(succ)
                frontier = next_frontier
                if not frontier:
                    break

        if direction in ("in", "both"):
            # BFS from source following incoming edges
            seen = {article_id}
            frontier = {article_id}
            for _ in range(hops):
                next_frontier = set()
                for node in frontier:
                    for pred in self.graph.predecessors(node):
                        if pred not in seen:
                            seen.add(pred)
                            next_frontier.add(pred)
                            nodes.add(pred)
                frontier = next_frontier
                if not frontier:
                    break

        return list(nodes)

    def expand_context(self, article_ids: List[str], hops: int = 1) -> List[str]:
        """
        Given a set of retrieved article IDs, expand to include
        articles they reference (within N hops).

        Returns deduplicated, topologically sorted list of article IDs.
        """
        expanded: Set[str] = set(article_ids)
        for art_id in article_ids:
            neighbors = self.get_neighbors(art_id, hops=hops, direction="both")
            expanded.update(neighbors)

        # Sort: articles in same law first, then externals
        same_law = [n for n in expanded if not n.startswith("ext_")]
        external = [n for n in expanded if n.startswith("ext_")]
        return same_law + external

    def get_article_chain(self, article_id: str, max_depth: int = 5) -> List[List[str]]:
        """
        Get all reference chains starting from an article.

        Returns list of paths (each path is a list of node IDs).
        Useful for debugging and visualization.
        """
        if article_id not in self.graph:
            return []

        paths = []
        for target in self.graph.successors(article_id):
            try:
                # Find all simple paths up to max_depth
                for path in nx.all_simple_paths(self.graph, article_id, target,
                                                cutoff=max_depth):
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue

        return paths

    def most_referenced(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Find articles with the most incoming references (highest authority)."""
        in_degrees = dict(self.graph.in_degree())
        sorted_items = sorted(in_degrees.items(), key=lambda x: -x[1])
        return [(node, deg) for node, deg in sorted_items[:top_n] if deg > 0]

    def find_orphans(self) -> List[str]:
        """Find article nodes with no outgoing references."""
        return [n for n in self.graph.nodes()
                if self.graph.out_degree(n) == 0
                and self.graph.nodes[n].get("node_type") == "article"]

    @property
    def stats(self) -> Dict:
        """Graph statistics."""
        return {
            **self._stats,
            "density": nx.density(self.graph) if self._stats["nodes"] > 1 else 0,
            "components": nx.number_weakly_connected_components(self.graph),
        }

    def to_dict(self) -> Dict:
        """Export graph to a serializable format for visualization."""
        return {
            "nodes": [
                {"id": n, **d}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
