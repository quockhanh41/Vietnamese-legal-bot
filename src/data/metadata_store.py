"""
Document-level metadata store for Phase 2 enrichment data.

Persists cross-reference graphs, timeline data, and amendment instructions
as structured JSON, keyed by document ID.

Design: Simple JSON file store — good for hundreds to low-thousands of docs.
For larger scale (>10k docs), swap to SQLite or Postgres.
"""

from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import date, datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ── Data Classes ────────────────────────────────────────────────


@dataclass
class DocumentMeta:
    """Aggregated Phase 2 metadata for a single document."""

    doc_id: int
    title: str

    # Cross-references
    ref_count: int = 0
    internal_refs: int = 0
    external_refs: int = 0
    articles_referenced: List[str] = field(default_factory=list)  # Articles this doc references

    # Temporal
    effective_date: Optional[str] = None   # ISO format
    expiry_date: Optional[str] = None
    is_in_effect: bool = False
    replaced_laws: List[str] = field(default_factory=list)
    version_chain: List[str] = field(default_factory=list)  # Ordered list of law names

    # Amendment
    is_amendment: bool = False
    target_laws: List[str] = field(default_factory=list)    # Laws this amendment modifies
    amendment_count: int = 0                                 # Number of instructions
    amended_articles: List[str] = field(default_factory=list)

    # Graph
    graph_node_id: Optional[str] = None
    graph_in_degree: int = 0
    graph_out_degree: int = 0

    # Metadata
    enriched_at: Optional[str] = None   # ISO timestamp
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DocumentMeta':
        """Deserialize from dict."""
        # Handle legacy keys
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MetadataStore:
    """
    JSON-backed document metadata store for Phase 2 enrichment.

    Usage:
        store = MetadataStore("data/metadata")
        store.save(doc_id, meta)
        meta = store.get(doc_id)
        all_meta = store.get_all()
    """

    def __init__(self, data_dir: str = "data/metadata"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Separate files for different data types
        self._docs_file = self.data_dir / "documents.json"
        self._graph_file = self.data_dir / "graph.json"
        self._timeline_file = self.data_dir / "timelines.json"
        self._amendments_file = self.data_dir / "amendments.json"
        self._index_file = self.data_dir / "index.json"

        # In-memory caches
        self._documents: Dict[int, DocumentMeta] = {}
        self._graph_data: Dict[str, Any] = {}
        self._timelines: List[Dict] = []
        self._amendments: List[Dict] = []

        # Load existing data
        self._load()

    # ── Document Metadata ──────────────────────────────────────

    def save_document(self, doc_id: int, meta: DocumentMeta) -> None:
        """Save document metadata."""
        meta.enriched_at = datetime.now().isoformat()
        self._documents[doc_id] = meta
        self._save_json(self._docs_file, {
            str(k): v.to_dict() for k, v in self._documents.items()
        })

    def get_document(self, doc_id: int) -> Optional[DocumentMeta]:
        """Get metadata for a single document."""
        return self._documents.get(doc_id)

    def get_all_documents(self) -> Dict[int, DocumentMeta]:
        """Get all document metadata."""
        return dict(self._documents)

    def get_documents_by_target(self, law_name: str) -> List[DocumentMeta]:
        """Find amendment documents that target a specific law."""
        norm = law_name.lower()
        results = []
        for meta in self._documents.values():
            if meta.is_amendment:
                for target in meta.target_laws:
                    if norm in target.lower() or target.lower() in norm:
                        results.append(meta)
                        break
        return results

    def get_articles_amended(self, doc_id: int) -> List[str]:
        """Get list of articles amended by a document."""
        meta = self._documents.get(doc_id)
        return meta.amended_articles if meta else []

    # ── Graph Data ─────────────────────────────────────────────

    def save_graph(self, graph_dict: Dict[str, Any]) -> None:
        """Save graph data (nodes + edges in serializable format)."""
        self._graph_data = graph_dict
        self._save_json(self._graph_file, graph_dict)

    def get_graph(self) -> Dict[str, Any]:
        """Get the full graph data."""
        return dict(self._graph_data)

    def get_article_neighbors(self, node_id: str, hops: int = 1) -> List[str]:
        """
        Get neighboring article node IDs from the graph.
        Requires the graph to be loaded (rebuild from graph_data if needed).
        """
        # This is a query-time method — use the NetworkX graph for actual traversal
        # Here we just provide the stored adjacency data
        edges = self._graph_data.get("edges", [])
        neighbors = set()
        for e in edges:
            if e.get("source") == node_id:
                neighbors.add(e.get("target"))
            if e.get("target") == node_id:
                neighbors.add(e.get("source"))
        return list(neighbors)

    # ── Timeline Data ──────────────────────────────────────────

    def save_timelines(self, timelines: List[Dict]) -> None:
        """Save timeline data (list of serialized LawTimeline dicts)."""
        self._timelines = timelines
        self._save_json(self._timeline_file, timelines)

    def get_timelines(self) -> List[Dict]:
        """Get all timeline data."""
        return list(self._timelines)

    def get_timeline(self, doc_id: int) -> Optional[Dict]:
        """Get timeline for a specific document."""
        for tl in self._timelines:
            if tl.get("doc_id") == doc_id:
                return tl
        return None

    def get_effective_laws(self, at_date: Optional[date] = None) -> List[Dict]:
        """Get laws effective at a given date."""
        if at_date is None:
            at_date = date.today()

        results = []
        for tl in self._timelines:
            eff_str = tl.get("effective_date")
            exp_str = tl.get("expiry_date")
            if not eff_str:
                continue

            eff = date.fromisoformat(eff_str)
            exp = date.fromisoformat(exp_str) if exp_str else None

            if eff <= at_date and (exp is None or exp > at_date):
                results.append(tl)

        return results

    # ── Amendment Data ─────────────────────────────────────────

    def save_amendments(self, amendments: List[Dict]) -> None:
        """Save amendment instructions (serialized)."""
        self._amendments = amendments
        self._save_json(self._amendments_file, amendments)

    def get_amendments(self) -> List[Dict]:
        """Get all amendment data."""
        return list(self._amendments)

    def get_amendments_for_doc(self, doc_id: int) -> List[Dict]:
        """Get amendments produced by a specific document."""
        return [a for a in self._amendments if a.get("amending_doc_id") == doc_id]

    # ── Index / Summary ────────────────────────────────────────

    def build_index(self) -> Dict[str, Any]:
        """Build a summary index of all stored data."""
        index = {
            "total_documents": len(self._documents),
            "amendment_laws": sum(1 for m in self._documents.values() if m.is_amendment),
            "base_laws": sum(1 for m in self._documents.values() if not m.is_amendment),
            "with_effective_date": sum(1 for m in self._documents.values() if m.effective_date),
            "currently_in_effect": sum(1 for m in self._documents.values() if m.is_in_effect),
            "total_graph_nodes": self._graph_data.get("nodes", []).__len__(),
            "total_graph_edges": self._graph_data.get("edges", []).__len__(),
            "total_amendments": sum(m.amendment_count for m in self._documents.values()),
            "total_articles_amended": len(set(
                a for m in self._documents.values() for a in m.amended_articles
            )),
            "last_enriched": max(
                (m.enriched_at for m in self._documents.values() if m.enriched_at),
                default=None,
            ),
            "documents": sorted(
                [
                    {
                        "doc_id": m.doc_id,
                        "title": m.title[:80],
                        "is_amendment": m.is_amendment,
                        "effective_date": m.effective_date,
                        "is_in_effect": m.is_in_effect,
                        "ref_count": m.ref_count,
                        "amendment_count": m.amendment_count,
                    }
                    for m in self._documents.values()
                ],
                key=lambda d: d["doc_id"],
            ),
        }
        self._save_json(self._index_file, index)
        return index

    def get_index(self) -> Dict[str, Any]:
        """Get the summary index."""
        if not self._index_file.exists():
            return self.build_index()
        return self._load_json(self._index_file)

    # ── Persistence ────────────────────────────────────────────

    def _load(self) -> None:
        """Load all data from disk."""
        if self._docs_file.exists():
            raw = self._load_json(self._docs_file)
            self._documents = {
                int(k): DocumentMeta.from_dict(v)
                for k, v in raw.items()
            }

        if self._graph_file.exists():
            self._graph_data = self._load_json(self._graph_file)

        if self._timeline_file.exists():
            self._timelines = self._load_json(self._timeline_file)

        if self._amendments_file.exists():
            self._amendments = self._load_json(self._amendments_file)

        if self._documents:
            logger.info(f"Loaded {len(self._documents)} document metadata records")

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        """Save data to a JSON file atomically."""
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)

    def clear(self) -> None:
        """Clear all stored data."""
        self._documents.clear()
        self._graph_data.clear()
        self._timelines.clear()
        self._amendments.clear()
        for f in [self._docs_file, self._graph_file, self._timeline_file,
                   self._amendments_file, self._index_file]:
            if f.exists():
                f.unlink()
