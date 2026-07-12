"""
Document-level metadata store for Phase 2 enrichment data.

Persists cross-reference graphs, timeline data, and amendment instructions
in a single SQLite file, keyed by document ID.

Design: SQLite-backed store — scales to tens of thousands of docs on a
single file, no separate JSON files or in-memory caches to keep in sync.
"""

from __future__ import annotations

import json
import sqlite3
import logging
import threading
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
    # None = unknown (kept by the temporal filter, which excludes only an
    # explicit False). The old `False` default silently hid current docs
    # whose effective_date extraction failed (Luật Đầu tư 2025, NĐ 144/2021
    # on the 100-doc corpus, 2026-07-12).
    is_in_effect: Optional[bool] = None
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


_LIST_FIELDS = ("articles_referenced", "replaced_laws", "version_chain",
                "target_laws", "amended_articles", "errors")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id INTEGER PRIMARY KEY, title TEXT, ref_count INTEGER, internal_refs INTEGER,
    external_refs INTEGER, effective_date TEXT, expiry_date TEXT, is_in_effect INTEGER,
    is_amendment INTEGER, amendment_count INTEGER, graph_node_id TEXT,
    graph_in_degree INTEGER, graph_out_degree INTEGER, enriched_at TEXT,
    articles_referenced TEXT, replaced_laws TEXT, version_chain TEXT,
    target_laws TEXT, amended_articles TEXT, errors TEXT);
CREATE INDEX IF NOT EXISTS ix_docs_amend ON documents(is_amendment);
CREATE TABLE IF NOT EXISTS timelines (
    doc_id INTEGER, effective_date TEXT, expiry_date TEXT, is_in_effect INTEGER, blob TEXT);
CREATE INDEX IF NOT EXISTS ix_tl_doc ON timelines(doc_id);
CREATE TABLE IF NOT EXISTS amendments (amending_doc_id INTEGER, target TEXT, blob TEXT);
CREATE INDEX IF NOT EXISTS ix_am_doc ON amendments(amending_doc_id);
CREATE TABLE IF NOT EXISTS graph (id INTEGER PRIMARY KEY CHECK(id=1), blob TEXT);
"""


class MetadataStore:
    """
    SQLite-backed document metadata store for Phase 2 enrichment.

    Usage:
        store = MetadataStore("data/metadata")
        store.save_document(doc_id, meta)
        meta = store.get_document(doc_id)
        all_meta = store.get_all_documents()
    """

    def __init__(self, data_dir: str = "data/metadata"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "store.db"
        # check_same_thread=False + an RLock: a single shared connection is
        # safe to use from FastAPI's threadpool workers (sync endpoints run
        # off the event-loop thread). RLock (reentrant) is required because
        # some methods call others (get_articles_amended→get_document,
        # build_index→get_all_documents/get_graph) and a plain Lock would
        # self-deadlock. Read-heavy / rare-write at ~600-doc demo scale.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── Document Metadata ──────────────────────────────────────

    @staticmethod
    def _to_bool(v: Optional[int]) -> Optional[bool]:
        return None if v is None else bool(v)

    def _row_to_meta(self, row: sqlite3.Row) -> DocumentMeta:
        d = dict(row)
        for f in _LIST_FIELDS:
            d[f] = json.loads(d[f]) if d[f] else []
        d["is_in_effect"] = self._to_bool(d["is_in_effect"])
        d["is_amendment"] = bool(d["is_amendment"])
        return DocumentMeta.from_dict(d)

    def save_document(self, doc_id: int, meta: DocumentMeta) -> None:
        """Save document metadata."""
        meta.enriched_at = datetime.now().isoformat()
        d = meta.to_dict()
        for f in _LIST_FIELDS:
            d[f] = json.dumps(d[f], ensure_ascii=False)
        cols = ["doc_id", "title", "ref_count", "internal_refs", "external_refs",
                "effective_date", "expiry_date", "is_in_effect", "is_amendment",
                "amendment_count", "graph_node_id", "graph_in_degree", "graph_out_degree",
                "enriched_at", *_LIST_FIELDS]
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO documents ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [d.get(c) for c in cols])
            self._conn.commit()

    def get_document(self, doc_id: int) -> Optional[DocumentMeta]:
        """Get metadata for a single document."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return self._row_to_meta(row) if row else None

    def get_all_documents(self) -> Dict[int, DocumentMeta]:
        """Get all document metadata."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return {r["doc_id"]: self._row_to_meta(r) for r in rows}

    def get_documents_by_target(self, law_name: str) -> List[DocumentMeta]:
        """Find amendment documents that target a specific law."""
        norm = law_name.lower()
        results = []
        with self._lock:
            rows = self._conn.execute("SELECT * FROM documents WHERE is_amendment=1").fetchall()
        for r in rows:
            meta = self._row_to_meta(r)
            for target in meta.target_laws:
                if norm in target.lower() or target.lower() in norm:
                    results.append(meta)
                    break
        return results

    def get_articles_amended(self, doc_id: int) -> List[str]:
        """Get list of articles amended by a document."""
        meta = self.get_document(doc_id)
        return meta.amended_articles if meta else []

    # ── Graph Data ─────────────────────────────────────────────

    def save_graph(self, graph_dict: Dict[str, Any]) -> None:
        """Save graph data (nodes + edges in serializable format)."""
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO graph (id, blob) VALUES (1, ?)",
                               (json.dumps(graph_dict, ensure_ascii=False, default=str),))
            self._conn.commit()

    def get_graph(self) -> Dict[str, Any]:
        """Get the full graph data."""
        with self._lock:
            row = self._conn.execute("SELECT blob FROM graph WHERE id=1").fetchone()
        return json.loads(row["blob"]) if row else {}

    def get_article_neighbors(self, node_id: str, hops: int = 1) -> List[str]:
        """
        Get neighboring article node IDs from the graph.
        Requires the graph to be loaded (rebuild from graph_data if needed).
        """
        # This is a query-time method — use the NetworkX graph for actual traversal
        # Here we just provide the stored adjacency data
        edges = self.get_graph().get("edges", [])
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
        with self._lock:
            self._conn.execute("DELETE FROM timelines")
            self._conn.executemany(
                "INSERT INTO timelines (doc_id, effective_date, expiry_date, is_in_effect, blob) VALUES (?,?,?,?,?)",
                [(t.get("doc_id"), t.get("effective_date"), t.get("expiry_date"),
                  None if t.get("is_in_effect") is None else int(bool(t.get("is_in_effect"))),
                  json.dumps(t, ensure_ascii=False, default=str)) for t in timelines])
            self._conn.commit()

    def get_timelines(self) -> List[Dict]:
        """Get all timeline data."""
        with self._lock:
            rows = self._conn.execute("SELECT blob FROM timelines").fetchall()
        return [json.loads(r["blob"]) for r in rows]

    def get_timeline(self, doc_id: int) -> Optional[Dict]:
        """Get timeline for a specific document."""
        with self._lock:
            row = self._conn.execute("SELECT blob FROM timelines WHERE doc_id=?", (doc_id,)).fetchone()
        return json.loads(row["blob"]) if row else None

    def get_effective_laws(self, at_date: Optional[date] = None) -> List[Dict]:
        """Get laws effective at a given date."""
        if at_date is None:
            at_date = date.today()
        iso = at_date.isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT blob FROM timelines WHERE effective_date IS NOT NULL AND effective_date<=? "
                "AND (expiry_date IS NULL OR expiry_date>?)", (iso, iso)).fetchall()
        return [json.loads(r["blob"]) for r in rows]

    # ── Amendment Data ─────────────────────────────────────────

    def save_amendments(self, amendments: List[Dict]) -> None:
        """Save amendment instructions (serialized)."""
        with self._lock:
            self._conn.execute("DELETE FROM amendments")
            self._conn.executemany(
                "INSERT INTO amendments (amending_doc_id, target, blob) VALUES (?,?,?)",
                [(a.get("amending_doc_id"), a.get("target"),
                  json.dumps(a, ensure_ascii=False, default=str)) for a in amendments])
            self._conn.commit()

    def get_amendments(self) -> List[Dict]:
        """Get all amendment data."""
        with self._lock:
            rows = self._conn.execute("SELECT blob FROM amendments").fetchall()
        return [json.loads(r["blob"]) for r in rows]

    def get_amendments_for_doc(self, doc_id: int) -> List[Dict]:
        """Get amendments produced by a specific document."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT blob FROM amendments WHERE amending_doc_id=?", (doc_id,)).fetchall()
        return [json.loads(r["blob"]) for r in rows]

    # ── Index / Summary ────────────────────────────────────────

    def build_index(self) -> Dict[str, Any]:
        """Build a summary index of all stored data."""
        with self._lock:
            documents = self.get_all_documents()
            graph_data = self.get_graph()
        index = {
            "total_documents": len(documents),
            "amendment_laws": sum(1 for m in documents.values() if m.is_amendment),
            "base_laws": sum(1 for m in documents.values() if not m.is_amendment),
            "with_effective_date": sum(1 for m in documents.values() if m.effective_date),
            "currently_in_effect": sum(1 for m in documents.values() if m.is_in_effect),
            "total_graph_nodes": graph_data.get("nodes", []).__len__(),
            "total_graph_edges": graph_data.get("edges", []).__len__(),
            "total_amendments": sum(m.amendment_count for m in documents.values()),
            "total_articles_amended": len(set(
                a for m in documents.values() for a in m.amended_articles
            )),
            "last_enriched": max(
                (m.enriched_at for m in documents.values() if m.enriched_at),
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
                    for m in documents.values()
                ],
                key=lambda d: d["doc_id"],
            ),
        }
        return index

    def get_index(self) -> Dict[str, Any]:
        """Get the summary index."""
        return self.build_index()

    # ── Persistence ────────────────────────────────────────────

    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            for t in ("documents", "timelines", "amendments", "graph"):
                self._conn.execute(f"DELETE FROM {t}")
            self._conn.commit()
