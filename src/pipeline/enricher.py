"""
Phase 2 Enrichment Pipeline — Orchestrate all 3 modules.

Runs cross-reference extraction, temporal versioning, and amendment tracking
on all documents in a Qdrant collection, storing results in MetadataStore.

Pipeline flow:
  1. Collect all documents from Qdrant (deduplicate by document_id)
  2. Reconstruct full text by concatenating chunks
  3. For each document:
     a. Extract cross-references → build knowledge graph
     b. Extract effective/expiry dates → timeline
     c. If amendment law: extract amendment instructions
  4. Store all results in MetadataStore
  5. Build summary index
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.db.qdrant_client import LegalVectorStore
from src.data.graph import ReferenceExtractor, LegalKnowledgeGraph, Reference
from src.data.timeline import DateExtractor, LawTimeline, VersionResolver
from src.data.amendment import (
    AmendmentExtractor,
    AmendmentInstruction,
    AmendmentConsolidator,
)
from src.data.metadata_store import MetadataStore, DocumentMeta

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Result of running the full enrichment pipeline."""

    total_docs: int = 0
    docs_processed: int = 0
    base_laws: int = 0
    amendment_laws: int = 0

    # Module results
    total_refs: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0

    with_effective_date: int = 0
    currently_in_effect: int = 0

    total_amendments: int = 0
    articles_amended: int = 0

    # Timing
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_seconds: float = 0

    # Issues
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class EnrichmentPipeline:
    """
    Orchestrates Phase 2 enrichment across all 3 modules.

    Usage:
        pipeline = EnrichmentPipeline(vector_store, metadata_store)
        result = pipeline.run()
        print(f"Enriched {result.docs_processed} documents")
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        metadata_store: Optional[MetadataStore] = None,
    ):
        self.vector_store = vector_store
        self.metadata_store = metadata_store or MetadataStore()

        # Module instances
        self.ref_extractor = ReferenceExtractor()
        self.date_extractor = DateExtractor()
        self.amend_extractor = AmendmentExtractor()
        self.consolidator = AmendmentConsolidator()

        # Knowledge graph (built incrementally)
        self.graph = LegalKnowledgeGraph()

        # Result tracking
        self.result = EnrichmentResult()

    def run(self) -> EnrichmentResult:
        """Execute the full enrichment pipeline."""
        self.result.started_at = datetime.now().isoformat()
        start_time = datetime.now()

        # Step 1: Collect all documents from Qdrant
        documents = self._collect_documents()
        self.result.total_docs = len(documents)

        if not documents:
            self.result.errors.append("No documents found in Qdrant collection")
            return self.result

        logger.info(f"Collected {len(documents)} unique documents")

        # Step 2: Reconstruct full text and extract metadata
        all_timelines = []

        for doc_id, info in documents.items():
            try:
                full_text = info["full_text"]
                title = info["title"]
                issuance_date = info.get("issuance_date", "")

                meta = DocumentMeta(doc_id=doc_id, title=title)

                # ── M1: Cross-References ──
                refs = self.ref_extractor.extract(
                    text=full_text, doc_id=doc_id, law_name=title
                )
                meta.ref_count = len(refs)
                meta.internal_refs = sum(1 for r in refs if r.ref_type.value == "internal")
                meta.external_refs = sum(1 for r in refs if r.ref_type.value == "external")
                meta.articles_referenced = list(set(
                    r.target_article for r in refs if r.target_article
                ))
                self.graph.add_references(refs)
                self.result.total_refs += len(refs)

                # ── M2: Temporal ──
                timeline = self.date_extractor.extract(
                    doc_id=doc_id,
                    content=full_text,
                    issuance_date=issuance_date,
                    law_name=title,
                )
                # Số hiệu văn bản cho exact replacement matching (M1.8b) —
                # name-based matching alone misfires at multi-version scale
                # (see VersionResolver.find_replacement).
                timeline.doc_number = info.get("document_number") or None
                all_timelines.append(timeline)

                if timeline.effective_date:
                    meta.effective_date = timeline.effective_date.isoformat()
                    meta.is_in_effect = timeline.is_in_effect
                    self.result.with_effective_date += 1
                    if meta.is_in_effect:
                        self.result.currently_in_effect += 1
                if timeline.expiry_date:
                    meta.expiry_date = timeline.expiry_date.isoformat()
                if timeline.replaced_laws:
                    meta.replaced_laws = timeline.replaced_laws

                # ── M3: Amendments ──
                if self.amend_extractor.is_amendment_law(title):
                    meta.is_amendment = True
                    meta.target_laws = self.amend_extractor.extract_target_laws(
                        title, full_text
                    )

                    instructions = self.amend_extractor.extract(
                        content=full_text, doc_id=doc_id, title=title
                    )
                    meta.amendment_count = len(instructions)
                    meta.amended_articles = list(set(
                        i.target_article for i in instructions
                        if i.target_article
                    ))
                    self.result.amendment_laws += 1
                    self.result.total_amendments += len(instructions)
                    self.result.articles_amended += len(meta.amended_articles)
                else:
                    self.result.base_laws += 1

                # Save this document's metadata
                self.metadata_store.save_document(doc_id, meta)
                self.result.docs_processed += 1

            except Exception as e:
                msg = f"Error processing doc {doc_id} ({info.get('title', '?')[:50]}): {e}"
                logger.error(msg)
                self.result.errors.append(msg)

        # Step 3: Resolve version chains
        if all_timelines:
            resolver = VersionResolver(all_timelines)
            for doc_id, info in documents.items():
                meta = self.metadata_store.get_document(doc_id)
                if meta and not meta.is_amendment:
                    chain = resolver.version_chain(info["title"])
                    meta.version_chain = [
                        f"{t.law_name} ({t.effective_date.isoformat() if t.effective_date else '?'})"
                        for t in chain
                    ]

                    # Cross-link supersession (Plan §3.6): is_in_effect above
                    # was only self-reported from each doc's own expiry text,
                    # so a law that was replaced by a *newer* law (which
                    # names it in ITS `replaced_laws`) was never marked
                    # expired. VersionResolver.find_replacement already does
                    # this name-matching — it just wasn't wired in before.
                    replacement = resolver.find_replacement(doc_id)
                    # `is not False`: flip True AND None (unknown) — a found
                    # replacement is positive knowledge either way; only skip
                    # docs already known superseded (their own expiry text).
                    if replacement and replacement.effective_date and meta.is_in_effect is not False:
                        if meta.is_in_effect:
                            self.result.currently_in_effect -= 1
                        meta.is_in_effect = False
                        meta.expiry_date = meta.expiry_date or replacement.effective_date.isoformat()

                    self.metadata_store.save_document(doc_id, meta)

            # Save timelines
            timeline_dicts = [_serialize_timeline(tl) for tl in all_timelines]
            self.metadata_store.save_timelines(timeline_dicts)

        # Step 4: Save graph
        self.metadata_store.save_graph(self.graph.to_dict())
        self.result.graph_nodes = self.graph.stats.get("nodes", 0)
        self.result.graph_edges = self.graph.stats.get("edges", 0)

        # Step 5: Save amendment data
        # (amendments are stored in individual document metadata)

        # Step 6: Build index
        self.metadata_store.build_index()

        # Step 7 (M2.5): push known is_in_effect onto chunk payloads so
        # retrieval can filter server-side (exclude_superseded=True) instead
        # of relying on a client-side post-filter.
        n_backfilled = self._backfill_is_in_effect()
        print(f"  ↳ is_in_effect backfilled on {n_backfilled} docs")

        # Finalize
        self.result.finished_at = datetime.now().isoformat()
        self.result.elapsed_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Enrichment complete: {self.result.docs_processed} docs in "
            f"{self.result.elapsed_seconds:.1f}s"
        )

        return self.result

    # ── Helpers ─────────────────────────────────────────────────

    def _backfill_is_in_effect(self) -> int:
        """Push each doc's known is_in_effect onto its chunk payloads so
        retrieval can filter server-side (LegalVectorStore.search_*(
        exclude_superseded=True)). Only docs with a KNOWN status (True/False)
        are written; None (unknown) is left unset so the filter keeps it
        (fail-open).
        """
        n = 0
        for doc_id, meta in self.metadata_store.get_all_documents().items():
            if meta.is_in_effect is not None:
                self.vector_store.set_is_in_effect(doc_id, bool(meta.is_in_effect))
                n += 1
        return n

    def _collect_documents(self) -> Dict[int, Dict[str, Any]]:
        """
        Collect all unique documents from Qdrant and reconstruct full text.

        Returns:
            Dict[int, Dict] mapping doc_id → {title, issuance_date, chunks, full_text}
        """
        documents: Dict[int, Dict] = {}

        # Scan all points in the collection
        offset = None
        while True:
            points, offset = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break

            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("document_id")
                if not doc_id:
                    continue

                if doc_id not in documents:
                    documents[doc_id] = {
                        "doc_id": doc_id,
                        "title": payload.get("document_title", ""),
                        "issuance_date": payload.get("issuance_date", ""),
                        "document_number": payload.get("document_number", ""),
                        "legal_type": payload.get("legal_type", ""),
                        "chunks": [],
                    }
                documents[doc_id]["chunks"].append(p)

            if offset is None:
                break

        # Sort chunks and build full text
        for doc_id in documents:
            documents[doc_id]["chunks"].sort(
                key=lambda p: p.payload.get("chunk_index", 0)
            )
            documents[doc_id]["full_text"] = "\n".join(
                p.payload.get("chunk_text", "") for p in documents[doc_id]["chunks"]
            )

        return documents

    def get_status(self) -> Dict[str, Any]:
        """Get current enrichment status."""
        index = self.metadata_store.get_index()
        return {
            "result": {
                "docs_processed": self.result.docs_processed,
                "total_docs": self.result.total_docs,
                "elapsed_seconds": self.result.elapsed_seconds,
                "started_at": self.result.started_at,
                "finished_at": self.result.finished_at,
            },
            "index": index,
            "errors": self.result.errors,
        }


def _serialize_timeline(tl: LawTimeline) -> Dict[str, Any]:
    """Serialize a LawTimeline to a JSON-safe dict."""
    return {
        "doc_id": tl.doc_id,
        "law_name": tl.law_name,
        "issuance_date": tl.issuance_date,
        "doc_number": tl.doc_number,
        "effective_date": tl.effective_date.isoformat() if tl.effective_date else None,
        "expiry_date": tl.expiry_date.isoformat() if tl.expiry_date else None,
        "is_in_effect": tl.is_in_effect,
        "replaced_laws": tl.replaced_laws,
        "article_dates": {
            k: v.isoformat() for k, v in (tl.article_dates or {}).items()
        },
        "is_relative": tl.is_relative,
        "relative_days": tl.relative_days,
        "source_text": tl.source_text[:200] if tl.source_text else "",
    }
