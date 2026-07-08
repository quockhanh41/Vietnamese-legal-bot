"""
Unit tests for src.pipeline.retriever.get_phase2_status — the Phase 2
(temporal + amendment) lookup shared by EnhancedRetriever and the chat
orchestrator (Plan §3.6). Pure-Python: MetadataStore is a lightweight fake,
no JSON files touched.

Run: .venv/bin/python -m pytest tests/test_retriever_phase2.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.metadata_store import DocumentMeta
from src.pipeline.retriever import get_phase2_status


class FakeMetadataStore:
    """Minimal stand-in for MetadataStore exposing only what get_phase2_status needs."""

    def __init__(self, documents=None, timelines=None):
        self._documents = documents or {}
        self._timelines = timelines or {}

    def get_document(self, doc_id):
        return self._documents.get(doc_id)

    def get_all_documents(self):
        return dict(self._documents)

    def get_timeline(self, doc_id):
        return self._timelines.get(doc_id)


def test_no_document_id_returns_unknown_status():
    status = get_phase2_status(None, "Điều 173", FakeMetadataStore())
    assert status["is_in_effect"] is None
    assert status["is_amended"] is False


def test_no_metadata_for_doc_returns_unknown_status():
    status = get_phase2_status(999, "Điều 173", FakeMetadataStore())
    assert status["is_in_effect"] is None


def test_effective_date_and_in_effect_from_document_meta():
    store = FakeMetadataStore(documents={
        1: DocumentMeta(doc_id=1, title="Bộ luật hình sự 2015", effective_date="2016-07-01", is_in_effect=True),
    })
    status = get_phase2_status(1, "Điều 173", store)
    assert status["effective_date"] == "2016-07-01"
    assert status["is_in_effect"] is True


def test_superseded_document_reports_is_in_effect_false():
    store = FakeMetadataStore(documents={
        46056: DocumentMeta(doc_id=46056, title="Bộ Luật Hình sự 1999", is_in_effect=False, expiry_date="2016-07-01"),
    })
    status = get_phase2_status(46056, "Điều 7", store)
    assert status["is_in_effect"] is False


def test_chunk_from_amendment_law_is_flagged_amended():
    """A chunk that IS a paragraph of an amendment law is itself amendment content."""
    store = FakeMetadataStore(documents={
        354053: DocumentMeta(
            doc_id=354053, title="Luật sửa đổi Bộ luật Hình sự 2017",
            is_amendment=True, target_laws=["Bộ luật Hình sự"],
        ),
    })
    status = get_phase2_status(354053, "Điều 1", store)
    assert status["is_amended"] is True
    assert status["amending_laws"] == ["Bộ luật Hình sự"]


def test_article_targeted_by_another_amendment_law_is_flagged():
    """A base-law article (Điều 173) that some OTHER amendment law's
    amended_articles list targets should be flagged as amended."""
    store = FakeMetadataStore(documents={
        296661: DocumentMeta(doc_id=296661, title="Bộ luật hình sự 2015", is_amendment=False),
        354053: DocumentMeta(
            doc_id=354053, title="Luật sửa đổi Bộ luật Hình sự 2017",
            is_amendment=True, amended_articles=["173", "174"],
        ),
    })
    status = get_phase2_status(296661, "Điều 173", store)
    assert status["is_amended"] is True
    assert "Luật sửa đổi Bộ luật Hình sự 2017"[:60] in status["amending_laws"]


def test_article_not_targeted_by_any_amendment_is_not_flagged():
    store = FakeMetadataStore(documents={
        296661: DocumentMeta(doc_id=296661, title="Bộ luật hình sự 2015", is_amendment=False),
        354053: DocumentMeta(
            doc_id=354053, title="Luật sửa đổi Bộ luật Hình sự 2017",
            is_amendment=True, amended_articles=["999"],
        ),
    })
    status = get_phase2_status(296661, "Điều 173", store)
    assert status["is_amended"] is False
    assert status["amending_laws"] == []


def test_timeline_fills_effective_date_when_document_meta_missing_it():
    store = FakeMetadataStore(
        documents={1: DocumentMeta(doc_id=1, title="Test")},
        timelines={1: {"effective_date": "2015-11-27"}},
    )
    status = get_phase2_status(1, None, store)
    assert status["effective_date"] == "2015-11-27"
