"""
Unit tests for src/pipeline/enricher.py — EnrichmentPipeline._backfill_is_in_effect
(M2.5). No network: vector_store is a fake that just records set_is_in_effect
calls; metadata_store is a real MetadataStore backed by a tmp_path (SQLite is
cheap and this exercises the real DocumentMeta.is_in_effect tri-state).

Run: .venv/bin/python -m pytest tests/test_enricher.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.enricher import EnrichmentPipeline
from src.data.metadata_store import MetadataStore, DocumentMeta


class _FakeVectorStore:
    """Records set_is_in_effect calls instead of talking to Qdrant."""

    def __init__(self):
        self.calls = []

    def set_is_in_effect(self, doc_id, value):
        self.calls.append((doc_id, value))


def _pipeline_with(tmp_path):
    vector_store = _FakeVectorStore()
    metadata_store = MetadataStore(data_dir=str(tmp_path))
    pipeline = EnrichmentPipeline(vector_store, metadata_store)
    return pipeline, vector_store, metadata_store


def test_backfill_pushes_only_known_status_docs(tmp_path):
    pipeline, vector_store, metadata_store = _pipeline_with(tmp_path)

    metadata_store.save_document(1, DocumentMeta(doc_id=1, title="Luật A", is_in_effect=True))
    metadata_store.save_document(2, DocumentMeta(doc_id=2, title="Luật B", is_in_effect=False))
    metadata_store.save_document(3, DocumentMeta(doc_id=3, title="Luật C", is_in_effect=None))

    n = pipeline._backfill_is_in_effect()

    assert n == 2
    assert set(vector_store.calls) == {(1, True), (2, False)}
    assert all(doc_id != 3 for doc_id, _ in vector_store.calls)


def test_backfill_returns_zero_when_all_unknown(tmp_path):
    pipeline, vector_store, metadata_store = _pipeline_with(tmp_path)

    metadata_store.save_document(1, DocumentMeta(doc_id=1, title="Luật A", is_in_effect=None))

    n = pipeline._backfill_is_in_effect()

    assert n == 0
    assert vector_store.calls == []


def test_extract_refs_per_article_sets_full_source_article():
    """Per-article extraction sets source_article to the FULL `article` payload
    string — the exact string query-time expansion builds its seed node id from."""
    from unittest.mock import MagicMock
    from src.pipeline.enricher import EnrichmentPipeline

    class Ch:
        def __init__(self, payload):
            self.payload = payload

    p = EnrichmentPipeline(vector_store=MagicMock(), metadata_store=MagicMock())
    chunks = [
        Ch({"article": "Điều 5. Giải thích từ ngữ",
            "parent_chunk_text": "Điều 5. ... áp dụng theo quy định tại Điều 10 của Luật này ..."}),
    ]
    refs = p._extract_refs_per_article(chunks, doc_id=1, title="Luật Test 2020")
    assert refs, "should extract at least one reference"
    assert all(r.source_article == "Điều 5. Giải thích từ ngữ" for r in refs)
    assert any(r.target_article == "Điều 10" for r in refs)


def test_extract_refs_per_article_dedupes_article_by_payload_string():
    """Multiple chunks of the same article extract that article's text once."""
    from unittest.mock import MagicMock
    from src.pipeline.enricher import EnrichmentPipeline

    class Ch:
        def __init__(self, payload):
            self.payload = payload

    p = EnrichmentPipeline(vector_store=MagicMock(), metadata_store=MagicMock())
    same = {"article": "Điều 7. X", "parent_chunk_text": "Điều 7. ... tại Điều 3 của Luật này ..."}
    refs = p._extract_refs_per_article([Ch(same), Ch(dict(same))], doc_id=1, title="Luật Test")
    # article 7 processed once → its single internal ref to Điều 3, not doubled
    assert sum(1 for r in refs if r.target_article == "Điều 3") == 1


def test_extract_refs_no_article_stays_doc_level():
    """A chunk with no `article` (preamble) keeps source_article=None."""
    from unittest.mock import MagicMock
    from src.pipeline.enricher import EnrichmentPipeline

    class Ch:
        def __init__(self, payload):
            self.payload = payload

    p = EnrichmentPipeline(vector_store=MagicMock(), metadata_store=MagicMock())
    chunks = [Ch({"article": None, "chunk_text": "... theo Điều 3 của Luật này ..."})]
    refs = p._extract_refs_per_article(chunks, doc_id=1, title="Luật Test")
    assert refs and all(r.source_article is None for r in refs)
