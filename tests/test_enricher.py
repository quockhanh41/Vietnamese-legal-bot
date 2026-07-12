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
