"""
Unit tests for scripts/ingest_zalo_corpus.py — corpus flattening only
(embedding/upsert paths hit models/Qdrant and are exercised by running the
script, not by unit tests).

Run: .venv/bin/python -m pytest tests/test_zalo_ingest.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ingest_zalo_corpus import flatten_corpus

CORPUS = [
    {"law_id": "01/2020/qh14", "articles": [
        {"article_id": "2", "title": "Điều 2. Phạm vi", "text": "Nội dung."},
    ]},
]


def test_flatten_embed_text_is_title_plus_text():
    rows = flatten_corpus(CORPUS)
    assert len(rows) == 1
    assert rows[0]["embed_text"] == "Điều 2. Phạm vi\nNội dung."


def test_flatten_metadata_payload_contract():
    md = flatten_corpus(CORPUS)[0]["metadata"]
    # Benchmark ids ride on existing SearchResult fields (Task 5 reads
    # result.document_number / result.article):
    assert md["document_number"] == "01/2020/qh14"
    assert md["article"] == "2"
    assert md["document_title"] == "Điều 2. Phạm vi"
    # Deterministic point identity for compute_point_id:
    assert md["document_id"] == "01/2020/qh14"
    assert md["chunk_index"] == "2"
