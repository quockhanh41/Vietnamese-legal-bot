"""
M3.4 — comparison queries must relax BOTH supersession layers (search-level
exclude_superseded AND the _filter_in_effect post-filter). Pure-Python: vector
store / embedder / LLM mocked, heavy post-processing stubbed.

Run: .venv/bin/python -m pytest tests/test_comparison_supersession.py -q
"""
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.orchestrator import AdaptiveRAGPipeline


def _adaptive_with_mocks() -> AdaptiveRAGPipeline:
    embedder = MagicMock()
    embedder.encode_queries.return_value = [np.array([0.1, 0.2])]
    embedder.encode.return_value = {"dense_vecs": [np.array([0.1, 0.2])], "lexical_weights": [{}]}
    llm = MagicMock()
    llm.generate.return_value = "answer"
    llm.generate_with_model.return_value = ("answer", "mock/model")
    return AdaptiveRAGPipeline(
        vector_store=MagicMock(), embedder=embedder,
        llm_gateway=llm, metadata_store=MagicMock(),
    )


def _drive(route_method: str, is_comparison: bool):
    """Run one non-stream route with heavy collaborators stubbed; capture the
    supersession flags passed to search + _filter_in_effect."""
    p = _adaptive_with_mocks()
    cap = {}

    def fake_search(*a, **k):
        cap["exclude_superseded"] = k.get("exclude_superseded")
        return []

    p.vector_store.search_dense = fake_search
    p.vector_store.search_hybrid = fake_search

    capf = {}

    def spy_filter(results, annotations, keep_superseded=False):
        capf["keep_superseded"] = keep_superseded
        return results

    p._filter_in_effect = spy_filter
    p._annotate_phase2 = lambda results: {}
    p._post_process = lambda *a, **k: (MagicMock(), SimpleNamespace(warnings=[]), "answer")
    p._enrich_response = lambda parsed, c, v: parsed

    getattr(p, route_method)(
        "Luật X 2013 và Luật X 2024 khác nhau thế nào?", time.time(),
        is_comparison=is_comparison,
    )
    return cap, capf


@pytest.mark.parametrize("route", ["_simple_pipeline", "_medium_pipeline", "_complex_pipeline"])
def test_comparison_relaxes_both_supersession_layers(route):
    cap, capf = _drive(route, is_comparison=True)
    assert cap["exclude_superseded"] is False
    assert capf["keep_superseded"] is True


@pytest.mark.parametrize("route", ["_simple_pipeline", "_medium_pipeline", "_complex_pipeline"])
def test_non_comparison_keeps_supersession_on(route):
    cap, capf = _drive(route, is_comparison=False)
    assert cap["exclude_superseded"] is True
    assert capf["keep_superseded"] is False


def _stream_pipeline(classify_line: str):
    """Drive query_stream end-to-end with mocks; return the exclude_superseded
    captured from the medium-route hybrid search."""
    p = _adaptive_with_mocks()
    p.llm.generate.return_value = classify_line          # _classify_and_scope call
    p.llm.generate_stream.return_value = iter(["ans"])   # token stream
    p._get_citation_lookup = lambda: SimpleNamespace(
        corpus_summary=lambda: "", corpus_summary_compact=lambda: "")
    cap = {}

    def fake_hybrid(*a, **k):
        cap["exclude_superseded"] = k.get("exclude_superseded")
        return []

    p.vector_store.search_hybrid = fake_hybrid
    p._annotate_phase2 = lambda r: {}
    p._filter_in_effect = lambda results, annotations, keep_superseded=False: results
    p._post_process = lambda *a, **k: (
        SimpleNamespace(score=0.9, label="high", refused=False),
        SimpleNamespace(warnings=[], entailment_ratio=1.0,
                        contradiction_ratio=0.0, grounding_ratio=1.0),
        "ans",
    )
    list(p.query_stream("Luật X 2013 và Luật X 2024 khác nhau về hạn mức?"))
    return cap


def test_stream_comparison_relaxes_supersession():
    cap = _stream_pipeline("SCOPE: IN\nCLASS: MEDIUM\nCOMPARE: YES")
    assert cap["exclude_superseded"] is False


def test_stream_non_comparison_keeps_supersession():
    cap = _stream_pipeline("SCOPE: IN\nCLASS: MEDIUM\nCOMPARE: NO")
    assert cap["exclude_superseded"] is True
