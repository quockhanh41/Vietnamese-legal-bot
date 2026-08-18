"""Parity between AdaptiveRAGPipeline.query() and .query_stream().

The project measures query() (scripts/eval.py:364, "non-streaming path") but
ships query_stream() (src/api/main.py, src/ui/app_gradio.py). Nothing used to
compare them, so divergences were invisible to every gate the project has.
These tests make "the two paths must agree" a machine-checked claim.

Pure-Python: no Qdrant, no LLM, no models.

Run: .venv/bin/python -m pytest tests/test_stream_nonstream_parity.py -q
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import src.rag.orchestrator as orch
from src.db.qdrant_client import SearchResult
from src.rag.orchestrator import AdaptiveRAGPipeline


class _Vec:
    """Stands in for the numpy vector the embedder returns."""

    def tolist(self):
        return [0.1, 0.2, 0.3]


class _Conf:
    """ConfidenceScorer result double — real floats, so the streaming path can
    JSON-serialize the __VERIFICATION__ payload."""

    label = "high"
    score = 0.9
    refused = False


class _Ver:
    """HallucinationVerifier result double."""

    passed = True
    warnings = []
    entailment_ratio = 0.0
    contradiction_ratio = 0.0
    grounding_ratio = 0.0


class _NoMeta:
    """MetadataStore double with nothing in it.

    get_document -> None keeps get_phase2_status on its fail-open path
    (is_in_effect stays None, so _filter_in_effect keeps every result), and an
    empty graph makes _expand_cross_references a no-op. Both keep the two paths
    comparing retrieval, not metadata lookups.
    """

    def get_document(self, doc_id):
        return None

    def get_timeline(self, doc_id):
        return None

    def get_all_documents(self):
        return {}

    def get_graph(self):
        return {"nodes": [], "edges": []}


class _Reranker:
    """Deterministic cross-encoder double: preserves input order."""

    def compute_score(self, pairs, normalize=True):
        return [1.0 - i * 0.01 for i in range(len(pairs))]


def _sr(i):
    return SearchResult(
        point_id=f"p{i}",
        score=1.0 - i * 0.01,
        chunk_text=f"chunk {i}",
        document_title="Bộ luật hình sự 2015",
        document_number="100/2015/QH13",
        legal_type="Bộ luật",
        article=f"Điều {100 + i}",
        document_id=1,
        parent_chunk_text=f"Điều {100 + i} full text",
    )


def _make_adaptive(n_results=20):
    """AdaptiveRAGPipeline wired to doubles only."""
    results = [_sr(i) for i in range(n_results)]

    vector_store = MagicMock()
    vector_store.search_dense.return_value = list(results)
    vector_store.search_hybrid.return_value = list(results)

    embedder = MagicMock()
    embedder.encode_queries.return_value = [_Vec()]
    embedder.encode.return_value = {"dense_vecs": [_Vec()], "lexical_weights": [{}]}

    llm = MagicMock()
    llm.generate.return_value = "1. truy van mo rong"
    llm.generate_with_model.return_value = ("cau tra loi", "groq/fake-model")
    llm.generate_stream.return_value = iter(["cau ", "tra ", "loi"])

    return AdaptiveRAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        llm_gateway=llm,
        metadata_store=_NoMeta(),
        reranker=_Reranker(),
        citation_lookup=MagicMock(),
    )


@pytest.fixture
def force_route(monkeypatch):
    """Pin the classifier so a test exercises one route on both paths."""

    def _force(complexity):
        monkeypatch.setattr(orch, "parse_citation", lambda q: None)
        monkeypatch.setattr(
            AdaptiveRAGPipeline,
            "_classify_and_scope",
            lambda self, q: (complexity, True, False),
        )

    return _force


@pytest.fixture
def spy_top_k(monkeypatch):
    """Record every (min_k, max_k) _dynamic_top_k is called with."""
    calls = []
    # Accessing a @staticmethod through the class yields the plain function
    # (Python 3.10+), so there is no .__func__ to unwrap here.
    original = AdaptiveRAGPipeline._dynamic_top_k

    def spy(results, min_k=3, max_k=20):
        calls.append({"min_k": min_k, "max_k": max_k})
        return original(results, min_k=min_k, max_k=max_k)

    monkeypatch.setattr(AdaptiveRAGPipeline, "_dynamic_top_k", staticmethod(spy))
    return calls


@pytest.fixture
def capture_post_process(monkeypatch):
    """Record the final result set each path hands to _post_process.

    That set is what becomes context + citations, so equal point_ids means the
    two paths retrieved the same evidence in the same order.
    """
    seen = []

    def fake(self, question, answer, results, context, citations=None,
             rerank_scores=None, generation_model=None):
        seen.append([r.point_id for r in results])
        return _Conf(), _Ver(), answer

    monkeypatch.setattr(AdaptiveRAGPipeline, "_post_process", fake)
    return seen


@pytest.mark.parametrize("route", ["simple", "medium", "complex"])
def test_both_paths_use_same_dynamic_top_k_bounds(
    route, force_route, spy_top_k, capture_post_process,
):
    """LOCK (green before and after the refactor).

    A divergence here is exactly how RERANK_MAX_K ended up honoured on one path
    and hardcoded on the other.

    `capture_post_process` is required even though this test ignores what it
    records: it replaces the real _post_process, which would otherwise build a
    HallucinationVerifier and reach for the Tier-2 NLI model over the network.
    """
    force_route(route)

    _make_adaptive().query("cau hoi")
    non_stream = list(spy_top_k)
    spy_top_k.clear()

    list(_make_adaptive().query_stream("cau hoi"))
    stream = list(spy_top_k)

    assert non_stream == stream


@pytest.mark.parametrize("route", ["simple", "medium", "complex"])
def test_both_paths_retrieve_same_points(route, force_route, capture_post_process):
    """LOCK (green before and after the refactor).

    The single most important guard in this suite: it turns "the two paths must
    agree" from a verbal convention into a machine-checked assertion.
    """
    force_route(route)

    _make_adaptive().query("cau hoi")
    list(_make_adaptive().query_stream("cau hoi"))

    assert len(capture_post_process) == 2
    assert capture_post_process[0] == capture_post_process[1]
