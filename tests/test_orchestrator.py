"""
Unit tests for the Plan §3.4 (parent-document / small-to-big retrieval) and
§3.6 (Phase 2 temporal/amendment wiring) additions to
src/rag/orchestrator.py. Pure-Python: no Qdrant, no LLM, no models.

Run: .venv/bin/python -m pytest tests/test_orchestrator.py -q
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.qdrant_client import SearchResult
from src.rag.orchestrator import NaiveRAGPipeline, AdaptiveRAGPipeline
from src.rag.verifier import VerificationResult


def _sr(point_id, document_id=1, article=None, chunk_text="chunk", parent_chunk_text=None, score=0.9):
    return SearchResult(
        point_id=point_id,
        score=score,
        chunk_text=chunk_text,
        document_title="Bộ luật hình sự 2015",
        document_number="100/2015/QH13",
        legal_type="Bộ luật",
        article=article,
        document_id=document_id,
        parent_chunk_text=parent_chunk_text,
    )


def _make_pipeline() -> NaiveRAGPipeline:
    return NaiveRAGPipeline(
        vector_store=MagicMock(), embedder=MagicMock(), llm_gateway=MagicMock(),
    )


# ── _expand_to_parent (Plan §3.4) ───────────────────────────────────────────

def test_expand_to_parent_swaps_chunk_text():
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173", chunk_text="khoản 1 text", parent_chunk_text="Điều 173 full text")]
    expanded = pipeline._expand_to_parent(results)
    assert expanded[0].chunk_text == "Điều 173 full text"


def test_expand_to_parent_falls_back_when_no_parent_text():
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173", chunk_text="khoản 1 text", parent_chunk_text=None)]
    expanded = pipeline._expand_to_parent(results)
    assert expanded[0].chunk_text == "khoản 1 text"


def test_expand_to_parent_dedupes_same_article():
    """Two Khoản-level hits on the same Điều collapse into one block."""
    pipeline = _make_pipeline()
    results = [
        _sr("a", article="Điều 173", chunk_text="khoản 1", parent_chunk_text="Điều 173 full text", score=0.9),
        _sr("b", article="Điều 173", chunk_text="khoản 2", parent_chunk_text="Điều 173 full text", score=0.8),
    ]
    expanded = pipeline._expand_to_parent(results)
    assert len(expanded) == 1
    assert expanded[0].point_id == "a"  # highest-scored (first) occurrence wins


def test_expand_to_parent_keeps_different_articles():
    pipeline = _make_pipeline()
    results = [
        _sr("a", article="Điều 173", parent_chunk_text="text 173"),
        _sr("b", article="Điều 174", parent_chunk_text="text 174"),
    ]
    expanded = pipeline._expand_to_parent(results)
    assert len(expanded) == 2


def test_expand_to_parent_no_article_never_dedupes():
    """Results without an `article` field (e.g. whole-document chunks) must
    never be collapsed against each other just because they share a None key."""
    pipeline = _make_pipeline()
    results = [
        _sr("a", article=None, chunk_text="chunk a"),
        _sr("b", article=None, chunk_text="chunk b"),
    ]
    expanded = pipeline._expand_to_parent(results)
    assert len(expanded) == 2


def test_expand_to_parent_guards_against_oversized_parent_text():
    """Regression guard for the ArticleLevelChunker bug where parent_chunk_text
    was accidentally set to an entire document (~267K chars) instead of one
    article — must fall back to chunk_text rather than blow up context size."""
    pipeline = _make_pipeline()
    huge = "x" * (NaiveRAGPipeline._PARENT_TEXT_MAX_LEN + 1)
    results = [_sr("a", article="Điều 1", chunk_text="small chunk", parent_chunk_text=huge)]
    expanded = pipeline._expand_to_parent(results)
    assert expanded[0].chunk_text == "small chunk"


# ── _filter_in_effect (Plan §3.6) ───────────────────────────────────────────

def test_filter_in_effect_excludes_explicit_false():
    pipeline = _make_pipeline()
    results = [_sr("a"), _sr("b")]
    annotations = {"a": {"is_in_effect": False}, "b": {"is_in_effect": True}}
    filtered = pipeline._filter_in_effect(results, annotations)
    assert [r.point_id for r in filtered] == ["b"]


def test_filter_in_effect_fails_open_on_unknown():
    """A chunk with no Phase 2 metadata (is_in_effect is None, or missing
    entirely) must be KEPT — unknown status is not the same as superseded."""
    pipeline = _make_pipeline()
    results = [_sr("a"), _sr("b"), _sr("c")]
    annotations = {"a": {"is_in_effect": None}}  # "b", "c" absent entirely
    filtered = pipeline._filter_in_effect(results, annotations)
    assert [r.point_id for r in filtered] == ["a", "b", "c"]


# ── _build_context / _build_citations amendment annotations (Plan §3.6) ────

def test_build_context_appends_amendment_note():
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173", chunk_text="nội dung điều luật")]
    annotations = {"a": {"is_amended": True, "amending_laws": ["Luật sửa đổi BLHS 2017", "Luật sửa đổi BLHS 2025"]}}
    context = pipeline._build_context(results, annotations)
    assert "⚠️ Đã được sửa đổi bởi: Luật sửa đổi BLHS 2017, Luật sửa đổi BLHS 2025" in context
    assert "nội dung điều luật" in context


def test_build_context_no_note_when_not_amended():
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173", chunk_text="nội dung điều luật")]
    annotations = {"a": {"is_amended": False, "amending_laws": []}}
    context = pipeline._build_context(results, annotations)
    assert "⚠️" not in context


def test_build_context_without_annotations_arg_still_works():
    """Backward compat: annotations is optional (defaults to None)."""
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173", chunk_text="nội dung điều luật")]
    context = pipeline._build_context(results)
    assert "nội dung điều luật" in context
    assert "⚠️" not in context


def test_build_citations_includes_amendment_fields():
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173")]
    annotations = {"a": {"is_amended": True, "amending_laws": ["Luật sửa đổi BLHS 2017"]}}
    citations = pipeline._build_citations(results, annotations)
    assert citations[0]["is_amended"] is True
    assert citations[0]["amending_laws"] == ["Luật sửa đổi BLHS 2017"]


def test_build_citations_defaults_without_annotations():
    pipeline = _make_pipeline()
    results = [_sr("a", article="Điều 173")]
    citations = pipeline._build_citations(results)
    assert citations[0]["is_amended"] is False
    assert citations[0]["amending_laws"] == []


# ── _post_process: grounding_ratio wiring into ConfidenceScorer (M0.2) ─────
#
# `_post_process` runs `HallucinationVerifier.verify()` BEFORE
# `ConfidenceScorer.score()`, so it can hand the verifier's grounding_ratio
# straight to the scorer — same convention already used for entailment_ratio.
# These tests mock BOTH collaborators so they can assert on the exact kwargs
# `_post_process` passes through, without loading the real NLI/reranker models.

def _make_adaptive_pipeline() -> AdaptiveRAGPipeline:
    return AdaptiveRAGPipeline(
        vector_store=MagicMock(), embedder=MagicMock(), llm_gateway=MagicMock(),
    )


def test_post_process_passes_grounding_ratio_when_claims_present():
    """When the verifier found groundable claims, its grounding_ratio must
    flow into ConfidenceScorer.score(grounding_ratio=...) verbatim."""
    pipeline = _make_adaptive_pipeline()

    mock_verifier = MagicMock()
    mock_verifier.verify.return_value = VerificationResult(
        grounded_claims=[True, False],
        grounding_ratio=0.5,
        entailment_labels=["ENTAILMENT", "NEUTRAL"],
        entailment_ratio=0.5,
    )
    pipeline._verifier = mock_verifier

    mock_scorer = MagicMock()
    mock_scorer.score.return_value = MagicMock(refused=False)
    pipeline._confidence_scorer = mock_scorer

    pipeline._post_process("Q?", "answer text", [], "context text")

    assert mock_scorer.score.call_args.kwargs["grounding_ratio"] == 0.5


def test_post_process_passes_none_when_no_groundable_claims():
    """When the answer had no groundable ("Chi tiết" bullet) claims at all —
    `grounded_claims` stays the empty-list default — the signal must be
    reported as unavailable (None), not as a spurious 0.0 (which would be
    indistinguishable from '0% grounded' and unfairly penalize the score)."""
    pipeline = _make_adaptive_pipeline()

    mock_verifier = MagicMock()
    mock_verifier.verify.return_value = VerificationResult(
        grounded_claims=[],
        grounding_ratio=0.0,  # verifier's early-return default — must NOT leak through
        entailment_labels=[],
        entailment_ratio=0.0,
    )
    pipeline._verifier = mock_verifier

    mock_scorer = MagicMock()
    mock_scorer.score.return_value = MagicMock(refused=False)
    pipeline._confidence_scorer = mock_scorer

    pipeline._post_process("Q?", "answer text", [], "context text")

    assert mock_scorer.score.call_args.kwargs["grounding_ratio"] is None
    # Same convention must already hold for entailment (pre-existing behavior).
    assert mock_scorer.score.call_args.kwargs["entailment_ratio"] is None


# ── retrieve_routed hybrid branch: server-side is_in_effect filter (M2.5) ──

def test_retrieve_routed_hybrid_excludes_superseded():
    """Non-citation query → hybrid search must request exclude_superseded=True."""
    import numpy as np
    from src.rag.citation_lookup import retrieve_routed

    class _Emb:
        def encode(self, qs, return_dense=True, return_sparse=True):
            # dense_vec.tolist() is called downstream → must be a numpy array
            return {"dense_vecs": [np.array([0.1])], "lexical_weights": [{1: 0.5}]}
        def encode_queries(self, qs): return [np.array([0.1])]

    class _Store:
        def __init__(self): self.kw = None
        def search_hybrid(self, dense, sparse, top_k=20, **kw):
            self.kw = kw
            return []

    store = _Store()
    retrieve_routed("Thế nào là tội cướp tài sản?", store, _Emb(), citation_lookup=None, fill_top_k=10)
    assert store.kw.get("exclude_superseded") is True


# ── NaiveRAGPipeline.query: server-side is_in_effect filter (Task 6 fix) ───
#
# Task 6 wired exclude_superseded=True into AdaptiveRAGPipeline's answer
# paths + retrieve_routed's hybrid branch, but left the base-class
# NaiveRAGPipeline.query()/query_stream() untouched. NaiveRAGPipeline is a
# LIVE answer path (src/api/main.py's get_naive_pipeline()), so it should
# request the same server-side filter for consistency. The client-side
# _filter_in_effect backstop still runs regardless (fail-open on unknown).

def test_naive_query_excludes_superseded():
    """NaiveRAGPipeline.query() must request exclude_superseded=True from
    search_dense."""
    import numpy as np

    class _Store:
        def __init__(self):
            self.kw = None

        def search_dense(self, vector, top_k=5, **kw):
            self.kw = kw
            return []

    class _Emb:
        def encode_queries(self, qs):
            return [np.array([0.1])]

    class _LLM:
        def generate(self, **kwargs):
            return "answer"

    store = _Store()
    pipeline = NaiveRAGPipeline(vector_store=store, embedder=_Emb(), llm_gateway=_LLM())
    pipeline.query("Thế nào là tội cướp tài sản?")
    assert store.kw.get("exclude_superseded") is True
