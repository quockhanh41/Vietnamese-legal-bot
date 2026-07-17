"""
Unit tests for the SCOPE gate merged into the complexity-classifier LLM call
(src/rag/orchestrator.py AdaptiveRAGPipeline._classify_and_scope,
src/llm/prompts.py COMPLEXITY_CLASSIFIER_PROMPT, src/rag/citation_lookup.py
CitationLookup.corpus_summary). Pure-Python: LLM gateway and vector store are
mocked — no network, no Qdrant, no models.

Context: retrieval on out-of-scope questions (e.g. "đất đai" procedural
questions against a criminal-law corpus) returns chunks that LOOK relevant,
fooling every confidence signal (see docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md,
"Calibrate ngưỡng confidence": gen-203 scored 0.8455, above every reasonable
threshold). The SCOPE gate runs BEFORE retrieval so it can't be fooled by
retrieval quality — these tests check it fails open on parse errors and
never runs retrieval when it refuses.

Run: .venv/bin/python -m pytest tests/test_scope_gate.py -q
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.citation_lookup import CitationLookup
from src.rag.orchestrator import AdaptiveRAGPipeline


FAKE_ALIASES = {
    46056: "Bộ Luật Hình sự 1999",
    296661: "Bộ luật hình sự 2015",
    296884: "Bộ luật tố tụng hình sự 2015",
}


def _lookup_with_fake_aliases():
    """Build a CitationLookup with a pre-populated alias index, bypassing Qdrant."""
    lookup = CitationLookup(vector_store=None)
    lookup._alias_index = dict(FAKE_ALIASES)
    return lookup


def _make_pipeline(llm_response: str) -> AdaptiveRAGPipeline:
    """
    Build an AdaptiveRAGPipeline with:
    - a mocked LLM gateway whose .generate() always returns `llm_response`
    - a mocked vector store whose search methods raise if called, so tests
      can assert retrieval never ran for the out-of-scope path
    - a pre-populated CitationLookup (no Qdrant scroll)
    """
    llm = MagicMock()
    llm.generate.return_value = llm_response

    vector_store = MagicMock()
    vector_store.search_dense.side_effect = AssertionError("search_dense must not be called")
    vector_store.search_hybrid.side_effect = AssertionError("search_hybrid must not be called")

    embedder = MagicMock()

    pipeline = AdaptiveRAGPipeline(vector_store=vector_store, embedder=embedder, llm_gateway=llm)
    pipeline._citation_lookup = _lookup_with_fake_aliases()
    return pipeline


# ── _classify_and_scope: parsing variants ───────────────────────────────────

def test_scope_out_class_simple():
    pipeline = _make_pipeline("SCOPE: OUT\nCLASS: SIMPLE")
    complexity, in_scope, _is_cmp = pipeline._classify_and_scope("Thủ tục cấp sổ đỏ mất bao lâu?")
    assert in_scope is False
    assert complexity == "simple"


def test_scope_parse_lowercase():
    pipeline = _make_pipeline("scope: out\nclass: medium")
    complexity, in_scope, _is_cmp = pipeline._classify_and_scope("Hồ sơ xin giấy phép xây dựng gồm những gì?")
    assert in_scope is False
    assert complexity == "medium"


def test_scope_parse_reversed_order():
    pipeline = _make_pipeline("CLASS: COMPLEX\nSCOPE: IN")
    complexity, in_scope, _is_cmp = pipeline._classify_and_scope(
        "Trộm cắp tài sản trị giá lớn kèm nhiều tình tiết thì xử lý thế nào?"
    )
    assert in_scope is True
    assert complexity == "complex"


def test_scope_garbage_defaults_in_scope_and_medium():
    """Unparseable response must NEVER block the pipeline — fail open."""
    pipeline = _make_pipeline("xin lỗi tôi không hiểu câu hỏi")
    complexity, in_scope, _is_cmp = pipeline._classify_and_scope("một câu hỏi bất kỳ")
    assert in_scope is True
    assert complexity == "medium"


def test_scope_in_class_simple():
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: SIMPLE")
    complexity, in_scope, _is_cmp = pipeline._classify_and_scope("Tội trộm cắp tài sản bị xử phạt thế nào?")
    assert in_scope is True
    assert complexity == "simple"


# ── query() routing ──────────────────────────────────────────────────────

def test_out_of_scope_refuses_without_retrieval():
    pipeline = _make_pipeline("SCOPE: OUT\nCLASS: SIMPLE")
    resp = pipeline.query("Thủ tục làm sổ đỏ mất bao lâu?")
    assert resp.refused is True
    assert resp.pipeline_used == "out-of-scope"
    assert resp.confidence == "low"
    assert resp.confidence_score == 0.0
    assert resp.citations == []
    pipeline.vector_store.search_dense.assert_not_called()
    pipeline.vector_store.search_hybrid.assert_not_called()


def test_out_of_scope_answer_mentions_corpus_titles():
    pipeline = _make_pipeline("SCOPE: OUT\nCLASS: SIMPLE")
    resp = pipeline.query("Thủ tục làm sổ đỏ mất bao lâu?")
    for title in FAKE_ALIASES.values():
        assert title in resp.answer


def test_in_scope_routes_to_simple_pipeline():
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: SIMPLE")
    sentinel = object()
    pipeline._simple_pipeline = MagicMock(return_value=sentinel)
    pipeline._medium_pipeline = MagicMock(side_effect=AssertionError("must not route to medium"))
    pipeline._complex_pipeline = MagicMock(side_effect=AssertionError("must not route to complex"))

    result = pipeline.query("Tội trộm cắp tài sản bị xử phạt thế nào?")

    assert result is sentinel
    pipeline._simple_pipeline.assert_called_once()


def test_in_scope_routes_to_medium_pipeline():
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: MEDIUM")
    sentinel = object()
    pipeline._medium_pipeline = MagicMock(return_value=sentinel)
    pipeline._simple_pipeline = MagicMock(side_effect=AssertionError("must not route to simple"))
    pipeline._complex_pipeline = MagicMock(side_effect=AssertionError("must not route to complex"))

    result = pipeline.query("So sánh tội trộm cắp và tội cướp tài sản khác nhau như thế nào?")

    assert result is sentinel
    pipeline._medium_pipeline.assert_called_once()


def test_in_scope_routes_to_complex_pipeline():
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: COMPLEX")
    sentinel = object()
    pipeline._complex_pipeline = MagicMock(return_value=sentinel)
    pipeline._simple_pipeline = MagicMock(side_effect=AssertionError("must not route to simple"))
    pipeline._medium_pipeline = MagicMock(side_effect=AssertionError("must not route to medium"))

    result = pipeline.query(
        "Nếu phạm nhiều tội trong cùng một vụ án thì hình phạt chung được tính thế nào?"
    )

    assert result is sentinel
    pipeline._complex_pipeline.assert_called_once()


# ── corpus_summary ─────────────────────────────────────────────────────────

def test_corpus_summary_lists_titles_from_alias_index():
    lookup = _lookup_with_fake_aliases()
    summary = lookup.corpus_summary()
    for title in FAKE_ALIASES.values():
        assert title in summary
    assert summary.startswith("- ")


def test_corpus_summary_is_cached():
    lookup = _lookup_with_fake_aliases()
    first = lookup.corpus_summary()
    lookup._alias_index[999999] = "Should not appear — cache already built"
    second = lookup.corpus_summary()
    assert first == second
    assert "Should not appear" not in second


# ── COMPARE parsing (M3.4) ──────────────────────────────────────────────────

def test_compare_yes_parsed():
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: MEDIUM\nCOMPARE: YES")
    complexity, in_scope, is_comparison = pipeline._classify_and_scope(
        "Luật Đất đai 2013 và Luật Đất đai 2024 khác nhau thế nào về hạn mức?")
    assert (complexity, in_scope, is_comparison) == ("medium", True, True)


def test_compare_no_parsed():
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: SIMPLE\nCOMPARE: NO")
    _c, _s, is_comparison = pipeline._classify_and_scope("Tội trộm cắp bị phạt thế nào?")
    assert is_comparison is False


def test_compare_missing_line_defaults_false():
    """Old 2-line responses (no COMPARE line) must default to NO — fail-safe:
    never broaden the pool on a classifier hiccup."""
    pipeline = _make_pipeline("SCOPE: IN\nCLASS: SIMPLE")
    _c, _s, is_comparison = pipeline._classify_and_scope("Đồng phạm là gì?")
    assert is_comparison is False
