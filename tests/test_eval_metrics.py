"""
Unit tests for src/eval/metrics.py — pure functions, no network, no models.

Run: .venv/bin/python -m pytest tests/test_eval_metrics.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.eval.metrics import (
    artnum,
    lenient_text_match,
    strict_match,
    lenient_match,
    recall_at_k,
    mrr,
    refusal_accuracy,
    false_refusal_rate,
    citation_accuracy,
)


# ── artnum ────────────────────────────────────────────────────────────────

def test_artnum_basic():
    assert artnum("Điều 173") == "173"


def test_artnum_with_trailing_text():
    assert artnum("Điều 173. Tội trộm cắp tài sản") == "173"


def test_artnum_with_letter_suffix():
    assert artnum("Điều 12a") == "12a"


def test_artnum_none_or_empty():
    assert artnum(None) is None
    assert artnum("") is None
    assert artnum("không có số") is None


# ── strict vs lenient match (chunker metadata-misalignment case) ─────────

def test_strict_match_true_when_doc_and_article_align():
    result = {"document_id": 296661, "article": "Điều 173", "chunk_text": "Điều 173. Tội trộm cắp tài sản..."}
    expected = {"doc_id": 296661, "article": "Điều 173"}
    assert strict_match(result, expected) is True
    assert lenient_match(result, expected) is True


def test_strict_match_false_wrong_document():
    result = {"document_id": 296884, "article": "Điều 173", "chunk_text": "Điều 173. Thời hạn tạm giam..."}
    expected = {"doc_id": 296661, "article": "Điều 173"}
    assert strict_match(result, expected) is False
    assert lenient_match(result, expected) is False


def test_lenient_rescues_misaligned_article_metadata():
    """Known chunker bug: `article` payload field is stale/wrong, but the
    chunk's own text still starts with the correct 'Điều <n>.' — lenient
    matching should catch what strict matching misses."""
    result = {
        "document_id": 296661,
        "article": "Điều 172",  # wrong/stale metadata
        "chunk_text": "Điều 173. Tội trộm cắp tài sản\n1. Người nào trộm cắp...",
    }
    expected = {"doc_id": 296661, "article": "Điều 173"}
    assert strict_match(result, expected) is False
    assert lenient_match(result, expected) is True


def test_lenient_text_match_requires_delimiter():
    # "Điều 17" must not fuzzy-match inside "Điều 173" (no `.`/`:` after "17")
    assert lenient_text_match("17", "Điều 173. Tội trộm cắp tài sản") is False
    assert lenient_text_match("173", "Điều 173. Tội trộm cắp tài sản") is True
    assert lenient_text_match("173", "Điều 173: Tội trộm cắp tài sản") is True


def test_lenient_text_match_empty_inputs():
    assert lenient_text_match(None, "Điều 173.") is False
    assert lenient_text_match("173", None) is False
    assert lenient_text_match("173", "") is False


# ── recall@k ──────────────────────────────────────────────────────────────

def _hit(doc_id, article, text=""):
    return {"document_id": doc_id, "article": article, "chunk_text": text}


def test_recall_at_k_full_hit_within_k():
    results = [_hit(1, "Điều 1"), _hit(296661, "Điều 173"), _hit(2, "Điều 2")]
    expected = [{"doc_id": 296661, "article": "Điều 173"}]
    assert recall_at_k(results, expected, k=5) == 1.0


def test_recall_at_k_respects_k_cutoff():
    # Match is at rank 6 — recall@5 must miss it, recall@10 must find it.
    results = [_hit(0, "Điều 0")] * 5 + [_hit(296661, "Điều 173")]
    expected = [{"doc_id": 296661, "article": "Điều 173"}]
    assert recall_at_k(results, expected, k=5) == 0.0
    assert recall_at_k(results, expected, k=10) == 1.0


def test_recall_at_k_partial_multi_article():
    results = [_hit(296661, "Điều 12")]
    expected = [
        {"doc_id": 296661, "article": "Điều 12"},
        {"doc_id": 296661, "article": "Điều 173"},
        {"doc_id": 296661, "article": "Điều 9"},
    ]
    assert recall_at_k(results, expected, k=5) == 1 / 3


def test_recall_at_k_no_expected_articles_is_vacuous():
    assert recall_at_k([_hit(1, "Điều 1")], [], k=5) == 1.0


def test_recall_at_k_lenient_vs_strict():
    results = [_hit(296661, "Điều 172", "Điều 173. Nội dung thật...")]
    expected = [{"doc_id": 296661, "article": "Điều 173"}]
    assert recall_at_k(results, expected, k=5, lenient=False) == 0.0
    assert recall_at_k(results, expected, k=5, lenient=True) == 1.0


# ── MRR ───────────────────────────────────────────────────────────────────

def test_mrr_first_rank_match():
    results = [_hit(296661, "Điều 173")]
    expected = [{"doc_id": 296661, "article": "Điều 173"}]
    assert mrr(results, expected) == 1.0


def test_mrr_third_rank_match():
    results = [_hit(0, "Điều 0"), _hit(0, "Điều 1"), _hit(296661, "Điều 173")]
    expected = [{"doc_id": 296661, "article": "Điều 173"}]
    assert mrr(results, expected) == pytest.approx(1 / 3)


def test_mrr_no_match_is_zero():
    results = [_hit(0, "Điều 0"), _hit(0, "Điều 1")]
    expected = [{"doc_id": 296661, "article": "Điều 173"}]
    assert mrr(results, expected) == 0.0


def test_mrr_best_rank_of_any_expected_article():
    # Điều 9 matches at rank 2 even though Điều 12 (also expected) never appears.
    results = [_hit(0, "Điều 0"), _hit(296661, "Điều 9")]
    expected = [
        {"doc_id": 296661, "article": "Điều 12"},
        {"doc_id": 296661, "article": "Điều 9"},
    ]
    assert mrr(results, expected) == 0.5


def test_mrr_no_expected_articles_is_vacuous():
    assert mrr([_hit(1, "Điều 1")], []) == 1.0


# ── refusal / false-refusal accuracy ─────────────────────────────────────

def test_refusal_accuracy_all_correct():
    rows = [
        {"expected_behavior": "refuse", "refused": True},
        {"expected_behavior": "refuse", "refused": True},
        {"expected_behavior": "answer", "refused": False},  # ignored
    ]
    assert refusal_accuracy(rows) == 1.0


def test_refusal_accuracy_partial():
    rows = [
        {"expected_behavior": "refuse", "refused": True},
        {"expected_behavior": "refuse", "refused": False},
    ]
    assert refusal_accuracy(rows) == 0.5


def test_refusal_accuracy_no_refuse_rows_is_none():
    rows = [{"expected_behavior": "answer", "refused": False}]
    assert refusal_accuracy(rows) is None


def test_false_refusal_rate():
    rows = [
        {"expected_behavior": "answer", "refused": True},   # false refusal
        {"expected_behavior": "answer", "refused": False},
        {"expected_behavior": "refuse", "refused": True},   # ignored
    ]
    assert false_refusal_rate(rows) == 0.5


def test_false_refusal_rate_no_answer_rows_is_none():
    rows = [{"expected_behavior": "refuse", "refused": True}]
    assert false_refusal_rate(rows) is None


# ── citation accuracy ─────────────────────────────────────────────────────

def test_citation_accuracy_full_match():
    answer = "Theo Điều 173 và Điều 12 Bộ luật Hình sự..."
    expected = [{"doc_id": 1, "article": "Điều 173"}, {"doc_id": 1, "article": "Điều 12"}]
    assert citation_accuracy(answer, expected) == 1.0


def test_citation_accuracy_partial_match():
    answer = "Theo Điều 173 Bộ luật Hình sự..."
    expected = [{"doc_id": 1, "article": "Điều 173"}, {"doc_id": 1, "article": "Điều 999"}]
    assert citation_accuracy(answer, expected) == 0.5


def test_citation_accuracy_no_expected_articles_is_none():
    assert citation_accuracy("Theo Điều 173...", []) is None
