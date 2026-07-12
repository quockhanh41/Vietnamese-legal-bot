"""Tests for the Tier-2 --retry merge logic in scripts/eval.py.

The retry mode exists because Groq free-tier TPD budgets die mid-run
(M1.7, 2026-07-12: 85/133 questions errored with RateLimitError). Instead
of re-running the whole set (re-burning the tokens of every question that
already succeeded), --retry re-runs ONLY the errored/missing questions and
merges the fresh rows into the previous report's rows.

Pure logic only — no LLM, no network.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval import (
    aggregate_tier2,
    failed_question_ids,
    merge_tier2_rows,
    row_citation_accuracy,
)


def _q(qid):
    return {"id": qid, "type": "single-hop", "question": f"q {qid}?"}


def _row(qid, error=None, refused=False, conf=0.8, behavior="answer"):
    return {
        "id": qid,
        "type": "single-hop",
        "expected_behavior": behavior,
        "error": error,
        "refused": None if error else refused,
        "confidence_score": None if error else conf,
        "confidence_label": None if error else "high",
        "pipeline_used": None if error else "simple",
        "latency_ms": 100.0,
        "citation_accuracy": None,
    }


QUESTIONS = [_q("a-1"), _q("a-2"), _q("a-3"), _q("a-4")]


class TestFailedQuestionIds:
    def test_errored_rows_are_selected(self):
        prev = [_row("a-1"), _row("a-2", error="RateLimitError"), _row("a-3"), _row("a-4")]
        assert failed_question_ids(QUESTIONS, prev) == ["a-2"]

    def test_questions_missing_a_row_are_selected(self):
        # e.g. the previous run crashed before reaching a-4
        prev = [_row("a-1"), _row("a-2", error="boom"), _row("a-3")]
        assert failed_question_ids(QUESTIONS, prev) == ["a-2", "a-4"]

    def test_all_ok_returns_empty(self):
        prev = [_row(q["id"]) for q in QUESTIONS]
        assert failed_question_ids(QUESTIONS, prev) == []


class TestMergeTier2Rows:
    def test_new_rows_replace_errored_prev_rows(self):
        prev = [_row("a-1", conf=0.9), _row("a-2", error="RateLimitError"),
                _row("a-3", conf=0.7), _row("a-4", error="RateLimitError")]
        new = [_row("a-2", conf=0.85), _row("a-4", error="RateLimitError again")]
        merged = merge_tier2_rows(QUESTIONS, prev, new)

        assert [r["id"] for r in merged] == ["a-1", "a-2", "a-3", "a-4"]
        assert merged[1]["confidence_score"] == 0.85 and merged[1]["error"] is None
        # a question that errors again keeps the NEW error row
        assert merged[3]["error"] == "RateLimitError again"
        # untouched ok rows carried over verbatim
        assert merged[0]["confidence_score"] == 0.9
        assert merged[2]["confidence_score"] == 0.7

    def test_merged_rows_follow_question_order(self):
        prev = [_row("a-3"), _row("a-1", error="x"), _row("a-2"), _row("a-4")]
        new = [_row("a-1")]
        merged = merge_tier2_rows(QUESTIONS, prev, new)
        assert [r["id"] for r in merged] == ["a-1", "a-2", "a-3", "a-4"]


class TestRowCitationAccuracy:
    """Citation accuracy is only measurable when the bot actually answered.

    A refusal has no citations to grade and is already priced into
    false_refusal_rate — scoring it 0 double-counts the failure (M1 gate
    discovery: 10 of 16 zero-citation sectors rows were refusals, dragging
    0.83 down to 0.67)."""

    EXPECTED = [{"doc_id": 1, "article": "Điều 117"}]

    def test_refused_row_is_none_not_zero(self):
        assert row_citation_accuracy(
            "answer", True, "⚠️ Tôi không tìm thấy đủ thông tin...", self.EXPECTED) is None

    def test_answered_row_is_scored(self):
        assert row_citation_accuracy(
            "answer", False, "Theo Điều 117 Bộ luật Dân sự...", self.EXPECTED) == 1.0

    def test_answered_row_missing_citation_scores_zero(self):
        assert row_citation_accuracy(
            "answer", False, "Hợp đồng cần điều kiện sau...", self.EXPECTED) == 0.0

    def test_refuse_expected_row_is_none(self):
        assert row_citation_accuracy(
            "refuse", False, "Theo Điều 117...", self.EXPECTED) is None


class TestAggregateTier2:
    def test_recomputes_counts_and_errors(self):
        rows = [
            _row("a-1", conf=0.9, behavior="answer"),
            _row("a-2", conf=0.5, refused=True, behavior="refuse"),
            _row("a-3", error="still dead"),
        ]
        agg = aggregate_tier2(rows)
        assert agg["n"] == 3
        assert agg["n_errors"] == 1
        dist = agg["confidence_distribution"]
        assert dist["answer"]["n"] == 1 and dist["answer"]["mean"] == 0.9
        assert dist["refuse"]["n"] == 1 and dist["refuse"]["mean"] == 0.5
