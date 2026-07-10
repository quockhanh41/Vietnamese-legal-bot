"""
Unit tests for src/rag/confidence.py — rubric-based LLM self-eval (M0.1 fix,
roadmap docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md "2 follow-up còn
lại"). Pure-Python: the LLM gateway is mocked — no network, no model load.

Covers:
  - `ConfidenceScorer._parse_rubric_score`: well-formed rubric text, garbled
    text (fail-safe → None), partial rubric (missing a criterion → None),
    score-aggregation boundaries (all-0 → 0.0, all-2 → 1.0).
  - `ConfidenceScorer._llm_self_eval`: defaults to 0.5 when the LLM response
    is unparseable (same semantics as the old free-form-number parser); uses
    block-based context selection (not blind `context[:2000]`) when calling
    the LLM.

Run: .venv/bin/python -m pytest tests/test_confidence.py -q
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.confidence import ConfidenceScorer, DEFAULT_WEIGHTS, REFUSAL_THRESHOLD


# ── _parse_rubric_score ──────────────────────────────────────────────

def test_parse_rubric_score_well_formed_all_high():
    raw = "CĂN CỨ: 2\nTRÍCH DẪN: 2\nSUY DIỄN: 2\nĐẦY ĐỦ: 2"
    assert ConfidenceScorer._parse_rubric_score(raw) == 1.0


def test_parse_rubric_score_well_formed_all_low():
    raw = "CĂN CỨ: 0\nTRÍCH DẪN: 0\nSUY DIỄN: 0\nĐẦY ĐỦ: 0"
    assert ConfidenceScorer._parse_rubric_score(raw) == 0.0


def test_parse_rubric_score_mixed_computes_correct_ratio():
    raw = "CĂN CỨ: 2\nTRÍCH DẪN: 1\nSUY DIỄN: 0\nĐẦY ĐỦ: 1"
    # (2+1+0+1) / 8 = 0.5
    assert ConfidenceScorer._parse_rubric_score(raw) == 0.5


def test_parse_rubric_score_garbage_returns_none():
    assert ConfidenceScorer._parse_rubric_score("Tôi không chắc, có lẽ 50%?") is None
    assert ConfidenceScorer._parse_rubric_score("") is None
    assert ConfidenceScorer._parse_rubric_score(None) is None


def test_parse_rubric_score_partial_rubric_returns_none():
    """Missing even one criterion must fail the whole parse (not silently
    average over 3) — avoids an unpredictable bias."""
    raw = "CĂN CỨ: 2\nTRÍCH DẪN: 2\nSUY DIỄN: 2"  # ĐẦY ĐỦ missing
    assert ConfidenceScorer._parse_rubric_score(raw) is None


def test_parse_rubric_score_out_of_range_value_returns_none():
    raw = "CĂN CỨ: 5\nTRÍCH DẪN: 2\nSUY DIỄN: 2\nĐẦY ĐỦ: 2"  # 5 not in {0,1,2}
    assert ConfidenceScorer._parse_rubric_score(raw) is None


def test_parse_rubric_score_tolerates_extra_surrounding_text():
    raw = (
        "Đây là đánh giá của tôi:\n"
        "CĂN CỨ: 1\n"
        "TRÍCH DẪN: 2\n"
        "SUY DIỄN: 1\n"
        "ĐẦY ĐỦ: 2\n"
        "Hy vọng hữu ích."
    )
    assert ConfidenceScorer._parse_rubric_score(raw) == 0.75


# ── _llm_self_eval ────────────────────────────────────────────────────

CONTEXT_2_BLOCKS = (
    "[1] Bộ luật Hình sự 2015 — Điều 17 (Chương IV)\n"
    "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm."
    "\n\n---\n\n"
    "[2] Bộ luật Hình sự 2015 — Điều 173 (Chương XVI)\n"
    "Tội trộm cắp tài sản bị phạt cải tạo không giam giữ đến 03 năm."
)


def _scorer_with_llm_response(response: str) -> ConfidenceScorer:
    llm = MagicMock()
    llm.generate.return_value = response
    return ConfidenceScorer(llm)


def test_llm_self_eval_well_formed_response():
    scorer = _scorer_with_llm_response(
        "CĂN CỨ: 2\nTRÍCH DẪN: 2\nSUY DIỄN: 2\nĐẦY ĐỦ: 1"
    )
    answer = "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm [1]."
    score = scorer._llm_self_eval("Đồng phạm là gì?", answer, CONTEXT_2_BLOCKS)
    assert score == 0.875  # (2+2+2+1)/8


def test_llm_self_eval_unparseable_response_defaults_to_neutral():
    """Fail-safe semantics must be preserved: an unparseable rubric response
    (or any other failure) falls back to the same 0.5 neutral default the
    old free-form-number parser used."""
    scorer = _scorer_with_llm_response("Tôi không thể đánh giá câu trả lời này.")
    score = scorer._llm_self_eval("Đồng phạm là gì?", "Một câu trả lời nào đó.", CONTEXT_2_BLOCKS)
    assert score == 0.5


def test_llm_self_eval_llm_exception_defaults_to_neutral():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("network error")
    scorer = ConfidenceScorer(llm)
    score = scorer._llm_self_eval("Đồng phạm là gì?", "Một câu trả lời nào đó.", CONTEXT_2_BLOCKS)
    assert score == 0.5


def test_llm_self_eval_passes_block_selected_context_not_blind_truncation():
    """The context handed to the LLM prompt must be the block(s) the answer
    actually cites (here, only [1]) — not a blind `context[:2000]` that
    would also include the unrelated trộm-cắp block [2]."""
    scorer = _scorer_with_llm_response("CĂN CỨ: 2\nTRÍCH DẪN: 2\nSUY DIỄN: 2\nĐẦY ĐỦ: 2")
    answer = "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm [1]."
    scorer._llm_self_eval("Đồng phạm là gì?", answer, CONTEXT_2_BLOCKS)

    prompt_used = scorer.llm.generate.call_args.kwargs["prompt"]
    assert "Đồng phạm là trường hợp có hai người" in prompt_used  # block 1 present
    assert "trộm cắp tài sản" not in prompt_used  # block 2 excluded


# ── score(): weighting/threshold constants untouched (guard) ───────────

def test_default_weights_and_threshold_unchanged():
    """Guard against accidental drift of the calibrated constants.

    M0.2 (2026-07-10) intentionally updated DEFAULT_WEIGHTS to add the new
    `grounding` signal (retriever 0.25->0.20, llm_self_eval 0.30->0.25,
    grounding +0.10 new). M0.3 (same day) recalibrated REFUSAL_THRESHOLD
    0.725 -> 0.700 on fresh Tier-2 data (clean separation, gap 0.0139,
    calibration report 20260710T085625Z) — this guard pins both on purpose;
    any further change must come with a new calibration report."""
    assert DEFAULT_WEIGHTS == {
        "retriever": 0.20,
        "reranker": 0.20,
        "llm_self_eval": 0.25,
        "context_overlap": 0.15,
        "entailment": 0.10,
        "grounding": 0.10,
    }
    assert REFUSAL_THRESHOLD == 0.700 or os.getenv("CONFIDENCE_REFUSAL_THRESHOLD")


def test_default_weights_sum_to_one():
    """Renormalization only preserves a 0-1 scale if the full weight set is
    a proper distribution to begin with."""
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


# ── score(): grounding signal (M0.2) ────────────────────────────────────

def _scorer_with_pinned_signals(llm_score: float = 1.0) -> ConfidenceScorer:
    """A ConfidenceScorer whose retriever/reranker/context_overlap/llm_self_eval
    signals are all pinned to a fixed value, so a test can isolate exactly the
    contribution of `grounding_ratio` in `score()`'s weighted aggregation
    without needing to reverse-engineer the real regex/keyword logic of the
    other signals."""
    scorer = ConfidenceScorer(MagicMock())
    scorer._retriever_signal = lambda results: llm_score
    scorer._reranker_signal = lambda scores: llm_score
    scorer._context_overlap = lambda answer, context: llm_score
    scorer._llm_self_eval = lambda question, answer, context: llm_score
    return scorer


def test_score_grounding_present_applies_010_weight():
    """With every other signal pinned at 1.0 and grounding at 0.5, the
    weighted score must drop by exactly weight(grounding) * (1.0 - 0.5)
    = 0.10 * 0.5 = 0.05 off the max — i.e. land at 0.95."""
    scorer = _scorer_with_pinned_signals(1.0)
    result = scorer.score(
        question="Q", answer="A", context="C",
        search_results=[], rerank_scores=[0.5],
        entailment_ratio=1.0, grounding_ratio=0.5,
    )
    assert result.grounding_score == 0.5
    assert result.score == pytest.approx(1.0 - DEFAULT_WEIGHTS["grounding"] * 0.5)
    assert result.score == pytest.approx(0.95)


def test_score_grounding_none_renormalizes_max_stays_1_0():
    """grounding_ratio=None must be DROPPED from aggregation (renormalized),
    not treated as a 0 — so a perfect answer on every other signal still
    reaches the full 1.0, not a value capped below it by a phantom penalty."""
    scorer = _scorer_with_pinned_signals(1.0)
    result = scorer.score(
        question="Q", answer="A", context="C",
        search_results=[], rerank_scores=[0.5],
        entailment_ratio=1.0, grounding_ratio=None,
    )
    assert result.score == pytest.approx(1.0)
    # Breakdown still reports a sane default (0.0) for the unavailable
    # signal, but it must not have been counted in the weighted average.
    assert result.grounding_score == 0.0
    assert result.refused is False
