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
    """Guard against accidental drift of the calibrated constants — this
    task must not touch them."""
    assert DEFAULT_WEIGHTS == {
        "retriever": 0.25,
        "reranker": 0.20,
        "llm_self_eval": 0.30,
        "context_overlap": 0.15,
        "entailment": 0.10,
    }
    assert REFUSAL_THRESHOLD == 0.725 or os.getenv("CONFIDENCE_REFUSAL_THRESHOLD")
