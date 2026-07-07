"""
F16: Confidence Scoring — Multi-signal weighted scoring + refusal gate.

Signals (from plan §4.3):
  ① Retriever Score (0.25): max cosine similarity from search results
  ② Reranker Score (0.20): max cross-encoder score (or 0 if no reranker)
  ③ LLM Self-Eval  (0.30): LLM rates own answer on 0-100 scale
  ④ Context Overlap (0.15): keyword overlap between answer and context
  ⑤ Claim Entailment Ratio (0.10): % claims entailed by NLI (from verifier)

Threshold: < 0.70 → refuse to answer ("Tôi không tìm thấy đủ thông tin...")

Usage:
    scorer = ConfidenceScorer(llm_gateway)
    result = scorer.score(question, answer, context, search_results, rerank_scores, entailment_ratio)
    if result.refused:
        # return fallback answer
"""

from __future__ import annotations

import os
import re
import statistics
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.db.qdrant_client import SearchResult
from src.llm.gateway import LLMGateway
from src.llm.prompts import CONFIDENCE_EVAL_PROMPT

logger = logging.getLogger(__name__)

# Default weights from the plan (§4.3)
DEFAULT_WEIGHTS = {
    "retriever": 0.25,
    "reranker": 0.20,
    "llm_self_eval": 0.30,
    "context_overlap": 0.15,
    "entailment": 0.10,
}

# Threshold for refusal — RE-VALIDATED 2026-07-07 WITH THE LIVE BGE RERANKER
# (transformers<5 pin, commit 958a777). The 0.725 value was originally picked
# while the 0.20-weight reranker signal was silently renormalized away (reranker
# crashed on every medium/complex query); this re-run confirms it holds now that
# the signal is real. Fresh Tier-2, 70 questions, 0 errors: 34 answer + 36 refuse
# (reports 20260707T110621Z criminal + 20260707T111111Z general, analysis
# 20260707T111144Z_calibration via scripts/calibrate_threshold.py).
#
# Grid Youden-optimal is t=0.675 (youden 0.9428, refusal_recall 0.9722,
# false_refusal 0.0294) — but 0.675 LEAKS the one genuine out-of-scope refuse
# question that clears the scope gate, gen-802 (land-dispute, score 0.7045):
# refusal_recall drops from 1.0 to 0.9722. For a legal bot, answering an
# out-of-scope question is worse than a conservative false-refusal, so we keep
# the threshold in the leak-safe band (t > 0.7045). 0.725 is the lowest sweep
# grid point that still refuses gen-802: refusal_recall 1.0, false_refusal
# 0.0588 (crim-203 + crim-201), youden 0.9412 — within 0.0016 of the grid
# optimum. New margin: the clean separation gap is now gen-802 (0.7045, top
# refuse) → crim-104 (0.7291, lowest robustly-answered) = 0.0246 wide; 0.725
# sits 0.0205 above gen-802 (fat leak-side cushion) and 0.0041 below crim-104
# (thin, benign false-refuse side). The reranker did NOT collapse the answer
# distribution (answer mean 0.8301); its only boundary effect is pushing
# comparison-type crim-203 to ~0.53-0.57 (verified: retrieval succeeds — both
# ground-truth murder articles present at top cosine 0.8333 — but the
# cross-encoder scores every single chunk low (max 0.2015) because a comparison
# answer synthesizes 2+ articles; stacked with a harsh self-eval draw + the
# NLI-NEUTRAL-on-short-Vietnamese-claims weakness). crim-203 is a flagged
# per-signal follow-up, not a threshold problem — rescuing it would need t<0.57
# and leak every refuse question. Still provisional at 6-doc corpus; recalibrate
# as the eval set grows. Override per-deployment with CONFIDENCE_REFUSAL_THRESHOLD.
REFUSAL_THRESHOLD = float(os.getenv("CONFIDENCE_REFUSAL_THRESHOLD", "0.725"))

REFUSAL_ANSWER = (
    "⚠️ **Tôi không tìm thấy đủ thông tin chính xác để trả lời câu hỏi này.**\n\n"
    "Mặc dù tôi đã tìm kiếm trong cơ sở dữ liệu, nhưng thông tin tìm được "
    "không đủ độ tin cậy để đưa ra câu trả lời chính xác.\n\n"
    "💡 **Gợi ý:**\n"
    "- Hãy thử diễn đạt lại câu hỏi của bạn\n"
    "- Đề cập cụ thể tên văn bản pháp luật hoặc điều khoản\n"
    "- Tham khảo ý kiến luật sư để được tư vấn chính xác"
)

# Out-of-scope refusal — distinct from REFUSAL_ANSWER (which fires AFTER
# retrieval+generation when the confidence gate isn't satisfied). This one
# fires BEFORE any retrieval/generation, when the SCOPE gate (merged into the
# complexity-classifier LLM call — see src/llm/prompts.py
# COMPLEXITY_CLASSIFIER_PROMPT) decides the question cannot be answered from
# this corpus at all — no confidence score involved, since retrieval was
# never even run. `{corpus_summary}` is interpolated with the actual document
# titles (CitationLookup.corpus_summary()) so the message is never generic.
OUT_OF_SCOPE_ANSWER_TEMPLATE = (
    "⚠️ **Câu hỏi này nằm ngoài phạm vi dữ liệu hiện có của tôi.**\n\n"
    "Cơ sở dữ liệu hiện tại chỉ bao gồm các văn bản pháp luật sau:\n\n"
    "{corpus_summary}\n\n"
    "💡 **Gợi ý:**\n"
    "- Hãy đặt câu hỏi trong phạm vi các văn bản nêu trên (ví dụ: về trách nhiệm hình sự, tội danh, hình phạt)\n"
    "- Tham khảo ý kiến luật sư hoặc cơ quan chuyên môn để được tư vấn chính xác về lĩnh vực này"
)


@dataclass
class ConfidenceResult:
    """Result of multi-signal confidence scoring."""

    # Overall
    score: float = 0.0
    refused: bool = False

    # Per-signal breakdown for debugging/UI
    retriever_score: float = 0.0
    reranker_score: float = 0.0
    llm_self_eval_score: float = 0.0
    context_overlap_score: float = 0.0
    entailment_score: float = 0.0

    # Human-readable label
    label: str = "low"  # "high" | "medium" | "low"

    def __post_init__(self):
        if self.score >= 0.80:
            self.label = "high"
        elif self.score >= 0.50:
            self.label = "medium"
        else:
            self.label = "low"


class ConfidenceScorer:
    """
    Multi-signal confidence scorer for Vietnamese legal RAG.

    Aggregates 5 signals into a weighted score. If score < 0.70,
    the answer should be refused.
    """

    def __init__(
        self,
        llm_gateway: LLMGateway,
        weights: Optional[Dict[str, float]] = None,
        refusal_threshold: float = REFUSAL_THRESHOLD,
    ):
        self.llm = llm_gateway
        self.weights = weights or DEFAULT_WEIGHTS
        self.refusal_threshold = refusal_threshold

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        search_results: List[SearchResult],
        rerank_scores: Optional[List[float]] = None,
        entailment_ratio: Optional[float] = None,
    ) -> ConfidenceResult:
        """
        Compute multi-signal confidence score.

        Args:
            question: The user's question
            answer: The generated answer
            context: The retrieved context used for generation
            search_results: Results from vector search (for scores)
            rerank_scores: Scores from cross-encoder reranker. None means
                reranking did not run for this query (signal unavailable) —
                distinct from an (impossible) empty-but-ran list.
            entailment_ratio: Ratio of entailed claims from verifier (F17).
                None means semantic (NLI) verification was skipped (signal
                unavailable), not "0% entailed".

        Returns:
            ConfidenceResult with score, refusal flag, and per-signal breakdown
        """
        sig_retriever = self._retriever_signal(search_results)
        sig_reranker = self._reranker_signal(rerank_scores)
        sig_overlap = self._context_overlap(answer, context)
        sig_entailment = entailment_ratio
        sig_llm = self._llm_self_eval(question, answer, context)

        # Weighted aggregation over available signals only — when a signal is
        # unavailable (None), drop it and renormalize the remaining weights so
        # the max achievable score stays 1.0 instead of silently capping below it.
        signals = {
            "retriever": sig_retriever,
            "reranker": sig_reranker,
            "context_overlap": sig_overlap,
            "entailment": sig_entailment,
            "llm_self_eval": sig_llm,
        }
        available = {k: v for k, v in signals.items() if v is not None}
        weight_sum = sum(self.weights[k] for k in available)
        if weight_sum > 0:
            weighted_score = sum(self.weights[k] * v for k, v in available.items()) / weight_sum
        else:
            weighted_score = 0.0

        # Clamp to [0, 1]
        weighted_score = max(0.0, min(1.0, weighted_score))

        return ConfidenceResult(
            score=weighted_score,
            refused=weighted_score < self.refusal_threshold,
            retriever_score=sig_retriever or 0.0,
            reranker_score=sig_reranker or 0.0,
            llm_self_eval_score=sig_llm or 0.0,
            context_overlap_score=sig_overlap or 0.0,
            entailment_score=sig_entailment or 0.0,
        )

    # ── Signal ①: Retriever Score ──────────────────────────

    @staticmethod
    def _retriever_signal(results: List[SearchResult]) -> Optional[float]:
        """
        Retriever confidence signal from search result scores (0-1).

        Qdrant returns cosine scores (0-1) for dense search, or RRF scores
        (typically << 0.1, saturating near 1/60 for the top hit) for hybrid
        search. RRF scores can't be rescaled by a fixed multiplier (`* 60`)
        without saturating at ~1.0 for virtually every query — instead, for
        RRF-scale scores we use a distribution-aware proxy: how far the top
        score stands above the median of the returned scores. A large margin
        means one result clearly dominates (confident); a small margin means
        the results are close together (ambiguous). This is monotone in the
        top score's separation and doesn't saturate.
        """
        if not results:
            return None
        scores = [r.score for r in results if r.score is not None]
        if not scores:
            return None
        max_s = max(scores)
        if max_s < 0.1:  # RRF-scale scores
            median_s = statistics.median(scores)
            if max_s <= 0:
                return 0.0
            # Relative margin of top score over median, mapped to [0, 1].
            margin = (max_s - median_s) / max_s
            return max(0.0, min(1.0, margin))
        return max_s

    # ── Signal ②: Reranker Score ──────────────────────────

    @staticmethod
    def _reranker_signal(scores: Optional[List[float]]) -> Optional[float]:
        """Max cross-encoder score from reranker (0-1), or None if unavailable."""
        if scores is None:
            return None
        if not scores:
            return None
        return max(scores)

    # ── Signal ③: LLM Self-Eval ───────────────────────────

    def _llm_self_eval(self, question: str, answer: str, context: str) -> float:
        """Ask LLM to rate its own answer confidence on 0-100 scale."""
        try:
            prompt = CONFIDENCE_EVAL_PROMPT.format(
                question=question,
                answer=answer[:1000],  # Keep prompt manageable
                context=context[:2000],
            )
            result = self.llm.generate(
                prompt=prompt, temperature=0, max_tokens=100, reasoning_effort="none",
            )
            # Extract number from response
            nums = re.findall(r'\d+', result.strip())
            if nums:
                score = int(nums[0]) / 100.0
                return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"LLM self-eval failed: {e}")
        return 0.5  # Neutral default on failure

    # ── Signal ④: Context Overlap ─────────────────────────

    @staticmethod
    def _context_overlap(answer: str, context: str) -> float:
        """
        Compute % of key terms from answer that appear in context.

        Uses Vietnamese-aware keyword extraction:
        - Extracts legal terms (Điều, Khoản, tội, etc.)
        - Extracts capitalized nouns (proper names)
        - Computes Jaccard-like overlap ratio
        """
        if not answer or not context:
            return 0.0

        # Normalize
        answer_lower = answer.lower()
        context_lower = context.lower()

        # Extract keywords: legal terms + capitalized words + numbers
        def extract_keywords(text: str) -> set:
            keywords = set()
            # Legal article references
            for m in re.finditer(r'điều\s+\d+|khoản\s+\d+|điểm\s+[a-zđ]', text):
                keywords.add(m.group())
            # Capitalized multi-word terms (Vietnamese proper names)
            for m in re.finditer(r'[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỪỨỬỮỰÝỲỶỸỴ]\w+(?:\s+[A-ZÀÁẢÃẠĂẰẮẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỪỨỬỮỰÝỲỶỸỴ]\w+)*', text):
                keywords.add(m.group().lower())
            # Numbers (percentage, amounts, years)
            for m in re.finditer(r'\b\d{2,4}\b', text):
                keywords.add(m.group())
            return keywords

        answer_kw = extract_keywords(answer_lower)
        context_kw = extract_keywords(context_lower)

        if not answer_kw:
            return 0.5  # No keywords to check → neutral

        overlap = len(answer_kw & context_kw)
        ratio = overlap / len(answer_kw)
        return min(1.0, ratio)
