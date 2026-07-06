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

import re
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

# Threshold for refusal
REFUSAL_THRESHOLD = 0.70

REFUSAL_ANSWER = (
    "⚠️ **Tôi không tìm thấy đủ thông tin chính xác để trả lời câu hỏi này.**\n\n"
    "Mặc dù tôi đã tìm kiếm trong cơ sở dữ liệu, nhưng thông tin tìm được "
    "không đủ độ tin cậy để đưa ra câu trả lời chính xác.\n\n"
    "💡 **Gợi ý:**\n"
    "- Hãy thử diễn đạt lại câu hỏi của bạn\n"
    "- Đề cập cụ thể tên văn bản pháp luật hoặc điều khoản\n"
    "- Tham khảo ý kiến luật sư để được tư vấn chính xác"
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
        entailment_ratio: float = 0.0,
    ) -> ConfidenceResult:
        """
        Compute multi-signal confidence score.

        Args:
            question: The user's question
            answer: The generated answer
            context: The retrieved context used for generation
            search_results: Results from vector search (for scores)
            rerank_scores: Scores from cross-encoder reranker (if available)
            entailment_ratio: Ratio of entailed claims from verifier (F17)

        Returns:
            ConfidenceResult with score, refusal flag, and per-signal breakdown
        """
        sig_retriever = self._retriever_signal(search_results)
        sig_reranker = self._reranker_signal(rerank_scores)
        sig_overlap = self._context_overlap(answer, context)
        sig_entailment = entailment_ratio
        sig_llm = self._llm_self_eval(question, answer, context)

        # Weighted aggregation
        weighted_score = (
            self.weights["retriever"] * sig_retriever
            + self.weights["reranker"] * sig_reranker
            + self.weights["context_overlap"] * sig_overlap
            + self.weights["entailment"] * sig_entailment
            + self.weights["llm_self_eval"] * sig_llm
        )

        # Clamp to [0, 1]
        weighted_score = max(0.0, min(1.0, weighted_score))

        return ConfidenceResult(
            score=weighted_score,
            refused=weighted_score < self.refusal_threshold,
            retriever_score=sig_retriever,
            reranker_score=sig_reranker,
            llm_self_eval_score=sig_llm,
            context_overlap_score=sig_overlap,
            entailment_score=sig_entailment,
        )

    # ── Signal ①: Retriever Score ──────────────────────────

    @staticmethod
    def _retriever_signal(results: List[SearchResult]) -> float:
        """Max cosine similarity score from search results."""
        if not results:
            return 0.0
        scores = [r.score for r in results if r.score is not None]
        if not scores:
            return 0.0
        # Qdrant scores are 0-1 for dense, or RRF score for hybrid
        # Normalize if needed: RRF scores are typically < 0.1
        max_s = max(scores)
        if max_s < 0.1:  # RRF score — rescale to 0-1
            return min(1.0, max_s * 60)  # RRF k=60 typically
        return max_s

    # ── Signal ②: Reranker Score ──────────────────────────

    @staticmethod
    def _reranker_signal(scores: Optional[List[float]]) -> float:
        """Max cross-encoder score from reranker (0-1)."""
        if not scores:
            return 0.0
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
            result = self.llm.generate(prompt=prompt, temperature=0, max_tokens=10)
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
