"""
F16: Confidence Scoring — Multi-signal weighted scoring + refusal gate.

Signals (from plan §4.3, weights updated M0.2 — see DEFAULT_WEIGHTS below):
  ① Retriever Score (0.20): max cosine similarity from search results
  ② Reranker Score (0.20): max cross-encoder score (or 0 if no reranker)
  ③ LLM Self-Eval  (0.25): LLM rates own answer on 0-100 scale
  ④ Context Overlap (0.15): keyword overlap between answer and context
  ⑤ Claim Entailment Ratio (0.10): % claims entailed by NLI (from verifier)
  ⑥ Grounding Ratio (0.10): % "Chi tiết" bullets that cite a valid context
    block [n] whose content backs the claim (from verifier Tier-1.5,
    machine-checked — see src/rag/verifier.py `_verify_grounding`)

Threshold: < 0.70 → refuse to answer ("Tôi không tìm thấy đủ thông tin...")

Usage:
    scorer = ConfidenceScorer(llm_gateway)
    result = scorer.score(question, answer, context, search_results, rerank_scores,
                           entailment_ratio, grounding_ratio)
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
from src.rag.context_blocks import parse_context_blocks, select_premise

logger = logging.getLogger(__name__)

# Default weights from the plan (§4.3).
#
# M0.2 (2026-07-10, docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md): added
# `grounding` (0.10) as a 6th signal — it was previously computed by the
# verifier (Tier-1.5, `_verify_grounding`) but only surfaced as
# observability, never fed into the confidence score. It's a machine-checked
# signal (does each "Chi tiết" bullet cite a valid context block whose text
# actually backs the claim?), so it deserves weight. The 0.10 was taken from
# the two least-discriminating signals: `retriever` (0.25→0.20 — saturates
# on hybrid/RRF-scale scores, see `_retriever_signal`) and `llm_self_eval`
# (0.30→0.25 — a self-report; M0.1 fixed its ceiling-at-0.85-1.00 bug but it
# is still the model grading itself, not an independent check). `reranker`,
# `context_overlap`, `entailment` are unchanged. Sum is still 1.00.
#
# REFUSAL_THRESHOLD (below) was recalibrated against these weights in M0.3
# (2026-07-10: 0.725 → 0.700) and again in M1.7 (2026-07-12, 100-doc corpus:
# 0.700 → 0.650) — see the provenance block above the constant.
DEFAULT_WEIGHTS = {
    "retriever": 0.20,
    "reranker": 0.20,
    "llm_self_eval": 0.25,
    "context_overlap": 0.15,
    "entailment": 0.10,
    "grounding": 0.10,
}

# Threshold for refusal — RECALIBRATED 2026-07-12 (M1.7) TO 0.650 at the
# 100-doc multi-sector scale (collection luat_law_ctx_v3, 23.7K points).
# Fresh Tier-2, 133 questions, 0 errors: criminal 41 + general 41 + sectors
# 51 (reports 20260712T004159Z / 20260712T002745Z / 20260712T003539Z,
# analysis 20260712T004208Z_calibration).
#
# THE THRESHOLD'S JOB CHANGED AT THIS SCALE. At 6 docs the confidence gate
# did scope work; at 100 docs the SCOPE GATE does it (20/21 expected-refuse
# rows are refused there and never reach this threshold — calibration now
# excludes those sentinel-0.0 rows from the sweep, scripts/
# calibrate_threshold.py M1.7 methodology fix). The confidence gate's only
# genuine customer is weak-evidence answers: exactly one expected-refuse row
# cleared the scope gate — crim-406 ("Điều 999 BLHS", non-existent article)
# at 0.625.
#
# t=0.650 is BOTH the grid Youden optimum AND the lowest leak-safe point:
# refusal_recall 1.0 (crim-406 caught, margin 0.025), false_refusal 0.028
# (3/107 — dd-201/ld-201/dn-201, ALL comparison-type: the known
# reranker-on-comparison weakness, M3.4 in the plan; benign refusals),
# youden 0.972, answer_precision 1.0. The old 0.700 false-refused 12/107
# (0.1121) for zero extra leak-safety — the answer distribution's floor
# dropped at 100 docs (min 0.4426 vs 0.7038 at 6 docs; more/harder
# questions, denser distractors). Answer-side margin: nearest kept answer
# hg-101 at 0.670 (+0.020). Both margins are within run-to-run noise —
# provisional as always; RECALIBRATE AT EVERY SCALE STEP (M2 gate,
# docs/Plan_Giai_Doan_Cuoi_2026-07-10.md).
#
# Known residue owned by OTHER workstreams, NOT this threshold: 5 scope-gate
# over-refusals (gen-301/304/802, dd-102/301 — root-cause B, classifier
# workstream); comparison-type low scores (M3.4). Eval-label lesson: crim-401
# /402 "refuse" labels were stale TWICE (SHTT and BHXH laws ARE in the
# 100-doc corpus — the bot answered them correctly at 0.8453/0.7045); every
# out-of-scope label must be verified against corpus_100docs.parquet.
#
# ── Previous calibration (M0.3, 2026-07-10, 6-doc corpus — kept for
# provenance; full history in git and the roadmap doc) ──
# 0.725 → 0.700 after the M0 overhaul (luat_law_ctx_v2 rebuild, per-claim
# NLI premise, rubric self-eval, grounding as 6th signal). 70 questions,
# 0 errors; first-ever clean separation: refuse max gen-802 0.6899 <
# answer min crim-203 0.7038 (gap +0.0139); t=0.700 the only grid point in
# the gap — refusal_recall 1.0, false_refusal 0.0, youden 1.0 (reports
# 20260710T085125Z + 20260710T085328Z, analysis 20260710T085625Z).
# Override per-deployment with CONFIDENCE_REFUSAL_THRESHOLD.
REFUSAL_THRESHOLD = float(os.getenv("CONFIDENCE_REFUSAL_THRESHOLD", "0.650"))

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
# never even run. `{corpus_summary}` is interpolated with the corpus
# inventory (CitationLookup.corpus_summary_compact() — sector names + counts
# on sector-labeled collections, full title list on older ones) so the
# message is never generic yet stays readable at 100+ docs (M1.4).
OUT_OF_SCOPE_ANSWER_TEMPLATE = (
    "⚠️ **Câu hỏi này nằm ngoài phạm vi dữ liệu hiện có của tôi.**\n\n"
    "Cơ sở dữ liệu hiện tại bao gồm:\n\n"
    "{corpus_summary}\n\n"
    "💡 **Gợi ý:**\n"
    "- Hãy đặt câu hỏi trong phạm vi các lĩnh vực nêu trên (ví dụ: tội danh và "
    "hình phạt, hợp đồng dân sự, quyền lợi người lao động, thành lập doanh "
    "nghiệp, sử dụng đất, ly hôn và nuôi con)\n"
    "- Tham khảo ý kiến luật sư hoặc cơ quan chuyên môn để được tư vấn chính "
    "xác về lĩnh vực này"
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
    grounding_score: float = 0.0

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

    Aggregates 6 signals into a weighted score. If score < 0.70,
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
        grounding_ratio: Optional[float] = None,
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
            grounding_ratio: Ratio of groundable ("Chi tiết" bullet) claims
                that cite a valid context block [n] backing them, from
                verifier Tier-1.5 (`VerificationResult.grounding_ratio`).
                None means the signal is unavailable — either the verifier
                didn't run for this route, or the answer had no groundable
                claims at all — NOT "0% grounded". Caller is responsible for
                making that distinction (see orchestrator._post_process,
                which mirrors the existing entailment_ratio convention).

        Returns:
            ConfidenceResult with score, refusal flag, and per-signal breakdown
        """
        sig_retriever = self._retriever_signal(search_results)
        sig_reranker = self._reranker_signal(rerank_scores)
        sig_overlap = self._context_overlap(answer, context)
        sig_entailment = entailment_ratio
        sig_grounding = grounding_ratio
        sig_llm = self._llm_self_eval(question, answer, context)

        # Weighted aggregation over available signals only — when a signal is
        # unavailable (None), drop it and renormalize the remaining weights so
        # the max achievable score stays 1.0 instead of silently capping below it.
        signals = {
            "retriever": sig_retriever,
            "reranker": sig_reranker,
            "context_overlap": sig_overlap,
            "entailment": sig_entailment,
            "grounding": sig_grounding,
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
            grounding_score=sig_grounding or 0.0,
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

    # Rubric criteria, in the order they appear in `CONFIDENCE_EVAL_PROMPT`.
    # Each is graded 0/1/2 by the LLM; combined in code into a 0-1 score
    # (sum / (2 * len(criteria))) instead of trusting a single self-reported
    # 0-100 number.
    _RUBRIC_CRITERIA = ["CĂN CỨ", "TRÍCH DẪN", "SUY DIỄN", "ĐẦY ĐỦ"]
    _RUBRIC_LINE_PATTERN = {
        label: re.compile(rf'{re.escape(label)}\s*:\s*([0-2])\b', re.IGNORECASE)
        for label in _RUBRIC_CRITERIA
    }

    def _llm_self_eval(self, question: str, answer: str, context: str) -> float:
        """
        Ask the LLM to grade its own answer against a rubric of separate,
        coarse (0/1/2) criteria, then combine them into a 0-1 score in code.

        Replaces the old single self-reported 0-100 number, which ceilinged
        at 0.85-1.00 across 8/8 manually inspected cases — including ones
        where the verifier flagged a contradiction — despite carrying 0.30
        of the total signal weight (see roadmap M0.1 follow-up).

        Context passed to the LLM is also no longer a blind `context[:2000]`
        truncation: it's the union of context block(s) the answer actually
        cites via `[n]` (same selector as the per-claim NLI premise), capped
        to ~3000 chars — falling back to lexical-overlap / blind truncation
        per `context_blocks.select_premise`.
        """
        try:
            blocks = parse_context_blocks(context)
            eval_context = select_premise(answer, context, blocks=blocks, cap=3000)
            prompt = CONFIDENCE_EVAL_PROMPT.format(
                question=question,
                answer=answer[:1000],  # Keep prompt manageable
                context=eval_context,
            )
            result = self.llm.generate(
                prompt=prompt, temperature=0, max_tokens=150, reasoning_effort="none",
            )
            score = self._parse_rubric_score(result)
            if score is not None:
                return score
        except Exception as e:
            logger.warning(f"LLM self-eval failed: {e}")
        return 0.5  # Neutral default on failure / unparseable rubric response

    @classmethod
    def _parse_rubric_score(cls, raw: str) -> Optional[float]:
        """
        Parse a `CONFIDENCE_EVAL_PROMPT` rubric response into a 0-1 score.

        Fail-safe: returns None unless ALL 4 criteria are found with a value
        in {0, 1, 2} — a partial/garbled parse is treated the same as a total
        failure (caller falls back to the 0.5 neutral default) rather than
        silently averaging over fewer criteria, which would bias the score
        in an unpredictable direction.
        """
        if not raw:
            return None
        scores = []
        for label in cls._RUBRIC_CRITERIA:
            m = cls._RUBRIC_LINE_PATTERN[label].search(raw)
            if not m:
                return None
            scores.append(int(m.group(1)))
        return sum(scores) / (2 * len(cls._RUBRIC_CRITERIA))

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
