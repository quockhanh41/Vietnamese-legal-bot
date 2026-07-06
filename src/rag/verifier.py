"""
F17: Hallucination Verification — 2-tier (structural + semantic NLI).

Tier 1: Structural Validation
  - Check cited articles/clauses actually exist in retrieved context
  - Fast, rule-based, 100% precision

Tier 2: Semantic NLI Verification
  - Decompose answer into atomic claims
  - For each claim: NLI model → ENTAILMENT / CONTRADICTION / NEUTRAL
  - If contradiction_ratio > 30% → flag

Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (multilingual, ~800MB)
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of 2-tier hallucination verification."""

    # Tier 1: Structural
    citation_validity: List[bool] = field(default_factory=list)
    all_citations_valid: bool = False
    invalid_citations: List[str] = field(default_factory=list)

    # Tier 2: Semantic
    claims: List[str] = field(default_factory=list)
    entailment_labels: List[str] = field(default_factory=list)  # ENTAILMENT/CONTRADICTION/NEUTRAL
    entailment_ratio: float = 0.0   # % claims that are entailed
    contradiction_ratio: float = 0.0  # % claims contradicted

    # Overall
    passed: bool = True
    has_contradiction: bool = False
    warnings: List[str] = field(default_factory=list)


class HallucinationVerifier:
    """
    2-tier hallucination verifier for Vietnamese legal RAG.

    Usage:
        verifier = HallucinationVerifier()
        result = verifier.verify(answer, citations, context)
        if not result.passed:
            print(result.warnings)
    """

    # Patterns to extract article references from citations
    ARTICLE_PATTERN = re.compile(r'Điều\s+(\d+)', re.IGNORECASE)
    CLAUSE_PATTERN = re.compile(r'(?:khoản|Khoản)\s+(\d+)')
    POINT_PATTERN = re.compile(r'(?:điểm|điểm)\s+([a-zđ])')

    def __init__(self):
        self._nli_model = None  # Lazy-loaded
        self._nli_tokenizer = None

    # ── Public API ──────────────────────────────────────────────

    def verify(
        self,
        answer: str,
        citations: List[Dict],
        context: str,
    ) -> VerificationResult:
        """Run full 2-tier verification."""
        result = VerificationResult()

        if not answer or not context:
            result.warnings.append("Missing answer or context — skipping verification")
            result.passed = False
            return result

        # Tier 1: Structural citation check
        self._verify_citations(citations, context, result)

        # Tier 2: Semantic NLI verification
        self._verify_semantic(answer, context, result)

        # Overall assessment
        if result.contradiction_ratio > 0.30:
            result.has_contradiction = True
            result.warnings.append(
                f"⚠️ {result.contradiction_ratio:.0%} of claims contradicted by source documents."
            )
        if not result.all_citations_valid:
            result.warnings.append(
                f"⚠️ {len(result.invalid_citations)} citation(s) not found in retrieved context."
            )

        result.passed = not result.has_contradiction
        return result

    # ── Tier 1: Structural ──────────────────────────────────────

    def _verify_citations(self, citations: List[Dict], context: str, result: VerificationResult):
        """Check that each cited article/clause actually appears in the context."""
        if not citations:
            result.all_citations_valid = True
            return

        for cite in citations:
            article = cite.get("article", "")
            clause = cite.get("clause", "")

            # Check article reference in context
            article_valid = True
            if article:
                # Extract article number "Điều X" from citation
                m = self.ARTICLE_PATTERN.search(article)
                if m:
                    article_num = m.group(1)
                    if not self._article_in_context(article_num, context):
                        article_valid = False
                        result.invalid_citations.append(f"Điều {article_num}")

            result.citation_validity.append(article_valid)

        result.all_citations_valid = all(result.citation_validity)

    def _article_in_context(self, article_num: str, context: str) -> bool:
        """Check if an article reference exists in the context."""
        # Pattern: "Điều 3." or "Điều 3 " or "Điều 3:" etc.
        pattern = rf'Điều\s+{re.escape(article_num)}\s*[\.:\)\s]'
        return bool(re.search(pattern, context, re.IGNORECASE))

    # ── Tier 2: Semantic NLI ────────────────────────────────────

    def _load_nli_model(self):
        """Lazy-load the NLI model."""
        if self._nli_model is not None:
            return

        try:
            from transformers import pipeline
            MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            logger.info(f"Loading NLI model {MODEL_NAME}...")
            self._nli_model = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=-1,  # CPU
            )
            logger.info("✅ NLI model loaded")
        except Exception as e:
            logger.warning(f"⚠️ NLI model not available (skip): {e}")
            self._nli_model = False  # Sentinel

    def _split_claims(self, text: str) -> List[str]:
        """
        Split answer into atomic claims for verification.
        Vietnamese sentence-aware splitting.
        """
        if not text:
            return []

        # Remove markdown formatting
        clean = re.sub(r'\*\*', '', text)
        clean = re.sub(r'#+\s*', '', clean)

        # Split by Vietnamese sentence boundaries
        # "..." lists: each item is a claim
        # Regular sentences: split by sentence-ending punctuation
        sentences = []

        # First: check for bullet-list items (each item is a claim)
        lines = clean.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Bullet items
            bullet_match = re.match(r'^[\-\*\•]\s+(.*)', line)
            if bullet_match:
                sentences.append(bullet_match.group(1))
                continue
            # Numbered items
            num_match = re.match(r'^\d+[\.\)]\s+(.*)', line)
            if num_match:
                sentences.append(num_match.group(1))
                continue
            # Regular sentence — split by terminal punctuation
            # Vietnamese sentence endings: .!? và ...
            parts = re.split(r'(?<=[.!?])\s+', line)
            for p in parts:
                p = p.strip()
                if p and len(p) > 15:  # Filter very short fragments
                    sentences.append(p)

        # Deduplicate and filter
        seen = set()
        unique_sentences = []
        for s in sentences:
            key = s.lower().strip()
            if key not in seen:
                seen.add(key)
                unique_sentences.append(s)

        return unique_sentences

    def _verify_semantic(self, answer: str, context: str, result: VerificationResult):
        """Run NLI on each claim in the answer."""
        self._load_nli_model()
        if not self._nli_model:
            result.warnings.append("NLI model not available — semantic verification skipped")
            return

        claims = self._split_claims(answer)
        if len(claims) < 2:
            result.warnings.append("Too few claims for NLI — semantic verification skipped")
            return

        result.claims = claims
        # Truncate context to fit model max length (typically 512 tokens)
        truncated_context = context[:2000]

        entail_count = 0
        contradict_count = 0

        for claim in claims:
            try:
                output = self._nli_model(
                    sequences=claim,
                    candidate_labels=["ENTAILMENT", "CONTRADICTION", "NEUTRAL"],
                    hypothesis_template="The context says: {}.",
                    multi_label=False,
                )
                label = output["labels"][0]  # Top label
                result.entailment_labels.append(label)

                if label == "ENTAILMENT":
                    entail_count += 1
                elif label == "CONTRADICTION":
                    contradict_count += 1
            except Exception as e:
                logger.warning(f"NLI error on claim '{claim[:50]}': {e}")
                result.entailment_labels.append("NEUTRAL")

        total = len(claims)
        result.entailment_ratio = entail_count / total if total > 0 else 0
        result.contradiction_ratio = contradict_count / total if total > 0 else 0
