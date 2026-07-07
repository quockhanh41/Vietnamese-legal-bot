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

    # Tier 1.5: Grounding — each detail-claim cites a numbered context block [n],
    # and the cited block actually contains what the claim attributes to it.
    grounded_claims: List[bool] = field(default_factory=list)
    grounding_ratio: float = 0.0
    invalid_block_refs: List[int] = field(default_factory=list)
    ungrounded_article_citations: List[str] = field(default_factory=list)

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

        # Tier 1: Structural citation check — verify articles cited IN THE ANSWER
        # actually appear in the retrieved context (citations list is not used here,
        # since it's built from the same retrieval results as context and would be vacuous).
        self._verify_citations(answer, context, result)

        # Tier 1.5: Grounding — does each claim point at a specific, valid
        # context block, and does that block actually back the claim?
        self._verify_grounding(answer, context, result)

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
        if result.invalid_block_refs:
            result.warnings.append(
                f"⚠️ {len(result.invalid_block_refs)} trích dẫn nguồn [n] không hợp lệ."
            )
        if result.ungrounded_article_citations:
            result.warnings.append(
                f"⚠️ {len(result.ungrounded_article_citations)} trích dẫn Điều không khớp với khối nguồn đã dẫn."
            )

        # Grounding is an observability/quality signal only — it must NOT
        # change `passed`, which stays governed solely by contradiction (the
        # calibrated refusal-confidence pipeline depends on that behavior).
        result.passed = not result.has_contradiction
        return result

    # ── Tier 1: Structural ──────────────────────────────────────

    def _verify_citations(self, answer: str, context: str, result: VerificationResult):
        """
        Check that each article reference CITED IN THE ANSWER actually appears
        in the retrieved context. Parses "Điều <n>" mentions directly from the
        LLM-generated answer text (not from citations built off the same
        retrieval results as context, which would make this check vacuous).
        """
        article_nums = self.ARTICLE_PATTERN.findall(answer)
        if not article_nums:
            result.all_citations_valid = True
            return

        # Deduplicate while preserving order
        seen = set()
        unique_articles = []
        for num in article_nums:
            if num not in seen:
                seen.add(num)
                unique_articles.append(num)

        for article_num in unique_articles:
            valid = self._article_in_context(article_num, context)
            result.citation_validity.append(valid)
            if not valid:
                result.invalid_citations.append(f"Điều {article_num}")

        result.all_citations_valid = all(result.citation_validity)

    def _article_in_context(self, article_num: str, context: str) -> bool:
        """Check if an article reference exists in the context."""
        # Pattern: "Điều 3." or "Điều 3 " or "Điều 3:" etc.
        pattern = rf'Điều\s+{re.escape(article_num)}\s*[\.:\)\s]'
        return bool(re.search(pattern, context, re.IGNORECASE))

    def _article_in_text(self, article_num: str, text: str) -> bool:
        """Same check as `_article_in_context` but scoped to an arbitrary
        text snippet (used to check a single context block)."""
        return self._article_in_context(article_num, text)

    # ── Tier 1.5: Grounding ─────────────────────────────────────

    # Matches a block header line like "[1] Bộ luật Hình sự 2015 — Điều 17 (...)"
    # at the start of each `_build_context()` block (src/rag/orchestrator.py).
    BLOCK_HEADER_PATTERN = re.compile(r'^\[(\d+)\]')
    BLOCK_REF_PATTERN = re.compile(r'\[(\d+)\]')

    def _parse_context_blocks(self, context: str) -> Dict[int, str]:
        """
        Parse the `_build_context()` output — blocks numbered "[i] ..." joined
        by "\n\n---\n\n" — into an {index: block_text} dict. Derives the block
        count from what's actually parsed, not from an assumed format.
        """
        blocks: Dict[int, str] = {}
        if not context:
            return blocks

        for raw_block in context.split('\n\n---\n\n'):
            raw_block = raw_block.strip()
            if not raw_block:
                continue
            match = self.BLOCK_HEADER_PATTERN.match(raw_block)
            if not match:
                continue
            idx = int(match.group(1))
            blocks[idx] = raw_block

        return blocks

    def _verify_grounding(self, answer: str, context: str, result: VerificationResult):
        """
        Tier 1.5: check that each detail-claim in the answer cites a valid
        numbered context block [n], and that a block cited alongside an
        "Điều X" reference actually contains that article.

        Observability-only: does not affect `result.passed`.
        """
        blocks = self._parse_context_blocks(context)
        k = len(blocks)

        claims = self._split_claims(answer)
        if not claims:
            result.grounding_ratio = 0.0
            return

        invalid_refs_seen = set()
        ungrounded_articles: List[str] = []
        grounded_flags: List[bool] = []

        for claim in claims:
            refs = [int(n) for n in self.BLOCK_REF_PATTERN.findall(claim)]
            valid_refs = [n for n in refs if 1 <= n <= k]
            invalid_refs = [n for n in refs if not (1 <= n <= k)]
            invalid_refs_seen.update(invalid_refs)

            is_grounded = len(valid_refs) > 0
            grounded_flags.append(is_grounded)

            # For each article cited in this claim, verify it's actually
            # present in (at least one of) the claim's cited block(s).
            article_nums = self.ARTICLE_PATTERN.findall(claim)
            if article_nums and valid_refs:
                for article_num in article_nums:
                    found_in_any = any(
                        self._article_in_text(article_num, blocks[n])
                        for n in valid_refs
                    )
                    if not found_in_any:
                        article_label = f"Điều {article_num}"
                        if article_label not in ungrounded_articles:
                            ungrounded_articles.append(article_label)

        result.grounded_claims = grounded_flags
        result.invalid_block_refs = sorted(invalid_refs_seen)
        result.ungrounded_article_citations = ungrounded_articles
        total = len(claims)
        grounded_count = sum(1 for g in grounded_flags if g)
        result.grounding_ratio = grounded_count / total if total > 0 else 0.0

    # ── Tier 2: Semantic NLI ────────────────────────────────────

    # Maps the model's native output labels to our ENTAILMENT/NEUTRAL/CONTRADICTION strings.
    NLI_LABEL_MAP = {
        "entailment": "ENTAILMENT",
        "neutral": "NEUTRAL",
        "contradiction": "CONTRADICTION",
    }

    def _load_nli_model(self):
        """Lazy-load the NLI model."""
        if self._nli_model is not None:
            return

        try:
            from transformers import pipeline
            MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
            logger.info(f"Loading NLI model {MODEL_NAME}...")
            # This is a native NLI model (premise/hypothesis pair classification),
            # NOT a zero-shot classifier — use "text-classification" and pass the
            # context as `text` (premise) and the claim as `text_pair` (hypothesis).
            self._nli_model = pipeline(
                "text-classification",
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
            # Stop at the "related questions" section (SYSTEM_PROMPT mandates
            # every answer end with "### 💡 Câu hỏi liên quan" + 3 suggested
            # follow-up questions). These are UI suggestions for the user's
            # NEXT query, not factual claims about the law — they're
            # interrogative, so NLI can never sensibly label them ENTAILMENT
            # against a legal-context premise. Since every single answer this
            # pipeline produces has this trailer, leaving it in the claim set
            # systematically (and meaninglessly) deflates entailment_ratio
            # for every question, not just these 4. The section is always
            # last per the fixed prompt structure, so breaking here drops it
            # (and everything after it) without touching real claims above.
            # Match only a standalone header line (nothing but non-word chars
            # around the phrase) — "câu hỏi liên quan đến X" occurs in natural
            # Vietnamese mid-sentence, and truncating there would silently
            # drop every real claim after it; missing a malformed header just
            # falls back to the old noisy-but-harmless behavior.
            if re.match(r'^\W*câu\s*hỏi\s*liên\s*quan\W*$', line, re.IGNORECASE):
                break
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
        # Truncate context to fit model max length (typically 512 tokens).
        # Keep the char-based truncation as a cheap first pass; the tokenizer
        # will also truncate (truncation=True) so we don't overflow the model.
        truncated_context = context[:2000]

        entail_count = 0
        contradict_count = 0

        for claim in claims:
            try:
                # Native NLI pair: premise=context, hypothesis=claim.
                output = self._nli_model(
                    {"text": truncated_context, "text_pair": claim},
                    truncation=True,
                )
                # pipeline() may return a dict or a list with one dict depending on version
                if isinstance(output, list):
                    output = output[0]
                raw_label = str(output["label"]).lower()
                label = self.NLI_LABEL_MAP.get(raw_label, "NEUTRAL")
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
