"""
Pure metric/matching functions for the retrieval + answer/refusal eval (scripts/eval.py).

No I/O, no models, no network — importable and unit-testable on their own
(see tests/test_eval_metrics.py). scripts/eval.py normalizes whatever it gets
(SearchResult objects, RAGResponse, eval-set question dicts) into plain dicts
before calling these functions.

Matching convention:
    `artnum()` mirrors scripts/validate_eval_set.py so eval matching and eval-set
    grounding validation agree on what "the same article" means.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ── Article-number matching ──────────────────────────────────────────────

def artnum(article: Optional[str]) -> Optional[str]:
    """Extract the article number from 'Điều 123' / 'Điều 123. Tội ...'.

    Same regex as scripts/validate_eval_set.py — keep them identical so a
    question graded "reachable" by the eval-set validator is matched the
    same way here.
    """
    m = re.search(r"\d+[a-z]?", article or "")
    return m.group() if m else None


_LENIENT_RE_CACHE: Dict[str, "re.Pattern"] = {}


def _lenient_pattern(article_num: str) -> "re.Pattern":
    pat = _LENIENT_RE_CACHE.get(article_num)
    if pat is None:
        pat = re.compile(r"Điều\s+" + re.escape(article_num) + r"\s*[\.:]")
        _LENIENT_RE_CACHE[article_num] = pat
    return pat


def lenient_text_match(article_num: Optional[str], chunk_text: Optional[str]) -> bool:
    """True if 'Điều <n>.' / 'Điều <n>:' appears verbatim in the chunk text.

    This catches the known chunker metadata-misalignment issue: the `article`
    payload field may be wrong/stale for a chunk while the chunk's own text
    still correctly starts with "Điều <n>." — lenient matching measures how
    often that rescues an otherwise-missed strict match.
    """
    if not article_num or not chunk_text:
        return False
    return _lenient_pattern(article_num).search(chunk_text) is not None


# ── Result matching (result dict vs expected {doc_id, article} dict) ────
#
# `result` is expected to have keys: document_id, article, chunk_text
# `expected` is expected to have keys: doc_id, article
# (this mirrors SearchResult's field names vs the eval-set schema's field names)

def strict_match(result: Dict, expected: Dict) -> bool:
    """result.document_id == expected.doc_id AND artnum(result.article) == artnum(expected.article)."""
    if result.get("document_id") != expected.get("doc_id"):
        return False
    return artnum(result.get("article")) == artnum(expected.get("article"))


def lenient_match(result: Dict, expected: Dict) -> bool:
    """result.document_id == expected.doc_id AND expected article number literally in chunk_text."""
    if result.get("document_id") != expected.get("doc_id"):
        return False
    return lenient_text_match(artnum(expected.get("article")), result.get("chunk_text"))


def any_match(result: Dict, expected: Dict, lenient: bool = False) -> bool:
    return lenient_match(result, expected) if lenient else strict_match(result, expected)


# ── Retrieval metrics ─────────────────────────────────────────────────────

def recall_at_k(
    results: List[Dict],
    expected_articles: List[Dict],
    k: int,
    lenient: bool = False,
) -> float:
    """Fraction of expected_articles found among the top-k results.

    A question with multiple expected_articles: recall is the fraction of
    ITS articles found (not all-or-nothing).
    """
    if not expected_articles:
        return 1.0  # nothing to find — vacuously satisfied
    topk = results[:k]
    found = sum(
        1 for exp in expected_articles
        if any(any_match(r, exp, lenient=lenient) for r in topk)
    )
    return found / len(expected_articles)


def mrr(
    results: List[Dict],
    expected_articles: List[Dict],
    lenient: bool = False,
) -> float:
    """Reciprocal rank (1-indexed) of the FIRST result matching ANY expected article.

    0.0 if no result matches any expected article ("no-match → 0").
    """
    if not expected_articles:
        return 1.0
    for rank, r in enumerate(results, start=1):
        if any(any_match(r, exp, lenient=lenient) for exp in expected_articles):
            return 1.0 / rank
    return 0.0


# ── Answer/refusal metrics (Tier 2) ──────────────────────────────────────

def refusal_accuracy(rows: List[Dict]) -> Optional[float]:
    """Among rows with expected_behavior=='refuse', fraction where refused is True.

    rows: [{"expected_behavior": "answer"|"refuse", "refused": bool}, ...]
    Returns None if there are no refuse-expected rows (undefined metric).
    """
    refuse_rows = [r for r in rows if r.get("expected_behavior") == "refuse"]
    if not refuse_rows:
        return None
    correct = sum(1 for r in refuse_rows if r.get("refused") is True)
    return correct / len(refuse_rows)


def false_refusal_rate(rows: List[Dict]) -> Optional[float]:
    """Among rows with expected_behavior=='answer', fraction incorrectly refused."""
    answer_rows = [r for r in rows if r.get("expected_behavior") == "answer"]
    if not answer_rows:
        return None
    refused = sum(1 for r in answer_rows if r.get("refused") is True)
    return refused / len(answer_rows)


def citation_accuracy(answer_text: str, expected_articles: List[Dict]) -> Optional[float]:
    """Fraction of expected article numbers that appear as 'Điều <n>' in the answer text."""
    expected_nums = {artnum(e.get("article")) for e in expected_articles}
    expected_nums.discard(None)
    if not expected_nums:
        return None
    found_nums = set(re.findall(r"Điều\s+(\d+[a-z]?)", answer_text or ""))
    hit = sum(1 for n in expected_nums if n in found_nums)
    return hit / len(expected_nums)
