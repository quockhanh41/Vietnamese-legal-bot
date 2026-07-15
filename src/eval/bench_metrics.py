"""
Exact-ID retrieval metrics for external benchmarks (M3.1 — Zalo/ALQAC).

Operates on (law_id, article_id) tuples with binary relevance. Kept separate
from src/eval/metrics.py, whose recall/MRR use the internal eval's
legal-domain strict/lenient matching (any_match) — different semantics.
"""
import math
from typing import List, Set, Tuple

IdPair = Tuple[str, str]


def recall_at_k(retrieved: List[IdPair], relevant: Set[IdPair], k: int) -> float:
    """Fraction of relevant ids found in the top-k. Vacuously 1.0 when
    relevant is empty (mirrors src/eval/metrics.recall_at_k)."""
    if not relevant:
        return 1.0
    topk = set(retrieved[:k])
    found = sum(1 for r in relevant if r in topk)
    return found / len(relevant)


def mrr_at_k(retrieved: List[IdPair], relevant: Set[IdPair], k: int = 10) -> float:
    """Reciprocal rank (1-indexed) of the first relevant id within top-k."""
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[IdPair], relevant: Set[IdPair], k: int = 10) -> float:
    """Binary-relevance nDCG@k: DCG over hit ranks / ideal DCG."""
    if not relevant:
        return 1.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
