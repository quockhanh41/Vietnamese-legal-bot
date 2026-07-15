"""
Unit tests for src/eval/bench_metrics.py — exact-ID retrieval metrics for
external benchmarks. Hand-computed expectations documented inline.

Run: .venv/bin/python -m pytest tests/test_bench_metrics.py -q
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.bench_metrics import recall_at_k, mrr_at_k, ndcg_at_k

REL = {("L1", "A1"), ("L2", "A2")}
# ranks:      1            2            3
RETRIEVED = [("L9", "A9"), ("L1", "A1"), ("L2", "A2")]


def test_recall_at_k():
    assert recall_at_k(RETRIEVED, REL, k=1) == 0.0
    assert recall_at_k(RETRIEVED, REL, k=2) == 0.5   # found L1:A1 only
    assert recall_at_k(RETRIEVED, REL, k=3) == 1.0


def test_recall_empty_relevant_is_vacuous():
    assert recall_at_k(RETRIEVED, set(), k=3) == 1.0


def test_mrr_first_hit_at_rank_2():
    assert mrr_at_k(RETRIEVED, REL, k=10) == 0.5


def test_mrr_no_hit_within_k():
    assert mrr_at_k(RETRIEVED, REL, k=1) == 0.0


def test_ndcg_hand_computed():
    # DCG  = 1/log2(2+1) + 1/log2(3+1)          (hits at ranks 2 and 3)
    # IDCG = 1/log2(1+1) + 1/log2(2+1)          (2 relevant, ideal ranks 1,2)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RETRIEVED, REL, k=10) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_1():
    assert ndcg_at_k([("L1", "A1"), ("L2", "A2")], REL, k=10) == pytest.approx(1.0)


def test_ndcg_no_hits_is_0():
    assert ndcg_at_k([("L9", "A9")], REL, k=10) == 0.0


def test_ndcg_k_smaller_than_relevant_set():
    # 3 relevant, k=2, hits at ranks 1 and 2 → DCG == IDCG over 2 slots → 1.0
    rel3 = {("L1", "A1"), ("L2", "A2"), ("L3", "A3")}
    assert ndcg_at_k([("L1", "A1"), ("L2", "A2")], rel3, k=2) == pytest.approx(1.0)
