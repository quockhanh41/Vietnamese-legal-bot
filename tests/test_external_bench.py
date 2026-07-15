"""
Unit tests for src/eval/external_bench.py — frozen split for the Zalo benchmark.
Pure functions; determinism is the whole point.

Run: .venv/bin/python -m pytest tests/test_external_bench.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.eval.external_bench import make_splits

IDS = [f"q{i}" for i in range(100)]


def test_split_sizes_80_10_10():
    s = make_splits(IDS)
    assert len(s["train"]) == 80
    assert len(s["dev"]) == 10
    assert len(s["test"]) == 10


def test_split_disjoint_and_complete():
    s = make_splits(IDS)
    union = set(s["train"]) | set(s["dev"]) | set(s["test"])
    assert union == set(IDS)
    assert len(s["train"]) + len(s["dev"]) + len(s["test"]) == len(IDS)


def test_split_deterministic_across_calls_and_input_order():
    a = make_splits(IDS)
    b = make_splits(list(reversed(IDS)))  # input order must not matter
    assert a["train"] == b["train"]
    assert a["dev"] == b["dev"]
    assert a["test"] == b["test"]


def test_split_records_seed_and_ratios():
    s = make_splits(IDS)
    assert s["seed"] == 20260715
    assert s["ratios"] == [0.8, 0.1, 0.1]


def test_split_different_seed_differs():
    assert make_splits(IDS, seed=1)["train"] != make_splits(IDS, seed=2)["train"]
