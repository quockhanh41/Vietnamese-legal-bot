import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.measure_latency import percentiles


def test_percentiles_basic():
    out = percentiles([float(x) for x in range(1, 101)])  # 1..100
    assert out["n"] == 100
    assert out["max"] == 100.0
    assert 49 <= out["p50"] <= 52
    assert 94 <= out["p95"] <= 96


def test_percentiles_empty():
    assert percentiles([]) == {"p50": 0.0, "p95": 0.0, "max": 0.0, "n": 0}
