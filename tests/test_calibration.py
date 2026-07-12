"""
Unit tests for scripts/calibrate_threshold.py — pure sweep math, no network, no models.

Run: .venv/bin/python -m pytest tests/test_calibration.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest

from calibrate_threshold import (
    sweep_thresholds,
    pick_best_threshold,
    histogram,
    overlap_region,
    outliers,
    split_scope_gate_rows,
    scope_gate_stats,
)


def _row(id_, behavior, score, pipeline=None):
    r = {"id": id_, "expected_behavior": behavior, "confidence_score": score}
    if pipeline is not None:
        r["pipeline_used"] = pipeline
    return r


# ── sweep_thresholds ──────────────────────────────────────────────────────

def test_sweep_perfect_separation():
    # Refuse rows all < 0.5, answer rows all >= 0.5 -> some threshold gives
    # perfect youden = 1.0 (refusal_recall=1, false_refusal_rate=0).
    rows = [
        _row("r1", "refuse", 0.1),
        _row("r2", "refuse", 0.2),
        _row("a1", "answer", 0.8),
        _row("a2", "answer", 0.9),
    ]
    sweep = sweep_thresholds(rows, t_min=0.3, t_max=0.7, step=0.1)
    # at t=0.5: refuse scores (0.1,0.2) both < 0.5 -> recall 1.0
    # answer scores (0.8,0.9) both >= 0.5 -> false_refusal_rate 0.0
    row_05 = next(r for r in sweep if abs(r["threshold"] - 0.5) < 1e-9)
    assert row_05["refusal_recall"] == 1.0
    assert row_05["false_refusal_rate"] == 0.0
    assert row_05["youden"] == 1.0
    assert row_05["answer_precision"] == 1.0


def test_sweep_thresholds_ascending_order_and_bounds():
    rows = [_row("a1", "answer", 0.6), _row("r1", "refuse", 0.4)]
    sweep = sweep_thresholds(rows, t_min=0.3, t_max=0.5, step=0.1)
    thresholds = [r["threshold"] for r in sweep]
    assert thresholds == sorted(thresholds)
    assert thresholds[0] == pytest.approx(0.3)
    assert thresholds[-1] == pytest.approx(0.5)


def test_sweep_empty_refuse_group_gives_none():
    rows = [_row("a1", "answer", 0.8), _row("a2", "answer", 0.9)]
    sweep = sweep_thresholds(rows, t_min=0.3, t_max=0.5, step=0.1)
    for row in sweep:
        assert row["refusal_recall"] is None
        assert row["youden"] is None
        # false_refusal_rate and answer_precision still computable
        assert row["false_refusal_rate"] is not None


def test_sweep_empty_answer_group_gives_none():
    rows = [_row("r1", "refuse", 0.2), _row("r2", "refuse", 0.3)]
    sweep = sweep_thresholds(rows, t_min=0.3, t_max=0.5, step=0.1)
    for row in sweep:
        assert row["false_refusal_rate"] is None
        assert row["youden"] is None
        assert row["refusal_recall"] is not None


def test_sweep_no_rows_at_all():
    sweep = sweep_thresholds([], t_min=0.3, t_max=0.5, step=0.1)
    for row in sweep:
        assert row["refusal_recall"] is None
        assert row["false_refusal_rate"] is None
        assert row["answer_precision"] is None
        assert row["youden"] is None


def test_sweep_answer_precision_no_rows_above_threshold():
    rows = [_row("r1", "refuse", 0.1)]
    sweep = sweep_thresholds(rows, t_min=0.5, t_max=0.5, step=0.1)
    assert sweep[0]["answer_precision"] is None  # nothing scores >= 0.5


# ── pick_best_threshold ───────────────────────────────────────────────────

def test_pick_best_threshold_maximizes_youden():
    rows = [
        _row("r1", "refuse", 0.1),
        _row("r2", "refuse", 0.2),
        _row("a1", "answer", 0.8),
        _row("a2", "answer", 0.9),
    ]
    sweep = sweep_thresholds(rows, t_min=0.3, t_max=0.7, step=0.1)
    best = pick_best_threshold(sweep)
    assert best is not None
    assert best["youden"] == 1.0


def test_pick_best_threshold_tie_break_lower_false_refusal():
    # Two thresholds with equal youden; the one with lower false_refusal_rate wins.
    sweep = [
        {"threshold": 0.4, "refusal_recall": 0.5, "false_refusal_rate": 0.1, "answer_precision": 0.9, "youden": 0.4},
        {"threshold": 0.5, "refusal_recall": 0.6, "false_refusal_rate": 0.2, "answer_precision": 0.8, "youden": 0.4},
    ]
    best = pick_best_threshold(sweep)
    assert best["threshold"] == 0.4


def test_pick_best_threshold_all_none_returns_none():
    sweep = [{"threshold": 0.5, "refusal_recall": None, "false_refusal_rate": None,
              "answer_precision": None, "youden": None}]
    assert pick_best_threshold(sweep) is None


def test_pick_best_threshold_empty_sweep_returns_none():
    assert pick_best_threshold([]) is None


# ── histogram ─────────────────────────────────────────────────────────────

def test_histogram_buckets_scores():
    scores = [0.0, 0.04, 0.05, 0.99, 1.0]
    hist = histogram(scores, bucket_size=0.05)
    assert hist["0.00-0.05"] == 2  # 0.0, 0.04
    assert hist["0.05-0.10"] == 1  # 0.05
    assert hist["0.95-1.00"] == 2  # 0.99, 1.0 (clamped to last bucket)


def test_histogram_empty_scores():
    assert histogram([]) == {}


# ── overlap_region ────────────────────────────────────────────────────────

def test_overlap_region_clean_separation():
    rows = [
        _row("r1", "refuse", 0.2),
        _row("r2", "refuse", 0.3),
        _row("a1", "answer", 0.8),
        _row("a2", "answer", 0.9),
    ]
    overlap = overlap_region(rows)
    assert overlap["max_refuse_score"] == 0.3
    assert overlap["min_answer_score"] == 0.8
    assert overlap["overlaps"] is False
    assert overlap["gap"] == pytest.approx(0.5)


def test_overlap_region_overlapping_distributions():
    rows = [
        _row("r1", "refuse", 0.673),  # crim-001-like false refusal case
        _row("a1", "answer", 0.6),
    ]
    overlap = overlap_region(rows)
    assert overlap["overlaps"] is True
    assert overlap["gap"] < 0


def test_overlap_region_missing_group_is_none():
    rows = [_row("a1", "answer", 0.8)]
    overlap = overlap_region(rows)
    assert overlap["max_refuse_score"] is None
    assert overlap["min_answer_score"] is None
    assert overlap["gap"] is None
    assert overlap["overlaps"] is None


def test_overlap_region_empty_rows():
    overlap = overlap_region([])
    assert overlap["overlaps"] is None


# ── outliers ──────────────────────────────────────────────────────────────

def test_outliers_top_n():
    rows = [
        _row("a1", "answer", 0.9),
        _row("a2", "answer", 0.5),
        _row("a3", "answer", 0.6),
        _row("a4", "answer", 0.95),
        _row("r1", "refuse", 0.1),
        _row("r2", "refuse", 0.6),
        _row("r3", "refuse", 0.3),
        _row("r4", "refuse", 0.05),
    ]
    out = outliers(rows, n=2)
    lowest_answer_ids = [r["id"] for r in out["lowest_scoring_answer_questions"]]
    assert lowest_answer_ids == ["a2", "a3"]
    highest_refuse_ids = [r["id"] for r in out["highest_scoring_refuse_questions"]]
    assert highest_refuse_ids == ["r2", "r3"]


def test_outliers_empty_groups():
    out = outliers([], n=3)
    assert out["lowest_scoring_answer_questions"] == []
    assert out["highest_scoring_refuse_questions"] == []


# ── scope-gate split (M1.7 methodology fix) ──────────────────────────────
#
# Scope-gate refusals (pipeline_used == "out-of-scope") carry a SENTINEL
# confidence_score of 0.0 — they never reached the confidence gate. Mixing
# them into the sweep inflates refusal_recall at every threshold and drags
# the Youden optimum toward meaningless lows (observed: 0.475/0.550 on the
# contaminated 2026-07-11 runs). The threshold must be calibrated only on
# rows the confidence gate actually scored.

def test_split_scope_gate_rows_separates_sentinels():
    rows = [
        _row("a1", "answer", 0.8, pipeline="simple"),
        _row("r1", "refuse", 0.0, pipeline="out-of-scope"),
        _row("r2", "refuse", 0.55, pipeline="medium"),
    ]
    gate_rows, scope_rows = split_scope_gate_rows(rows)
    assert [r["id"] for r in gate_rows] == ["a1", "r2"]
    assert [r["id"] for r in scope_rows] == ["r1"]


def test_split_scope_gate_rows_missing_pipeline_field_is_gate_row():
    # Old reports / hand-built rows without pipeline_used stay in the sweep.
    rows = [_row("a1", "answer", 0.8)]
    gate_rows, scope_rows = split_scope_gate_rows(rows)
    assert len(gate_rows) == 1 and scope_rows == []


def test_scope_gate_stats_over_refusals_and_escapes():
    scope_rows = [
        _row("r1", "refuse", 0.0, pipeline="out-of-scope"),   # correct refusal
        _row("a9", "answer", 0.0, pipeline="out-of-scope"),   # OVER-refusal (root cause B)
    ]
    gate_rows = [
        _row("a1", "answer", 0.8, pipeline="simple"),
        _row("r2", "refuse", 0.835, pipeline="medium"),       # escaped scope gate (crim-402 pattern)
    ]
    stats = scope_gate_stats(scope_rows, gate_rows)
    assert stats["n_scope_refused"] == 2
    assert stats["n_expected_refuse"] == 1
    assert stats["over_refused_ids"] == ["a9"]
    assert stats["escaped_refuse"] == [{"id": "r2", "confidence_score": 0.835}]


def test_scope_gate_stats_empty():
    stats = scope_gate_stats([], [])
    assert stats["n_scope_refused"] == 0
    assert stats["over_refused_ids"] == []
    assert stats["escaped_refuse"] == []
