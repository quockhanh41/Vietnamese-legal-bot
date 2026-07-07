"""
Calibrate the confidence refusal threshold (src/rag/confidence.py REFUSAL_THRESHOLD)
using real Tier-2 (`scripts/eval.py --answers`) confidence-score data.

Pure analysis — no LLM calls, no network. Reads per-question rows
(confidence_score + expected_behavior) out of one or more Tier-2 report JSON
files (data/eval_reports/<ts>_eval.json with config.tier == "tier2_answers"),
sweeps candidate thresholds, and reports which threshold best separates
"answer" from "refuse" questions (Youden's J statistic).

Usage:
    python scripts/calibrate_threshold.py                       # auto-discover latest tier2 reports
    python scripts/calibrate_threshold.py --reports a.json b.json
"""

from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import click

REPORT_DIR = "data/eval_reports"

T_MIN = 0.30
T_MAX = 0.90
T_STEP = 0.025

BUCKET_SIZE = 0.05


# ── Loading ──────────────────────────────────────────────────────────────

def discover_latest_reports(report_dir: str = REPORT_DIR) -> List[str]:
    """Find the most recent tier2_answers report per eval `set` (criminal/general).

    Scans report_dir for *_eval.json files with config.tier == "tier2_answers",
    groups by config.set, and keeps only the newest (by config.generated_at_utc,
    falling back to filename) per group.
    """
    candidates = sorted(glob.glob(os.path.join(report_dir, "*_eval.json")))
    latest_by_set: Dict[str, Tuple[str, str]] = {}  # set_name -> (generated_at, path)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        config = data.get("config", {})
        if config.get("tier") != "tier2_answers":
            continue
        set_name = config.get("set", "unknown")
        generated_at = config.get("generated_at_utc", "") or path
        prev = latest_by_set.get(set_name)
        if prev is None or generated_at > prev[0]:
            latest_by_set[set_name] = (generated_at, path)
    return [path for _, path in latest_by_set.values()]


def load_rows(report_paths: List[str]) -> List[Dict]:
    """Load and merge per-question rows from one or more Tier-2 report JSONs.

    Each returned row keeps at least: id, expected_behavior, confidence_score,
    error, plus a `_source` tag (the report path) for traceability.
    """
    rows: List[Dict] = []
    for path in report_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tier2 = data.get("tier2")
        if not tier2:
            continue
        for row in tier2.get("rows", []):
            r = dict(row)
            r["_source"] = path
            rows.append(r)
    return rows


def _usable_rows(rows: List[Dict]) -> List[Dict]:
    """Filter to rows with no pipeline error and a usable confidence_score."""
    return [
        r for r in rows
        if r.get("error") is None
        and r.get("confidence_score") is not None
        and r.get("expected_behavior") in ("answer", "refuse")
    ]


# ── Sweep math (pure — unit tested in tests/test_calibration.py) ─────────

def sweep_thresholds(
    rows: List[Dict],
    t_min: float = T_MIN,
    t_max: float = T_MAX,
    step: float = T_STEP,
) -> List[Dict]:
    """Sweep candidate thresholds t in [t_min, t_max] (inclusive) by `step`.

    rows: [{"confidence_score": float, "expected_behavior": "answer"|"refuse"}, ...]
    (rows with error != null / missing score must already be filtered out by
    the caller — this function assumes every row has a usable score.)

    For each t, computes:
      - refusal_recall: fraction of expected_behavior=="refuse" rows with
        score < t (i.e. would be refused). None if there are no refuse rows.
      - false_refusal_rate: fraction of expected_behavior=="answer" rows with
        score < t (incorrectly refused). None if there are no answer rows.
      - answer_precision: among rows with score >= t (i.e. would be
        answered), fraction that are actually expected_behavior=="answer".
        None if no row scores >= t.
      - youden: refusal_recall - false_refusal_rate. None if either operand
        is None (can't compute without both groups).

    Returns a list of dicts, one per threshold, in ascending order of t.
    """
    refuse_scores = [r["confidence_score"] for r in rows if r["expected_behavior"] == "refuse"]
    answer_scores = [r["confidence_score"] for r in rows if r["expected_behavior"] == "answer"]

    results = []
    t = t_min
    n_steps = round((t_max - t_min) / step)
    for i in range(n_steps + 1):
        t = round(t_min + i * step, 10)

        refusal_recall = (
            sum(1 for s in refuse_scores if s < t) / len(refuse_scores)
            if refuse_scores else None
        )
        false_refusal = (
            sum(1 for s in answer_scores if s < t) / len(answer_scores)
            if answer_scores else None
        )

        all_scores_with_behavior = [(s, "refuse") for s in refuse_scores] + \
                                    [(s, "answer") for s in answer_scores]
        would_answer = [b for s, b in all_scores_with_behavior if s >= t]
        answer_precision = (
            sum(1 for b in would_answer if b == "answer") / len(would_answer)
            if would_answer else None
        )

        youden = (
            refusal_recall - false_refusal
            if refusal_recall is not None and false_refusal is not None
            else None
        )

        results.append({
            "threshold": round(t, 4),
            "refusal_recall": refusal_recall,
            "false_refusal_rate": false_refusal,
            "answer_precision": answer_precision,
            "youden": youden,
        })
    return results


def pick_best_threshold(sweep: List[Dict]) -> Optional[Dict]:
    """Pick the sweep row maximizing youden; tie-break on lower false_refusal_rate.

    Rows with youden None are ignored. Returns None if no row has a usable
    youden value (e.g. one of the two behavior groups is entirely empty).
    """
    candidates = [r for r in sweep if r["youden"] is not None]
    if not candidates:
        return None

    def sort_key(r: Dict):
        # Maximize youden, then minimize false_refusal_rate.
        fr = r["false_refusal_rate"] if r["false_refusal_rate"] is not None else 1.0
        return (-r["youden"], fr)

    return sorted(candidates, key=sort_key)[0]


def histogram(scores: List[float], bucket_size: float = BUCKET_SIZE) -> Dict[str, int]:
    """Bucket scores into fixed-width [lo, lo+bucket_size) bins, 0.0 to 1.0.

    Returns an ordered dict of "lo-hi" -> count. Empty input -> {} (no bars).
    """
    if not scores:
        return {}
    n_buckets = round(1.0 / bucket_size)
    counts: Dict[str, int] = {}
    for i in range(n_buckets):
        lo = round(i * bucket_size, 4)
        hi = round(lo + bucket_size, 4)
        label = f"{lo:.2f}-{hi:.2f}"
        counts[label] = 0
    for s in scores:
        idx = min(int(s / bucket_size), n_buckets - 1)
        lo = round(idx * bucket_size, 4)
        hi = round(lo + bucket_size, 4)
        label = f"{lo:.2f}-{hi:.2f}"
        counts[label] += 1
    return counts


def overlap_region(rows: List[Dict]) -> Dict:
    """Max score among refuse-rows vs min score among answer-rows.

    If max(refuse) >= min(answer), the two distributions overlap and no
    threshold can perfectly separate them; the gap is negative (or zero).
    If max(refuse) < min(answer), there's a clean separation gap.
    Either group missing -> all fields None.
    """
    refuse_scores = [r["confidence_score"] for r in rows if r["expected_behavior"] == "refuse"]
    answer_scores = [r["confidence_score"] for r in rows if r["expected_behavior"] == "answer"]
    if not refuse_scores or not answer_scores:
        return {"max_refuse_score": None, "min_answer_score": None, "gap": None, "overlaps": None}
    max_refuse = max(refuse_scores)
    min_answer = min(answer_scores)
    return {
        "max_refuse_score": round(max_refuse, 4),
        "min_answer_score": round(min_answer, 4),
        "gap": round(min_answer - max_refuse, 4),
        "overlaps": max_refuse >= min_answer,
    }


def outliers(rows: List[Dict], n: int = 3) -> Dict:
    """Top-n answer-questions with LOWEST scores + refuse-questions with HIGHEST scores.

    These are the rows most likely to be misclassified by any threshold —
    useful for diagnosing which signals are misbehaving.
    """
    answer_rows = sorted(
        (r for r in rows if r["expected_behavior"] == "answer"),
        key=lambda r: r["confidence_score"],
    )
    refuse_rows = sorted(
        (r for r in rows if r["expected_behavior"] == "refuse"),
        key=lambda r: -r["confidence_score"],
    )
    return {
        "lowest_scoring_answer_questions": [
            {"id": r.get("id"), "confidence_score": r["confidence_score"]} for r in answer_rows[:n]
        ],
        "highest_scoring_refuse_questions": [
            {"id": r.get("id"), "confidence_score": r["confidence_score"]} for r in refuse_rows[:n]
        ],
    }


# ── Printing ───────────────────────────────────────────────────────────────

def print_sweep_table(sweep: List[Dict], best: Optional[Dict]):
    click.echo("\n## Threshold sweep\n")
    click.echo("| t | refusal_recall | false_refusal_rate | answer_precision | youden | |")
    click.echo("|---|---|---|---|---|---|")
    for row in sweep:
        marker = " <-- chosen" if best is not None and row["threshold"] == best["threshold"] else ""
        rr = row["refusal_recall"]
        fr = row["false_refusal_rate"]
        ap = row["answer_precision"]
        yj = row["youden"]
        click.echo(
            f"| {row['threshold']:.3f} | {rr if rr is None else round(rr, 4)} | "
            f"{fr if fr is None else round(fr, 4)} | {ap if ap is None else round(ap, 4)} | "
            f"{yj if yj is None else round(yj, 4)} |{marker} |"
        )


def _text_bar(count: int, scale: int = 1) -> str:
    return "#" * count if count else ""


def print_histograms(rows: List[Dict]):
    click.echo("\n## Score histograms (bucket=0.05)\n")
    for behavior in ("answer", "refuse"):
        scores = [r["confidence_score"] for r in rows if r["expected_behavior"] == behavior]
        click.echo(f"\n### {behavior} (n={len(scores)})\n")
        hist = histogram(scores)
        if not hist:
            click.echo("(no data)")
            continue
        for label, count in hist.items():
            click.echo(f"{label} | {_text_bar(count)} ({count})")


def print_overlap(overlap: Dict):
    click.echo("\n## Overlap region\n")
    if overlap["max_refuse_score"] is None:
        click.echo("Cannot compute — one of the two behavior groups is empty.")
        return
    click.echo(f"max(refuse scores) = {overlap['max_refuse_score']}")
    click.echo(f"min(answer scores) = {overlap['min_answer_score']}")
    if overlap["overlaps"]:
        click.echo(
            f"⚠️ OVERLAP: some refuse-question scores ({overlap['max_refuse_score']}) exceed "
            f"some answer-question scores ({overlap['min_answer_score']}) — no threshold can "
            f"perfectly separate the two groups (gap = {overlap['gap']})."
        )
    else:
        click.echo(f"Clean separation — gap = {overlap['gap']}")


def print_outliers(out: Dict):
    click.echo("\n## Outliers\n")
    click.echo("Lowest-scoring answer-questions (should have scored HIGH, didn't):")
    for r in out["lowest_scoring_answer_questions"]:
        click.echo(f"  - {r['id']}: {r['confidence_score']}")
    click.echo("Highest-scoring refuse-questions (should have scored LOW, didn't):")
    for r in out["highest_scoring_refuse_questions"]:
        click.echo(f"  - {r['id']}: {r['confidence_score']}")


# ── CLI ────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--reports", multiple=True, type=click.Path(exists=True),
              help="Tier-2 report JSON paths. If omitted, auto-discovers the latest "
                   "tier2_answers report per eval set in data/eval_reports/.")
def main(reports: Tuple[str, ...]):
    report_paths = list(reports) if reports else discover_latest_reports()
    if not report_paths:
        click.echo("⚠️ No tier2_answers reports found. Run `scripts/eval.py --answers` first.")
        sys.exit(1)

    click.echo(f"📊 Calibrating from {len(report_paths)} report(s):")
    for p in report_paths:
        click.echo(f"  - {p}")

    all_rows = load_rows(report_paths)
    rows = _usable_rows(all_rows)
    n_errors = len(all_rows) - len(rows)
    click.echo(f"\nRows: {len(all_rows)} total, {n_errors} skipped (error/missing score), "
               f"{len(rows)} usable.")

    n_answer = sum(1 for r in rows if r["expected_behavior"] == "answer")
    n_refuse = sum(1 for r in rows if r["expected_behavior"] == "refuse")
    click.echo(f"By expected_behavior: answer={n_answer}, refuse={n_refuse}")

    sweep = sweep_thresholds(rows)
    best = pick_best_threshold(sweep)
    print_sweep_table(sweep, best)

    if best is not None:
        click.echo(f"\n**Youden-optimal threshold: t={best['threshold']:.3f} "
                   f"(youden={round(best['youden'], 4)}, "
                   f"refusal_recall={round(best['refusal_recall'], 4)}, "
                   f"false_refusal_rate={round(best['false_refusal_rate'], 4)})**")
    else:
        click.echo("\n⚠️ Could not pick a best threshold — one of the two behavior "
                   "groups (answer/refuse) is entirely empty.")

    print_histograms(rows)

    overlap = overlap_region(rows)
    print_overlap(overlap)

    out = outliers(rows)
    print_outliers(out)

    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(REPORT_DIR, f"{ts}_calibration.json")
    analysis = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_reports": report_paths,
        "n_rows_total": len(all_rows),
        "n_rows_errors_skipped": n_errors,
        "n_rows_usable": len(rows),
        "n_answer": n_answer,
        "n_refuse": n_refuse,
        "sweep": sweep,
        "best_threshold": best,
        "histograms": {
            "answer": histogram([r["confidence_score"] for r in rows if r["expected_behavior"] == "answer"]),
            "refuse": histogram([r["confidence_score"] for r in rows if r["expected_behavior"] == "refuse"]),
        },
        "overlap": overlap,
        "outliers": out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    click.echo(f"\n💾 Calibration report saved: {out_path}")


if __name__ == "__main__":
    main()
