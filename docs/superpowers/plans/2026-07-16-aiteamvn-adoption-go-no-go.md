# AITeamVN adoption GO/NO-GO — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the evidence to decide whether to adopt AITeamVN (mixed embedding + reranker) as production default — recalibrated threshold, deploy RAM/latency verdict, criminal-set eval, and root-cause of the 2 OOS leaks — ending in one GO/NO-GO recommendation. **No production change.**

**Architecture:** Mostly runs of existing tools (`eval.py`, `calibrate_threshold.py`) against the already-built `luat_law_ctx_v4_aiteamvn` collection, plus one new measurement script (`measure_stack_memory.py`) and an analysis task. Nothing in the production code path is modified.

**Tech Stack:** Python 3.13, existing eval/calibration harness, FlagEmbedding, sentence-transformers, psutil 7.2.2, Qdrant Cloud, pytest.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-16-aiteamvn-adoption-go-no-go-design.md`.
- **Evidence-only.** Do NOT touch `DEFAULT_COLLECTION_NAME`, `REFUSAL_THRESHOLD`, the production embedder wiring, or deploy config. Report numbers only.
- Stack under test (fixed everywhere): `--embedder mixed` (AITeamVN dense + bge-m3 sparse) + `--reranker AITeamVN/Vietnamese_Reranker`, collection `luat_law_ctx_v4_aiteamvn`.
- Gate #3 runs on **CPU** (HF Spaces deploy parity — no GPU).
- Groq budget is fragile (per-org TPD). Use `--sleep 3`; on mid-run errors use `scripts/eval.py --retry <report>`.
- Baselines to compare against: criminal Tier-1 `20260714T160205Z`, criminal Tier-2 `20260714T182637Z`; general Tier-2 `20260716T034715Z` (new-stack, 1 error — cleaned in Task 1); sectors Tier-2 `20260716T030111Z` (new-stack, clean).
- Current suite: 385 tests; must stay green (only Task 4 adds code).

---

### Task 1: Clean the new-stack general Tier-2 report

The gate #1 general run (`20260716T034715Z`) had 1 Groq error. Calibration needs clean rows.

**Files:** none (produces a new merged report in `data/eval_reports/`).

- [ ] **Step 1: Retry the errored question(s)**

Run:
```bash
.venv/bin/python scripts/eval.py --retry data/eval_reports/20260716T034715Z_eval.json --sleep 3
```

- [ ] **Step 2: Verify the merged report is clean**

Confirm the printed summary shows `n=41` and `errors=0`. Note the new report timestamp (call it `<GENERAL_CLEAN>` — a `data/eval_reports/<ts>_eval.json`). If it still errors, re-run this same `--retry` on the newest merged report until `errors=0`.

---

### Task 2: Criminal Tier-1 + Tier-2 on the new stack

**Files:** none (produces two reports).

- [ ] **Step 1: Tier-1 (no LLM)**

Run:
```bash
.venv/bin/python scripts/eval.py --set criminal \
  --collection luat_law_ctx_v4_aiteamvn --embedder mixed
```
Record recall@1/3/5/10 + MRR (routed/hybrid/dense). Baseline to compare: `20260714T160205Z` (v4 criminal Tier-1).

- [ ] **Step 2: Tier-2 (LLM — Groq budget)**

Run:
```bash
.venv/bin/python scripts/eval.py --set criminal --answers \
  --collection luat_law_ctx_v4_aiteamvn --embedder mixed \
  --reranker AITeamVN/Vietnamese_Reranker --sleep 3
```
If it errors mid-run: `.venv/bin/python scripts/eval.py --retry <that_report>.json --sleep 3` until `errors=0`. Call the clean report `<CRIMINAL_T2>`. Baseline: `20260714T182637Z` (v4 criminal Tier-2).

- [ ] **Step 3: Record the deltas**

Extract and note (for the writeup): criminal Tier-1 recall@5 per mode (new vs v4), and Tier-2 refusal_accuracy / false_refusal / mean_citation_accuracy (new vs v4).

---

### Task 3: Recalibrate the threshold on the new stack

**Files:** none (calibration prints a table; produces a calibration report).

- [ ] **Step 1: Run calibration over the 3 new-stack Tier-2 reports**

Run (substitute the timestamps from Tasks 1–2):
```bash
.venv/bin/python scripts/calibrate_threshold.py \
  --reports data/eval_reports/<GENERAL_CLEAN>.json \
            data/eval_reports/20260716T030111Z_eval.json \
            data/eval_reports/<CRIMINAL_T2>.json
```

- [ ] **Step 2: Record the recommended threshold**

From the printed sweep, note the chosen threshold and its refusal_recall + false_refusal, **and** what refusal_recall/false_refusal would be at the current **0.700** on the new stack. This is the gate #2 result. Do NOT edit `REFUSAL_THRESHOLD` — record only.

---

### Task 4: `measure_stack_memory.py` — deploy RAM/latency (gate #3)

**Files:**
- Create: `scripts/measure_stack_memory.py`
- Test: `tests/test_measure_stack_memory.py`

**Interfaces:**
- Produces: `classify_headroom(peak_rss_mb: float, budget_mb: int = 16384) -> str` returning `"fits-headroom"` (<70%), `"tight"` (70–90%), or `"doesn't-fit"` (≥90%); and a `main()` that loads `MixedHybridEmbedder(device="cpu")` + `FlagReranker("AITeamVN/Vietnamese_Reranker", device="cpu")`, runs an end-to-end query N times, and reports peak RSS + P50/P95 latency.

- [ ] **Step 1: Write the failing test**

Create `tests/test_measure_stack_memory.py`:

```python
from scripts.measure_stack_memory import classify_headroom


def test_fits_headroom_below_70pct():
    assert classify_headroom(10_000, budget_mb=16384) == "fits-headroom"


def test_tight_between_70_and_90pct():
    assert classify_headroom(13_000, budget_mb=16384) == "tight"


def test_doesnt_fit_at_or_above_90pct():
    assert classify_headroom(15_000, budget_mb=16384) == "doesn't-fit"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_measure_stack_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.measure_stack_memory'`.

- [ ] **Step 3: Write the script**

Create `scripts/measure_stack_memory.py`:

```python
"""Gate #3 — peak RAM + latency of the AITeamVN mixed stack on CPU (HF Spaces
deploy parity). Loads MixedHybridEmbedder (AITeamVN dense + BGE-M3 sparse) +
AITeamVN reranker, runs an end-to-end query (embed -> hybrid search -> rerank),
and reports peak RSS vs the 16GB budget. Local RSS approximates HF Spaces —
read trend + margin, not an absolute.

Usage: .venv/bin/python scripts/measure_stack_memory.py \
    --collection luat_law_ctx_v4_aiteamvn --repeats 20
"""
import argparse, json, os, statistics, sys, time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

REPORT_DIR = "data/eval_reports"
QUERY = "Điều kiện chuyển mục đích sử dụng đất nông nghiệp sang đất ở?"


def _percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def classify_headroom(peak_rss_mb, budget_mb=16384):
    ratio = peak_rss_mb / budget_mb
    if ratio < 0.70:
        return "fits-headroom"
    if ratio < 0.90:
        return "tight"
    return "doesn't-fit"


def main(collection, repeats, budget_mb):
    import psutil
    from src.data.mixed_embedder import MixedHybridEmbedder
    from src.db.qdrant_client import LegalVectorStore
    from FlagEmbedding import FlagReranker

    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 1e6

    embedder = MixedHybridEmbedder(device="cpu")
    reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=False, device="cpu")
    store = LegalVectorStore(collection_name=collection)

    lat = []
    peak = proc.memory_info().rss / 1e6
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = embedder.encode([QUERY], return_dense=True, return_sparse=True)
        cands = store.search_hybrid(out["dense_vecs"][0].tolist(),
                                    out["lexical_weights"][0], top_k=10)
        reranker.compute_score([[QUERY, c.chunk_text] for c in cands], normalize=True)
        lat.append((time.perf_counter() - t0) * 1000)
        peak = max(peak, proc.memory_info().rss / 1e6)

    report = {
        "device": "cpu",
        "collection": collection,
        "repeats": repeats,
        "budget_mb": budget_mb,
        "rss_baseline_mb": round(rss0, 1),
        "rss_peak_mb": round(peak, 1),
        "rss_stack_mb": round(peak - rss0, 1),
        "headroom_ratio": round(peak / budget_mb, 3),
        "verdict": classify_headroom(peak, budget_mb),
        "latency_ms_p50": round(_percentile(lat, 50), 1),
        "latency_ms_p95": round(_percentile(lat, 95), 1),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{ts}_stack_memory.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ report → {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="luat_law_ctx_v4_aiteamvn")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--budget-mb", type=int, default=16384)
    a = ap.parse_args()
    main(a.collection, a.repeats, a.budget_mb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_measure_stack_memory.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the measurement**

Run:
```bash
.venv/bin/python scripts/measure_stack_memory.py --collection luat_law_ctx_v4_aiteamvn --repeats 20
```
Record `rss_peak_mb`, `rss_stack_mb`, `verdict`, `latency_ms_p95`. This is the gate #3 result.

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_stack_memory.py tests/test_measure_stack_memory.py
git commit -m "feat(m3): measure_stack_memory.py — AITeamVN mixed stack RAM/latency (gate #3)"
```

---

### Task 5: Root-cause the 2 OOS leaks (gen-907, gen-912)

**Files:** none (analysis → notes in the writeup).

- [ ] **Step 1: Pull both rows from v4 vs new-stack general Tier-2**

Run (substitute `<GENERAL_CLEAN>`):
```bash
.venv/bin/python -c "
import json
v4 = {r['id']: r for r in json.load(open('data/eval_reports/20260714T181746Z_eval.json'))['tier2']['rows']}
ai = {r['id']: r for r in json.load(open('data/eval_reports/<GENERAL_CLEAN>.json'))['tier2']['rows']}
for q in ('gen-907','gen-912'):
    print('===', q, '===')
    for name, d in (('v4', v4[q]), ('ai', ai[q])):
        print(f'  {name}: refused={d[\"refused\"]} conf={d[\"confidence_score\"]} label={d.get(\"confidence_label\")} pipe={d.get(\"pipeline_used\")}')
"
```

- [ ] **Step 2: Classify the cause**

For each: did the pipeline route it differently (scope gate vs confidence gate), and did the confidence score cross 0.700 because the new embedding pulled an in-corpus-looking doc into context, or because a confidence signal (reranker/context/self-eval) shifted? Write 1–2 sentences of root cause per question for the writeup. (No code fix — evidence only.)

---

### Task 6: GO/NO-GO writeup

**Files:**
- Create: `docs/notes/aiteamvn-adoption-go-no-go.md`

- [ ] **Step 1: Write the consolidated writeup**

Create `docs/notes/aiteamvn-adoption-go-no-go.md` with these sections, filled from Tasks 1–5:

```markdown
# AITeamVN adoption — GO/NO-GO

Spec: docs/superpowers/specs/2026-07-16-aiteamvn-adoption-go-no-go-design.md
Stack: AITeamVN dense + bge-m3 sparse (mixed) + AITeamVN reranker, collection luat_law_ctx_v4_aiteamvn.

## Gate #2 — Threshold recalibration
| threshold | refusal_recall | false_refusal |
|---|---|---|
| 0.700 (current, new stack) | _ | _ |
| <recommended> | _ | _ |
Pooled sets: general + sectors + criminal (new stack).

## Gate #3 — Deploy RAM/latency (CPU)
peak RSS _ MB / stack _ MB vs 16GB budget → verdict: _. P95 latency _ ms.
Caveat: local RSS approximates HF Spaces.

## Criminal eval (new vs v4)
Tier-1 recall@5 (routed): _ vs _. Tier-2 refusal/false_refusal/citation: _ vs _.

## gen-907 / gen-912 root cause
- gen-907: _
- gen-912: _

## Recommendation
GO (mixed) if gate #3 = fits-headroom/tight AND criminal holds AND recalibrated
threshold is stable; else GO (dense-only fallback 0.898) if only gate #3 fails;
else NO-GO with reasons.
```

- [ ] **Step 2: Commit**

```bash
git add docs/notes/aiteamvn-adoption-go-no-go.md
git commit -m "docs(m3): AITeamVN adoption GO/NO-GO writeup"
```

---

## Notes for the executor
- Tasks 1–3, 5 are runs/analysis (Groq budget on 1–3); Task 4 is the only code (CPU, no LLM); Task 6 consolidates.
- If gate #3 says `doesn't-fit`, that is NOT a NO-GO for AITeamVN — it's a NO-GO for *mixed*; record the dense-only fallback (recall@5 0.898, one model, still > v4 hybrid 0.874) as the recommendation instead.
- The production swap itself (rewire embedder, flip default, adopt threshold) is explicitly out of scope — a separate milestone if this returns GO.
