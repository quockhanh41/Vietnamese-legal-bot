# M2 Completion (600-docs ingest) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare everything needed to grow the corpus from 100 to ~600 documents — capacity gate, semi-automated corpus selection, pre-ingest validation, eval expansion, and Kaggle handoff — up to (but not including) the GPU ingest the user runs.

**Architecture:** Five in-session deliverables feeding one human-run GPU step. (A) a telemetry-based capacity script; (B) a corpus selector that keeps the 100-doc spine locked, auto-generates ~500 candidates within the same 6 sectors, budgets them by **chunk count bounded to ≤~3× across sectors** (frequency-bias control), and emits a review table for the user to prune; (C) chunker-alignment + capacity + eval-grounding validation on the frozen 600-doc parquet, plus a cross-sector leakage diagnostic; (D) a standalone enrichment runner + updated Kaggle guide. Phase E (post-ingest enrichment, recalibration, gates) is a **separate follow-up plan** because it depends on the user's ingest.

**Tech Stack:** Python 3.13, `pyarrow`/`pandas` (streaming parquet), `qdrant-client` + REST `/telemetry`, `click`, `pytest`. No GPU/LLM in this plan.

**Reference spec:** [`docs/superpowers/specs/2026-07-13-m2-600docs-completion-design.md`](../specs/2026-07-13-m2-600docs-completion-design.md)

## Global Constraints

- **No live mutation** — nothing in this plan writes to any Qdrant collection. `measure_capacity.py` and any read against `luat_law_ctx_v3` are **read-only**. Ingest to `luat_law_ctx_v4` is the user's Kaggle step, out of this plan.
- **Spine is immutable** — the 100 doc-ids in `scripts/select_corpus.py::CORPUS` appear in the 600 corpus unchanged, with their hand-assigned `sector`.
- **Deepen the same 6 sectors only** — `hinh_su, dan_su, lao_dong, doanh_nghiep, dat_dai, hngd`. No new sectors → OOS eval anchors stay valid; do not touch the 14 `expected_behavior: refuse` general questions.
- **Frequency-bias budget** — per-sector total chunk count (spine + added) must satisfy `max/min ≤ ~3.0`. Budget on **estimated** chunks in Phase B (`len(content)//700`), verify on **real** chunks in Phase C.
- **`article_number` / article labels are strings** — Vietnamese articles carry letter suffixes ("Điều 43a"); never cast to int.
- **No silent doc loss** — every candidate dropped (superseded guard, content guard, chunker) is logged, matching `scripts/ingest.py` curated-mode behavior.
- **Grounding is verified against the parquet** — every new eval question's `expected_articles` must pass `validate_eval_set.py --corpus-parquet data/raw/corpus_600docs.parquet` (labels have gone stale twice — always verify).
- **Test invocation** — `.venv/bin/python -m pytest ...`. Baseline: **333 tests collected**, all green.
- **Commits** — branch `m2-600docs-ingest` (already checked out), one commit per task, message `feat(M2.x): ...` or `docs(M2): ...`, footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `scripts/measure_capacity.py` (create) | Read Qdrant `/telemetry`; per-collection bytes/point; project 600-doc RAM/disk vs free-tier | 1 |
| `tests/test_measure_capacity.py` (create) | Telemetry parsing + projection math (fixtures, no network) | 1 |
| `scripts/select_corpus_600.py` (create) | Sector map, candidate filter, chunk-budget allocation, review-table export, parquet build | 2–5 |
| `tests/test_select_corpus_600.py` (create) | sector_of / estimate_chunks / is_candidate / allocate_budget invariants | 2, 3 |
| `data/raw/corpus_600docs.parquet` (build artifact) | Frozen 600-doc corpus (after user review) | 5 |
| `scripts/verify_chunker_alignment.py` (modify) | Add per-sector chunk-count + max/min ratio report when `sector` column present | 6 |
| `data/eval/eval_set_sectors_v2.json` (create) | v1 (51) + ~30–50 new Thông tư-targeted questions | 7 |
| `scripts/eval.py` (modify) | Register sectors_v2; add cross-sector leakage@k to `run_tier1` | 7, 8 |
| `tests/test_eval_metrics.py` (modify) | leakage@k computation | 8 |
| `scripts/run_enrichment.py` (create) | Standalone enrichment runner for the Kaggle notebook | 9 |
| `requirements-enrich.txt` (create) | Deps to run enrichment on Kaggle (adds `networkx`) | 9 |
| `docs/notes/kaggle-ingest-guide.md` (modify) | 600-doc ingest + drop stale collections + enrichment cell + bring `store.db` back | 10 |

**Task order:** 1(A1) → 2(B1) → 3(B2) → 4(B3) → **[USER REVIEW GATE]** → 5(B4) → 6(C1) → 7(C2) → 8(C3) → 9(D1) → 10(D2). Task 1 is independent. Tasks 6–8 depend on Task 5's parquet. Tasks 9/10 are independent of 1–8 but land last so the guide references the final artifacts.

---

### Task 1 (A1): Capacity gate from Qdrant telemetry (2.1)

Read-only script that parses `/telemetry?details_level=10` for real per-collection byte sizes and projects the 600-doc footprint against the 1 GB RAM / 4 GB disk free-tier limits. Confirmed available: each shard/segment exposes `vectors_size_bytes` + `payloads_size_bytes`.

**Files:**
- Create: `scripts/measure_capacity.py`
- Test: `tests/test_measure_capacity.py`

**Interfaces:**
- Consumes: env `QDRANT_URL`, `QDRANT_API_KEY` (via `python-dotenv`).
- Produces:
  - `parse_collection_sizes(telemetry: dict) -> dict[str, dict]` → `{name: {"vectors_bytes": int, "payloads_bytes": int}}`
  - `project_capacity(points: int, dim: int = 1024, sparse_bytes_per_point: int = 400, hnsw_bytes_per_point: int = 256) -> dict` → `{"ram_bytes": int, "disk_bytes": int, "ram_pct": float, "disk_pct": float}`
  - `FREE_TIER_RAM = 1024**3`, `FREE_TIER_DISK = 4 * 1024**3`, `GATE_PCT = 70.0`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_measure_capacity.py
from scripts.measure_capacity import parse_collection_sizes, project_capacity, GATE_PCT

_TELEMETRY = {
    "result": {"collections": {"collections": [
        {"id": "luat_law_ctx_v3",
         "shards": [{"local": {"vectors_size_bytes": 8559563, "payloads_size_bytes": 3034112}}]},
        {"id": "vietnamese_laws",
         "shards": [{"local": {"vectors_size_bytes": 8722368, "payloads_size_bytes": 2456320}}]},
    ]}}
}

def test_parse_collection_sizes_sums_shards():
    sizes = parse_collection_sizes(_TELEMETRY)
    assert sizes["luat_law_ctx_v3"]["vectors_bytes"] == 8559563
    assert sizes["luat_law_ctx_v3"]["payloads_bytes"] == 3034112
    assert set(sizes) == {"luat_law_ctx_v3", "vietnamese_laws"}

def test_project_capacity_quantized_ram_and_pct():
    p = project_capacity(points=142_000)
    # int8 quantized dense = points * dim bytes = 142000 * 1024
    assert p["ram_bytes"] >= 142_000 * 1024
    # 600-doc worst case must stay under the 70% gate on the 1GB tier
    assert p["ram_pct"] < GATE_PCT
    assert 0 < p["disk_pct"] < GATE_PCT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_measure_capacity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.measure_capacity'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/measure_capacity.py
"""M2.1 — Capacity gate from Qdrant telemetry (read-only).

Reads GET {QDRANT_URL}/telemetry?details_level=10 for real per-collection
byte sizes and projects a target point count against the free-tier limits
(1GB RAM / 4GB disk). Gate: projected usage < 70%.

Usage:
    .venv/bin/python scripts/measure_capacity.py                 # table + project 142k
    .venv/bin/python scripts/measure_capacity.py --target-points 150000
"""
import json, os, sys, time
import click, requests
from dotenv import load_dotenv

load_dotenv()

FREE_TIER_RAM = 1024 ** 3          # 1 GB
FREE_TIER_DISK = 4 * 1024 ** 3     # 4 GB
GATE_PCT = 70.0


def parse_collection_sizes(telemetry: dict) -> dict:
    """Sum vectors_size_bytes + payloads_size_bytes across every shard of
    every collection in a /telemetry?details_level=10 response."""
    out = {}
    cols = telemetry.get("result", {}).get("collections", {}).get("collections", [])
    for col in cols:
        name = col.get("id")
        v = p = 0
        for shard in col.get("shards", []):
            local = (shard or {}).get("local") or {}
            v += local.get("vectors_size_bytes", 0) or 0
            p += local.get("payloads_size_bytes", 0) or 0
        out[name] = {"vectors_bytes": v, "payloads_bytes": p}
    return out


def project_capacity(points: int, dim: int = 1024,
                     sparse_bytes_per_point: int = 400,
                     hnsw_bytes_per_point: int = 256) -> dict:
    """Estimate RAM/disk for `points` chunks under the v3 config:
    int8-quantized dense always-RAM (dim bytes/point) + sparse + HNSW graph;
    fp32 original on-disk (dim*4 bytes/point) + payload (~130 B/point observed)."""
    ram = points * dim + points * sparse_bytes_per_point + points * hnsw_bytes_per_point
    disk = points * dim * 4 + points * 130
    return {
        "ram_bytes": ram,
        "disk_bytes": disk,
        "ram_pct": 100.0 * ram / FREE_TIER_RAM,
        "disk_pct": 100.0 * disk / FREE_TIER_DISK,
    }


def _fetch(url: str, key: str, path: str) -> dict:
    r = requests.get(f"{url.rstrip('/')}{path}", headers={"api-key": key}, timeout=30)
    r.raise_for_status()
    return r.json()


@click.command()
@click.option("--target-points", type=int, default=142_000,
              help="Projected chunk count for the 600-doc corpus (same-mix extrapolation).")
def main(target_points):
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url:
        sys.exit("QDRANT_URL not set")
    tel = _fetch(url, key, "/telemetry?details_level=10")
    sizes = parse_collection_sizes(tel)

    print(f"{'collection':<22}{'points':>10}{'vectors MB':>12}{'payload MB':>12}{'B/point':>10}")
    stale = []
    for name, s in sorted(sizes.items()):
        try:
            pts = _fetch(url, key, f"/collections/{name}/points/count")
            n = pts.get("result", {}).get("count", 0)
        except Exception:
            n = 0
        vb, pb = s["vectors_bytes"], s["payloads_bytes"]
        bpp = (vb + pb) / n if n else 0
        print(f"{name:<22}{n:>10,}{vb/1e6:>12.1f}{pb/1e6:>12.1f}{bpp:>10.0f}")
        if name in {"luat_law", "luat_law_ctx", "vietnamese_laws"}:
            stale.append(name)

    proj = project_capacity(target_points)
    print(f"\nProjection @ {target_points:,} points (600-doc corpus):")
    print(f"  RAM  ~{proj['ram_bytes']/1e6:8.0f} MB  ({proj['ram_pct']:5.1f}% of 1GB)  "
          f"{'PASS' if proj['ram_pct'] < GATE_PCT else 'FAIL'} (<{GATE_PCT}%)")
    print(f"  Disk ~{proj['disk_bytes']/1e6:8.0f} MB  ({proj['disk_pct']:5.1f}% of 4GB)  "
          f"{'PASS' if proj['disk_pct'] < GATE_PCT else 'FAIL'} (<{GATE_PCT}%)")
    if stale:
        print(f"\n⚠️  Drop before ingest to reclaim space: {', '.join(stale)}")

    os.makedirs("data/eval_reports", exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with open(f"data/eval_reports/capacity_{ts}.json", "w") as f:
        json.dump({"sizes": sizes, "projection": proj, "target_points": target_points}, f, indent=2)
    print(f"\nWrote data/eval_reports/capacity_{ts}.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_measure_capacity.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run it live (read-only) to capture the v3 baseline**

Run: `.venv/bin/python scripts/measure_capacity.py`
Expected: a table of all 5 collections with bytes/point, a projection that shows RAM/Disk **PASS (<70%)**, and the stale-collection warning. A `data/eval_reports/capacity_*.json` is written.

- [ ] **Step 6: Commit**

```bash
git add scripts/measure_capacity.py tests/test_measure_capacity.py data/eval_reports/capacity_*.json
git commit -m "feat(M2.1): capacity gate from Qdrant telemetry (bytes/point + 600 projection)"
```

---

### Task 2 (B1): Corpus selector — sector map + candidate filter

Pure, testable core of `select_corpus_600.py`: classify a dataset row into one of the 6 sectors, estimate its chunk count, and decide if it's a candidate. No I/O yet.

**Files:**
- Create: `scripts/select_corpus_600.py`
- Test: `tests/test_select_corpus_600.py`

**Interfaces:**
- Consumes: `scripts/select_corpus.py::CORPUS` (the spine list of `(id, sector)`).
- Produces:
  - `SECTOR_KEYWORDS: dict[str, set[str]]` (sector → legal_sectors tokens)
  - `HNGD_TITLE_HINTS: set[str]`
  - `RELEVANT_TYPES: set[str]`
  - `SPINE_IDS: set[int]`, `SPINE_SECTOR: dict[int, str]`
  - `sector_of(legal_sectors: str | None, title: str) -> str | None`
  - `estimate_chunks(content: str) -> int`
  - `is_candidate(row: dict) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_select_corpus_600.py
from scripts.select_corpus_600 import (
    sector_of, estimate_chunks, is_candidate, SPINE_IDS, RELEVANT_TYPES,
)

def test_sector_of_maps_token():
    assert sector_of("Doanh nghiệp, Đầu tư", "Nghị định về đăng ký doanh nghiệp") == "doanh_nghiep"
    assert sector_of("Bất động sản", "Nghị định quy định về giá đất") == "dat_dai"

def test_sector_of_hngd_via_title_when_no_token():
    # HNGĐ has no dedicated legal_sectors token → fall back to title hints
    assert sector_of("Văn hóa - Xã hội", "Luật Phòng, chống bạo lực gia đình") == "hngd"

def test_sector_of_returns_none_when_out_of_scope():
    assert sector_of("Giao thông - Vận tải", "Luật Giao thông đường bộ") is None

def test_estimate_chunks_uses_700_char_rule():
    assert estimate_chunks("x" * 6999) == 9    # 6999 // 700 = 9
    assert estimate_chunks("") == 1            # floor of 1

def test_is_candidate_rejects_spine_and_noise_types():
    spine_id = next(iter(SPINE_IDS))
    assert is_candidate({"id": spine_id, "legal_type": "Luật",
                         "legal_sectors": "Doanh nghiệp", "title": "x",
                         "content": "y" * 1000}) is False       # already in spine
    assert is_candidate({"id": -1, "legal_type": "Công văn",
                         "legal_sectors": "Doanh nghiệp", "title": "x",
                         "content": "y" * 1000}) is False       # noise type
    assert is_candidate({"id": -1, "legal_type": "Thông tư",
                         "legal_sectors": "Doanh nghiệp",
                         "title": "Thông tư hướng dẫn Luật Doanh nghiệp",
                         "content": "y" * 5000}) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_select_corpus_600.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.select_corpus_600'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/select_corpus_600.py  (top of file — core only; B2/B3/B4 append)
"""M2.2 prep — build the ~600-doc corpus by deepening the same 6 sectors.

Spine (100 hand-curated ids in scripts/select_corpus.py) is locked in.
~500 candidates are auto-generated (relevant legal_type + sector match +
content guard), budgeted by estimated chunk count with max/min <= ~3x
across sectors (frequency-bias control, spec B.5), exported to a review
table, then (after user prune) frozen into corpus_600docs.parquet.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.constants import MIN_CONTENT_LENGTH
from scripts.select_corpus import CORPUS  # spine

SPINE_SECTOR = {doc_id: sector for doc_id, sector in CORPUS}
SPINE_IDS = set(SPINE_SECTOR)

RELEVANT_TYPES = {"Luật", "Bộ luật", "Nghị định", "Thông tư",
                  "Thông tư liên tịch", "Văn bản hợp nhất"}

# legal_sectors tokens (comma-separated, noisy) → our 6 curated sectors.
SECTOR_KEYWORDS = {
    "hinh_su":      {"Trách nhiệm hình sự", "Thủ tục Tố tụng", "Vi phạm hành chính"},
    "dan_su":       {"Quyền dân sự", "Sở hữu trí tuệ", "Dịch vụ pháp lý"},
    "lao_dong":     {"Lao động - Tiền lương", "Bảo hiểm"},
    "doanh_nghiep": {"Doanh nghiệp", "Đầu tư", "Thương mại", "Chứng khoán",
                     "Kế toán - Kiểm toán"},
    "dat_dai":      {"Bất động sản", "Tài nguyên - Môi trường"},
    "hngd":         set(),   # no dedicated token → title hints only
}
# Priority order when a doc matches several sectors' tokens (most specific first).
_SECTOR_ORDER = ["hinh_su", "hngd", "dat_dai", "lao_dong", "doanh_nghiep", "dan_su"]
HNGD_TITLE_HINTS = {"hôn nhân", "gia đình", "trẻ em", "nuôi con nuôi", "bình đẳng giới"}


def _tokens(legal_sectors):
    return {t.strip() for t in (legal_sectors or "").split(",") if t.strip()}


def sector_of(legal_sectors, title):
    """Return one of the 6 sectors, or None if out of scope."""
    title_l = (title or "").lower()
    if any(h in title_l for h in HNGD_TITLE_HINTS):
        return "hngd"
    toks = _tokens(legal_sectors)
    for sector in _SECTOR_ORDER:
        if toks & SECTOR_KEYWORDS[sector]:
            return sector
    return None


def estimate_chunks(content):
    """Cheap chunk estimate (v3: ~700 chars/chunk). Floor of 1."""
    return max(1, len(content or "") // 700)


def is_candidate(row):
    if row.get("id") in SPINE_IDS:
        return False
    if row.get("legal_type") not in RELEVANT_TYPES:
        return False
    content = row.get("content")
    if not isinstance(content, str) or len(content) < MIN_CONTENT_LENGTH:
        return False
    return sector_of(row.get("legal_sectors"), row.get("title")) is not None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_select_corpus_600.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/select_corpus_600.py tests/test_select_corpus_600.py
git commit -m "feat(M2.2): corpus_600 selector core — sector map + candidate filter"
```

---

### Task 3 (B2): Chunk-budget allocation (bounded ratio ≤3×)

Add the frequency-bias control: rank candidates per sector and greedily allocate by feeding the currently-smallest sector, capped so `max/min ≤ ratio_cap`. Consolidated docs (Văn bản hợp nhất) and spine-anchored docs rank first to cut near-duplicate density.

**Files:**
- Modify: `scripts/select_corpus_600.py`
- Test: `tests/test_select_corpus_600.py` (add cases)

**Interfaces:**
- Consumes: `estimate_chunks` (Task B1).
- Produces:
  - `rank_key(cand: dict) -> tuple` (sort key; higher = picked earlier via `reverse=True`)
  - `allocate_budget(ranked_by_sector: dict[str, list[dict]], spine_chunks: dict[str, int], target_added_chunks: int, ratio_cap: float = 3.0) -> tuple[dict[str, list[dict]], dict[str, int]]` → `(picks_by_sector, totals_by_sector)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_select_corpus_600.py  (append)
from scripts.select_corpus_600 import allocate_budget

def _cand(cid, chunks):
    return {"id": cid, "est_chunks": chunks, "legal_type": "Thông tư",
            "is_consolidated": False, "anchored": False, "year": 2020}

def test_allocate_budget_keeps_ratio_within_cap():
    ranked = {s: [_cand(f"{s}{i}", 100) for i in range(50)]
              for s in ["hinh_su", "dan_su", "lao_dong", "doanh_nghiep", "dat_dai", "hngd"]}
    spine = {"hinh_su": 2800, "dan_su": 3300, "lao_dong": 1900,
             "doanh_nghiep": 4100, "dat_dai": 2200, "hngd": 1000}
    picks, totals = allocate_budget(ranked, spine, target_added_chunks=20000, ratio_cap=3.0)
    assert max(totals.values()) / min(totals.values()) <= 3.0 + 1e-9
    # small sector (hngd) must have grown the most
    assert len(picks["hngd"]) >= len(picks["doanh_nghiep"])

def test_allocate_budget_stops_when_candidates_exhausted():
    ranked = {s: [_cand(f"{s}{i}", 100) for i in range(2)] for s in ["hinh_su", "hngd"]}
    picks, totals = allocate_budget(ranked, {"hinh_su": 500, "hngd": 500},
                                    target_added_chunks=99999, ratio_cap=3.0)
    assert all(len(picks[s]) <= 2 for s in picks)   # never exceed available
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_select_corpus_600.py -k allocate_budget -v`
Expected: FAIL with `ImportError: cannot import name 'allocate_budget'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/select_corpus_600.py  (append)

def rank_key(cand):
    """Higher sorts first (use sorted(..., key=rank_key, reverse=True)):
    spine-anchored > consolidated > newer > longer-but-not-absurd."""
    return (
        1 if cand.get("anchored") else 0,
        1 if cand.get("is_consolidated") else 0,
        cand.get("year") or 0,
        -abs(cand.get("est_chunks", 0) - 60),   # prefer ~mid-size docs
    )


def allocate_budget(ranked_by_sector, spine_chunks, target_added_chunks, ratio_cap=3.0):
    """Greedily add the next-best candidate to the currently-smallest sector,
    never letting any sector exceed ratio_cap * current_min (unless it IS the
    min). Stops at target_added_chunks or when no sector can add feasibly."""
    totals = {s: spine_chunks.get(s, 0) for s in ranked_by_sector}
    picks = {s: [] for s in ranked_by_sector}
    idx = {s: 0 for s in ranked_by_sector}
    added = 0
    while added < target_added_chunks:
        cur_min = min(totals.values())
        feasible = []
        for s, cands in ranked_by_sector.items():
            if idx[s] >= len(cands):
                continue
            nxt = cands[idx[s]]
            over = totals[s] + nxt["est_chunks"] > ratio_cap * cur_min
            if over and totals[s] > cur_min:
                continue   # would break ratio and this sector isn't the min
            feasible.append(s)
        if not feasible:
            break
        s = min(feasible, key=lambda x: totals[x])
        nxt = ranked_by_sector[s][idx[s]]
        picks[s].append(nxt)
        totals[s] += nxt["est_chunks"]
        idx[s] += 1
        added += nxt["est_chunks"]
    return picks, totals
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_select_corpus_600.py -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add scripts/select_corpus_600.py tests/test_select_corpus_600.py
git commit -m "feat(M2.2): chunk-budget allocation with bounded ratio (frequency-bias control)"
```

---

### Task 4 (B3): Generate the candidate review table → USER REVIEW GATE

Wire the core into a CLI that scans the real 518K-row parquet, builds candidates, allocates the budget, and writes a reviewable table grouped by sector. **This task ends at the review gate** — the user prunes the table before B4.

**Files:**
- Modify: `scripts/select_corpus_600.py` (add `scan_candidates`, `main`/CLI)

**Interfaces:**
- Consumes: `is_candidate`, `sector_of`, `estimate_chunks`, `rank_key`, `allocate_budget`, `SPINE_SECTOR`.
- Produces:
  - `scan_candidates(merged_path: str) -> list[dict]` (row-group streaming, like `select_corpus.py::extract_corpus`)
  - `spine_chunks_from_parquet(corpus_100_path: str) -> dict[str, int]`
  - CLI writing `data/raw/corpus_600_candidates.csv` + a per-sector summary to stdout.

- [ ] **Step 1: Implement scan + CLI**

```python
# scripts/select_corpus_600.py  (append)
import csv, re
import pyarrow.parquet as pq
import click

MERGED_PATH = "data/raw/legal_docs_merged.parquet"
SPINE_PARQUET = "data/raw/corpus_100docs.parquet"
CANDIDATES_CSV = "data/raw/corpus_600_candidates.csv"

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _year(row):
    d = row.get("issuance_date") or ""
    m = re.search(r"/(\d{4})$", d.strip()) or _YEAR_RE.search(row.get("title", ""))
    return int(m.group(1) if m.lastindex else m.group(0)) if m else 0


def _anchored(title):
    return "hướng dẫn" in (title or "").lower()


def scan_candidates(merged_path=MERGED_PATH):
    pf = pq.ParquetFile(merged_path)
    cands = []
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i).to_pandas()
        for _, r in df.iterrows():
            row = r.to_dict()
            if not is_candidate(row):
                continue
            cands.append({
                "id": int(row["id"]),
                "sector": sector_of(row.get("legal_sectors"), row.get("title")),
                "legal_type": row["legal_type"],
                "title": row.get("title", ""),
                "year": _year(row),
                "est_chunks": estimate_chunks(row.get("content")),
                "is_consolidated": row["legal_type"] == "Văn bản hợp nhất",
                "anchored": _anchored(row.get("title")),
            })
    return cands


def spine_chunks_from_parquet(corpus_100_path=SPINE_PARQUET):
    df = pq.read_table(corpus_100_path, columns=["sector", "content"]).to_pandas()
    out = {}
    for _, r in df.iterrows():
        out[r["sector"]] = out.get(r["sector"], 0) + estimate_chunks(r["content"])
    return out


@click.command()
@click.option("--target-added-chunks", type=int, default=115000,
              help="Approx chunks to add on top of the ~23.7k spine (tune to hit ~600 docs).")
@click.option("--ratio-cap", type=float, default=3.0)
def main(target_added_chunks, ratio_cap):
    cands = scan_candidates()
    by_sector = {s: [] for s in SECTOR_KEYWORDS}
    for c in cands:
        by_sector[c["sector"]].append(c)
    for s in by_sector:
        by_sector[s].sort(key=rank_key, reverse=True)
    spine = spine_chunks_from_parquet()
    picks, totals = allocate_budget(by_sector, spine, target_added_chunks, ratio_cap)

    with open(CANDIDATES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keep", "id", "sector", "legal_type", "year", "est_chunks",
                    "anchored", "is_consolidated", "title"])
        for s in _SECTOR_ORDER:
            for c in picks[s]:
                w.writerow(["1", c["id"], s, c["legal_type"], c["year"], c["est_chunks"],
                            int(c["anchored"]), int(c["is_consolidated"]), c["title"]])

    print(f"{'sector':<14}{'spine ch':>10}{'+added ch':>10}{'total ch':>10}{'+docs':>7}")
    for s in _SECTOR_ORDER:
        print(f"{s:<14}{spine.get(s,0):>10,}{totals[s]-spine.get(s,0):>10,}"
              f"{totals[s]:>10,}{len(picks[s]):>7}")
    ratio = max(totals.values()) / min(totals.values())
    print(f"\nmax/min chunk ratio = {ratio:.2f} (cap {ratio_cap})  "
          f"{'OK' if ratio <= ratio_cap + 1e-9 else 'OVER'}")
    print(f"total added docs = {sum(len(picks[s]) for s in picks)}  "
          f"(+100 spine = {100 + sum(len(picks[s]) for s in picks)})")
    print(f"\nReview + prune: {CANDIDATES_CSV}  (set keep=0 to drop a row)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the real dataset**

Run: `.venv/bin/python scripts/select_corpus_600.py`
Expected: a per-sector table with `max/min chunk ratio ≤ 3.00 OK`, total docs ≈ 500 (≈600 with spine), and `data/raw/corpus_600_candidates.csv` written. If total docs is far from ~500, re-run with `--target-added-chunks` tuned.

- [ ] **Step 3: Commit the generator (not the CSV yet)**

```bash
git add scripts/select_corpus_600.py
git commit -m "feat(M2.2): candidate review-table generator over the 518K-row pool"
```

- [ ] **Step 4: USER REVIEW GATE — hand the CSV to the user**

Present `data/raw/corpus_600_candidates.csv` to the user (open it, or summarize per-sector). The user sets `keep=0` on any doc to drop (off-topic, dead law whose replacement is absent, duplicate). **Do not proceed to B4 until the user confirms the pruned CSV.** Ask explicitly: "Approve this candidate list, or edit `keep` flags first?"

---

### Task 5 (B4): Freeze approved IDs + build `corpus_600docs.parquet`

After the user approves the CSV, read the kept rows, union with the spine, and build the final parquet — reusing the proven row-group extraction from `select_corpus.py`.

**Files:**
- Modify: `scripts/select_corpus_600.py` (add `build_corpus_600`)

**Interfaces:**
- Consumes: approved `data/raw/corpus_600_candidates.csv`, `SPINE_SECTOR`, `scripts/select_corpus.py::extract_corpus` pattern.
- Produces: `data/raw/corpus_600docs.parquet` with columns matching `corpus_100docs.parquet` (+ `sector`).

- [ ] **Step 1: Implement the builder**

```python
# scripts/select_corpus_600.py  (append)
import pandas as pd

OUTPUT_PATH = "data/raw/corpus_600docs.parquet"


def build_corpus_600(candidates_csv=CANDIDATES_CSV, merged_path=MERGED_PATH,
                     output_path=OUTPUT_PATH):
    wanted = dict(SPINE_SECTOR)   # spine locked in
    with open(candidates_csv) as f:
        for r in csv.DictReader(f):
            if r["keep"].strip() == "1":
                wanted[int(r["id"])] = r["sector"]

    pf = pq.ParquetFile(merged_path)
    parts = []
    for i in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(i).to_pandas()
        hit = t[t["id"].isin(wanted)]
        if len(hit):
            parts.append(hit)
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="id")

    missing = set(wanted) - set(df["id"])
    if missing:
        raise SystemExit(f"NOT FOUND {len(missing)} ids in dataset: {sorted(missing)}")
    assert SPINE_IDS <= set(df["id"]), "spine doc dropped!"

    df["sector"] = df["id"].map(wanted)
    df = df.sort_values("sector").reset_index(drop=True)
    df.to_parquet(output_path)

    print(f"Wrote {output_path}: {len(df)} docs")
    est = df.groupby("sector")["content"].apply(lambda s: sum(estimate_chunks(c) for c in s))
    for sector, g in df.groupby("sector"):
        n_tt = (g["legal_type"] == "Thông tư").sum()
        print(f"  {sector:<14}{len(g):>4} docs  {n_tt:>3} Thông tư  ~{est[sector]:>6,} chunks")
    return df
```

Add a `--build` flag to `main` that calls `build_corpus_600()` instead of the scan, so both modes live in one CLI:

```python
# in main(): add  @click.option("--build", is_flag=True)  and at the top of main():
#     if build:
#         build_corpus_600(); return
```

- [ ] **Step 2: Build the parquet from the approved CSV**

Run: `.venv/bin/python scripts/select_corpus_600.py --build`
Expected: `Wrote data/raw/corpus_600docs.parquet: ~600 docs` + a per-sector summary listing Thông tư counts and estimated chunks. No "spine doc dropped" or "NOT FOUND".

- [ ] **Step 3: Commit**

```bash
git add scripts/select_corpus_600.py
git commit -m "feat(M2.2): freeze approved ids + build corpus_600docs.parquet"
```

---

### Task 6 (C1): Chunker alignment + per-sector chunk balance on the 600 corpus

Extend `verify_chunker_alignment.py` to also report per-sector chunk totals + max/min ratio when a `sector` column is present, then run it on the 600 corpus. This is the gate that catches **Thông tư structure breakage** and confirms the **≤3× balance** on real chunks.

**Files:**
- Modify: `scripts/verify_chunker_alignment.py`

**Interfaces:**
- Consumes: `corpus_600docs.parquet`; existing `run(data_path, quiet)`, `load_documents`, and the per-doc chunk lists it already computes.
- Produces: `per_sector_chunk_report(rows: list[dict], chunk_counts: dict[int, int]) -> dict[str, int]` and printed ratio line.

- [ ] **Step 1: Add the per-sector aggregation (pure function + wire into run)**

```python
# scripts/verify_chunker_alignment.py  (add function)
def per_sector_chunk_report(rows, chunk_counts):
    """rows: list of loaded doc dicts (must carry 'id' and 'sector').
    chunk_counts: {doc_id: n_chunks}. Returns {sector: total_chunks}."""
    totals = {}
    for r in rows:
        sec = r.get("sector")
        if sec is None:
            continue
        totals[sec] = totals.get(sec, 0) + chunk_counts.get(r["id"], 0)
    return totals
```

Wire it into `run()`: accumulate `chunk_counts[doc_id] = len(chunks)` in the existing per-doc loop, and after the loop, if any row has a `sector`, print:

```python
    sectors_present = any(r.get("sector") is not None for r in documents)
    if sectors_present:
        totals = per_sector_chunk_report(documents, chunk_counts)
        print("\n--- per-sector chunk balance ---")
        for s, n in sorted(totals.items()):
            print(f"  {s:<14}{n:>8,}")
        ratio = max(totals.values()) / min(totals.values())
        print(f"  max/min = {ratio:.2f}  {'OK (<=3x)' if ratio <= 3.0 else 'OVER 3x — rebalance corpus'}")
```

(`load_documents` reads the parquet with pandas; ensure it keeps the `sector` and `id` columns in the dicts it returns — add them to the projection if it selects columns.)

- [ ] **Step 2: Run the alignment gate on the 600 corpus**

Run: `.venv/bin/python scripts/verify_chunker_alignment.py --data-path data/raw/corpus_600docs.parquet --quiet`
Expected: `GATE PASS`, `WRONG_LABEL=0` overall, and a per-sector chunk table with `max/min <= 3x OK`.

- [ ] **Step 3: If Thông tư breaks alignment (WRONG_LABEL > 0)**

Diagnose exactly as M1.2b (`docs/notes/corpus-100docs-selection-2026-07-10.md` §"Gate alignment"): inspect the printed `WRONG_LABEL examples`, widen the chunker's `ARTICLE_PATTERN`/header regex to cover the Thông tư structure (e.g. `Điều`/`Mục` numbering, quoted headers), keeping hierarchical strict for base laws. Re-run Step 2 until `WRONG_LABEL=0`. Write a short `docs/notes/chunker-thongtu-diagnosis-2026-07-13.md` if a fix was needed. If the balance line shows `OVER 3x`, adjust the candidate CSV (`keep` flags) and rebuild (B4) before continuing.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_chunker_alignment.py docs/notes/chunker-thongtu-diagnosis-2026-07-13.md 2>/dev/null; \
git add scripts/verify_chunker_alignment.py
git commit -m "feat(M2.7): per-sector chunk-balance report in alignment gate; validate 600 corpus"
```

---

### Task 7 (C2): Eval expansion — Thông tư-targeted questions (2.7)

Author ~30–50 new questions targeting Thông tư content + newly-added docs (5–10 per sector: citation-lookup, penalty amount, procedure/forms, comparison), keep every OOS anchor, and validate grounding against the 600 parquet.

**Files:**
- Create: `data/eval/eval_set_sectors_v2.json` (= v1's 51 questions + new ones)
- Modify: `scripts/eval.py` (point `SECTORS_SET` at v2; keep v1 file for the M1 baseline)

**Interfaces:**
- Consumes: `data/raw/corpus_600docs.parquet`, `scripts/validate_eval_set.py`.
- Produces: v2 eval set; `eval.py::SECTORS_SET = "data/eval/eval_set_sectors_v2.json"`.

- [ ] **Step 1: Seed v2 from v1**

```bash
cp data/eval/eval_set_sectors_v1.json data/eval/eval_set_sectors_v2.json
```

- [ ] **Step 2: Author new questions (per-sector, grounded in the parquet)**

For each of the 6 sectors, pick added Thông tư/Nghị định from `corpus_600docs.parquet`, read their articles, and append question objects to `questions` in v2. Each MUST follow this exact schema (from `ds-001`), with `expected_articles.doc_id` present in the 600 parquet and `article` a real `Điều N` header in that doc:

```json
{
  "id": "ld-401",
  "question": "Thông tư ... quy định mức xử phạt đối với hành vi ... là bao nhiêu?",
  "type": "single-hop",
  "difficulty": "medium",
  "legal_sector": "lao_dong",
  "expected_behavior": "answer",
  "ground_truth": "<1-3 sentence answer paraphrased from the article>",
  "expected_articles": [{"doc_id": 479312, "article": "Điều 18"}]
}
```

Distribution: ≥5 per sector, mixing `type` ∈ {citation-lookup, single-hop, comparison}. Add 1–2 new OOS traps under the general set only if they target a **non-covered** Thông tư domain (thuế/giao thông) — do not add in-scope refusals. Use ids `<sector-prefix>-4xx` to avoid collisions with existing 001–301 ranges.

- [ ] **Step 3: Point eval.py at v2**

```python
# scripts/eval.py  — change the one constant
SECTORS_SET = "data/eval/eval_set_sectors_v2.json"  # M2.7 — v1 + Thông tư questions
```

- [ ] **Step 4: Validate grounding (the gate — must be 0 errors)**

Run: `.venv/bin/python scripts/validate_eval_set.py --corpus-parquet data/raw/corpus_600docs.parquet`
Expected: `eval_set_sectors_v2` prints stats and **0 grounding errors**. Every `expected_articles` entry resolves to a real article header in its `doc_id`. Fix any flagged id (stale label) before continuing.

- [ ] **Step 5: Commit**

```bash
git add data/eval/eval_set_sectors_v2.json scripts/eval.py
git commit -m "feat(M2.7): eval set v2 — Thông tư-targeted questions, grounded on 600 corpus"
```

---

### Task 8 (C3): Cross-sector leakage@k diagnostic (frequency bias)

Add the frequency-bias measurement to `run_tier1`: for each question with a known `legal_sector`, what fraction of its top-k hits belong to a **different** sector, mapped via `doc_id → sector` from the corpus parquet. First-stage recall is already measured (dense/hybrid/routed pre-rerank); this adds the leakage lens.

**Files:**
- Modify: `scripts/eval.py`
- Test: `tests/test_eval_metrics.py` (add)

**Interfaces:**
- Consumes: `run_tier1` rows (each hit dict already carries `document_id`), a `doc_sector: dict[int, str]` loaded from `corpus_600docs.parquet`.
- Produces: `cross_sector_leakage(hit_doc_ids: list[int], query_sector: str, doc_sector: dict[int, str], k: int) -> float`; per-sector `leakage@k` in the tier-1 aggregate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_metrics.py  (append)
from scripts.eval import cross_sector_leakage

def test_cross_sector_leakage_counts_foreign_docs():
    doc_sector = {10: "lao_dong", 11: "lao_dong", 20: "dat_dai", 21: "hinh_su"}
    # top-4 for a lao_dong query: two in-sector, two foreign → 0.5
    assert cross_sector_leakage([10, 11, 20, 21], "lao_dong", doc_sector, k=4) == 0.5
    # unknown doc ids are ignored in the denominator
    assert cross_sector_leakage([10, 999], "lao_dong", doc_sector, k=4) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -k leakage -v`
Expected: FAIL with `ImportError: cannot import name 'cross_sector_leakage'`

- [ ] **Step 3: Implement + wire into run_tier1 aggregation**

```python
# scripts/eval.py  (add function)
def cross_sector_leakage(hit_doc_ids, query_sector, doc_sector, k):
    """Fraction of the top-k hits whose doc's sector != query_sector.
    Hits whose doc_id is not in doc_sector (out-of-corpus) are ignored."""
    known = [doc_sector[d] for d in hit_doc_ids[:k] if d in doc_sector]
    if not known:
        return 0.0
    foreign = sum(1 for s in known if s != query_sector)
    return foreign / len(known)
```

Load `doc_sector` once in `run_tier1` (guard: only if the parquet exists) and, in the per-row loop where `mode == "routed"` and `q.get("legal_sector")` is set, record `row["leakage@5"] = cross_sector_leakage([h["document_id"] for h in hit_dicts], q["legal_sector"], doc_sector, 5)`. In `_aggregate_tier1`'s `by_sector` block, add `"leakage@5": _mean([r["leakage@5"] for r in srows if "leakage@5" in r])`. Print it in the per-sector table.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_eval_metrics.py -v`
Expected: PASS (leakage tests + existing metric tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval.py tests/test_eval_metrics.py
git commit -m "feat(M2.7): cross-sector leakage@k diagnostic in tier-1 eval (frequency bias)"
```

---

### Task 9 (D1): Standalone enrichment runner + Kaggle deps

Phase E defaults to running enrichment on Kaggle right after ingest. Today enrichment is only reachable via the FastAPI endpoint; provide a CLI runner and the extra deps (`networkx`, absent from `requirements-ingest.txt`).

**Files:**
- Create: `scripts/run_enrichment.py`
- Create: `requirements-enrich.txt`

**Interfaces:**
- Consumes: `src.pipeline.enricher.EnrichmentPipeline(vector_store, metadata_store).run()`, `LegalVectorStore(collection_name=...)`, `MetadataStore(data_dir=...)`.
- Produces: CLI `--collection` / `--metadata-dir`; prints the `EnrichmentResult` summary; leaves `store.db` + JSON in `--metadata-dir`.

- [ ] **Step 1: Write the runner**

```python
# scripts/run_enrichment.py
"""M2 Phase E — run Phase-2 enrichment on a collection (Kaggle-friendly).

Reads chunks FROM Qdrant, builds graph/timeline/amendment into the SQLite
MetadataStore, and backfills is_in_effect into the collection's payloads.
Run this in the Kaggle notebook right after ingest (avoids scrolling ~140K
chunks over a home connection), then download the metadata dir back to repo.

Usage:
    python scripts/run_enrichment.py --collection luat_law_ctx_v4
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()
import click
from src.db.qdrant_client import LegalVectorStore
from src.data.metadata_store import MetadataStore
from src.pipeline.enricher import EnrichmentPipeline


@click.command()
@click.option("--collection", required=True, help="Qdrant collection to enrich (e.g. luat_law_ctx_v4)")
@click.option("--metadata-dir", default="data/metadata")
def main(collection, metadata_dir):
    store = LegalVectorStore(collection_name=collection)
    meta = MetadataStore(data_dir=metadata_dir)
    result = EnrichmentPipeline(vector_store=store, metadata_store=meta).run()
    print(f"docs_processed={result.docs_processed}/{result.total_docs}  "
          f"base={result.base_laws} amend={result.amendment_laws}  "
          f"graph={result.graph_nodes}n/{result.graph_edges}e  "
          f"in_effect={result.currently_in_effect}  "
          f"elapsed={result.elapsed_seconds:.0f}s")
    for e in result.errors:
        print("ERROR:", e)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `requirements-enrich.txt`**

```
# Extra deps to run scripts/run_enrichment.py on Kaggle (on top of
# requirements-ingest.txt). Enrichment needs the knowledge-graph library;
# it does NOT embed, so no FlagEmbedding here.
-r requirements-ingest.txt

# Knowledge graph construction/traversal — src/data/graph/graph_builder.py
networkx>=3.0
```

- [ ] **Step 3: Smoke-check it imports and shows help**

Run: `.venv/bin/python scripts/run_enrichment.py --help`
Expected: click help text listing `--collection` and `--metadata-dir` (no import errors).

- [ ] **Step 4: Commit**

```bash
git add scripts/run_enrichment.py requirements-enrich.txt
git commit -m "feat(M2.7): standalone enrichment runner + requirements-enrich for Kaggle"
```

---

### Task 10 (D2): Update the Kaggle ingest guide for 600 docs

Extend `docs/notes/kaggle-ingest-guide.md` for the 600-doc run: new dataset, drop-stale-collections step, `luat_law_ctx_v4`, and a new enrichment section (run `run_enrichment.py`, then save the metadata dir back as a Kaggle Dataset output to commit to the repo).

**Files:**
- Modify: `docs/notes/kaggle-ingest-guide.md`

- [ ] **Step 1: Edit the guide**

Make these concrete changes:
1. Retitle to cover 600 (M2.2); note the data file is now `corpus_600docs.parquet`.
2. Cell 4: attach `data/raw/corpus_600docs.parquet` as the dataset; update `DATA_PATH`.
3. **New Cell 4b — drop stale collections** (reclaim free-tier space before ingest):
   ```python
   from qdrant_client import QdrantClient
   c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
   for name in ["luat_law", "luat_law_ctx", "vietnamese_laws"]:
       c.delete_collection(name); print("dropped", name)
   ```
4. Cell 5: change `--collection luat_law_ctx_v3` → `--collection luat_law_ctx_v4` (keep `--quantized --device cuda --embed-batch-size 128 --resume`). Note `--quantized` is mandatory on this first run.
5. **New Cell 7 — enrichment** (Phase E, same session):
   ```python
   !pip install -q -r requirements-enrich.txt
   !python scripts/run_enrichment.py --collection luat_law_ctx_v4
   ```
   Then zip `data/metadata/` (`store.db`, `*.json`) as a Kaggle Dataset **output** and download it, to commit into the repo so local MetadataStore + eval use the enriched data.
6. Cell 6 (verify): change collection name to `luat_law_ctx_v4`.

- [ ] **Step 2: Verify the guide reads coherently**

Read the full file top-to-bottom; confirm every cell references `corpus_600docs.parquet` / `luat_law_ctx_v4` consistently and the drop-stale and enrichment cells are present.

- [ ] **Step 3: Commit**

```bash
git add docs/notes/kaggle-ingest-guide.md
git commit -m "docs(M2.2): Kaggle guide for 600-doc ingest — drop stale, v4, enrichment cell"
```

---

## Phase E (post-ingest) — separate follow-up plan

Not in this plan (depends on the user's Kaggle ingest). After v4 exists: run `run_enrichment.py`, re-run `measure_capacity.py` on v4 (final <70% gate), Tier-1 per-sector recall + leakage diagnostic on v4 (≤5pt drop gate), Tier-2 + `calibrate_threshold.py` (new REFUSAL_THRESHOLD), flip `DEFAULT_COLLECTION_NAME` → `luat_law_ctx_v4` + threshold in one commit, and write the 6→100→600 scale-journey table. Spec Phase E has the details.

---

## Self-Review

**Spec coverage:**
- 2.1 capacity gate → Task A1 (+ re-run in Phase E). ✓
- 2.2 corpus selection/build → Tasks B1–B4 (spine lock, sector map, ≤3× budget, review gate, parquet). ✓
- Frequency bias (spec B.5) → B2 budget invariant + C1 real-chunk ratio + C3 leakage diagnostic. ✓
- Thông tư chunker risk (spec C.1) → C1 gate + contingent fix. ✓
- Eval expansion + grounding (spec C.3) → C2. ✓
- Kaggle handoff + enrichment-on-Kaggle default (spec D, E.1) → D1 (runner + deps) + D2 (guide). ✓
- Post-ingest recalibrate/gates (spec E) → explicitly deferred to a follow-up plan. ✓

**Placeholder scan:** No TBD/TODO. Contingent chunker fix (C1 Step 3) references the concrete M1.2b procedure, not a vague "handle edge cases." Eval authoring (C2) gives the exact schema + a worked example + validation gate rather than pre-writing 40 legal answers (which require reading real articles).

**Type consistency:** `estimate_chunks`, `sector_of`, `is_candidate`, `rank_key`, `allocate_budget`, `scan_candidates`, `build_corpus_600`, `cross_sector_leakage`, `parse_collection_sizes`, `project_capacity`, `per_sector_chunk_report` — each defined once and consumed with matching signatures. `est_chunks` key name consistent across B2/B3. `SECTORS_SET` constant name matches `eval.py`.

---

## Execution Handoff

See the offer after this plan is saved.
