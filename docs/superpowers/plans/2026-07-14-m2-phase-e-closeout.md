# M2 Phase E — Post-ingest Close-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the 600-doc `luat_law_ctx_v4` collection against the M2 gates, recalibrate the refusal threshold, and promote v4 to the default — closing out M2.

**Architecture:** Phase E is a **runbook**, not a feature build. The v4 collection (60,958 pts, enriched with `is_in_effect` + `sector`) and its local `store.db` (605 docs) already exist. Each task runs an existing script against v4, checks a gate, and records the real numbers. Only Task 14 changes product code (a two-line config flip). Gates that FAIL stop the plan for a user decision — no auto-remediation.

**Tech Stack:** Python (click CLIs), Qdrant Cloud (v4 + v3 baseline), BGE-M3 embeddings (local CPU for query embedding only), LiteLLM (free-tier Groq/Gemini for Tier-2), pytest.

## Global Constraints

- **Gates (verbatim from spec §4):**
  - Capacity: Qdrant `< 70%` free-tier ceiling (RAM 1GB + disk 4GB).
  - Retrieval recall: `recall@5` drops `≤ 5 points` vs M1, **reported per-sector**.
  - Refusal: `refusal recall = 1.0` on the OOS set.
  - Threshold: new threshold ships with a calibration table + margin.
- **Branch:** all work on a dedicated branch off `main` (e.g. `m2-phase-e-closeout`). Never commit Phase E on `main`.
- **$0 rule:** Tier-2 runs on free-tier providers only (Groq/Gemini via LiteLLM); respect `--sleep` rate-limiting. No paid API usage.
- **Do NOT touch v3 or v2:** `luat_law_ctx_v3` (M1, 100 docs) and `luat_law_ctx_v2` (M0.5, 6 docs) stay as baselines. Never `--recreate`/delete them.
- **Enrichment (E.1) is already done** this session: `metadata_v4/store.db` (605 docs) copied to `data/metadata/store.db`; v4 payloads carry `is_in_effect`. No re-run needed.

## Preconditions (verified 2026-07-14, do not re-litigate)

- v4 reachable in Qdrant Cloud: `luat_law_ctx_v4` = 60,958 points; payloads have `is_in_effect`, `sector`, `document_id`.
- v3 baseline present: `luat_law_ctx_v3` = 23,727 points.
- `.env` has `QDRANT_URL`, `QDRANT_API_KEY`, and LLM keys (`GROQ_API_KEYS`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`).
- Local `data/metadata/store.db` = 605 docs (559 base + 46 amendment, 578 in-effect).
- Config to flip lives at: `src/data/constants.py:47` (`DEFAULT_COLLECTION_NAME = "luat_law_ctx_v3"`) and `src/rag/confidence.py:109` (`REFUSAL_THRESHOLD = float(os.getenv("CONFIDENCE_REFUSAL_THRESHOLD", "0.650"))`).

---

### Task 11 (E.2): Capacity re-measure on v4

**Files:**
- Run only: `scripts/measure_capacity.py` (no code change expected)
- Record: `.superpowers/sdd/progress.md`, artifact `data/eval_reports/capacity_<ts>.json` (auto-written by the script)

**Interfaces:**
- Consumes: live Qdrant `/telemetry` (v4 now real at 60,958 pts).
- Produces: recorded RAM% / disk% for v4 → feeds the M2 gate summary + scale-journey table (Task 15).

- [ ] **Step 1: Run the capacity measure against v4's real point count**

Run:
```bash
.venv/bin/python scripts/measure_capacity.py --target-points 60958
```
Expected: a per-collection table that now includes a `luat_law_ctx_v4` row (~60,958 pts), then a projection block printing `RAM ... PASS (<70.0%)` and `Disk ... PASS (<70.0%)` (napkin estimate ~10% RAM, ~6% disk).

- [ ] **Step 2: Gate check**

Confirm both `RAM` and `Disk` lines read `PASS`. If either reads `FAIL` → **STOP**, report to user (v4 exceeds free tier; corpus trim needed — do not proceed to flip).

- [ ] **Step 3: Record the real numbers**

Append v4 actual RAM% / disk% (and the `capacity_<ts>.json` filename) to `.superpowers/sdd/progress.md` under a new `## Phase E log` section. No commit yet (progress.md is gitignored under `.superpowers/`; verify with `git check-ignore .superpowers/sdd/progress.md`).

---

### Task 12 (E.3): Tier-1 retrieval + frequency-bias diagnostic on v4

**Files:**
- Run only: `scripts/eval.py` (no code change expected; leakage@5 + per-sector already wired in Tasks 6/8)
- Artifacts: `data/eval_reports/<ts>_eval.json` (auto-written per run)
- Record: `.superpowers/sdd/progress.md` Phase E log

**Interfaces:**
- Consumes: v4 (candidate) and v3 (baseline) collections; `data/eval/eval_set_sectors_v2.json` (83 q), `eval_set_criminal_v1.json`, `eval_set_general_v1.json`.
- Produces: per-sector `recall@5` (strict + lenient), `leakage@5`, MRR for v4 → gate result + scale-journey rung (Task 15).

- [ ] **Step 1: Run Tier-1 on v4 for all three sets**

Run (each writes a timestamped report to `data/eval_reports/`):
```bash
.venv/bin/python scripts/eval.py --set sectors  --collection luat_law_ctx_v4
.venv/bin/python scripts/eval.py --set criminal --collection luat_law_ctx_v4
.venv/bin/python scripts/eval.py --set general  --collection luat_law_ctx_v4
```
Expected: each prints an aggregate + a **per-sector** recall table (with a `leakage@5` column for the routed sectors run). No LLM calls (Tier-1 is retrieval-only). Note the report paths printed at the end.

- [ ] **Step 2: Establish the apples-to-apples M1 baseline**

The `sectors` set grew v1→v2 (Task 7). To isolate distractor-dilution (not corpus-coverage) run the **same v2 set against v3** and compare on the **shared v1 subset** (the 51 v1 questions are byte-identical inside v2, and their target docs are in both v3's 100-doc corpus and v4's superset):
```bash
.venv/bin/python scripts/eval.py --set sectors --collection luat_law_ctx_v3
```
Expected: a v3 per-sector recall table on the same questions. (New v2-only Thông tư questions target 600-corpus-only docs → v3 misses them by absence, so exclude them from the drop gate.)

- [ ] **Step 3: Gate check — per-sector recall@5 drop ≤ 5 points (v3→v4, v1-subset)**

For each sector, compute `recall@5(v3) − recall@5(v4)` on the shared v1-subset questions. Gate: every sector's drop `≤ 5 points`. **doanh_nghiep is the watch-item.** If any sector drops `> 5 points` → **STOP**, report the sector + magnitude to the user (spec risk table: trim that sector or investigate; do NOT jump to M3 retrieval-level mitigation). Also read the `leakage@5` column: high cross-sector leakage on a thin sector is a **watch-item to record**, not a blocker.

- [ ] **Step 4: Record**

Append to progress.md Phase E log: full v4 per-sector recall@5 (strict+lenient) for all three sets, the v3-baseline v1-subset numbers, the computed drops, leakage@5 observations, and the report filenames.

---

### Task 13 (E.4): Tier-2 answer/refusal eval + recalibrate threshold

**Files:**
- Run only: `scripts/eval.py --answers`, then `scripts/calibrate_threshold.py`
- Artifacts: `data/eval_reports/<ts>_eval.json` (Tier-2, with confidence dump) + `data/eval_reports/<ts>_calibration.json`
- Record: `.superpowers/sdd/progress.md` Phase E log; the new threshold value is carried into Task 14

**Interfaces:**
- Consumes: v4 collection; LLM via `AdaptiveRAGPipeline` (free-tier). `calibrate_threshold.py --reports` consumes the Tier-2 report JSON(s) containing per-question confidence scores.
- Produces: new `REFUSAL_THRESHOLD` value + calibration table + margin → Task 14.

- [ ] **Step 1: Run Tier-2 (answer/refusal, LLM calls) on v4**

This makes real (free-tier, rate-limited) LLM calls — slower, `--sleep` between questions. Run the answer-accuracy sets + the OOS/refusal set:
```bash
.venv/bin/python scripts/eval.py --answers --set sectors  --collection luat_law_ctx_v4 --sleep 2
.venv/bin/python scripts/eval.py --answers --set criminal --collection luat_law_ctx_v4 --sleep 2
.venv/bin/python scripts/eval.py --answers --set general  --collection luat_law_ctx_v4 --sleep 2
```
Expected: each prints refusal accuracy, false-refusal rate, citation accuracy, and writes a report with a confidence dump. If a provider rate-limits/errors mid-run, re-run that set with `--retry <partial_report>` (eval.py supports resume via `--retry`).

- [ ] **Step 2: Gate check — refusal recall = 1.0 on OOS**

Confirm the `general` (OOS/refusal) set reports `refusal recall = 1.0` (every out-of-scope question refused). Record `false_refusal_rate` explicitly. If refusal recall `< 1.0` → **STOP**, report which OOS questions were answered instead of refused (this is a correctness regression, not just a threshold tweak).

- [ ] **Step 3: Recalibrate the threshold on v4's confidence distribution**

Run calibrate over the Tier-2 report(s):
```bash
.venv/bin/python scripts/calibrate_threshold.py --reports data/eval_reports/<tier2_report_1>.json --reports data/eval_reports/<tier2_report_2>.json
```
Expected: prints a calibration table (threshold sweep with refusal/false-refusal at each cut) + a recommended `REFUSAL_THRESHOLD` for v4 + the margin, and writes `data/eval_reports/<ts>_calibration.json`.

- [ ] **Step 4: Gate check + record**

Confirm the new threshold has a calibration table + margin (spec gate). Record the new threshold value, the table, and the margin in progress.md Phase E log. **Do not edit config yet** — the flip is Task 14 (one commit).

---

### Task 14 (E.5): Flip default collection + threshold (one commit)

**Files:**
- Modify: `src/data/constants.py:47`
- Modify: `src/rag/confidence.py:109`
- Test: whichever test pins the collection/threshold (find with grep in Step 1)

**Interfaces:**
- Consumes: new threshold value from Task 13.
- Produces: v4 as the product default → any consumer calling `MetadataStore()` / `LegalVectorStore()` / the API now serves v4.

- [ ] **Step 1: Find tests that pin the old default/threshold**

Run:
```bash
grep -rn "luat_law_ctx_v3\|0.650\|0\.65\b\|CONFIDENCE_REFUSAL_THRESHOLD\|DEFAULT_COLLECTION_NAME" tests/ src/
```
Expected: list of any test asserting the v3 name or 0.650. These must be updated in the same commit so `make test` stays green.

- [ ] **Step 2: Flip the collection default**

Edit `src/data/constants.py:47`: `DEFAULT_COLLECTION_NAME: str = "luat_law_ctx_v3"` → `"luat_law_ctx_v4"`.

- [ ] **Step 3: Flip the threshold default**

Edit `src/rag/confidence.py:109`: change the default in `float(os.getenv("CONFIDENCE_REFUSAL_THRESHOLD", "0.650"))` to the new value from Task 13 (keep it env-overridable — only the fallback default changes).

- [ ] **Step 4: Update any pinned tests, then run the suite**

Update the tests found in Step 1 to the new values. Run:
```bash
make test
```
Expected: full suite passes (baseline 347). If a test fails only because it asserted v3/0.650, update the assertion; if it fails for a real reason → **STOP**, investigate.

- [ ] **Step 5: Smoke-check the new default end-to-end**

Run:
```bash
.venv/bin/python -c "from src.data.constants import DEFAULT_COLLECTION_NAME; from src.db.qdrant_client import LegalVectorStore; s=LegalVectorStore(); print(DEFAULT_COLLECTION_NAME, s.get_collection_info()['points_count'])"
```
Expected: prints `luat_law_ctx_v4 60958` (default now resolves to v4).

- [ ] **Step 6: Commit both changes together**

```bash
git add src/data/constants.py src/rag/confidence.py tests/
git commit -m "feat(M2.7): promote luat_law_ctx_v4 to default + recalibrated refusal threshold"
```

---

### Task 15 (E.6): Close-out docs — scale journey + status

**Files:**
- Create: `docs/notes/scale-journey-6-100-600.md` (the M2 portfolio artefact)
- Modify: the roadmap/plan status table (`docs/superpowers/plans/2026-07-13-m2-600docs-completion.md` §7 and/or `docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md`)
- Modify: project memory `~/.claude/.../memory/vlb-project-state.md` + MEMORY.md pointer
- Record: `.superpowers/sdd/progress.md` Phase E log (mark E.1–E.6 done)

**Interfaces:**
- Consumes: recorded numbers from Tasks 11–13.
- Produces: the "scale with discipline" narrative artefact + updated project state.

- [ ] **Step 1: Write the scale-journey table**

Create `docs/notes/scale-journey-6-100-600.md` with a table across the three rungs — 6 docs (v2, M0.5) → 100 docs (v3, M1) → 600 docs (v4, M2) — columns: corpus size, points, per-sector recall@5, refusal threshold, capacity %. Fill the 600 rung from Tasks 11–13; pull 6/100 rungs from the M0/M1 numbers in `docs/` + prior eval reports. State the story: recall held within gate as the corpus scaled 100×.

- [ ] **Step 2: Update the roadmap/plan status → M2 ✅**

Mark M2 complete in the plan §7 table and the roadmap doc (which gates passed, the v4 default flip, the new threshold). Surgical edits only.

- [ ] **Step 3: Update project memory**

Update `vlb-project-state.md`: M2 done (v4 default, 605 docs, new threshold, gates passed), point to this plan + the scale-journey artefact. Update the MEMORY.md one-line pointer.

- [ ] **Step 4: Commit the docs**

```bash
git add docs/notes/scale-journey-6-100-600.md docs/superpowers/plans/2026-07-13-m2-600docs-completion.md docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md
git commit -m "docs(M2): scale-journey 6→100→600 + mark M2 complete"
```

- [ ] **Step 5: Finish the branch**

Use superpowers:finishing-a-development-branch to verify tests green and present merge/PR options for the `m2-phase-e-closeout` branch.

---

## Self-Review

**Spec coverage (Phase E §137-148):**
- E.1 enrichment + is_in_effect backfill + store.db to repo → **already done this session** (noted in Global Constraints). ✓
- E.2 capacity re-measure <70% → Task 11. ✓
- E.3 Tier-1 per-sector recall ≤5pt + frequency-bias diagnostic (first-stage recall / leakage@5) → Task 12. ✓
- E.4 Tier-2 + calibrate_threshold → new REFUSAL_THRESHOLD + table + margin, refusal recall=1.0 → Task 13. ✓
- E.5 flip DEFAULT_COLLECTION_NAME + threshold in one commit → Task 14. ✓
- E.6 update plan §7 + memory + scale-journey 6→100→600 artefact → Task 15. ✓

**Placeholder scan:** No TBD/TODO. Report filenames are `<ts>`/`<tier2_report_N>` placeholders resolved at runtime (the scripts print exact paths). Gate contingencies name the concrete STOP condition, not "handle errors."

**Gate/value consistency:** `luat_law_ctx_v4` / `luat_law_ctx_v3` used consistently; `DEFAULT_COLLECTION_NAME` (constants.py:47) and `REFUSAL_THRESHOLD` (confidence.py:109) locations match the Preconditions block; `--target-points 60958` matches the verified v4 point count; the ≤5pt gate methodology (v1-subset, v3-vs-v4 same set) is defined once in Task 12 and referenced by Task 15's rung.

**Open decisions for the execution handoff:** (1) branch name; (2) confirm Tier-2 runs locally on free-tier now (time/rate-limit) vs. deferring; (3) any FAILED gate stops for a user call.
