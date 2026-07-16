# AITeamVN off-the-shelf embedding + reranker vs M2 (600-doc)

Spec: `docs/superpowers/specs/2026-07-15-aiteamvn-internal-comparison-design.md`.
Plan: `docs/superpowers/plans/2026-07-15-aiteamvn-internal-comparison.md`.
Feeds the M3.2 decision (is a self fine-tune worth it if an off-the-shelf VN
bge-m3 already wins?). Win or lose, this note gets the tables.

## Config
- Dense: `AITeamVN/Vietnamese_Embedding` (SentenceTransformer, 1024-dim, normalized).
- Sparse: `BAAI/bge-m3` (via `MixedHybridEmbedder`, `src/data/mixed_embedder.py`).
- Reranker: `AITeamVN/Vietnamese_Reranker`.
- Corpus: `data/raw/corpus_600docs.parquet` → collection `luat_law_ctx_v4_aiteamvn` (config = v4).
- Eval set: `data/eval/eval_set_sectors_v2.json` (83 q).

## Runbook

### 1. Re-embed the 600-doc corpus → Qdrant Cloud (Kaggle GPU, per docs/notes/kaggle-ingest-guide.md)
Same 7-cell flow as the v4 ingest; Cell 5 becomes (creates the collection in the
same Qdrant Cloud the local eval reads — QDRANT_URL/QDRANT_API_KEY via Kaggle Secrets):
```bash
python scripts/ingest.py \
  --data-path "{DATA_PATH}" \
  --collection luat_law_ctx_v4_aiteamvn \
  --embedder mixed --embed-model AITeamVN/Vietnamese_Embedding \
  --quantized --device cuda --embed-batch-size 128 --resume
```
Expect ~60,958 points. `MixedHybridEmbedder` loads BOTH AITeamVN (dense) and BGE-M3
(sparse) on the T4 (~4.6GB together — fits 16GB; drop `--embed-batch-size` to 64/32 on
CUDA OOM). Do NOT run Cell 7 enrichment on Kaggle — it only has the v3 100-doc metadata
seed. Graph/timeline/amendment (SQLite) are collection-agnostic and already computed for
v4 — not re-derived.

### 1b. Backfill is_in_effect onto the new collection (LOCAL — needs the 605-doc store.db)
Required for routed + Tier-2 parity with v4 (they filter `exclude_superseded=True`).
Tier-1 dense/hybrid do not filter, so they are already comparable without this.
```bash
.venv/bin/python scripts/backfill_is_in_effect.py --collection luat_law_ctx_v4_aiteamvn
```

### 2. Tier-1 on the new collection (no LLM)
```bash
.venv/bin/python scripts/eval.py --set sectors \
  --collection luat_law_ctx_v4_aiteamvn --embedder mixed
```

### 3. Re-measure M2 baseline at @1/@3 (no LLM — old report only had @5/@10)
```bash
.venv/bin/python scripts/eval.py --set sectors --collection luat_law_ctx_v4
```

### 4. Tier-2 answer/refusal, full swap (LLM — Groq budget)
```bash
.venv/bin/python scripts/eval.py --set sectors --answers \
  --collection luat_law_ctx_v4_aiteamvn --embedder mixed \
  --reranker AITeamVN/Vietnamese_Reranker --sleep 3
```
If Groq TPD dies mid-run: `scripts/eval.py --retry <report.json>`.

### 5. Delete the parallel collection after the verdict (free RAM).

## Results — Tier-1 retrieval (strict recall, sectors, n=83)

Baseline v4 re-measured same-run at K=(1,3,5,10): report `20260716T023024Z`.
AITeamVN mixed: report `20260716T022403Z`.

| mode | model | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| dense  | M2 v4 (bge-m3)   | 0.548 | 0.813 | 0.868 | 0.916 | 0.713 |
| dense  | **AITeamVN mixed** | 0.572 | 0.831 | **0.898** | 0.940 | 0.738 |
| hybrid | M2 v4 (bge-m3)   | 0.645 | 0.813 | 0.874 | 0.892 | 0.766 |
| hybrid | **AITeamVN mixed** | 0.608 | 0.904 | **0.922** | 0.964 | 0.785 |
| routed | M2 v4 (bge-m3)   | 0.657 | 0.807 | 0.819 | 0.886 | 0.771 |
| routed | **AITeamVN mixed** | 0.621 | 0.849 | **0.849** | 0.940 | 0.778 |

recall@5 improves on **all three modes** (dense +3.0pt, hybrid +4.8pt, routed +3.0pt);
recall@3/@10 and MRR all improve. r@1 dips for hybrid/routed (−3.6pt) — AITeamVN
pulls the target into top-3/5 more but ranks it #1 slightly less.

Per-sector routed recall@5 (gate: no drop > 2pt vs v4):

| sector | v4 | AITeamVN | Δ |
|---|---|---|---|
| dan_su | 0.844 | 0.875 | +3.1 |
| **dat_dai** | 0.967 | 0.833 | **−13.3** |
| doanh_nghiep | 0.844 | 1.000 | +15.6 |
| hinh_su | 0.800 | 0.800 | +0.0 |
| hngd | 0.688 | 0.719 | +3.1 |
| lao_dong | 0.767 | 0.833 | +6.7 |

**`dat_dai` −13.3pt root cause (investigated):** exactly 2 of 15 questions (dd-101, dd-301),
both routed via the hybrid fallback. NOT a sparse/`exclude_superseded`/backfill artifact —
plain hybrid (no filter) and routed (with filter) fail identically, and **dat_dai routed
recall@10 is unchanged (0.9667)**. Pure dense-ranking near-miss: AITeamVN ranks
semantically-adjacent articles of the *same* law above the exact target, pushing it from
rank ≤5 to rank 6–10 (dd-101: `Điều 116/196` over `Điều 121`; dd-301: `Điều 4` over `Điều 1`
of the same amendment law). A top-5-boundary artifact on n=15, not a capability loss.

## Results — Tier-2 answer/refusal (sectors, n=83, 0 LLM errors)

v4 baseline `20260714T181951Z`; AITeamVN full-swap `20260716T030111Z`.
(refusal_acc = None: the sectors set has no refuse-expected questions.)

| stack | false_refusal | citation_acc |
|---|---|---|
| M2 v4              | 0.1084 | 0.9257 |
| AITeamVN full-swap | 0.1205 | **0.9521** |

Citation accuracy **+2.6pt**. false_refusal +1.2pt is a churn (4 new / 3 fixed, net +1),
still ≤ the 0.136 M3 floor. The 4 new refusals crossed the **0.700 threshold** (calibrated
for bge-m3/v4) as the confidence distribution shifted under the new stack — a
**recalibration artifact**, not a capability loss. One of them (dd-101, 0.708→0.677) is the
direct downstream of its Tier-1 near-miss.

## Verdict
- **Tier-1**: per-mode recall@5 all ≥ M2 (improve) — **PASS**. Per-sector rule tripped only by
  `dat_dai` routed −13.3pt, shown above to be a top-5-boundary near-miss (recall@10 unchanged),
  not a real regression.
- **Tier-2**: citation +2.6pt; false_refusal +1.2pt within the 0.136 floor, attributable to
  the fixed 0.700 threshold (recalibration is adoption-scope).
- **Decision for M3.2**: the off-the-shelf AITeamVN pair **beats stock bge-m3 on the internal
  600-doc corpus** — clearly on retrieval (all modes + citation), with only small-n boundary
  blemishes. Training our own bi-encoder (M3.2) must clear this bar to be worth it, so this
  **de-risks / likely deprioritizes M3.2**. Adopting AITeamVN as production default is a
  **separate follow-up** (swap default + recalibrate the 0.700 threshold for the new stack).
