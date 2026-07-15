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

| mode | model | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|---|
| dense  | M2 v4 (bge-m3)   | _ | _ | 0.8675 | 0.9157 | _ |
| dense  | AITeamVN mixed   | _ | _ | _ | _ | _ |
| hybrid | M2 v4 (bge-m3)   | _ | _ | 0.8675 | 0.8976 | _ |
| hybrid | AITeamVN mixed   | _ | _ | _ | _ | _ |
| routed | M2 v4 (bge-m3)   | _ | _ | 0.8193 | 0.8916 | _ |
| routed | AITeamVN mixed   | _ | _ | _ | _ | _ |

(M2 v4 @5/@10 from report `20260714T164130Z`; @1/@3 and MRR filled by runbook step 3.)

Per-sector routed recall@5 (must not drop > 2pt vs v4): dan_su 0.844, dat_dai
0.967, doanh_nghiep 0.844, hinh_su 0.800, hngd 0.688, lao_dong 0.767.

## Results — Tier-2 answer/refusal (sectors)

| stack | refusal_acc | false_refusal | citation_acc |
|---|---|---|---|
| M2 v4              | _ | _ | _ |
| AITeamVN full-swap | _ | _ | _ |

## Verdict
- Tier-1 gate (per-mode ≥ M2): _
- Tier-2 gate (not worse than M2): _
- Decision for M3.2: _
