# M3.5 Task 2B — Re-enrichment validation

**Date**: 2026-07-17 · **Collection**: `luat_law_ctx_v4_aiteamvn` (605 docs, 60,958 pts)
**Command**: `scripts/run_enrichment.py --collection luat_law_ctx_v4_aiteamvn` (elapsed 1092s)
**Backup**: `data/metadata.bak-pre-m3.5-20260717T215947` (pre-run metadata dir)

## Why

The enriched graph (SQLite `store.db`, the runtime source — the JSON files in
`data/metadata/` are stale legacy) had **zero article→article edges**: every edge
originated at a document node, because `enricher.py` extracted references over
whole-document text with no `current_article`. Task 1's query-time expansion seeds
from a retrieved *article*, so it needs article-source edges. Task 2A fixed the
enricher to extract per-article; this task re-ran enrichment to rebuild the graph.

## Results

### 1. Article-source edges now exist (the fix)
`get_graph()` edges by SOURCE node_type:

| | article-source | document-source |
|---|---|---|
| **Before** (runtime SQLite) | 0 | 10,878 |
| **After** | **15,599** | 313 |

Graph grew to 16,622 nodes / 15,912 edges (from 10,436 / 10,878) — per-article
extraction attributes each reference to its containing article.

### 2. Expansion fetches on the live collection
Seed = a real article node (`doc 687595, "Điều 1. Phạm vi điều chỉnh"`):

- `_expand_cross_references([seed])` → **4 blocks**: Điều 3, Điều 10, Điều 22, Điều 28
  (all doc 687595). Confirms the node-id match (enricher's full `article` string ==
  expansion seed node id) works end-to-end.
- **Latency**: ~0.9s per query **with a warm CitationLookup** — the production
  condition, since the scope-gate classifier calls `corpus_summary()` on every query
  before routing. Cold (first-ever call) pays a one-time ~134s alias-index build
  (`get_document_index()` full-collection scroll) — a **pre-existing** CitationLookup
  cost, NOT introduced by M3.5, and amortized across the session.
  - *Eval note*: `run_multihop` (Task 4) bypasses the scope gate, so its first
    question pays the ~134s warm-up lazily; subsequent questions are ~1s. Optionally
    pre-warm with `pipeline._get_citation_lookup().corpus_summary()`.

### 3. M2 no-regression
`is_in_effect` counts (True / False / None):

| | True | False | None |
|---|---|---|---|
| **Pre** | 580 | 18 | 7 |
| **Post** | 580 | 18 | 7 |

Identical. Sentinel: **Luật nuôi con nuôi 2010 (108082) = `is_in_effect: True`** —
the M2 VBHN-supersession/hngd fix (`f55dc9e`) is preserved. `in_effect=580` also
matches the enrichment run's own report.

## Verdict: **PASS**

Article-level graph rebuilt, expansion fires in ~1s warm, M2 gates intact. Proceed to
the M3.5 measurement tasks (cohort + before/after eval). `store.db` updated locally
(git-ignored); the stale `data/metadata/*.json` legacy files were not written by the
enricher and are left as-is.
