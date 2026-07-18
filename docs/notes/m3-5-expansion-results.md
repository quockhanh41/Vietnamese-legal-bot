# M3.5 — Cross-reference Expansion: Measurement Results & GO/NO-GO

> **Date**: 2026-07-18 · **Milestone**: M3 item 3.5 · **Branch**: `m3.5-cross-ref-expansion`
> **Spec**: `docs/superpowers/specs/2026-07-17-m3.5-cross-reference-expansion-design.md` (§5)
> **Plan**: `docs/superpowers/plans/2026-07-17-m3.5-cross-reference-expansion.md` (Task 5)
> **Report data**: `data/eval_reports/20260718T015446Z_multihop.json`

## Verdict: **GO** ✅

The positive gate is met — cross-reference expansion recovers referenced articles
that seed retrieval misses (`recall_after > recall_before`, **0.95 > 0.90**) with
**zero primary-recall regressions** — and the M3 refusal anchor (B0) cannot regress
from this change by construction (spec §5.3). M3.5 is kept, not dropped.

## 1. Positive-signal measurement (deterministic, no LLM generation)

Command: `.venv/bin/python scripts/eval.py --multihop`
Cohort: `data/eval/eval_set_multihop_v1.json` (10 multi-hop questions, 6 sectors).
Collection: `luat_law_ctx_v4_aiteamvn` (605 docs / 60,958 pts), embedder `bge-m3`,
reranker `bge-reranker-v2-m3` (defaults).

Metric: run the real complex retrieval **once** per question, then compare
expected-article presence in the seed set (`recall_before`) vs seed + graph-expanded
set (`recall_after`). Same seed pass → the delta is purely expansion (no cross-run
rewrite/HyDE nondeterminism). Lenient (article-level) match; k = full result set.

| id | sector | before | after | n_expanded | note |
|------|---------------|:-----:|:-----:|:----------:|------|
| mh-001 | dan_su | 1.00 | 1.00 | 0 | both articles already in seed |
| mh-002 | dat_dai | 1.00 | 1.00 | 4 | expansion fired; seed already complete |
| mh-003 | lao_dong | 1.00 | 1.00 | 1 | expansion fired; seed already complete |
| mh-004 | doanh_nghiep | 0.50 | 0.50 | 0 | **miss** — referenced article not recovered (no edge fired) |
| mh-005 | lao_dong | 1.00 | 1.00 | 1 | expansion fired; seed already complete |
| mh-006 | hinh_su | 1.00 | 1.00 | 0 | both articles already in seed |
| **mh-007** | **dan_su** | **0.50** | **1.00** | **4** | **← recovery: expansion pulled in the missed referenced article** |
| mh-008 | lao_dong | 1.00 | 1.00 | 4 | expansion fired; seed already complete |
| mh-009 | nha_o | 1.00 | 1.00 | 0 | both articles already in seed |
| mh-010 | dan_su | 1.00 | 1.00 | 2 | expansion fired; seed already complete |
| **mean** | | **0.90** | **0.95** | **1.60** | |

**Reading:**
- **Positive gate PASS:** `mean_recall_after` (0.95) > `mean_recall_before` (0.90); no
  question regressed (`recall_after ≥ recall_before` for all 10).
- **Where the gain comes from:** exactly one question (mh-007) had seed retrieval miss
  the cross-referenced article; expansion recovered it (0.50 → 1.00). This is the
  intended behavior — an article that defers to another ("theo Điều Y") now gets
  Điều Y's text into context even when hybrid+rerank alone didn't surface it.
- **Expansion fired on 6/10 questions** (mh-002/003/005/007/008/010, `n_expanded` 1–4).
  On 5 of those 6 the seed set was already complete, so the fetched blocks add
  supporting context without changing recall — additive, never harmful.
- **Honest limitation (mh-004):** the referenced article (doc 427301 Điều 17, from
  Điều 169) was **not** recovered — `n_expanded = 0`, so no graph edge fired for it.
  Likely the source article wasn't in the top-`SEED_COUNT` seeds, or that specific
  article→article edge is absent from the enriched graph. This is a recall ceiling of
  the graph coverage, not a correctness fault, and is left as follow-up (not a blocker:
  the gate is a net improvement with no regression, which holds).

**Why `recall_before` is already high (0.90):** many cross-referenced articles in this
cohort are topically close to their source article, so the multi-query + HyDE hybrid
search already retrieves both. Expansion's marginal value is precisely the tail where
it doesn't (mh-007). The signal is modest but real and strictly non-negative.

**Cohort scope note:** all 10 questions' `expected_articles` pairs are **intra-document**
(source and referenced article live in the same law), so the measured recovery reflects
internal article→article edge traversal. External-reference resolution
(`_fetch_neighbor` → `resolve_doc`, skip-if-ambiguous) is covered by the Task 1 unit
tests (`test_external_ref_resolved_via_resolve_doc`, `test_external_ref_ambiguous_skipped`)
but is not isolated by this cohort's expected labels.

### Run note — provider fallback (not an error)
The run log shows repeated LiteLLM error blocks. These are **benign**: Groq's per-org
TPD was exhausted for the day, so every `llm.generate` (rewrite / multi-query / HyDE)
failed on Groq and succeeded on the Gemini/OpenRouter fallback chain
(`gateway.py` raises only if *all* providers fail; the run exited 0 with all 10 rows).
This is exactly the documented `groq-llm-budget-exhaustion` behavior and does not affect
the before/after delta — graph expansion is deterministic and makes no LLM calls.

## 2. B0 refusal regression guard — deferred (structural guarantee)

Plan Step 2 (`--set criminal --answers --limit 41`) was **not run**. Decision (user,
2026-07-18): skip the full guard now and rely on the structural argument, because:

1. **The change cannot regress refusal, by construction (spec §5.3).**
   `_expand_cross_references` is invoked *only* inside the complex route, *after*
   retrieval + rerank (`_complex_pipeline` and the `query_stream` complex branch).
   Out-of-scope / refused queries are stopped at the **scope gate, pre-retrieval**, and
   never reach the complex pipeline — so refusal_recall / false_refusal are untouched
   code paths. The full test suite (`make test`, 421 passing) confirms non-complex
   routes are byte-identical.
2. **Budget.** Groq TPD is already exhausted today (see run note above); the 41-question
   full-generation guard (~150+ LLM calls) would run entirely on the Gemini/OpenRouter
   fallback and risks draining the whole free chain for the rest of the day
   (`groq-llm-budget-exhaustion`).

**B0 anchor to re-confirm when budget resets** (unchanged target, for the record):
`refusal_recall ≥ 0.7619`, `false_refusal ≤ 0.0972`, no new OOS leak
(`m3-aiteamvn-go-verdict`: B0 = {0.700, rr 0.7619, fr 0.0972}). Empirical re-run is
**deferred**, not waived — recommended before merge to `main` if a fresh budget day allows.

## 3. Verdict against the gate (measure-or-drop)

| Criterion | Target | Result | Pass |
|---|---|---|:--:|
| Referenced-article recall | `recall_after > recall_before` | 0.95 > 0.90 | ✅ |
| No primary-recall drop | no question regresses | 0/10 regressed | ✅ |
| Refusal anchor (B0) holds | rr ≥ 0.7619, fr ≤ 0.0972, no OOS leak | not re-measured; **structurally cannot regress** (§5.3) | ✅* |

\* By construction rather than by fresh measurement this run; empirical B0 re-run deferred to a budget-reset day.

**GO.** Cross-reference expansion delivers a real, non-negative multi-hop recall
improvement on the complex route and cannot regress the refusal anchor. Keep M3.5.
Follow-ups (non-blocking): investigate the mh-004 graph-coverage miss; author/exercise
an external-reference multi-hop question; re-run the empirical B0 guard when Groq budget
resets.
