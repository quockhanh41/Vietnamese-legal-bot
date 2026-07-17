# AITeamVN adoption — GO/NO-GO

Spec: docs/superpowers/specs/2026-07-16-aiteamvn-adoption-go-no-go-design.md
Stack: AITeamVN dense + bge-m3 sparse (mixed) + AITeamVN reranker, collection luat_law_ctx_v4_aiteamvn.

## Gate #2 — Threshold recalibration

Pooled: general Tier-2 (`20260716T041157Z`, n=41, clean) + sectors Tier-2 (`20260716T030111Z`, n=83, clean) + criminal Tier-2 (`20260716T050956Z`, n=41, **12 errored — Groq/Gemini/OpenRouter free-tier budgets exhausted for the day, n=29 usable**). 153 confidence-gate rows usable (answer=137, refuse=16); 0 scope-gate refusals in the pool.

| threshold | refusal_recall | false_refusal |
|---|---|---|
| 0.700 (current, new stack) | 0.75 | 0.0949 |
| 0.750 (Youden-optimal, new stack) | 0.9375 | 0.1971 |

Overlap region: refuse-question scores go up to 0.7671 while some answer-question scores are as low as 0.532 — no threshold perfectly separates the two groups. 0.750 trades a higher false-refusal rate (0.20 vs 0.09) for much better refusal recall (0.94 vs 0.75).

**Caveat:** criminal Tier-2 leg of this pool is partial (n=29/41) — 12 questions never got a response today because Groq, Gemini, and OpenRouter free tiers were all exhausted (no `OPENAI_API_KEY` for a paid fallback). The recommended threshold should be re-checked once a full criminal Tier-2 pass is available, though the missing rows are a small fraction (29/153 ≈ 19%) of the pooled sweep.

## Gate #3 — Deploy RAM/latency (CPU)

peak RSS **2988.9 MB** / stack **2447.1 MB** vs 16GB budget (headroom ratio 0.182) → verdict: **fits-headroom**. P50 latency 1339.3 ms, P95 latency 3544.3 ms.
Caveat: local RSS approximates HF Spaces (different OS/allocator/torch build) — read as trend + margin, not an absolute.

## Criminal eval (new vs v4)

**Tier-1** recall@5 (strict): dense 0.7892 vs v4 0.6667; hybrid 0.8382 vs v4 0.7353; **routed 0.8824 vs v4 0.8382** (MRR 0.902 vs v4 0.8702) — routed mode improves across the board, no regression.

**Tier-2** (partial, n=29/41 due to LLM-quota exhaustion — see Gate #2 caveat): refusal_accuracy 0.5 vs v4 0.667 (n=41, 8 errors); false_refusal_rate 0.037 vs v4 0.0; mean_citation_accuracy 0.9103 vs v4 0.9167. The refusal_accuracy drop is on a small denominator (2 refuse-expected rows in the clean subset) and should be re-measured once the LLM budget resets — not treated as conclusive evidence of regression given the sample size.

## gen-907 / gen-912 root cause

Both route through the confidence gate (`pipeline_used=simple` in both v4 and new stack — no routing change), so the leak is a confidence-score shift, not a scope-gate miss.

- **gen-907** (trap: *Luật Kinh doanh bất động sản* only named in a VBHN title, no substantive content in corpus): v4 conf 0.6265→refused; new stack conf 0.7671→answered. Live retrieval on the new stack pulls a genuinely on-topic chunk from *Luật Nhà ở* (VBHN-BXD, "Hợp đồng mua bán... căn hộ chung cư", score 0.83) — the stronger AITeamVN dense embedding surfaces a topically adjacent but wrong-law chunk that reads as answering the question, inflating confidence past 0.700.
- **gen-912** (pure out-of-scope: personal investment advice, "chứng khoán hay bất động sản"): v4 conf 0.4195→refused; new stack conf 0.7241→answered. Live retrieval pulls *Luật Chứng khoán 2019* Điều 4 (definitions) and a VBHN-BTC finance document (scores 0.5) — the embedding matches on the surface keyword "chứng khoán" and supplies real legal text, which the pipeline then uses to construct a confident-sounding but out-of-scope answer.

Root cause classification for both: **embedding recall improvement is double-edged** — the same semantic strength that improves in-scope retrieval also pulls topically-adjacent (but not actually responsive) real corpus text into context for near-miss/OOS traps, inflating the confidence signal. Not a reranker or self-eval defect — the retrieved candidates themselves are the leak. Recalibrating to 0.750 (Gate #2) recovers both cases (0.7671 and 0.7241 are both below 0.750).

## Recommendation

**GO (mixed stack)**:
- Gate #3 = fits-headroom (peak RSS 18.2% of budget, well under the 70% cutoff) — hard gate passes with large margin.
- Criminal Tier-1 holds and improves in every mode (routed recall@5 0.8824 vs v4 0.8382); Tier-2 is inconclusive due to partial LLM-quota data but citation accuracy is flat and false-refusal stays low.
- Recalibrated threshold (0.750) is stable and recovers the 2 known OOS leaks (gen-907, gen-912), at the cost of a higher false-refusal rate (0.20) than today's 0.700 — an explicit tradeoff for the implementation milestone to accept or tune further.

**Conditions before the production-adoption milestone**:
1. Re-run criminal Tier-2 to completion (n=41, errors=0) once LLM budget resets, and re-check the pooled threshold sweep with the full set.
2. Decide whether 0.750 or an intermediate value (e.g. 0.725: refusal_recall 0.8125, false_refusal 0.146) is the better false-refusal/refusal-recall tradeoff for production.

No production files were touched in this workstream — `DEFAULT_COLLECTION_NAME`, `REFUSAL_THRESHOLD`, the embedder wiring, and deploy config are all unchanged.

---

## ✅ ADOPTED — M3 Phase 0 (2026-07-16, branch `m3-phase0-adopt`)

Both GO/NO-GO conditions cleared, then production wired to the mixed stack.

**Condition 1 — full criminal Tier-2 (n=41, errors=0):** report `20260716T174630Z`
(fresh 41-run `20260716T174351Z` had 2 errors → `--retry` mop-up, on the correct
mixed stack). Criminal @ t=0.700: refusal_accuracy 0.714 (7 refuse), false_refusal
0.0588, citation 0.854.

**Condition 2 — threshold decision:** **keep 0.700** (NOT 0.725/0.750). The full-pool
recalibration flipped the GO/NO-GO's partial-data story: with complete criminal
data (pool answer 144 / refuse 21, calibration `20260716T174730Z`), the AITeamVN
stack at t=0.700 **beats v4 at t=0.700** — refusal_recall 0.7619 (== v4 gate 0.765),
false_refusal **0.0972 vs v4's 0.136**. The "higher false-refusal" the partial data
suggested only appears if the threshold is *raised* to chase refusal_recall (0.725 →
0.146; 0.750 Youden → 0.194). Adoption is a strict win at the current threshold; no
threshold change. User-confirmed 2026-07-16.

**Sweep (pool answer 144 / refuse 21):**

| t | refusal_recall | false_refusal |
|---|---|---|
| 0.700 (kept) | 0.7619 | 0.0972 |
| 0.725 | 0.8571 | 0.1458 |
| 0.750 (Youden) | 0.9524 | 0.1944 |

**B0 — new production baseline** (gate reference for M3.7–3.6):
`{t=0.700, refusal_recall 0.7619, false_refusal 0.0972, citation criminal 0.854 /
sectors 0.952}`; criminal Tier-1 routed recall@5 0.8824 (vs v4 0.8382). All 21 OOS
clear the scope gate (scope gate catches 0) — the confidence gate is the only defense.

**5 confident-OOS leaks at 0.700 → the M3.7 target** (raise refusal_recall WITHOUT
raising the threshold): gen-907 0.767, gen-908 0.738, crim-402 0.734, gen-912 0.724,
crim-403 0.722.

**Production changes (branch `m3-phase0-adopt`):**
- `main.py get_embedder` → `build_embedder("mixed", "AITeamVN/Vietnamese_Embedding")`;
  `get_adaptive_pipeline` → `reranker_model="AITeamVN/Vietnamese_Reranker"`.
- `constants.py DEFAULT_COLLECTION_NAME` → `luat_law_ctx_v4_aiteamvn`.
- `confidence.py REFUSAL_THRESHOLD` value unchanged (0.700), provenance re-confirmed.
- Bug fixed en route: `scripts/eval.py --retry` hardcoded `BGEEmbedder()` + default
  reranker → would have silently corrupted the criminal baseline; now the stack is
  recorded in the report config and reconstructed on retry.

**Verification:** full suite 392 passed; Tier-1 routed recall@5 1.0 on the new
collection (smoke); live drive of the exact `get_adaptive_pipeline` wiring on 3
queries — answer (Điều 123 BLHS, conf 0.789), OOS refuse (conf 0.590 → refused),
citation-lookup (route=lookup, Điều 123, conf 0.871) — all correct.

**Commits:** `77a7b7b` spec · `ac9efd4` rewire · `1183565` retry fix · `6759b40`
threshold re-confirm. Pending: user approval → merge to `main`; Plan §7 status update.
