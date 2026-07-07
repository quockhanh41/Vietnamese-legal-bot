# Diagnosis: per-signal false refusals in Tier-2 criminal eval (2026-07-07)

Source report: `data/eval_reports/20260707T020605Z_eval.json` (criminal set,
`luat_law_ctx`, `REFUSAL_THRESHOLD=0.725`). Four answer-expected questions
were falsely refused: crim-109 (0.7032), crim-201 (0.6986), crim-202 (0.6679),
crim-302 (0.6954). None of the eval-report rows carry a per-signal
breakdown, so `scripts/inspect_confidence.py` was written to run the real
pipeline (real Qdrant + real Groq) on these 4 IDs and dump all 5 signals,
retrieved articles vs. ground truth, the raw answer, and the NLI claim/label
list.

## Headline finding: a real bug in claim splitting for the entailment signal

Every answer this pipeline generates ends with a mandatory
`### 💡 Câu hỏi liên quan` section (see `SYSTEM_PROMPT` in
`src/llm/prompts.py`) — 3 LLM-suggested follow-up **questions** for the user.
`HallucinationVerifier._split_claims()` (`src/rag/verifier.py`) was splitting
the *entire* answer into claims by sentence/bullet boundary with no section
awareness, so the section header itself and all 3 follow-up questions were
being fed to the NLI model as if they were factual assertions to verify
against the retrieved context.

This is wrong on its face: interrogative sentences don't assert anything, so
there is no coherent notion of them being "entailed" by a premise. In
practice the NLI model produced essentially random labels for these
fragments — observed variously as NEUTRAL, but also (nonsensically) as
**ENTAILMENT** and even **CONTRADICTION** for the bare 4-word header string
`💡 Câu hỏi liên quan`. Since this trailer is present in **every** answer
(not just these 4 questions), it systematically injects noise into
`entailment_ratio` (confidence weight 0.10) for the entire eval set.

**Fix**: `_split_claims()` now stops consuming lines once it hits a
standalone `câu hỏi liên quan` header line (case-insensitive; only non-word
characters around the phrase, so a real claim containing "câu hỏi liên quan
đến X" mid-sentence never truncates the list — see Fable verification notes)
— the section is always last per the fixed prompt structure, so this cleanly
drops the header + all 3 follow-up questions without touching any real claim
above it.

This is a genuine correctness fix, not a threshold/weight tune: it doesn't
uniformly push scores in one direction (see isolated deltas below — 2 of 4
questions actually score *lower* after the fix, because the removed noise
happened to include a bogus ENTAILMENT label inflating their old ratio).

## Per-question signal breakdown

Two real-pipeline runs were captured (pipeline is stochastic —
temperature=0.1 generation, so the raw answer text and its downstream
self-eval/entailment differ run to run even with unchanged code). To isolate
**just** the effect of the fix from that natural variance, an "isolated"
column recomputes `entailment_ratio` on the **before-fix run's own claim
list + NLI labels**, simply excluding the claims the fix would now drop —
same generation, same NLI outputs, only the claim-set boundary changes.

| id | retriever | reranker | self_eval | overlap | entailment (before) | entailment (isolated, fix only) | score (before) | score (isolated fix) | refused (before) |
|---|---|---|---|---|---|---|---|---|---|
| crim-109 | 0.5968 | unavailable | 0.95 | 1.00 | 0.100 (1/10) | 0.167 (1/6) | 0.7428 | 0.7512 | No (this run) |
| crim-201 | 0.6462 | unavailable | 1.00 | 0.8235 | 0.125 (1/8, incl. 1 bogus ENTAILMENT on the header) | 0.000 (0/4) | 0.7470 | 0.7313 | No (this run) |
| crim-202 | 0.8333 | unavailable (reranker crashed, see below) | 0.85 | 0.8235 | 0.167 (3/18, incl. 1 bogus ENTAILMENT on a follow-up question) | 0.143 (2/14) | 0.7544 | 0.7514 | No (this run) |
| crim-302 | 0.6911 | unavailable | 1.00 | 1.00 | 0.400 (4/10, incl. 1 bogus CONTRADICTION on the header) | 0.667 (4/6) | 0.8285 | 0.8618 | No (this run) |

A second, fully-fresh real-pipeline run *with the fix applied* (new
generation each time, so numbers move again from LLM stochasticity on top of
the fix):

| id | retriever | reranker | self_eval | overlap | entailment | score | refused |
|---|---|---|---|---|---|---|---|
| crim-109 | 0.5968 | unavailable | 0.95 | 1.00 | 0.000 (0/9) | 0.7303 | No |
| crim-201 | 0.6462 | unavailable | 1.00 | 0.9231 | 0.000 (0/6, contradiction 0.75) | 0.7500 | No |
| crim-202 | 0.8333 | unavailable | 0.60 | 0.8571 | 0.111 (1/9) | 0.6600 | **Yes** |
| crim-302 | 0.6911 | unavailable | 1.00 | 1.00 | 0.714 (5/7) | 0.8678 | No |

Note crim-202 refused on this run purely because `llm_self_eval_score`
dropped to 0.60 (weight 0.30, the largest) — a different generation drew a
harsher self-rating. This is the clearest evidence that **self-eval
variance**, not the entailment bug, is the single largest source of
run-to-run flips near the 0.725 boundary (see "not fixed" section).

Neither real-pipeline run reproduced an actual refusal for 3 of the 4
IDs — reinforcing that the specific 0.6679–0.7032 scores in the eval report
reflect one stochastic draw each, not a deterministic property of these
questions. crim-302's originally-reported dip to 0.435 is very plausibly the
same entailment-contamination bug compounding with an unlucky
self-eval/generation draw on that run (a bogus low/negative entailment label
on non-claim content, stacked with a harsh self-eval).

## Root cause per question

- **crim-109** ("Đồng phạm là gì?", single-hop, simple pipeline): retrieval
  ranks the correct article (Điều 17, BLHS 2015) **3rd**, behind two
  related-but-wrong articles (Điều 18 "Che giấu tội phạm", Điều 20 of the old
  1999 code) — a genuine, if minor, dense-retrieval ranking weakness (no
  hybrid/rerank on the "simple" route). `retriever_score` (0.60) is honestly
  reporting this. `entailment` is also genuinely low even after the fix
  (0.167) — a secondary, separate issue: `HallucinationVerifier._verify_semantic`
  truncates the NLI premise to `context[:2000]` chars, and with 15 chunks
  concatenated ahead of it, Điều 17's actual text may be truncated out of
  the premise the NLI model sees. **Not fixed here** (different code path,
  larger design question of per-claim premise selection vs. global
  truncation, weight is only 0.10) — flagged as a follow-up.

- **crim-201** (age-threshold multi-hop, simple pipeline): retrieval is
  reasonable (Điều 173 top-1, Điều 12 5th, Điều 9 not observed in top-12 but
  question only needs 12/173/9 per eval set — recall was partial). The
  generated answer is substantively **correct** (matches ground truth: not
  criminally liable). The concerning finding is `llm_self_eval_score` sits at
  1.00 in both runs despite the verifier flagging a CONTRADICTION on the
  answer's core claim in the before-fix run — self-eval shows a **ceiling
  effect** (0.85–1.00 across all 4 questions, both runs, 8/8 observations)
  that gives it little power to discriminate a genuinely shaky multi-hop
  inference from a solid one. This is a real calibration concern, but per
  the task's explicit constraint we do not touch the self-eval prompt unless
  it's "demonstrably broken" — ceiling-heavy-but-plausible isn't that bar.
  Flagged, not changed.

- **crim-202** (comparison type, medium pipeline): retrieval is actually
  strong (0.83, correct articles top-ranked). The reranker **crashes on
  every call** in this environment: `⚠️ Reranking failed (skip):
  XLMRobertaTokenizer has no attribute prepare_for_model` (FlagEmbedding /
  transformers version mismatch in `.venv`) — meaning `reranker_score` is
  `None`/unavailable for 100% of medium+complex-pipeline queries, not just
  this question, and its 0.20 weight is silently renormalized away
  corpus-wide. This is a significant, previously-undocumented finding, but
  it lives in a dependency-version mismatch (`FlagEmbedding`/`transformers`
  compatibility), not in `src/rag/confidence.py`'s scoring logic — fixing it
  safely would mean pinning/upgrading dependencies, which is outside this
  task's scope (risk of side effects, not what was asked). Recommended as a
  separate follow-up. The refusal seen in the after-fix run traces to a low
  `llm_self_eval_score` (0.60) draw, not to overlap/entailment/retriever.

- **crim-302** (2025-amendment question, simple pipeline, the priority
  case): retrieval is reasonable — both expected doc/article pairs (Điều 1
  of the 2025 amendment law, Điều 40 of BLHS 2015) appear in the top 7 of 15
  results. The generated answer is substantively correct and matches ground
  truth (khoản 3 Điều 40, bổ sung ung thư giai đoạn cuối). This question
  benefits the **most** from the claim-splitting fix: the before-fix run's
  entailment_ratio (0.4) was suppressed by a bogus CONTRADICTION label on
  the bare section-header string; isolating just the fix lifts it to 0.667,
  and a fresh post-fix run scored 0.714. No other signal shows a genuine
  quality problem here — this looks like the fixed bug was the dominant
  drag for this specific question.

## What was changed

- `src/rag/verifier.py` — `HallucinationVerifier._split_claims()`: stop
  consuming lines once the `Câu hỏi liên quan` section header is reached, so
  the mandatory follow-up-questions trailer (present in every answer) is
  never fed to the NLI entailment check as if it were a factual claim.
- `tests/test_verifier.py` (new) — 4 unit tests: the related-questions
  section is excluded while real claims survive; answers without the
  section are unaffected; a mid-sentence "câu hỏi liên quan đến X" does NOT
  truncate the claim list (only a standalone header line does — see Fable
  verification notes below); empty/None input still returns `[]`.

Weights, the 0.725 threshold, and all prompts (including
`CONFIDENCE_EVAL_PROMPT`) were left untouched, per the task's constraints.

## What was NOT fixed (and why)

1. **NLI premise truncation to `context[:2000]`** (`verifier.py`,
   `_verify_semantic`) can drop the actual supporting article from the NLI
   premise when it's not among the first ~3-4 concatenated chunks (crim-109).
   Fixing this well needs a design decision (per-claim relevant-chunk
   selection vs. a bigger global truncation) beyond a minimal patch.
2. **`llm_self_eval_score` ceiling effect** (0.85–1.00 in 8/8 observations
   across both runs, including on an answer the verifier flagged with a
   contradiction) — a real calibration concern, but the task explicitly
   restricts touching the self-eval prompt to "demonstrably broken" cases,
   and a ceiling-heavy-but-plausible signal doesn't clear that bar on this
   evidence alone.
3. **Reranker crash** (`FlagEmbedding`/`transformers` version mismatch)
   disables the entire 0.20-weight reranker signal for every
   medium/complex-pipeline query corpus-wide. Real and significant, but it's
   a dependency/environment issue, not a `confidence.py` scoring bug —
   flagged for separate follow-up rather than fixed here.
4. Threshold (0.725) and signal weights were deliberately left unchanged per
   the task's explicit instructions.

## Verification

- Full test suite: `.venv/bin/python -m pytest tests/ -q` → **86 passed**
  (82 pre-existing + 4 new in `tests/test_verifier.py`).
- Real-pipeline before/after captured via `scripts/inspect_confidence.py`
  (real Qdrant `luat_law_ctx` + real Groq) — see tables above.
- Full 41-question Tier-2 criminal eval re-run with the fix (since
  `_split_claims` runs on every generated answer):
  `caffeinate -i .venv/bin/python scripts/eval.py --set criminal --answers --sleep 3`
  → `data/eval_reports/20260707T052327Z_eval.json`, 0 errors.

  | metric | before (20260707T020605Z) | after (20260707T052327Z) |
  |---|---|---|
  | refusal_accuracy | 1.0 | 1.0 (no regression on refuse side) |
  | false_refusal_rate | 0.1176 (4/34) | **0.0588 (2/34)** |
  | mean_citation_accuracy | 0.8529 | 0.8824 |
  | answer-side conf mean / median / min | 0.8056 / 0.8060 / 0.6679 | 0.8301 / 0.8300 / 0.6698 |
  | refuse-side conf mean / max | 0.1817 / 0.7055 | 0.0760 / 0.5319 |

  Per-target-question scores in the after run: crim-109 **0.7303
  (answered)**, crim-201 0.6938 (still refused), crim-202 0.6698 (still
  refused), crim-302 **0.7993 (answered)**. The two that still refuse are
  the two whose signals are honestly reporting real weaknesses (multi-hop
  inference shakiness for crim-201; a low self-eval draw plus the corpus-wide
  dead reranker for crim-202) — consistent with the per-question diagnosis
  above. Notably the refuse-side max *dropped* (0.7055 → 0.5319), i.e. the
  fix also removed bogus ENTAILMENT inflation from should-refuse questions —
  evidence it is a debiasing fix, not a score-boosting tune. (All answer-side
  numbers also carry ordinary run-to-run generation variance; see the
  stochasticity note above.)

## Fable verification notes (independent, 2026-07-07)

- Re-verified the eval aggregates directly from both report JSONs — they
  match the table above exactly.
- Confirmed SYSTEM_PROMPT mandates the `### 💡 Câu hỏi liên quan` trailer,
  and confirmed via `data/eval_reports/inspect_confidence_output.json` that
  post-fix claim lists are clean while real claims survive (crim-302: 7
  claims, no interrogatives, entailment 0.714).
- Independently confirmed the reranker crash mechanism WITHOUT loading model
  weights: under `transformers 5.12.1`,
  `AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')` returns an
  `XLMRobertaTokenizer` with `hasattr(tok, 'prepare_for_model') == False`,
  which FlagEmbedding 1.4.0's FlagReranker still calls. The 0.20-weight
  reranker signal is dead for every medium/complex query until deps are
  pinned (`transformers<5`) or FlagEmbedding catches up — roadmap follow-up.
- **Tightened the fix's regex** during verification: the original
  `re.search(r'câu\s*hỏi\s*liên\s*quan', line)` matched any line *containing*
  the phrase, but "câu hỏi liên quan đến X" occurs mid-sentence in natural
  Vietnamese, where breaking would silently drop every real claim after it.
  Now `re.match(r'^\W*câu\s*hỏi\s*liên\s*quan\W*$', ...)` — a standalone
  header line only; a missed malformed header merely falls back to the old
  noisy-but-harmless behavior. Regression test added
  (`test_split_claims_phrase_mid_sentence_does_not_truncate`) → suite is now
  **86 passed**.
