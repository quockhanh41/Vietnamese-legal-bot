# Scale journey: 6 → 100 → 600 documents

The M2 story is **scaling with discipline**: a Vietnamese legal-RAG corpus grown
100× (6 → 600 docs) across three milestones, with a measured gate at every step
so quality never silently regressed. This note is the portfolio artefact for
that journey.

## The three rungs

| | M0.5 (v2) | M1 (v3) | M2 (v4) |
|---|---|---|---|
| Documents | 6 (criminal only) | 100 (6 sectors) | **600 (6 sectors)** |
| Qdrant points | 2,242 | 23,727 | **60,958** |
| Collection | `luat_law_ctx_v2` | `luat_law_ctx_v3` | `luat_law_ctx_v4` |
| Free-tier capacity | trivial | ~40% projected | **9.5% RAM / 6.0% disk** |
| REFUSAL_THRESHOLD | 0.700 | 0.650 | **0.700** |
| Date | 2026-07-10 | 2026-07-10 | 2026-07-14 |

Each collection is kept live in Qdrant as a comparison baseline; `v4` is the
default since 2026-07-15.

## Retrieval quality held within gate

The M2 gate: **routed recall@5 drops ≤ 5 points per sector vs M1.** Because the
eval set expanded (v1 51q → v2 83q, adding Thông tư questions that target
600-only docs), the honest comparison is the **v1 subset** — the questions both
the 100- and 600-doc corpora can answer — so the number isolates
distractor-dilution, not corpus coverage.

**Routed recall@5, v1-subset (100 → 600):**

| sector | v3 (100) | v4 (600) | Δ |
|---|---|---|---|
| dan_su (dân sự) | 1.000 | 0.950 | −5.0pt |
| dat_dai (đất đai) | 1.000 | 0.950 | −5.0pt |
| doanh_nghiep (doanh nghiệp) | 0.773 | 0.773 | 0.0 |
| hngd (hôn nhân–gia đình) | 0.850 | 0.800 | −5.0pt |
| lao_dong (lao động) | 0.750 | 0.750 | 0.0 |
| **overall** | **0.875** | **0.845** | **−3.0pt** |

Every sector is within the ≤5pt gate. Holding recall within 3 points overall
while the corpus grew 6× — and the per-document distractor pool grew with it —
is the whole point of "disciplined scaling."

**Full v2 set (83q), v4 routed recall@5** (for reference, includes the harder
600-only Thông tư questions): overall **0.819**; per sector dat_dai 0.967,
dan_su 0.844, doanh_nghiep 0.844, hinh_su 0.800, lao_dong 0.767, hngd 0.688.
Criminal set (separate domain): **0.838**. Earlier rungs, overall routed
recall@5: 6-doc 0.922 (criminal), 100-doc 0.897 criminal / 0.892 sectors.

## A real bug the scale-up surfaced (and we fixed)

The first v4 Tier-1 run failed the gate on **hngd at −25pt** (0.85 → 0.60,
routed). Investigation (not a guess) traced it through the routed retrieval
path — which filters `exclude_superseded=True` — to a temporal-enrichment bug:
`VersionResolver.find_replacement` treated a **Văn bản hợp nhất** (a
*consolidation* document — merges a law + its amendments into one readable file;
never repeals anything) as a *replacement* of the law it consolidates. So
**Luật Nuôi con nuôi 2010** was flagged `is_in_effect=False` and dropped from the
routed pool, taking hngd recall down with it (also mislabelled: Luật Thương mại
2005, Luật THAHS 2019).

The VBHN slipped past the two existing guards (amendment titles, legal
hierarchy). Fix: a dedicated consolidation guard (`_CONSOLIDATION_TITLE_RE`) +
TDD tests, then re-enrichment. **hngd recovered 0.60 → 0.80.** This is the
system's temporal-reasoning working as designed once the data was correct —
and a demonstration that scaling revealed a latent correctness issue the gate
caught before it shipped.

## Refusal calibration: an honest ceiling

The M2 refusal gate is stated as "refusal_recall = 1.0 on the OOS set." At
600-doc scale, with a rigorous **17-case** OOS set (M1's calibration used a
single OOS case), that is **not reachable at a usable threshold**: the scope
gate catches 0 of the 17 (all are "in-domain-looking but unanswerable"), so the
confidence gate is the only defense, and one question (gen-903, a confident
mis-answer at confidence 0.882) forces refusal_recall=1.0 only at t≈0.900 —
which false-refuses ~79% of valid questions.

We ship **t = 0.700** (balanced): refusal_recall 0.765, false_refusal 0.136.
The Youden-optimal 0.775 (recall 0.941, false_refusal 0.271) was rejected for
refusing 27% of valid questions — bad UX for a demo. The confident-OOS ceiling
(gen-903 and peers) is a **documented known gap**, not a hidden one: closing it
needs scope-classifier / retrieval-confidence work, tracked for M3.

## What this demonstrates

- **Gate-driven scaling** — capacity, retrieval, and refusal each measured and
  recorded at every rung; no rung shipped on vibes.
- **Root-cause discipline** — the hngd regression was debugged to a specific
  line in the version resolver, not patched at the symptom.
- **Honest evaluation** — the refusal gate's literal target is unreachable at
  this scale, and that is stated plainly with the operating point and its
  tradeoff, rather than reported as a pass.

_Numbers: capacity `capacity_20260714T155040Z.json`; Tier-1 v4
`20260714T164130Z_eval.json`, v3 `20260714T160023Z_eval.json`; Tier-2 +
calibration `20260714T182658Z_calibration.json`._
