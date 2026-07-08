# 🏛️ Vietnamese Legal Knowledge Bot

An **Adaptive RAG** chatbot for querying Vietnamese legal documents — with query-complexity
routing, a multi-signal confidence gate that *refuses rather than hallucinate*, two-tier
hallucination verification, and grounded `[n]` citations that tie every claim back to a
specific source passage.

> **Status:** portfolio / research project. Runs on a 6-document Vietnamese criminal-law
> corpus (1,670 chunks). The architecture is built to scale; see
> [`docs/`](docs/) for the production roadmap and the corpus-scaling checklist.

---

## Why it's interesting

Most RAG demos happily answer even when retrieval failed. A legal assistant must not — a
confidently wrong answer about the law is worse than "I don't know." This project is built
around that constraint:

- **Adaptive routing** — a single LLM call classifies each question as `SIMPLE` /
  `MEDIUM` / `COMPLEX` (and screens out-of-scope questions), routing each to a pipeline of
  the right depth instead of paying for the full pipeline every time. Pure citation lookups
  (`"Điều 173 BLHS 2015 quy định gì?"`) skip vector search entirely.
- **Confidence gate** — a 5-signal weighted score (retriever margin, reranker, LLM
  self-eval, context overlap, NLI entailment) decides whether to answer or refuse, at a
  threshold calibrated on real eval data (`0.725`).
- **Hallucination verification** — every answer is checked structurally (does each cited
  `Điều N` actually appear in the retrieved context?), semantically (NLI entailment via
  mDeBERTa), and by **grounding** (each detail claim must cite the numbered source block
  `[n]` it came from).
- **Out-of-scope detection** — questions the corpus can't answer are refused *before*
  retrieval, so the bot doesn't dress up an irrelevant chunk as an answer.

---

## Architecture

```
                          ┌─────────────────────────────────────────────┐
  question ──▶ Citation?  │  no  ▶  Classify + Scope gate (1 LLM call)   │
               │ yes             │        │ OUT ▶ refuse (pre-retrieval) │
               ▼                 │        ▼ IN                            │
        Metadata lookup          │   SIMPLE  MEDIUM  COMPLEX             │
        (Qdrant filter)          │   dense   hybrid  rewrite+HyDE+multi  │
               │                 │           +rerank +rerank +compress   │
               └────────┬────────┴───────────────┬─────────────────────-┘
                        ▼                         ▼
                  Build context  [1]…[k]  ──▶  LLM generate (streamed)
                        │                         │
                        ▼                         ▼
        ┌───────────────────────────────────────────────────────────┐
        │  Verify:  structural citations · NLI entailment · [n]       │
        │  Confidence: 5 signals ▶ score ≥ 0.725 ? answer : refuse    │
        └───────────────────────────────────────────────────────────┘
```

**Retrieval** uses BGE-M3 (dense + sparse hybrid) over a *contextual* embedding collection
— each chunk is prefixed with its `Luật | Chương | Điều | Khoản` header before embedding —
followed by a BGE-reranker-v2-m3 cross-encoder on MEDIUM/COMPLEX routes.

**Generation** runs on Groq (Qwen 3 32B) through a LiteLLM gateway with a fallback chain
(`qwen3-32b → llama-3.3-70b → gpt-4o-mini`) and comma-separated multi-key rotation to ride
out free-tier rate limits.

---

## Measured results

On a 34-question grounded criminal-law eval set (routed retrieval, strict matching):

| Metric | Score |
|---|---|
| recall@5 | **0.922** |
| recall@10 | **0.976** |
| MRR | **0.863** |
| Citation-lookup recall@5 (routed) | **1.0** |

Answer quality (Tier-2, with generation + confidence gate):

| Metric | Score |
|---|---|
| Out-of-scope refusal accuracy (general set, 29 Qs) | **29 / 29** |
| Criminal false-refusal rate | **2 / 34** |
| Refusal recall (never answers an out-of-scope question) | **1.0** |

Reproduce with `python scripts/eval.py` (Tier 1, retrieval only) and
`python scripts/eval.py --answers` (Tier 2). Threshold calibration lives in
`scripts/calibrate_threshold.py`.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI + SSE streaming |
| UI | Streamlit |
| Vector DB | Qdrant (dense + sparse / hybrid) |
| Embeddings | BGE-M3 (1024-d, `FlagEmbedding`) |
| Reranker | BGE-reranker-v2-m3 |
| LLM | Groq Qwen 3 32B via LiteLLM (fallback: Llama 3.3 70B, GPT-4o-mini) |
| NLI verifier | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` |
| Tests | pytest |

---

## Quickstart

**Prerequisites:** Python 3.11+, a Groq API key, and a Qdrant instance (Qdrant Cloud, or
local via Docker). Embedding + reranking + NLI models run on CPU and need ~6–8 GB RAM.

```bash
# 1. Install
make install                      # pip install -e ".[dev]"

# 2. Configure
cp .env.example .env              # then fill in GROQ_API_KEY + QDRANT_URL/QDRANT_API_KEY

# 3. (Optional) local Qdrant + Redis instead of Qdrant Cloud
make qdrant                       # docker compose up -d qdrant redis

# 4. Ingest the corpus into Qdrant
make download                     # fetch the dataset
make ingest                       # chunk → contextualize → embed → upsert

# 5. Run — backend + UI
make api                          # FastAPI on :8000
make ui                           # Streamlit on :8501   (separate terminal)
#   or both at once:
make dev
```

Then open **http://localhost:8501** and ask a question. Or hit the API directly:

```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"question":"Tội trộm cắp tài sản bị xử phạt như thế nào?","use_adaptive":true}'
```

Full container stack: `docker compose up --build`.

---

## Project structure

```
src/
  api/          FastAPI app (chat, streaming, documents, stats, graph/timeline endpoints)
  ui/           Streamlit chat UI (streaming, citation cards, confidence badges)
  rag/          orchestrator (adaptive routing) · confidence · verifier · citation_lookup
  data/         chunker · contextualizer · embedder (BGE-M3) · metadata_store · constants
  db/           Qdrant client (hybrid search, payload filters)
  llm/          LiteLLM gateway (fallback chain + key rotation) · prompts
  pipeline/     enricher · EnhancedRetriever (graph / timeline / amendment — Phase 2)
  eval/         retrieval + RAGAS metrics
scripts/        ingest · eval · calibrate_threshold · inspect_confidence · reindex · …
tests/          pytest suite
docs/           evaluation report, production roadmap, design notes
data/eval/      grounded eval sets (criminal + general)
```

📖 **New to the codebase?** Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
(in Vietnamese) — a guided code walkthrough: module map, the life of a query
(file-by-file), the ingest pipeline, and the key design decisions. The knowledge-graph /
versioning layer has its own deep-dive:
[`docs/KNOWLEDGE_GRAPH_VERSIONING.md`](docs/KNOWLEDGE_GRAPH_VERSIONING.md).

---

## Key API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/chat` | Ask a question (non-streaming) |
| `POST /api/chat/stream` | Ask a question (SSE: pipeline steps, citations, tokens, verification) |
| `GET /api/documents` | Browse ingested documents |
| `GET /api/stats` | Collection / corpus stats |
| `GET /api/article/history` · `/api/timeline/effective` | Temporal / amendment lookups (Phase 2) |
| `GET /api/graph/neighbors` · `/api/graph/stats` | Cross-reference graph (Phase 2) |
| `GET /health` | Liveness |

---

## Evaluation

```bash
python scripts/eval.py                 # Tier 1: retrieval metrics (recall@k, MRR) — no LLM
python scripts/eval.py --answers       # Tier 2: refusal + citation accuracy + confidence dist.
python scripts/calibrate_threshold.py  # sweep the refusal threshold (Youden-optimal)
python scripts/validate_eval_set.py    # schema + grounding checks on the eval sets
```

Reports land in `data/eval_reports/<UTC>_eval.json`. See
[`docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md`](docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md)
for the full evaluation history and the roadmap toward production.

---

## Design notes & limitations

- **Small corpus, on purpose.** Six criminal-law documents (1,670 chunks) is a deliberate
  working scale for tuning the confidence/verification machinery against ground truth. The
  `docs/` roadmap includes a step-by-step checklist for scaling to the full corpus
  (deterministic point IDs, payload indexes, on-disk quantization, GPU embedding, per-domain
  eval sets, threshold re-calibration).
- **Refusal is a feature.** A ~6% false-refusal rate is an accepted trade-off for never
  answering an out-of-scope legal question. The threshold is tunable via
  `CONFIDENCE_REFUSAL_THRESHOLD`.
- **Phase 2 (temporal / amendment)** is wired into the chat pipeline: superseded laws are
  filtered out of retrieval by default (fail-open on unknown status), and amended articles
  are flagged inline (context + citation cards). The cross-reference graph is still
  API-only (`/api/graph/*`), not yet used for context expansion in chat.
- **Parent-document retrieval.** Retrieval matches at the precise Khoản (clause) level, but
  context sent to the LLM expands to the full parent Điều (article), deduplicated so a
  multi-Khoản hit on one Điều doesn't repeat it as several blocks.
- **Not legal advice.** Outputs are for reference only.

---

## License

[MIT](LICENSE) © 2026 quockhanh41
