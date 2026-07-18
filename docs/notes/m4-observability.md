# M4.3 — Langfuse Observability: Live Verification

> **Date**: 2026-07-18 · **Milestone**: M4 item 4.3 · **Branch**: `m4.3-langfuse-observability`
> **Spec**: [`docs/superpowers/specs/2026-07-18-m4.3-langfuse-observability-design.md`](../superpowers/specs/2026-07-18-m4.3-langfuse-observability-design.md)
> **Langfuse project**: `jp.cloud.langfuse.com`, project id `cmrqc6lpy001ead0ddwcfxwmb`

## Verdict: implemented, live-verified — with two documented gaps

`/api/chat` produces one Langfuse trace per query with the expected stage spans
correctly nested, and the underlying LiteLLM call data (model, prompt, tokens)
travels in the **same trace** rather than an orphan trace — the design's core
risk (§5.1, two tracer providers must share one trace) is resolved for the
synchronous path. Two gaps surfaced during live verification and are documented
below rather than silently accepted: (1) `/api/chat/stream` still splits into
multiple orphan traces (the design's stated risk #1 — accepted per spec
fallback, gate demo runs on `/api/chat`); (2) even on the synchronous path,
LiteLLM's generation calls land as `type: SPAN` observations, not Langfuse's
native `type: GENERATION` — model/token data is present but not in Langfuse's
structured Generation UI fields (cost/token dashboard columns). Root cause
identified precisely below (§4).

Live verification also surfaced **two real, pre-existing production bugs**,
unrelated to the M4.3 code itself — the kind of thing this feature exists to
catch:

1. **`.env` had `LANGFUSE_BASE_URL` but not `LANGFUSE_HOST`.** LiteLLM's
   `langfuse_otel` callback only reads `LANGFUSE_HOST`/`LANGFUSE_OTEL_HOST`
   (the langfuse Python SDK's own `@observe` decorator prefers
   `LANGFUSE_BASE_URL`, deprecated-falls-back to `LANGFUSE_HOST` — two
   different env var conventions in the same feature). Without `LANGFUSE_HOST`,
   LiteLLM defaulted to the US cloud endpoint while the project actually lives
   on `jp.cloud.langfuse.com` — every LiteLLM generation call was silently
   shipped to the wrong region and vanished (no error raised; §4 of the spec:
   "network failures log a warning, never raise"). **Fixed**: added
   `LANGFUSE_HOST="https://jp.cloud.langfuse.com"` to `.env` (config-only,
   mirrors the existing `LANGFUSE_BASE_URL`; not committed — `.env` is
   gitignored).
2. **Groq's primary model `qwen/qwen3-32b` (first entry in `LLMGateway.MODEL_CHAIN`)
   returns 404 `model_not_found`** ("does not exist or you do not have access
   to it") — this is a `NotFoundError`, not a rate limit, so
   `_is_rate_limit()` in `src/llm/gateway.py` correctly treats it as "move to
   next model" rather than "rotate key," and every real query in this
   verification run silently fell back to `groq/llama-3.3-70b-versatile`
   (Fallback 1). Confirmed via a Langfuse `GENERATION` observation with
   `level: ERROR` and the full Groq error body attached (see §3.3). **Not
   fixed** — out of scope for Task 4 (verification/docs only); flagged for a
   follow-up to update `MODEL_CHAIN`. This is exactly the "fallback attempts
   visible" value the design spec called out as the highest-value diagnostic
   (§3.1) — it worked, on the first real query.

## 1. Stage map (spec §3.2, for reference)

| Stage | Decorated point | Notes |
|---|---|---|
| trace root | `AdaptiveRAGPipeline.query` | non-stream root |
| trace root | `AdaptiveRAGPipeline.query_stream` | `@observe(capture_output=False)` |
| route | `_classify_and_scope` | |
| retrieval | `LegalVectorStore.search_hybrid` | covers simple/medium/complex routes at one point |
| lookup | `_lookup_pipeline` | citation-lookup route visibility |
| rerank | `_rerank_results` | |
| expand | `_expand_cross_references` | M3.5 feature traced from day one |
| generate | LiteLLM `langfuse_otel` callback (no decorator) | model, tokens, cost, latency per attempt, incl. failures |
| verify | `_post_process` | 6-signal confidence + verifier |

## 2. Preflight

```
python -c "from dotenv import load_dotenv; load_dotenv(); from langfuse import get_client; print(get_client().auth_check())"
→ True
```

## 3. `/api/chat` (non-stream) — the guaranteed-correct nesting path

### 3.1. Pre-fix trace (evidence of bug #1)

Question: mh-001 (`data/eval/eval_set_multihop_v1.json`) — "Theo Bộ luật Tố
tụng dân sự 2015, khi phát hiện bản án đã có hiệu lực pháp luật cần xem xét
lại theo thủ tục giám đốc thẩm, những người nào có thẩm quyền kháng nghị bản
án đó?" `pipeline_used: "simple"` (classifier routed this single-document
multi-hop question to SIMPLE, not COMPLEX — see §5).

Trace `cfd03ddf99143555059a57d13b4cf0d0`
(`https://jp.cloud.langfuse.com/project/cmrqc6lpy001ead0ddwcfxwmb/traces/cfd03ddf99143555059a57d13b4cf0d0`):

```
query (root)
├── _classify_and_scope
└── _post_process
```

Only 3 observations, no generation data anywhere — confirms bug #1
(`LANGFUSE_HOST` missing → LiteLLM shipped generation spans to
`us.cloud.langfuse.com`, a different Langfuse region, where they are
invisible to this project). Stable across 6 polls over 90s (not a flush-delay
artifact).

### 3.2. Post-fix trace (medium route — rerank exercised)

Question: mh-002 — "Theo Nghị định 101/2024/NĐ-CP, trong trình tự, thủ tục
cấp Giấy chứng nhận quyền sử dụng đất lần đầu, cơ quan có chức năng quản lý
đất đai cấp huyện phải thực hiện những công việc gì thuộc khâu đăng ký đất đai
lần đầu?" `pipeline_used: "medium"`, `confidence: high (0.956)`, 8 citations.

Trace `ddbea94403e103e0d585b80df1cd4294`
(`https://jp.cloud.langfuse.com/project/cmrqc6lpy001ead0ddwcfxwmb/traces/ddbea94403e103e0d585b80df1cd4294`),
8 observations, all correctly nested under the single `query` root — **no
orphan trace**:

```
query (root, f6478d7214084e84)
├── _classify_and_scope (f6437cc63ed2c5c2)
│   └── raw_gen_ai_request (b49495894d1140c4) — model gemini-3.5-flash, 36,150 tokens
├── search_hybrid (6ed6a55c7d50cec5)
├── _rerank_results (504b52a5aa4f6e64)
├── raw_gen_ai_request (fc70593f18bdc53a) — model llama-3.3-70b-versatile, 4,122 tokens (main answer)
└── _post_process (89f269558cbaf71d)
    └── raw_gen_ai_request (a8ec7c8ecf435bec) — model llama-3.3-70b-versatile, 1,819 tokens (confidence rubric)
```

`_expand_cross_references` did not fire — the classifier routed this
question MEDIUM, not COMPLEX (see §5). `search_hybrid` and `_rerank_results`
both fired and nested correctly, which is the higher-risk half of the
dashboard check (the same call site the M3.5 cross-ref feature already has
extensive standalone coverage for — `docs/notes/m3-5-expansion-results.md`).

## 4. Generation typing: data present, but `type: SPAN` not `type: GENERATION`

Read directly off the observation JSON for the three `raw_gen_ai_request`
spans in trace `ddbea94403e103e0d585b80df1cd4294`:

- classify call: `llm.gemini.modelVersion=gemini-3.5-flash`,
  `llm.gemini.usageMetadata.totalTokenCount=36150` (32,739 cached — the
  static corpus-summary prefix)
- main answer: `llm.groq.model=llama-3.3-70b-versatile`,
  `llm.groq.usage.total_tokens=4122`
- confidence rubric: `llm.groq.model=llama-3.3-70b-versatile`,
  `llm.groq.usage.total_tokens=1819`

All three are `"type": "SPAN"`, and their parent `@observe` spans
(`_classify_and_scope`, `_post_process`, `query`) are also `"type": "SPAN"`
— none is Langfuse's native `"type": "GENERATION"`. Top-level `model`/`usage`/
`cost` fields on these observations are `null`/zero; the real data lives only
in `metadata.attributes` (provider-specific keys: `llm.groq.*`,
`llm.gemini.*`).

**Root cause** (read from
`litellm/integrations/opentelemetry.py::_handle_success`): LiteLLM only
creates its own primary `litellm_request` span (the one whose creation path
calls `LangfuseOtelLogger.set_langfuse_otel_attributes`, which sets
`langfuse.observation.type=generation`) when **no parent OTel span is
already active** (`should_create_primary_span = parent_span is None or
USE_OTEL_LITELLM_REQUEST_SPAN`). Because Tasks 1-3 wrap every calling method
in `@observe`, a parent span is *always* active when LiteLLM's callback
fires on the sync path — so LiteLLM instead calls
`set_attributes(parent_span, ...)` (which appears to be a no-op: the
`@observe` span has typically already been `.end()`-ed by the time
LiteLLM's callback runs, and OTel silently drops attribute writes on ended
spans) and only successfully creates the small `raw_gen_ai_request` child
span (`set_raw_request_attributes`, not `set_langfuse_otel_attributes` — no
`langfuse.observation.type` set, so it stays `SPAN`).

Confirmed by contrast on the stream path (§5): when context is lost
entirely, LiteLLM finds **no** parent span, takes the
`should_create_primary_span=True` branch, and correctly produces
`"type": "GENERATION"` observations with populated `model`/`usage`/`cost`
top-level fields (e.g. `e12fdffbce141e3c`, `model: llama-3.3-70b-versatile`)
— including one with `level: ERROR` for the failed `qwen/qwen3-32b` 404
(see §3, bug #2). So the mechanism works; it just doesn't fire when a parent
span already exists, which is always, on the sync path.

**Net effect**: the M4 gate's literal requirement ("Langfuse trace shows the
full stage set per query") is met — model, prompt, and token counts are
readable in the trace, nested correctly, no orphan. But Langfuse's
dashboard-native cost/token summary columns (which read the structured
`model`/`usage` fields) will show empty/zero for these calls. **Not fixed in
this task** (Task 4 is verification-only; the fix, if wanted, is likely
`USE_OTEL_LITELLM_REQUEST_SPAN=true`, unverified — would need its own
live-query cycle to confirm, and budget did not allow chasing it this run).

## 5. `/api/chat/stream` — SSE nesting (spec §5, risk #1): broken as predicted

Question: mh-001 (reused). Curl returned 200 in 61s with a full streamed
answer (`__STEP__` events + citations + confidence). Classifier again routed
it SIMPLE (consistent with the pre-fix run — classification itself appears
deterministic at `temperature=0`, contra the earlier confidence-score noise
noted in project memory, which is a different code path).

Instead of one trace under `query_stream`, **6 separate root-level traces**
appeared:

| Trace id | Root observation | Children |
|---|---|---|
| `369e1da5fc7a91a00ea0a00c8f089b75` | `query_stream` (SPAN) | **none** — zero nested children |
| `39f744f4d42be0b52b4e530c94f188a2` | `_classify_and_scope` (SPAN) | 1× `raw_gen_ai_request` |
| `191b85d7702ad085c191257ebae003ce` | `_post_process` (SPAN) | 1× `raw_gen_ai_request` |
| `89a81433b6f474f60515791158ab73ba` | `litellm_request` (**GENERATION**, `level: ERROR`) | model `qwen/qwen3-32b`, 404 model_not_found |
| `434fdd383eaeca84011fa6b87c6e94a2` | `litellm_request` (GENERATION) | model `llama-3.3-70b-versatile` (main answer) |
| `e1a4c401e7a4f41443045039ee2956e5` | `litellm_request` (GENERATION) | model `llama-3.3-70b-versatile` + 1× `raw_gen_ai_request` child (confidence rubric) |

This is `iterate_in_threadpool` context loss exactly as anticipated: each
`next()` on the stream generator can run on a different worker thread with
its own contextvars copy, so the OTel context carrying "you are inside
`query_stream`" is not visible when `_classify_and_scope`, `_post_process`,
or the LiteLLM calls execute — each becomes its own trace root instead of a
child of `query_stream`. Within each individually-broken-off piece, nesting
*within that piece* is still correct (e.g. `_classify_and_scope`'s own
`raw_gen_ai_request` child nests fine) — only the cross-boundary link back to
`query_stream` is lost.

**Per the spec's accepted fallback (§5.1)**: this is not a task failure. The
M4 gate demo runs on `/api/chat` (§3), which is confirmed correct. SSE
nesting stays a known, documented limitation.

## 6. Operational notes

- **Cost figures are notional.** LiteLLM prices Groq/Gemini calls at list
  price for its cost-estimation fields; actual spend on all keys used in this
  verification is $0 (free tier). The `costDetails.total: 0` / `totalPrice: 0`
  fields seen in the raw observation JSON reflect $0 free-tier usage, not a
  broken cost calculator.
- **Orphan traces from scripts** (spec §5.2): any `search_hybrid` call or
  LiteLLM call with no active `@observe` parent (eval Tier-1 runs, the
  enricher) creates its own one-span trace, same as the stream-path
  `litellm_request` traces above. Use `LANGFUSE_TRACING_ENVIRONMENT=<tag>`
  (e.g. `eval`, `dev`) to filter these in the Langfuse UI — tagging only,
  never a kill-switch (a hijacked `LANGFUSE_TRACING_ENVIRONMENT` would
  silently drop traces for anyone else using it as documented).
- **Kill-switch**: `LANGFUSE_TRACING_ENABLED=false` disables both the
  `@observe` decorators and the LiteLLM callback with one env var (no key
  removal needed) — verified by code path in `src/observability.py`
  (`_tracing_enabled()`); not re-verified live in this task since it's
  covered by the existing unit tests (Tasks 1-3).
- **Test isolation**: `pytest tests/ -q` → 450 passed (no regressions; the
  `.env` change is config-only and gitignored, not part of the diff).

## 7. Screenshot

`docs/assets/m4-langfuse-trace.png` — reserved path; user to drop in a
dashboard screenshot of trace `ddbea94403e103e0d585b80df1cd4294` (§3.2)
showing the nested stage spans.

## 8. Budget used

3 real LLM-backed queries against the live stack (2× `/api/chat`, 1×
`/api/chat/stream`), all via Groq (with one live 404 fallback and one Gemini
classify fallback observed, both captured in Langfuse). No 429s encountered
— Groq TPD was not exhausted at verification time (2026-07-18, ~13:40-13:59
UTC).
