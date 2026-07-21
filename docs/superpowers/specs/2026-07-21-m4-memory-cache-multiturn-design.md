# M4 — Production polish: multi-turn memory · query condensation · semantic cache · multi-turn eval

> Ngày: 2026-07-21 · Trạng thái: design đã duyệt (user "all oke"), chờ implementation plan.
> Tiền đề: [`Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §3 M4
> (4.1 memory + condensation, 4.2 semantic cache, 4.4 eval multi-turn — 4.3 Langfuse ✅ xong).

## Bối cảnh & hiện trạng

M4.3 (Langfuse) đã live-verified. Ba mục còn lại **chưa bắt đầu**:

- **4.1** — `session_id` chỉ được khai báo trong `ChatRequest` (`src/api/main.py:35`) rồi **vứt đi**;
  Redis service có sẵn trong `docker-compose.yml` (committed, `REDIS_URL=redis://redis:6379`)
  nhưng **0 dòng redis trong `src/`**; UI (`src/ui/pages/chat.py`) giữ `st.session_state.messages`
  client-side nhưng **không gửi `session_id`** → backend xử lý mỗi câu độc lập.
- **4.2** — chưa có cache.
- **4.4** — `eval.py` có `--multihop` (M3.5) nhưng chưa có runner multi-turn.

Facts đã verify: `MODEL_QUERY_TRANSFORM` tồn tại (`orchestrator.py:53`, Groq
`llama-3.3-70b-versatile` primary — hợp cho condensation); `LLMGateway.generate`
nhận `(prompt, system_prompt, model, temperature, max_tokens, json_mode, reasoning_effort)`;
`redis` **chưa** là python dependency (phải thêm vào `pyproject.toml`, import lazy);
chưa có `fakeredis` → Redis store nhận **injected client** để test được không cần Redis thật.

## Nguyên tắc dẫn đường

Ba năng lực mới đều là **enhancement ở tầng serving (API)**. Pipeline RAG
(`orchestrator.py`) giữ nguyên là hàm thuần của một `question: str` — **không đụng**.
Hệ quả:
- `eval.py` (gọi thẳng `pipeline.query`) không bị memory/cache xen vào → số đo sạch.
- Mỗi năng lực test độc lập, tắt độc lập.
- **Fail-open toàn tuyến**: nếu Redis / condensation / cache hỏng hết, chat chạy
  **y hệt hôm nay**. Best-effort — khớp tinh thần graceful-degradation của M5, quan
  trọng với free tier ($0).

Giữ nguyên 2 cổng từ chối (scope gate + confidence gate) — condensation chạy TRƯỚC
routing nên mọi route phía sau chỉ thấy câu standalone; không cổng nào bị nới.

## Component 1 — `src/rag/session_store.py` (conversation memory, 4.1)

```
SessionStore (ABC):
    get_history(session_id) -> list[Turn]        # tối đa MAX_TURNS gần nhất
    append_turn(session_id, question, answer)    # cap + TTL

MemorySessionStore   # dict[session_id -> deque(maxlen=MAX_TURNS)], process-local, ephemeral
RedisSessionStore    # RPUSH turn JSON; LTRIM giữ MAX_TURNS; EXPIRE TTL mỗi lần ghi
SessionStore.from_env()   # REDIS_URL có → Redis (client injectable), else Memory
```

- `Turn = {"question": str, "answer": str}` — lưu **câu gốc user gõ** (bản ghi trung
  thực, đúng thứ UI hiển thị).
- `redis` import **lazy** trong nhánh Redis → path memory không cần redis cài đặt.
- Redis lỗi giữa request → catch, coi như history rỗng (không bao giờ 500 vì Redis).
- `MAX_TURNS = 10`, `TTL = 3600s` (constants trong module).

## Component 2 — `src/rag/condense.py` (query condensation, 4.1)

```
condense_query(history, question, llm) -> str:
    if not history:                              # turn 1: KHÔNG gọi LLM
        return question
    rewritten = llm.generate(
        CONDENSE_PROMPT.format(history=fmt(history), question=question),
        model=MODEL_QUERY_TRANSFORM, temperature=0,
        reasoning_effort="none", max_tokens=120,
    )
    return clean(rewritten) or question          # fail-open: rỗng/lỗi -> câu gốc
```

- Prompt: "Cho lịch sử hội thoại + câu hỏi tiếp theo, viết lại thành **câu hỏi độc
  lập** bằng tiếng Việt; nếu đã độc lập thì giữ nguyên; chỉ xuất ra câu hỏi."
- `@observe`-wrapped → hiện thành 1 stage trong trace Langfuse (khớp 4.3).
- Đây là hàm **thuần, importable** → eval 4.4 gọi trực tiếp.

## Component 3 — `src/rag/semantic_cache.py` (semantic cache, 4.2)

```
SemanticCache (ABC):
    get(query, collection) -> RAGResponse | None
    set(query, collection, response)

MemorySemanticCache  # list[(embedding, response_dict)], FIFO cap CACHE_MAX
RedisSemanticCache   # cùng cấu trúc, lưu list trong Redis; cosine scan client-side
SemanticCache.from_env(embedder_getter)   # REDIS_URL -> Redis, else Memory; None nếu SEMANTIC_CACHE_DISABLED
```

- Key **theo câu ĐÃ condensed** (post-condensation) → follow-up cũng hưởng cache.
- `namespace = collection_name` → đổi collection **tự invalidate** (không lẫn v4 cũ).
- Hit khi `cosine(embed(query), cached_emb) ≥ SIM_THRESHOLD` (**0.95**).
- Lưu **RAGResponse serialize thành dict** (`dataclasses.asdict`) — dùng chung cho
  cả Memory và Redis (Redis buộc JSON); `get` dựng lại RAGResponse. RAGResponse là
  dataclass field đơn giản (`citations` = list dict) nên asdict/JSON round-trip an toàn.
- Redis 7-alpine **không** có vector search → **quét client-side** ≤ `CACHE_MAX` (500)
  vector (dot-product, micro-giây). Chi phí thật duy nhất = embed câu ngắn trên
  embedder **đã nạp sẵn** → hit path bỏ qua retrieval+rerank+generate+verify nên
  dư sức < 500ms.
- Cache lỗi → coi như miss. Không bao giờ chặn request.

## Component 4 — `src/api/main.py` (wiring — thay đổi orchestration DUY NHẤT)

Khởi tạo trong `lifespan`: `app.state.session_store = SessionStore.from_env()`,
`app.state.semantic_cache = SemanticCache.from_env(get_embedder)`.

Luồng `/api/chat` (non-stream):
```
sid       = request.session_id
history   = session_store.get_history(sid)              # [] nếu sid None / lỗi
condensed = condense_query(history, request.question, llm)   # bỏ qua nếu history rỗng
hit       = semantic_cache.get(condensed, collection)
resp      = hit if hit else pipeline.query(condensed)
if not hit: semantic_cache.set(condensed, collection, resp)
session_store.append_turn(sid, request.question, resp.answer)
return resp (+ cache_hit: bool, condensed_question: str)     # 2 field mới ChatResponse
```

Luồng `/api/chat/stream` (SSE):
- Lookup y hệt (condense + cache.get).
- **Hit → replay** cached RAGResponse thành SSE: `citations` → `token`(cả câu) →
  `verification` → `done` — đúng các event UI đã xử lý → < 500ms.
- **Miss → stream bình thường**, gom `(full_answer, citations, confidence)` sau vòng
  lặp → dựng RAGResponse tối thiểu → `cache.set` + `append_turn`.
- Mọi call store/cache/condense bọc `try/except → degrade`. Không refuse do memory/cache.

`condensed_question` trả về để observability/demo (thấy follow-up được hiểu thành gì).

## Component 5 — `src/ui/pages/chat.py` (tối thiểu)

- `st.session_state.session_id = uuid4()` **một lần** khi khởi tạo session.
- Gửi kèm `session_id` trên cả `/api/chat` và `/api/chat/stream`.
- Caption nhỏ khi `cache_hit` (⚡ cached) hoặc khi `condensed_question != question`
  (🔗 hiểu là: …) — tín hiệu demo/portfolio. Không đổi gì khác.

## Component 6 — `scripts/eval.py --multiturn` + `data/eval/eval_set_multiturn_v1.json` (4.4)

- Eval set: **~10 kịch bản** 2–3 lượt, đa lĩnh vực (hình sự, dân sự, lao động, doanh
  nghiệp, đất đai, hôn nhân gia đình). Mỗi turn: `{question, expected_articles}`.
- Runner: thread history cục bộ; mỗi follow-up `condensed = condense_query(history, q)`;
  chấm **recall@5 của câu condensed** vs `expected_articles` (tái dùng
  `metrics.recall_row`); append `{question: q, answer: ""}` vào history.
  - History dùng **câu hỏi gốc các lượt trước** (answer rỗng) — đủ để test resolve
    tham chiếu ("vậy còn X thì sao?" trỏ về chủ đề lượt trước), deterministic,
    **0 quota generation**. `condense_query`/prompt chịu được history answer-rỗng.
- **Pass** một follow-up turn: `recall@5 ≥ 0.5` (bắt được ≥ nửa expected_articles;
  với turn 1 article thì nhị phân — trúng top-5 hay không). **Gate: ≥ 80% follow-up
  turns pass** (chỉ tính lượt follow-up, không tính turn 1 vì turn 1 không condense).
- Deterministic trừ call condense temp-0 (gate ≥80% dung sai nhiễu nhỏ).
- Eval **không** đụng API/cache — gọi thẳng `condense_query` + pipeline.
- **Verify mọi `expected_articles`** với corpus thật (memory ghi label từng stale
  2 lần — mọi label phải kiểm chứng, không copy từ set khác).

## Config (tối thiểu, không env-explosion)

Constants trong module: `SIM_THRESHOLD=0.95`, `MAX_TURNS=10`, `TTL=3600`,
`CACHE_MAX=500`, recall pass-threshold `0.5` (eval). Đọc env: `REDIS_URL` (chọn
backend), `SEMANTIC_CACHE_DISABLED` (toggle tắt cache → NoOp). Thêm `redis` vào
`pyproject.toml`.

## Error handling (tóm tắt)

| Điểm hỏng | Hành vi |
|---|---|
| condense lỗi/rỗng | trả câu gốc |
| session_store lỗi | coi như history rỗng |
| cache lỗi | coi như miss |
| session_id = None | không memory, chat 1-lượt bình thường |

Request chat **không bao giờ** fail vì memory/cache/condense.

## Testing

- `tests/test_session_store.py` — memory cap/get/TTL logic; Redis path qua **injected
  fake client** (không cần redis thật).
- `tests/test_condense.py` — history rỗng ⇒ **không gọi LLM** (assert), có history ⇒
  rewrite + clean, fail-open khi LLM raise/rỗng (mock LLM).
- `tests/test_semantic_cache.py` — hit/miss quanh SIM_THRESHOLD, namespace isolation
  (đổi collection), FIFO eviction (fake embedder vector deterministic).
- Smoke test runner `--multiturn` (1 kịch bản, mock/stub).
- `make test` giữ xanh (suite hiện ~421+); gate multi-turn eval ≥ 80%; latency
  cache-hit đo vào **`docs/notes/m4-cache-bench.md`** (số artefact, không phải assert
  CI dễ flaky).

## Gate M4 (từ plan)

- Kịch bản follow-up eval pass **≥ 80%**.
- Trace Langfuse đủ stage (4.3 ✅) — condensation thêm 1 stage.
- Cache hit trả lời **< 500ms** (đo, ghi số thật).

## Cố tình KHÔNG làm (YAGNI)

- Không RediSearch / vector-DB cho cache (client-side scan đủ ở quy mô demo).
- Không auth/ownership cho session (ngoài scope portfolio).
- Không cache pre-warm; không chia sẻ memory cross-process ngoài phần Redis cho sẵn.
- Không thread `session_id` vào `orchestrator` (giữ pipeline thuần).

## Artefact để lại

- `docs/notes/m4-cache-bench.md` — số latency cache hit/miss.
- Eval report multi-turn (pass-rate follow-up) — bằng chứng gate ≥80%.
- Screenshot trace Langfuse có stage condensation + demo multi-turn (M6 dùng lại).
