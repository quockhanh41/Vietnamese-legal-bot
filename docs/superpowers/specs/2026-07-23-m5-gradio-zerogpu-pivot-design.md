# M5 — Pivot deploy: HF Docker Space ($0) → Gradio + ZeroGPU ($0)

> Ngày: 2026-07-23 · Trạng thái: design đã duyệt, chờ implementation plan.
> Tiền đề: [`2026-07-22-m5-security-deploy-design.md`](2026-07-22-m5-security-deploy-design.md)
> (spec M5 gốc — Phase 1 hardening đã xong, Task 1-15 trong
> [`docs/superpowers/plans/m5.md`](../plans/m5.md) đã implement & commit).
> Đây là **addendum**, không thay thế spec gốc: Phase 1 giữ nguyên 100%; chỉ
> Phase 2 (deploy target) đổi.

## Vì sao pivot

Khi thực thi Task 16 (deploy), `hf repo create --repo-type space --space_sdk
docker` trả về `402 Payment Required`. Xác minh qua tài liệu HF chính thức
(`spaces-overview`, `spaces-gpus`):

> "Static Spaces are free for everyone. Gradio and Docker Spaces run on
> compute and require a paid plan to create: PRO for personal accounts...
> Free personal accounts in good standing can still host up to **2 Gradio
> Spaces running on ZeroGPU**."

Đây là chính sách hiện hành của HF (khác với giả định "$0" lúc viết spec
gốc) — **Docker SDK không có ngoại lệ free nào**, bất kể hardware. Ngoại lệ
free duy nhất là **Gradio SDK + hardware `zero-a10g` (ZeroGPU)**. Tài khoản
`quockhanh41`: `isPro: false`, không org — không đủ điều kiện Docker/cpu-basic
free.

**Ràng buộc cứng đã chốt với user (2026-07-23): $0/tháng, không thẻ ở bất kỳ
đâu.** Loại toàn bộ HF PRO ($9/mo) và mọi cloud "free tier nhưng cần thẻ"
(GCP Cloud Run, AWS, Azure, Oracle Always Free). Các platform free-không-thẻ
khác (Render, Streamlit Community Cloud) đã bị loại từ roadmap gốc
(`Danh_Gia_va_Roadmap_Production_2026-07-06.md` §5.2: "Render 512MB bất khả
thi") vì RAM model (~7 GB: AITeamVN dense + BGE-M3 sparse + reranker + NLI)
vượt xa trần RAM của các platform đó (~512 MB–1 GB).

⇒ **Gradio + ZeroGPU trên HF là lựa chọn khả thi duy nhất** thỏa cả hai ràng
buộc (đủ RAM/compute, $0 không thẻ). Bonus: model chạy trên GPU (A10G/
Blackwell) thay vì CPU — nhanh hơn số P95 mà spec gốc dự tính cho CPU.

## Quyết định khung

| Quyết định | Lựa chọn | Hệ quả |
|---|---|---|
| **Deploy target sống (public)** | Space mới, **`sdk: gradio`**, hardware `zero-a10g` sau khi tạo | Dùng 1/2 slot free Gradio+ZeroGPU của account. |
| **FastAPI + Streamlit + Docker stack (Phase 1 M5)** | **Giữ nguyên, KHÔNG xoá** | Không phải "live" demo nữa, nhưng vẫn đúng, test được (484+ test), mô tả là "self-hosted Docker deploy" trong README — vẫn là portfolio-worthy (CORS/rate-limit/admin-key/degraded là code thật). |
| **App mới cho Gradio Space** | File mới `src/ui/app_gradio.py`, **additive** — gọi thẳng pipeline in-process (không qua HTTP) | Tái dùng `LegalVectorStore`, `build_embedder`, `MetadataStore`, `CitationLookup`, `LLMGateway`, `NaiveRAGPipeline`/`AdaptiveRAGPipeline`, `run_chat_turn`, `condense_query`, `SemanticCache` — **không sửa** các module RAG lõi. |
| **Session/history** | Bỏ `SessionStore` cho path này, dùng `history` built-in của `gr.ChatInterface` | Đơn giản hơn — Gradio quản lý theo session sẵn. `SessionStore`/FastAPI path vẫn giữ `SessionStore` nguyên vẹn. |
| **Degraded-message logic** | Extract `DEGRADED_MSG` + `_is_exhaustion()` từ `main.py` → module mới `src/rag/degraded.py`, cả hai import | **Có sửa `main.py`** (đổi 1 import) — không phải zero-diff, nhưng không đổi hành vi/test. |

## Kiến trúc

```
HF Gradio Space (hardware: zero-a10g, free)
┌───────────────────────────────────────────────────────────┐
│  app_gradio.py (import spaces TRƯỚC torch/mọi lib CUDA)    │
│                                                               │
│  Module scope (main process, không GPU thật):                │
│    embedder = build_embedder(..., device="cuda")  ← "eager"  │
│    reranker = FlagReranker(..., device="cuda")               │
│    llm = LLMGateway(); metadata_store; citation_lookup; ...  │
│    (spaces monkey-patch torch.cuda.* → .to("cuda") không lỗi,│
│     weight "pack" xuống đĩa, KHÔNG chiếm VRAM tới khi có     │
│     @spaces.GPU call thật — xem "ZeroGPU boundary" bên dưới) │
│                                                               │
│  @spaces.GPU(duration=<đo, xem §Duration>)                   │
│  def handle_turn(message, history):                          │
│      # TRONG boundary: condense → cache lookup (embed) →     │
│      #   [hit: trả cached] / [miss: retrieve→rerank→         │
│      #   generate(Groq, network)→NLI verify→cache.set]       │
│      # Toàn bộ turn 1 GPU allocation (kể cả Groq network call)│
│      yield ...                                                │
│                                                               │
│  gr.ChatInterface(fn=handle_turn, type="messages")            │
│    .launch()                                                  │
└───────────────────────────────────────────────────────────┘
        │ Secrets (HF Space settings, giống Phase 1):
        │   QDRANT_URL/API_KEY · GROQ_API_KEY(S) · fallback keys
        ▼
  Qdrant Cloud (luat_law_ctx_v4)  ·  Groq + fallback chain
```

## ZeroGPU boundary — chi tiết (đã verify qua `references/zerogpu.md`)

**Model loading — `device="cuda"` tại module scope LÀ ĐÚNG, không phải bug.**
`import spaces` monkey-patch `torch.cuda.*` trong main process, nên
`.to("cuda")`/`device="cuda"` chạy được **không cần GPU thật** lúc import:
tensor ở lại RAM, "pack" xuống đĩa ở bước startup, main process không giữ
weight nào sau đó. Khi có `@spaces.GPU` call, scheduler fork một GPU worker,
stream weight đĩa→VRAM. Đây là pattern HF khuyến nghị chính thức (ví dụ
`DiffusionPipeline(...).to("cuda")` ở module scope trong tài liệu HF). Dùng
chuỗi `"cuda"`, không dùng device id nguyên (`.to(0)`) — ZeroGPU cấp device id
khác nhau mỗi request.

**1 boundary cho cả turn, bao gồm cache lookup.** `@spaces.GPU` decorate hàm
`handle_turn` — bao trọn: condense câu hỏi → **cache lookup (gọi
`_embed_query`, cần embedder)** → nếu miss thì retrieve/rerank/generate
(Groq, network I/O)/NLI verify → cache set. Đặt cache lookup TRONG boundary
(không phải trước/ngoài) để không có model call nào chạy ngoài GPU
allocation — đơn giản, đúng, đổi lại: 1 cache-hit vẫn tốn 1 allocation nhỏ chỉ
để tính embedding (rẻ, ~vài chục ms). Giữ Groq network call trong cùng
boundary luôn (không tách riêng) — tách sẽ phải refactor pipeline nội bộ,
không đáng so với lãng phí quota nhỏ khi giữ GPU trong lúc chờ network.

**Duration — đo, không đoán.** Theo đúng methodology của HF doc: ship
placeholder (180s), instrument bằng `time.perf_counter()`, chạy 2-3 câu đại
diện (MEDIUM + COMPLEX) qua `gradio_client` sau khi deploy, set
`duration = round(measured_max × 1.4)`. Nếu MEDIUM/COMPLEX lệch nhiều, dùng
callable input-dependent (`@spaces.GPU(duration=_estimate)`) thay vì hằng số.
Khớp đúng tinh thần "measure-and-report" đã dùng cho `RERANK_MAX_K`/P95 ở
spec gốc — không đặt số trước khi có dữ liệu thật.

**Quota-exceeded UX.** Visitor có hạn mức ZeroGPU hàng ngày (số phút cụ thể:
xem `spaces-zerogpu` doc lúc implement, đổi theo thời gian). Khi
`duration` vượt hạn mức còn lại, `spaces` raise lỗi riêng (không phải
`RuntimeError("All models failed...")` của gateway) — bắt lỗi này trong
`handle_turn`, trả message tiếng Việt thân thiện KHÁC với `DEGRADED_MSG`
(vd: "Bạn đã dùng hết lượt dùng thử miễn phí hôm nay, quay lại vào ngày mai
nhé!"), không để lộ traceback lên UI.

**Concurrency.** ZeroGPU chạy handler đồng thời theo mặc định — không dùng
mutable global state ngoài các model đã load 1 lần (read-only, an toàn); state
theo-turn (câu hỏi, cache) đều local trong hàm, không module-level mutable.

## Những gì KHÔNG đổi

- Toàn bộ `src/rag/`, `src/pipeline/`, `src/data/`, `src/db/`, `src/llm/`:
  không sửa. `app_gradio.py` chỉ **gọi** các module này.
- `src/api/main.py`, `src/ui/app.py`/`pages/chat.py`, `Dockerfile`,
  `start.sh`: giữ nguyên, vẫn test được, mô tả trong README là deploy path
  tự-host (Docker, full security hardening) — không phải path public sống.
  Ngoại lệ 1 dòng: import `DEGRADED_MSG`/`_is_exhaustion` từ `src/rag/
  degraded.py` thay vì định nghĩa tại chỗ (không đổi hành vi/test).
- 484+ test hiện có: không đổi, không có test nào cần xoá.

## Testing / verify

- Logic thuần (adapter Gradio-history→`condense_query`, `src/rag/
  degraded.py`): unit test pytest bình thường, không cần GPU — cùng convention
  hiện tại của repo.
- `app_gradio.py` (phần `@spaces.GPU`): **không test local được** — theo đúng
  hướng dẫn chính thức của HF ("iterate ON the Space, not locally"). Check
  local tối đa: `python3 -m py_compile src/ui/app_gradio.py` + import sanity
  (không load model thật, tốn thời gian/bandwidth vô ích).
- Verify thật sau khi push: `hf spaces logs` (build sạch, model pack xong) →
  `gradio_client.Client(...).predict(...)` gọi thử 1 câu MEDIUM + 1 câu
  COMPLEX (đo `duration` thật theo methodology ở trên) → xác nhận output có
  citation, đúng ngôn ngữ, không lỗi.
- CI (`smoke.yml`): job `functional-smoke` hiện tại (spin uvicorn local, hit
  `/api/chat`) **giữ nguyên** — vẫn test path Docker/FastAPI tự-host. Job
  `liveness` (hiện ping `/_stcore/health` Streamlit) **đổi** sang ping Space
  Gradio công khai + optional 1 lần gọi `gradio_client`. Chi tiết YAML để ở
  implementation plan, không chốt ở đây.

## Deploy mechanics

- `requirements.txt` (root Space repo) — chỉ dep runtime thật cần: FlagEmbedding,
  sentence-transformers, transformers, langchain*, qdrant-client, litellm,
  python-dotenv, networkx, langfuse, opentelemetry-*. **Không** liệt kê
  `gradio`/`spaces`/`huggingface_hub` (preinstalled, pin gây vỡ runtime).
  **Không** đem theo dep chỉ-dùng-eval (`ragas`, `deepeval`, `datasets`).
- README Space mới, file `README_GRADIO.md` ở repo root (khác `README_HF.md`
  — file đó giữ nguyên làm doc cho path Docker tự-host): `sdk: gradio`,
  `app_file: src/ui/app_gradio.py`,
  `python_version`, `startup_duration_timeout` nới cho model pre-download,
  và ghi rõ giới hạn (1/2 slot free ZeroGPU đã dùng, hạn mức GPU/ngày theo
  visitor, cold start lần đầu).
- Secrets: `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY` (+ fallback keys) —
  set qua HF Space Secrets, giống Phase 1. `ADMIN_KEY`/`RATE_LIMIT`/
  `ALLOWED_ORIGINS` chỉ còn ý nghĩa cho path Docker tự-host (không dùng ở
  Gradio Space).
- Tạo Space + set hardware: CLI hiện tại trên máy (`hf repo create`) không có
  cờ `--flavor`/`--secrets` (bản cũ hơn spec skill giả định) — set hardware
  `zero-a10g` qua Settings UI hoặc `hf spaces settings` nếu subcommand tồn
  tại; verify lúc implement (`hf spaces --help`), không chốt trước.

## Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| `duration` đặt sai (quá thấp = silent truncate; quá cao = lỗi "illegal duration") | Methodology đo-rồi-set (không đoán); ship placeholder 180s, đo qua `gradio_client`, chốt `measured_max × 1.4` |
| Visitor hết hạn mức GPU/ngày giữa demo | Message tiếng Việt riêng cho quota-exceeded (không phải `DEGRADED_MSG`), không crash UI |
| Cold worker (weight đĩa→VRAM) chậm lần đầu sau idle | Chấp nhận — ghi rõ trong README, tương tự cold-start disclaimer Phase 1 |
| 2 "câu chuyện" deploy song song (Docker tự-host không sống + Gradio sống) dễ gây nhầm README | README phân biệt rõ 2 mục: "Live demo (Gradio + ZeroGPU, free)" vs "Self-hosted (Docker, full security hardening)" |
| `FlagEmbedding`/`sentence-transformers` device="cuda" tại module scope có thể có edge-case khác `diffusers` (chưa verify thực nghiệm) | Verify bằng cách push thật + đọc `hf spaces logs`; không giả định — đúng tinh thần "iterate on the Space" |

## Ngoài scope

- Xoá/refactor `src/api/main.py`, `Dockerfile`, `start.sh`, Phase 1 hardening
  — giữ nguyên như đã chốt.
- Tối ưu/quantize model để fit platform free-RAM-thấp khác (Render, Streamlit
  Cloud) — đã loại ở bước so sánh phương án (rủi ro chất lượng retrieval đã
  calibrate).
- AoTI compile, real-time session ZeroGPU — không cần cho use case chat
  thông thường của demo này.
