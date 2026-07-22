# M5 — Security tối thiểu + Deploy public (HF Spaces $0)

> Ngày: 2026-07-22 · Trạng thái: design đã duyệt, chờ implementation plan.
> Tiền đề: [`Plan_Giai_Doan_Cuoi_2026-07-10.md`](../../Plan_Giai_Doan_Cuoi_2026-07-10.md) §M5
> (5.1–5.4 + gate) và [`Danh_Gia_va_Roadmap_Production_2026-07-06.md`](../../Danh_Gia_va_Roadmap_Production_2026-07-06.md)
> §5.2 (kết luận Render 512MB bất khả thi) + §6.4 #19 (RAM query-time).

## Bối cảnh & vấn đề

M0–M4 đã xong: lõi Adaptive RAG + 600 docs (`luat_law_ctx_v4`) + memory/cache/
observability. Chưa có **mặt tiền public** — đây là artefact số 1 của portfolio
(URL demo bấm được). M5 = làm cho hệ thống **an toàn đủ mức** rồi **đưa lên
public** trên free tier, đo P95 thật, và có smoke test giữ nó xanh.

Đối chiếu code (HEAD `21d33fa`, 2026-07-22) xác nhận hiện trạng M5 phải xử lý:

- **CORS mở toang**: `allow_origins=["*"]` (`src/api/main.py:110`).
- **Admin endpoint không auth**: `/api/pipeline/enrich` (`main.py:392`) chạy
  enrichment (đắt, mutate metadata store) — ai gọi cũng được.
- **Không rate limit** ở bất kỳ đâu.
- **Quota cạn → 500 thô**: gateway `raise RuntimeError("All models failed...")`
  (`src/llm/gateway.py:231`) → `/api/chat` bọc thành `HTTPException(500)` → UI hiện
  lỗi kỹ thuật, không graceful.
- **UI không có cold-start UX**: `src/ui/app.py:37` chỉ ping `/health`, hiện
  🟢/🔴 — không phân biệt "đang khởi động (nạp model ~1–2 phút)" với "offline".
- **Metadata deploy blocker**: runtime thật = `data/metadata/store.db` (11 MB,
  **git-ignored**); chỉ có JSON **cũ 100-doc M1** được commit. Fresh HF build →
  `MetadataStore` auto-seed từ JSON cũ (`metadata_store.py:136`) → chạy 600-doc
  vectors trên **100-doc metadata** (temporal filter / effective-laws / cross-ref
  graph SAI).
- **Chưa có HF Space config**: có `Dockerfile` (2 target `api`/`ui`) +
  `docker-compose.yml` (local), chưa có gì cho HF Spaces.

## Quyết định khung (đã chốt với user 2026-07-22)

| Quyết định | Lựa chọn | Hệ quả |
|---|---|---|
| **Deploy topology** | **1 HF Docker Space**: Streamlit public (`app_port` 8501), FastAPI nội bộ trên `localhost:8000` | 1 URL, 1 cold start, secrets 1 chỗ. Streamlit gọi FastAPI **server-side qua localhost** → CORS gần như vô nghĩa; rate-limit/CORS/admin-key thành **defense-in-depth** chứ không load-bearing. Smoke test "từ ngoài" **không** đánh được `/api/chat` JSON trực tiếp → phải thiết kế lại (xem §6). |
| **Metadata shipping** | **Commit `store.db`** (bỏ khỏi `.gitignore`) | 0 code mới; file 11 MB ship thẳng trong image; auto-seed bị skip vì db đã có data (đúng bằng runtime db). Đóng luôn nợ "store.db→JSON export". |
| **Session/cache persistence** | **In-memory fallback** (M4 đã build) | $0-đơn giản nhất; session mất khi Space sleep-restart (48h) — không quan trọng với demo. Upstash Redis là opt-in **0 code** qua env `REDIS_URL`. |
| **Spec/phase** | 1 spec, 2 phase thực thi | Phase 1 = hardening (test được local); Phase 2 = deploy + đo + smoke/liveness CI. |

## Kiến trúc

```
HF Docker Space (1 container, app_port = 8501)
┌──────────────────────────────────────────────────────────┐
│  start.sh:  uvicorn (background, :8000)                   │
│           + streamlit  (foreground, :8501)                │
│           + trap → container chết sạch nếu 1 process rớt  │
│                                                            │
│  Streamlit (public :8501) ──localhost:8000──▶ FastAPI     │
│                                               + models     │
│                                               + store.db   │
└──────────────────────────────────────────────────────────┘
        │ Secrets (HF Space settings):
        │   QDRANT_URL/API_KEY · GROQ_API_KEY(S) · (+fallback keys)
        │   LANGFUSE_* · ADMIN_KEY · (REDIS_URL optional)
        ▼
  Qdrant Cloud (luat_law_ctx_v4, 600 docs)  ·  Groq + fallback chain
```

Model chỉ nạp trong process FastAPI; Streamlit nhẹ. `store.db` (committed) nằm
sẵn trong image. Model **pre-fetch lúc build** (Dockerfile `RUN`) để bake vào
layer → cold start chỉ nạp-từ-đĩa, không tải lại ~5 GB.

## Sáu workstream

### 1. Security (5.1) — right-sized cho topology internal-API

Streamlit gọi FastAPI qua localhost server-side → những mục dưới là
defense-in-depth nhưng vẫn là code thật, mô tả được trong portfolio:

- **CORS**: `["*"]` → whitelist theo env `ALLOWED_ORIGINS` (comma-separated,
  mặc định rỗng = không cho cross-origin). `src/api/main.py:108`.
- **Admin auth**: header `X-Admin-Key` (so với env `ADMIN_KEY`) trên
  `/api/pipeline/enrich`. **Fail-safe**: `ADMIN_KEY` không set → endpoint trả
  **403** (disabled), không bao giờ mở. Dùng FastAPI dependency, không middleware.
- **Rate limit**: `slowapi` per-IP trên `/api/chat` + `/api/chat/stream`
  (mặc định 30/phút, env `RATE_LIMIT` override). README ghi rõ đây là
  defense-in-depth vì topology internal-API. `slowapi` là dep mới.

Không nhận secret từ client (đã đúng — không endpoint nào đọc secret từ body).

### 2. Resilience / graceful degradation (5.3)

- **Quota-exhaustion → thông báo tử tế thay vì 500**: bắt riêng
  `RuntimeError` "All models failed" từ gateway trong `/api/chat` (+ stream),
  trả 200 với payload degraded (`answer` = câu tiếng Việt "hệ thống đang quá tải
  hạn mức miễn phí, vui lòng thử lại sau", cờ `degraded: true`, `confidence:
  "low"`, `refused: false`). Non-degraded exception khác vẫn 500. Stream: emit
  1 `token` event + `done` với `degraded: true`.
- **Warm-up / readiness**: thêm `GET /ready` báo model đã nạp chưa (đọc
  `app.state.embedder`/`reranker` — lifespan đã background-load ở `main.py:94`).
  Trả `{"ready": bool, "models_loaded": {...}}`.
- **UI cold-start UX**: `src/ui/app.py` sidebar dùng `/ready` (không chỉ
  `/health`) → 3 trạng thái: 🟢 sẵn sàng / 🟡 đang khởi động (cold start ~1–2
  phút, model đang nạp) / 🔴 offline. Render `degraded` message như notice thân
  thiện, không stack trace.

### 3. Performance / P95 (gate)

- **Đo warm P50/P95 trên HF CPU** qua flow thật (script probe N câu MEDIUM/
  COMPLEX sau khi model đã nạp), ghi **số thật** vào README (trung thực kể cả
  chậm — không hứa <3s).
- **Không đổi code rerank pool một cách suy đoán**: pool hiện đã nhỏ (MEDIUM
  rerank dynamic 5–15 qua `_dynamic_top_k(min_k=5,max_k=15)`;
  `orchestrator.py:660`). Thêm env knob `RERANK_MAX_K` (mặc định 15) để **hạ
  xuống 10 CHỈ KHI** P95 đo được đòi hỏi. Measure-then-tune, không đổi mặc định.

### 4. Metadata shipping (blocker)

- Bỏ `data/metadata/store.db` khỏi `.gitignore`, `git add` (11 MB).
- Auto-seed (`metadata_store.py:136`) tự skip vì `documents` table đã có data.
- Dọn JSON cũ 100-doc gây hiểu nhầm: hoặc xóa khỏi tracking, hoặc thay bằng
  ghi chú "runtime = store.db". (Quyết định chi tiết ở plan; không để JSON cũ
  còn là "source of truth" đánh lừa auto-seed ở máy khác.)

### 5. Deploy (5.2)

- **HF Space `README.md` frontmatter**: `sdk: docker`, `app_port: 8501`, title/
  emoji. (File README riêng cho Space, hoặc frontmatter trên README repo — chốt
  ở plan.)
- **Dockerfile**: thêm target `space` gộp cả API + UI: cài deps, **pre-fetch
  models lúc build** (`RUN python -c "..."` nạp AITeamVN dense + BGE-M3 sparse +
  reranker + NLI để cache vào `~/.cache/huggingface` trong image), copy `src/` +
  `scripts/` + `data/metadata/store.db`, `CMD ["bash","start.sh"]`.
- **`start.sh`**: `uvicorn src.api.main:app --port 8000 &` rồi
  `streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0`
  (foreground) + `trap`/`wait -n` để container thoát nếu 1 process chết.
- **Secrets** qua HF Space settings (không commit). `.env` vẫn gitignored.
- **RAM budget**: ~7 GB model (AITeamVN dense ~2 + BGE-M3 sparse ~2 + reranker
  ~2 + NLI ~1) + app, trên 16 GB free. **Risk item**: đo RSS thật lúc deploy;
  nếu sát trần → cân nhắc bỏ NLI Tier 2 hoặc lazy hơn (ghi ở §Rủi ro).

### 6. Smoke test (5.4) — thích ứng topology internal-API

Vì API không public, "e2e từ ngoài" tách 2 phần:

- **Functional smoke** (`scripts/smoke_test.py`): tự launch API (uvicorn/hoặc
  image) với secrets, đánh `/api/chat` 4 câu chuẩn:
  1. Câu luật trả lời được → assert có citations **và** `refused=False`.
  2. Câu ngoài phạm vi (OOS) → assert `refused=True`.
  3. Câu dạng citation-lookup ("Điều X Bộ luật Y…") → assert route citation-lookup
     resolve (có citation đúng doc/điều).
  4. Follow-up 2 lượt (cùng `session_id`) → assert `condensed_question` khác câu
     gốc (condensation chạy) **và** vẫn trả lời được.
  Chạy **weekly trong CI** (Groq burn nhỏ, gated như Tier-2 — KHÔNG nightly).
- **Liveness**: ping URL Streamlit public (`GET /_stcore/health` → 200) → thỏa
  "URL public sống".

## Interfaces / thay đổi theo file (dự kiến — chi tiết ở plan)

| File | Thay đổi |
|---|---|
| `src/api/main.py` | CORS whitelist theo env; `slowapi` limiter + decorator; `X-Admin-Key` dependency trên enrich; `/ready`; bắt quota-exhaustion → degraded payload (chat + stream) |
| `src/ui/app.py`, `src/ui/pages/chat.py` | `/ready` 3-trạng-thái sidebar; render `degraded` notice |
| `Dockerfile` | target `space` + pre-fetch models + copy store.db |
| `start.sh` | mới — launcher 2 process + trap |
| `README.md` (Space) | frontmatter HF + P95 thật + ghi chú security defense-in-depth |
| `scripts/smoke_test.py` | mới — 4-query functional smoke |
| `.github/workflows/` | job smoke weekly (functional + liveness ping) |
| `.gitignore` | bỏ `data/metadata/store.db` |
| `pyproject.toml` | thêm `slowapi` |
| `.env.example` | thêm `ALLOWED_ORIGINS`, `ADMIN_KEY`, `RATE_LIMIT`, `RERANK_MAX_K` |

## Testing

- **Unit/integration (local, pytest)**: CORS whitelist enforce; enrich 403 khi
  thiếu `ADMIN_KEY` + 200 khi đúng key; rate-limit trả 429 khi vượt; `/ready`
  phản ánh trạng thái model; quota-exhaustion → degraded payload (mock gateway
  raise) **cả** chat lẫn stream; `RERANK_MAX_K` cắt pool đúng.
- **Smoke (CI weekly)**: 4 câu như §6, gated.
- **Verify-before-completion**: `make test` xanh trước khi claim; smoke chạy tay
  1 lần trước khi bật CI schedule.

## Gate mapping (M5)

| Gate (plan) | Thỏa bởi |
|---|---|
| URL public sống | Deploy (5) + liveness ping (6) |
| P95 warm ghi README | Workstream 3 |
| Smoke test xanh | Functional smoke (6) |
| Không secret trong repo/logs | Secret-scan Phase 2 + `.env` gitignored (verify lại) + Secrets qua HF settings |

## Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| HF CPU P95 xấu (reranker + NLI) | Đo sớm; `RERANK_MAX_K`→10; NLI chạy sau stream (không chặn token đầu); ghi số thật README |
| RSS model sát trần 16 GB | Đo RSS lúc deploy; nếu sát → NLI lazy/bỏ Tier 2 hoặc top-k nhỏ hơn |
| 2-process trong 1 container: 1 chết âm thầm | `trap`/`wait -n` trong start.sh → cả container restart; healthcheck |
| Cold start ~1–2 phút làm recruiter bỏ đi | UI cold-start notice rõ ràng; smoke weekly giữ ấm phần nào; ghi disclaimer |
| Commit store.db 11 MB phình git | Chấp nhận (repo portfolio gần đóng băng); 1 lần |
| Groq mất free tier | Fallback chain đã có; degraded message thay 500 |

## Ngoài scope (cố tình không làm)

- Auth/JWT/RBAC, multi-tenant, billing (ngoài định vị portfolio).
- Reverse-proxy để public cả API lẫn UI trên 1 Space (đã chọn UI-public/API-
  internal).
- Upstash Redis wiring (để opt-in 0-code; in-memory đủ cho demo).
- Đổi mặc định rerank pool (chỉ tune sau khi đo).
