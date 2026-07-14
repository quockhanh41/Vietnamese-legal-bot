# Kiến trúc mã nguồn — Vietnamese Legal Knowledge Bot

Tài liệu này dành cho người **mới đọc mã nguồn**: nó trả lời "module nào nằm ở đâu,
chịu trách nhiệm gì" và "một câu hỏi của người dùng chạy qua những hàm nào, theo thứ tự
nào". Mọi mô tả đều kèm đường dẫn file/hàm để đối chiếu trực tiếp với code.

Sơ đồ workflow chi tiết đến mức hàm (Mermaid) ở [`WORKFLOWS.md`](WORKFLOWS.md).
Nếu bạn muốn hiểu **tại sao** hệ thống được thiết kế như vậy (bài toán, số đo, lịch sử
quyết định), đọc kèm [`Danh_Gia_va_Roadmap_Production_2026-07-06.md`](Danh_Gia_va_Roadmap_Production_2026-07-06.md).
Kế hoạch giai đoạn cuối (milestone M0–M6) ở
[`Plan_Giai_Doan_Cuoi_2026-07-10.md`](Plan_Giai_Doan_Cuoi_2026-07-10.md).
Lớp Knowledge Graph / phiên bản hoá có tài liệu riêng:
[`KNOWLEDGE_GRAPH_VERSIONING.md`](KNOWLEDGE_GRAPH_VERSIONING.md).

---

## 1. Tổng quan hệ thống

Hệ thống gồm 3 tiến trình khi chạy (xem `docker-compose.yml` / `Makefile`):

```
┌──────────────┐  HTTP / SSE   ┌─────────────────┐   Python call   ┌──────────────────┐
│ Streamlit UI │ ────────────▶ │ FastAPI backend │ ──────────────▶ │ AdaptiveRAG      │
│ (port 8501)  │               │ (port 8000)     │                 │ Pipeline         │
└──────────────┘               └─────────────────┘                 └────────┬─────────┘
                                                                            │
                                              ┌─────────────┬──────────────┼─────────────┐
                                              ▼             ▼              ▼             ▼
                                        Qdrant Cloud   BGE-M3        BGE-reranker   LLM chain (Groq
                                        (vectors)      (embedder)    + mDeBERTa NLI  → Gemini → OpenRouter)
```

Corpus hiện có 3 thế hệ song song trong Qdrant (xem mục 4):

- `luat_law_ctx_v4` — corpus tuyển chọn **600 văn bản, 6 lĩnh vực** (siêu tập của spine
  100-doc), 60.958 points (M2). Đây là `DEFAULT_COLLECTION_NAME` từ 2026-07-15 — chuyển
  cùng lúc với recalibrate ngưỡng từ chối (M2 Phase E): ngưỡng **0.700** là hàm của phân
  bố điểm trên chính collection này. Payload mỗi chunk mang `sector` + `is_in_effect`.
- `luat_law_ctx_v3` — corpus **100 văn bản, 6 lĩnh vực**, 23.727 points (M1). Từng là
  default 2026-07-12 → 07-15 (ngưỡng 0.650); giữ làm mốc baseline M1 cho scale journey.
- `luat_law_ctx_v2` — 6 văn bản luật hình sự, 2.242 points (M0.5). Giữ để so sánh
  trước/sau; dùng qua flag `--collection`.

Mã nguồn chia thành **2 mặt phẳng** độc lập:

- **Offline (ingest)** — `scripts/` + `src/data/`: tải dataset, tuyển corpus, chunk,
  embed, index vào Qdrant. Chạy một lần (hoặc khi mở rộng corpus).
- **Online (query)** — `src/rag/` + `src/api/` + `src/ui/`: nhận câu hỏi, truy xuất,
  sinh câu trả lời, kiểm chứng, quyết định trả lời hay từ chối. Chạy theo từng request.

Nguyên tắc thiết kế xuyên suốt: **thà từ chối còn hơn trả lời sai** — có 2 cổng từ chối
riêng biệt (mục 3.2 và 3.5).

---

## 2. Bản đồ module

### `src/rag/` — lõi runtime (đọc đầu tiên)

| File | Vai trò |
|---|---|
| `orchestrator.py` | **Trái tim của hệ thống.** `NaiveRAGPipeline` (baseline: dense search → generate) và `AdaptiveRAGPipeline` (routing + verify + confidence). Hai entry point: `query()` (đồng bộ) và `query_stream()` (streaming, dùng cho UI). Mọi route đều đi qua 3 bước hậu-retrieval chung: annotate Phase-2 → lọc hiệu lực → mở rộng lên Điều cha (mục 3.3). |
| `citation_lookup.py` | Route tra cứu trực tiếp cho câu hỏi dạng trích dẫn: `parse_citation()` bóc Điều/Khoản/Điểm/tên luật/năm **và số hiệu văn bản** ("59/2020/QH14", "145/2020/NĐ-CP") bằng regex, hiểu viết tắt đa lĩnh vực (BLHS, BLTTHS, BLDS, LDN...); `CitationLookup.resolve_doc()` xếp hạng ứng viên văn bản theo alias/số hiệu/năm; `lookup()` tra thẳng metadata trong Qdrant thay vì vector search — dùng **server-side filter** (`scroll_article()` lọc `document_id`+`article_number` trên payload index, M2.3), không còn scroll+cache toàn collection client-side; chỉ giữ index cấp document trong RAM (`_load_doc_index()` qua `get_document_index()`). Cũng là nơi sinh `corpus_summary()` (nhóm theo lĩnh vực, nạp vào scope gate) và `corpus_summary_compact()` (1 đoạn ngắn cho câu từ chối out-of-scope). |
| `confidence.py` | `ConfidenceScorer` — điểm tin cậy **6 tín hiệu** có trọng số (bảng ở mục 3.5); `REFUSAL_THRESHOLD = 0.700` (recalibrate M2 Phase E trên corpus 600 docs — lai lịch đầy đủ trong block comment ngay trên hằng số); 2 mẫu câu từ chối (`REFUSAL_ANSWER`, `OUT_OF_SCOPE_ANSWER_TEMPLATE`). |
| `verifier.py` | `HallucinationVerifier` — kiểm chứng 3 tầng: structural (Điều được cite có trong context không), grounding (mỗi claim `[n]` trỏ đúng khối nguồn — **tỉ lệ này là tín hiệu confidence thứ 6**), NLI entailment (mDeBERTa, premise chọn theo từng claim thay vì cắt mù context). |
| `context_blocks.py` | Tiện ích chung cho verifier + confidence: `parse_context_blocks()` tách context thành các khối `[n]`, `select_premise()` chọn đúng khối nguồn làm premise cho NLI/self-eval theo từng claim (fix M0.1 — thay cho `context[:2000]`). |

### `src/llm/` — cổng LLM

| File | Vai trò |
|---|---|
| `gateway.py` | `LLMGateway` bọc LiteLLM: chuỗi fallback cross-provider $0 (M1.7) `groq/qwen3-32b` → `groq/llama-3.3-70b` → `gemini/gemini-3.5-flash` → `openrouter/qwen3-next-80b-instruct:free` → `openrouter/llama-3.3-70b:free` → `openai/gpt-4o-mini`. Routing theo **prefix provider** (không match substring); provider thiếu key bị bỏ qua. **Mọi provider đều xoay nhiều key** khi rate-limit (`GROQ_API_KEYS`/`GEMINI_API_KEYS`/`OPENROUTER_API_KEYS`, plural thắng singular). Lưu ý quota: Groq TPD tính **per-organization** (nhiều key cùng org = 1 bucket), Gemini/OpenRouter per-account. `reasoning_effort="none"` tắt thinking cho Groq-Qwen3 **và Gemini 3.x** (Gemini trả content RỖNG trên budget nhỏ nếu không tắt). Kiểm key sống: `scripts/check_llm_keys.py`. |
| `prompts.py` | Toàn bộ prompt template: `SYSTEM_PROMPT` (trợ lý pháp lý **mọi lĩnh vực**), `GENERATION_PROMPT` (yêu cầu cite `[n]`), `COMPLEXITY_CLASSIFIER_PROMPT` (phân loại + scope gate, kèm few-shot đa lĩnh vực — M1.4), `QUERY_REWRITE_PROMPT`, `MULTI_QUERY_PROMPT`, `HYDE_PROMPT`, `CONFIDENCE_EVAL_PROMPT` (self-eval theo rubric 4 tiêu chí, code tự cộng điểm — M0.1). |

### `src/db/` — vector store

| File | Vai trò |
|---|---|
| `qdrant_client.py` | `LegalVectorStore` bọc Qdrant: tạo collection (dense 1024-dim COSINE + sparse, tùy chọn int8 scalar quantization), `upsert_chunks()` với **point ID deterministic** (`uuid5(document_id:chunk_index)` — chạy lại ingest không tạo trùng), `search_dense()`, `search_hybrid()` (RRF dense+sparse), và các helper **server-side** cho citation/temporal: `get_document_index()`, `scroll_article()`, `scroll_docs()` (M2.3), `set_is_in_effect()` (backfill payload). `_build_filter(exclude_superseded=True)` thêm `must_not is_in_effect == False` để **lọc văn bản hết hiệu lực ngay trên Qdrant** (M2.5) thay vì post-filter client-side. Trả về dataclass `SearchResult` — kiểu dữ liệu chảy xuyên suốt pipeline. |

### `src/data/` — phía ingest

| File / thư mục | Vai trò |
|---|---|
| `constants.py` | Nguồn chân lý duy nhất cho hằng số: `DEFAULT_COLLECTION_NAME = "luat_law_ctx_v4"`, bộ lọc `LEGAL_TYPES`, `MIN_CONTENT_LENGTH`... |
| `chunker.py` + `chunking/` | 5 chiến lược chunking (`article_level`, `hierarchical`, `fixed_size`, `semantic`, `sentence`) cùng trả về dataclass `LegalChunk` (chunk_text + metadata Chương/Điều/Khoản + `parent_chunk_text`). Factory: `create_chunker()`. Ingest dùng `HierarchicalChunker` cho luật thường và `ArticleLevelChunker` cho luật sửa đổi (mục 4). |
| `contextualizer.py` | Contextual Retrieval kiểu Anthropic nhưng **rule-based, không tốn LLM call**: ghép header `Luật \| Chương \| Điều \| Khoản` vào chuỗi **đem đi embed** (text hiển thị giữ nguyên). Đây là lý do collection có hậu tố `_ctx`. |
| `embedder.py` | `BGEEmbedder` bọc BGE-M3: một lần encode ra cả dense (1024-dim) lẫn sparse (lexical weights) — không cần BM25 riêng. Có xử lý MPS OOM cho macOS. |
| `ingest_checkpoint.py` | Checkpoint SQLite theo document (`data/ingest_progress/{collection}.sqlite`) cho `--resume` — tiết kiệm GPU-hour khi rớt session Kaggle, không phải cơ chế đảm bảo đúng đắn (point ID đã deterministic). |
| `metadata_store.py` | Kho **SQLite** (`data/metadata/store.db`, gitignored) lưu metadata enrichment mức document (graph, timeline, amendment) — M2.4, thay kho JSON+RAM cũ, API giữ nguyên. Fresh clone/CI: `store.db` rỗng → `__init__` **auto-seed từ `documents.json` + `graph/timelines.json`** (các file JSON vẫn là tracked source-of-truth). `is_in_effect` mặc định **None = chưa rõ** (fail-open — bộ lọc temporal chỉ loại khi biết chắc `False`; mặc định `False` cũ từng giấu Luật Đầu tư 2025 + NĐ 144/2021 khỏi mọi câu trả lời khi trích ngày hiệu lực thất bại). |
| `graph/`, `timeline/`, `amendment/` | Lớp enrichment — xem [`KNOWLEDGE_GRAPH_VERSIONING.md`](KNOWLEDGE_GRAPH_VERSIONING.md). `find_replacement` đúng ở bậc 100 docs qua 2 đợt sửa: (M1.8b) khớp số hiệu là chuẩn nhất, khớp tên phải đảm bảo văn bản thay thế **mới hơn**, không rõ ngày → không kết luận; (M1, 2026-07-12) thêm **hierarchy guard** — Nghị định/Thông tư không bao giờ được "thay thế" Luật/Bộ luật, kể cả khi clause trích xuất nói vậy (NĐ 76/2023 hướng dẫn từng đánh chết chính Luật PCBLGĐ 2022 mà nó hướng dẫn); (M2 Phase E, 2026-07-15) thêm **consolidation guard** — **Văn bản hợp nhất (VBHN)** chỉ hợp nhất luật + các bản sửa đổi vào một file, KHÔNG thay thế gì, nên không bao giờ được coi là văn bản thay thế (VBHN "hợp nhất Nghị định hướng dẫn Luật Nuôi con nuôi" từng đánh chết chính Luật Nuôi con nuôi 2010 ở bậc 600 docs → routed drop → hngd recall −25pt). |

### `src/pipeline/` — lớp enrichment lúc chạy

| File | Vai trò |
|---|---|
| `enricher.py` | Chạy 3 module enrichment (graph/timeline/amendment) trên toàn bộ document trong Qdrant, lưu vào `MetadataStore`. |
| `retriever.py` | `get_phase2_status()` — tra trạng thái temporal/amendment của một (document, Điều); **được pipeline chat chính dùng** để annotate + làm **backstop client-side** cho bộ lọc hiệu lực (server-side `exclude_superseded=True` là chính từ M2.5, mục 3.3) + badge "đã sửa đổi". `EnhancedRetriever` (mở rộng ngữ cảnh theo graph, lọc `at_date=`) vẫn chỉ phục vụ các endpoint Phase 2 (`/api/search/enhanced`...). |

### `src/api/` — backend

`main.py` là toàn bộ FastAPI app. Endpoint chính:

- `POST /api/chat` — non-streaming, chọn pipeline theo flag `use_adaptive`.
- `POST /api/chat/stream` — SSE; dịch các marker nội bộ của `query_stream()` thành event
  (chi tiết ở mục 3.6).
- `GET /api/documents`, `/api/stats` — phục vụ trang Documents.
- Nhóm Phase 2: `/api/pipeline/enrich`, `/api/search/enhanced`, `/api/graph/*`,
  `/api/timeline/effective`, `/api/article/history`.

Model nặng (BGE-M3, reranker, NLI) đều **lazy-load** ở lần dùng đầu — server khởi động
nhanh, request đầu tiên chậm.

### `src/ui/` — frontend Streamlit

- `app.py` — khung điều hướng 3 trang + kiểm tra backend `/health`.
- `pages/chat.py` — trang Q&A chính: gọi `/api/chat/stream`, hiện các bước trung gian
  (classification, routing, context...), render citation card `[n]`, badge confidence.
- `pages/documents.py`, `pages/eval_dashboard.py` — duyệt văn bản, dashboard eval.

### `src/eval/` + `scripts/` — đánh giá & tiện ích

- `src/eval/metrics.py` — hàm metric thuần (không I/O, không model) cho eval:
  `artnum()`, matching strict/lenient. Unit-test được độc lập.
- `scripts/eval.py` — runner 2 tầng: Tier 1 (retrieval: recall@5/@10, MRR, đo cả 3 chế
  độ dense/hybrid/routed — không tốn LLM) và Tier 2 (`--answers`: chạy full pipeline,
  đo refusal accuracy, citation accuracy — tốn LLM). Chọn bộ câu hỏi bằng
  `--set criminal|general|sectors`, chọn collection bằng `--collection`; báo cáo Tier 1
  có bảng **recall theo từng lĩnh vực** (artifact Gate M1). Report ghi vào
  `data/eval_reports/`. Hai quy ước Tier 2 (M1.7): `--retry <report>` chạy lại **chỉ**
  các câu lỗi (quota chết giữa run) và merge vào report cũ (`config.retry_of` giữ chuỗi
  lai lịch); citation accuracy = `None` khi bot từ chối — từ chối đã được đo ở
  false_refusal_rate, chấm 0 là đếm trùng một lỗi hai lần.
- `scripts/calibrate_threshold.py` — quét ngưỡng từ chối trên report Tier 2 (nguồn gốc
  con số `REFUSAL_THRESHOLD`, hiện 0.700 ở M2). Từ M1.7, sweep **chỉ chạy trên rows qua
  confidence gate** — refusal
  của scope gate mang điểm sentinel 0.0 bị tách ra section riêng (over-refusals +
  escaped-refuse), không được làm bẩn phân bố.
- `scripts/inspect_confidence.py` — soi breakdown 6 tín hiệu của từng câu hỏi (debug
  false-refusal).
- `scripts/check_llm_keys.py` — bắn 1 completion nhỏ cho từng cặp (model, key), bỏ qua
  fallback chain để key hỏng không trốn được; chạy mỗi khi đổi key trong `.env`.
- `scripts/select_corpus.py` — M1.1: trích 100 văn bản tuyển chọn tay ra
  `corpus_100docs.parquet` (lai lịch chọn lọc:
  [`notes/corpus-100docs-selection-2026-07-10.md`](notes/corpus-100docs-selection-2026-07-10.md)).
- `scripts/select_corpus_600.py` — M2.2: dựng `corpus_600docs.parquet` bằng cách đào sâu
  cùng 6 lĩnh vực (sector map + ngân sách chunk bounded ratio ≤3× chống frequency-bias +
  cổng review CSV cho user duyệt trước khi freeze).
- `scripts/measure_capacity.py` (M2.1: đo bytes/point từ telemetry Qdrant → chiếu vs trần
  free tier), `scripts/run_enrichment.py` (M2 Phase E: runner enrichment standalone cho
  Kaggle), `scripts/backfill_is_in_effect.py` (backfill payload độc lập).
- `scripts/ingest.py` — pipeline ingest chính (mục 4); `scripts/download_dataset.py`,
  `scripts/download_all_chunks.py`, `scripts/reindex_contextual.py`,
  `scripts/validate_eval_set.py` — tiện ích kèm theo. Chạy ingest trên Kaggle GPU:
  [`notes/kaggle-ingest-guide.md`](notes/kaggle-ingest-guide.md).

### `tests/`

Pytest cho các phần thuần logic: scope gate, citation lookup (gồm bộ đa lĩnh vực),
prompts đa lĩnh vực, calibration (gồm tách scope-gate rows), verifier, context blocks,
contextualizer, chunker (article-level + hierarchical), ingest checkpoint + bộ lọc
curated, orchestrator, retriever Phase-2, version resolver (gồm hierarchy guard),
eval metrics, eval retry/merge, gateway (routing provider + key rotation), metadata
store (fail-open temporal default). Chạy: `make test`.

**Thứ tự đọc code gợi ý:** `src/rag/orchestrator.py` (nắm luồng chính) →
`src/rag/confidence.py` + `verifier.py` (hiểu cổng từ chối) → `src/rag/citation_lookup.py`
→ `src/db/qdrant_client.py` → `scripts/ingest.py` + `src/data/contextualizer.py` →
phần còn lại khi cần.

---

## 3. Vòng đời một câu hỏi (luồng online)

Đường đi đầy đủ, lấy luồng streaming làm chuẩn (luồng non-streaming `query()` giống hệt,
chỉ khác không phát event trung gian). Sơ đồ chi tiết mức hàm:
[`WORKFLOWS.md`](WORKFLOWS.md) §1–§4.

```
UI chat.py ──POST /api/chat/stream──▶ api/main.py:chat_stream()
    └──▶ AdaptiveRAGPipeline.query_stream()   (src/rag/orchestrator.py)
```

### 3.1. Route trích dẫn (trước mọi thứ khác)

`parse_citation(question)` (`citation_lookup.py`) chạy **trước cả** classifier. Nếu câu
hỏi chứa "Điều N" hoặc số hiệu văn bản ("59/2020/QH14", "145/2020/NĐ-CP") → đi route
`lookup`: `resolve_doc()` xếp hạng văn bản ứng viên theo alias/viết tắt (BLHS, BLTTHS,
LDN... — hint quá dài được cắt tỉa về đúng một tên luật khớp, M1.3), số hiệu và năm, rồi
`lookup()` tra thẳng metadata Qdrant (tìm chunk **mở đầu** bằng "Điều N.") thay vì
vector search. Nếu lookup có kết quả, `retrieve_routed()` bổ sung thêm dense search
(khử trùng theo `point_id`) cho đủ top-10 để LLM có ngữ cảnh xung quanh. Lý do có route
riêng: vector search đo được recall@5 = 0.125 trên dạng câu này (chunk của luật sửa đổi
trích dẫn "Điều N" tràn lan làm nhiễu). Cố ý **không** fallback sang vector search khi
lookup rỗng ("Điều 999" không tồn tại → context rỗng → cổng tin cậy tự từ chối).

### 3.2. Phân loại + cổng phạm vi — 1 LLM call

`_classify_and_scope()` gọi LLM **một lần duy nhất** với `COMPLEXITY_CLASSIFIER_PROMPT`,
trả về đồng thời:

- `SCOPE: IN | OUT` — corpus có trả lời được không. Prompt được nhồi `corpus_summary()`
  — danh sách văn bản thực tế trong Qdrant, **nhóm theo lĩnh vực** khi collection có
  payload `sector` (v3+), không hard-code; kèm quy tắc + few-shot cho các ca dễ nhầm
  đa lĩnh vực (thủ tục có nghị định phủ là IN, trách nhiệm hình sự của hành vi ngoài
  lĩnh vực là IN, nhà ở/KDBĐS là OUT dù có Luật Đất đai...).
- `CLASS: SIMPLE | MEDIUM | COMPLEX` — độ sâu pipeline cần dùng.

Nếu `OUT` → từ chối **ngay, trước retrieval** bằng `OUT_OF_SCOPE_ANSWER_TEMPLATE`
(cổng từ chối thứ nhất; câu từ chối dùng `corpus_summary_compact()` — tên lĩnh vực +
số văn bản thay vì 100 tiêu đề). Parse bằng regex và **fail-open**: nếu không parse
được thì coi là `IN` + `MEDIUM` — lỗi classifier chỉ làm chọn sai route chứ không bao
giờ chặn nhầm câu hỏi thật.

> 🐞 **Bug đã ghi nhận (M1, chưa sửa):** ở corpus đa lĩnh vực, classifier thỉnh thoảng
> phán `OUT` nhầm cho câu có tài liệu thật trong corpus (Tier-2 v3 bắt được 5 ca, có ca
> retrieval recall@5 = 1.0). Là lỗi phán đoán của LLM ở biên đa lĩnh vực, không sửa
> được bằng ngưỡng — workstream riêng trong roadmap.

### 3.3. Ba route truy xuất + 3 bước hậu-retrieval chung

| Route | Các bước (đều trong `orchestrator.py`) |
|---|---|
| `_simple_pipeline` | dense search top-20 → `_dynamic_top_k()` cắt còn 3–15 |
| `_medium_pipeline` | hybrid search (dense + sparse) top-20 → dynamic top-k 5–15 → `_rerank_results()` (BGE-reranker-v2-m3 cross-encoder) |
| `_complex_pipeline` | rewrite query → multi-query (3 biến thể) → HyDE (văn bản giả định) → hybrid search cho cả 4 query → khử trùng lặp theo `point_id` → dynamic top-k 5–20 → rerank → `_compress_context()` (LLM lược context) |

`_dynamic_top_k()` co giãn số kết quả theo phân bố điểm: điểm cao tập trung → lấy ít,
điểm thấp/phẳng → lấy nhiều.

Sau retrieval, **mọi route** (kể cả lookup và naive) chạy 3 bước chung:

1. `_annotate_phase2()` — tra `get_phase2_status()` (temporal + amendment) cho từng kết quả.
2. `_filter_in_effect()` — loại chunk của văn bản **đã biết chắc** hết hiệu lực
   (`is_in_effect=False`). Từ M2.5 việc lọc này chạy **server-side ngay trên Qdrant**
   (`exclude_superseded=True` → `must_not is_in_effect==False`); `_filter_in_effect` giữ
   lại làm **backstop client-side**. Fail-open: thiếu metadata hoặc không rõ → giữ lại.
3. `_expand_to_parent()` — small-to-big: thay `chunk_text` (mức Khoản) bằng
   `parent_chunk_text` (nguyên Điều, đã tính sẵn lúc chunk), gộp các hit trùng Điều về
   một khối để `[n]` không lặp.

### 3.4. Dựng context và sinh câu trả lời

`_build_context()` đánh số các khối nguồn `[1] Tên luật — Điều N...`; đây chính là các
`[n]` mà LLM phải cite. `_build_citations()` gán `block_index` bằng **đúng phép đánh số
đó** (bất biến quan trọng: hai hàm phải nhận cùng một list kết quả, cùng thứ tự — xem
docstring `_build_citations`). Sau đó `GENERATION_PROMPT` + `SYSTEM_PROMPT` → LLM stream
từng token về client.

### 3.5. Hậu kiểm: verify + confidence (cổng từ chối thứ hai)

`_post_process()` chạy **sau khi** đã stream xong câu trả lời (không làm chậm token đầu):

1. **`HallucinationVerifier.verify()`** (`verifier.py`) — 3 tầng:
   - *Tier 1 structural*: mọi "Điều N" trong câu trả lời phải xuất hiện trong context.
   - *Tier 1.5 grounding*: mỗi bullet chi tiết phải cite `[n]` hợp lệ, và khối `[n]` đó
     phải thực sự chứa nội dung được gán cho nó. Tỉ lệ đạt (`grounding_ratio`) **đi vào
     điểm tin cậy** làm tín hiệu thứ 6 (M0.2).
   - *Tier 2 NLI*: tách câu trả lời thành claim, chấm ENTAILMENT/CONTRADICTION/NEUTRAL
     bằng mDeBERTa-v3-mnli-xnli; premise là **khối `[n]` được cite của từng claim**
     (`context_blocks.select_premise`), không cắt mù 2000 ký tự đầu context (M0.1).
2. **`ConfidenceScorer.score()`** (`confidence.py`) — 6 tín hiệu có trọng số:

   | Tín hiệu | Trọng số | Nguồn |
   |---|---|---|
   | retriever | 0.20 | max cosine từ search |
   | reranker | 0.20 | max điểm cross-encoder |
   | llm_self_eval | 0.25 | LLM tự chấm theo rubric 4 tiêu chí, code cộng điểm |
   | context_overlap | 0.15 | trùng lặp từ khóa answer↔context |
   | entailment | 0.10 | tỉ lệ claim được NLI xác nhận |
   | grounding | 0.10 | tỉ lệ bullet cite đúng khối nguồn (máy kiểm, từ Tier 1.5) |

   Tín hiệu thiếu (ví dụ route simple không có reranker) bị **loại và chuẩn hóa lại
   trọng số** — điểm tối đa luôn là 1.0, không bị "phạt ngầm".
3. Điểm `< 0.700` (`REFUSAL_THRESHOLD`, override bằng env `CONFIDENCE_REFUSAL_THRESHOLD`)
   → thay câu trả lời bằng `REFUSAL_ANSWER`. Ngưỡng recalibrate ở **M2 Phase E** trên
   corpus 600 docs (trước đó 0.650 ở M1.7 / 100 docs). Scope gate (mục 3.2) làm gần hết
   việc từ chối out-of-scope, nên confidence gate chỉ còn là lưới cuối cho câu **thiếu
   bằng chứng**; calibration vì vậy chỉ sweep trên rows qua confidence gate (xem
   `scripts/calibrate_threshold.py`). **Phát hiện M2**: trên OOS set nghiêm túc (17 câu,
   M1 chỉ có 1), gate "refusal_recall = 1.0" **không đạt được ở ngưỡng dùng được** —
   một câu confident-OOS (gen-903, conf 0.882) đẩy điểm 1.0 lên t≈0.900 (false-refuse
   ~79%). Chọn **t=0.700 balanced** (refusal_recall 0.765, false_refusal 0.136) thay vì
   Youden-optimal 0.775 (false-refuse 27%). Ceiling gen-903 là known gap → M3. Lai lịch
   đầy đủ trong block comment ngay trên hằng số ở `confidence.py`. Margin mỏng → **phải
   recalibrate ở mỗi bậc scale**.

### 3.6. Giao thức streaming UI ↔ backend

`query_stream()` yield chuỗi thường (token) xen kẽ các **marker** tiền tố:
`__STEP__{json}` (bước trung gian), `__CITATIONS__{json}`, `__VERIFICATION__{json}`
(kết quả hậu kiểm, đến sau cùng). `api/main.py:chat_stream()` dịch chúng thành SSE event
`step` / `citations` / `token` / `verification` / `done`; `ui/pages/chat.py` render
tương ứng (step vào expander, citation thành card, verification thành badge tin cậy /
thay câu trả lời bằng refusal nếu bị từ chối).

---

## 4. Pipeline ingest (luồng offline)

Thứ tự chạy (xem `Makefile`; corpus 100 docs xem thêm
[`notes/kaggle-ingest-guide.md`](notes/kaggle-ingest-guide.md)):

```
scripts/download_dataset.py      # tải dataset luật VN từ HuggingFace về data/raw/
        │
scripts/select_corpus.py         # (M1) tuyển 100 văn bản 6 lĩnh vực → corpus_100docs.parquet
        │
scripts/ingest.py                # đọc streaming bằng pyarrow (không load hết vào RAM)
        │   1. lọc: legal_type ∈ LEGAL_TYPES, độ dài ≥ MIN_CONTENT_LENGTH.
        │      File curated (có cột `sector`) tự nhận diện → bỏ lọc legal_type
        │      (đã tuyển tay), giữ guard độ dài, cảnh báo nếu rơi rớt document
        │   2. chunk: HierarchicalChunker (mức Khoản + parent_chunk_text nguyên Điều)
        │      cho luật thường; luật sửa đổi (nhận diện qua tiêu đề) dùng
        │      ArticleLevelChunker + đánh dấu is_amendment_chunk
        │   3. contextualize: prepend header "Luật | Chương | Điều | Khoản"
        │      vào bản đem đi embed (build_contextual_text) — text gốc giữ nguyên
        │   4. embed: BGE-M3 → dense + sparse trong 1 lần encode
        │   5. upsert vào Qdrant, point ID deterministic uuid5(document_id:chunk_index)
        ▼
Qdrant (payload: chunk_text, parent_chunk_text, document_title, article, article_number,
        chapter, clause, document_id, document_number, sector, url...)
```

Flag quan trọng của `scripts/ingest.py`: `--collection` (collection đích, mặc định
`DEFAULT_COLLECTION_NAME`), `--quantized` (int8 scalar quantization + fp32 on-disk khi
tạo collection mới — bắt buộc dùng ngay lần chạy đầu cho collection mới), `--resume`
(bỏ qua document đã xong theo checkpoint SQLite — dùng khi rớt session Kaggle/mạng),
`--device cuda` (chạy GPU).

Collection hiện có: `luat_law_ctx_v4` (600 docs, 60.958 points — corpus M2, **default
của code từ 2026-07-15**), `luat_law_ctx_v3` (100 docs, 23.727 points — baseline M1),
`luat_law_ctx_v2` (6 docs hình sự, 2.242 points — giữ để so sánh trước/sau), và các
collection cũ hơn. `scripts/reindex_contextual.py` re-embed một collection sẵn có sang
dạng contextual mà không phải ingest lại từ đầu.

Enrichment (graph/timeline/amendment) là bước offline **riêng**, chạy qua
`POST /api/pipeline/enrich`, `PipelineEnricher` trực tiếp, hoặc runner standalone
`scripts/run_enrichment.py --collection ...` (M2 Phase E — tiện chạy ngay trên Kaggle
sau ingest). Ghi metadata vào SQLite `data/metadata/store.db`, **và backfill
`is_in_effect` vào payload từng chunk trên Qdrant** (`set_is_in_effect`, M2.5) để lọc
hiệu lực chạy được server-side. Kết quả được pipeline chat dùng lúc chạy (mục 3.3).

---

## 5. Đánh giá & hiệu chuẩn

- Eval set (`data/eval/`): `eval_set_criminal_v1.json` (41 câu hình sự có ground-truth
  Điều luật), `eval_set_sectors_v2.json` (83 câu — v1 51 câu + câu nhắm Thông tư/văn bản 600-corpus,
  M2.7; `eval_set_sectors_v1.json` giữ làm subset apples-to-apples cho gate 100→600),
  `eval_set_general_v1.json` (41 câu biên/ngoài phạm vi — định nghĩa lại ở M1.6 cho
  corpus 100 docs: 27 câu nay phải **trả lời được**, 14 câu vẫn phải từ chối).
  `scripts/validate_eval_set.py` kiểm tra ground-truth có thực sự tồn tại trong corpus.
  **Quy tắc học bằng học phí (2 lần trả)**: mọi label out-of-scope và expected_articles
  phải verify trực tiếp với `corpus_100docs.parquet` + trạng thái `is_in_effect` — luật
  bị thay thế thì expected phải trỏ Điều tương ứng của luật kế nhiệm (ld-104/ld-105/
  dn-103 từng chấm 0 oan khi bot cite đúng luật mới).
- `scripts/eval.py` — Tier 1 đo retrieval trên cả 3 chế độ dense/hybrid/routed
  (strict = khớp payload, lenient = khớp text; khoảng cách giữa 2 số đo chính là độ
  lệch metadata của chunker), kèm bảng recall theo lĩnh vực; Tier 2 (`--answers`) đo
  hành vi trả lời/từ chối end-to-end. Số Tier 1 trên v3: routed recall@5 0.8971
  (criminal) / 0.8922 (sectors), recall@10 0.9216; citation-lookup và amendment 1.0.
  Số gate M1 (Tier 2, 2026-07-12): refusal recall 21/21, citation accuracy **0.8533**
  (pooled 2 full run × 51 câu sectors; criminal 0.8922) — nhiễu style trả lời giữa các
  model trong chain ±0.03-0.04/run, nên số gate đo bằng **mean của 2 run đầy đủ**,
  không bao giờ re-run chọn lọc câu fail.
- **Gate M2 (2026-07-15, corpus 600 docs v4)**: capacity 9.5% RAM / 6.0% disk (<70%);
  Tier-1 routed recall@5 (v1-subset, apples-to-apples 100→600) mọi lĩnh vực tụt ≤5pt
  (overall −3.0pt, doanh_nghiep watch-item giữ); refusal_recall=1.0 **không đạt được**
  (chọn t=0.700 balanced). Câu chuyện đầy đủ 6→100→600 + số per-sector từng bậc:
  [`notes/scale-journey-6-100-600.md`](notes/scale-journey-6-100-600.md).
- `scripts/calibrate_threshold.py` đọc report Tier 2 và quét ngưỡng **trên rows qua
  confidence gate** (sentinel scope-gate tách riêng), auto-discover report tier2 mới nhất
  mỗi set nếu không truyền `--reports` — quy trình đã chốt `REFUSAL_THRESHOLD = 0.700`
  (M2 Phase E; 0.650 ở M1.7) và phải chạy lại ở mỗi bậc scale.
- Số đo baseline hiện tại và lịch sử quyết định: xem
  [`Danh_Gia_va_Roadmap_Production_2026-07-06.md`](Danh_Gia_va_Roadmap_Production_2026-07-06.md).

---

## 6. Chạy dự án

```bash
make install          # pip install -e ".[dev]"
cp .env.example .env  # điền QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEYS
make qdrant           # Qdrant + Redis qua Docker (hoặc dùng Qdrant Cloud qua .env)
make ingest           # chỉ khi cần dựng lại index (corpus lớn: chạy Kaggle GPU, xem notes/)
make api              # uvicorn src.api.main:app --port 8000
make ui               # streamlit run src/ui/app.py --port 8501
make test             # pytest tests/ -v
make eval             # scripts/eval.py Tier 1
```

Biến môi trường quan trọng (`.env.example` có đủ danh sách): `QDRANT_URL` /
`QDRANT_API_KEY`; `GROQ_API_KEYS`, `GEMINI_API_KEYS`, `OPENROUTER_API_KEYS` (mỗi
provider nhận danh sách key phẩy ngăn cách — gateway tự xoay khi rate-limit; dạng
số ít `*_API_KEY` vẫn dùng được); `OPENAI_API_KEY` (tùy chọn, fallback cuối);
`CONFIDENCE_REFUSAL_THRESHOLD` (override ngưỡng 0.700).

---

## 7. Các quyết định thiết kế then chốt

1. **Từ chối thay vì bịa** — hai cổng từ chối độc lập: scope gate (trước retrieval,
   mục 3.2) và confidence gate (sau generation, mục 3.5). Một bot pháp lý trả lời sai
   nguy hiểm hơn nhiều so với trả lời "tôi không biết".
2. **Route trích dẫn riêng** — vì đo được vector search thất bại nặng (recall@5 0.125)
   trên dạng câu phổ biến nhất của tra cứu pháp luật. Bài học: đo trước, tối ưu sau.
3. **Contextual embedding không cần LLM** — metadata Chương/Điều/Khoản đã có sẵn từ
   chunker, chỉ cần ghép header rule-based; được chất lượng "Contextual Retrieval" mà
   chi phí bằng 0.
4. **Fail-open ở classifier và bộ lọc temporal, fail-safe ở confidence** — lỗi phân
   loại chỉ làm chọn sai route, thiếu metadata hiệu lực thì giữ chunk lại (kể cả ở
   tầng metadata: `is_in_effect` mặc định None = chưa rõ, không phải False); còn thiếu
   bằng chứng cho câu trả lời thì mặc định từ chối.
5. **Verify sau khi stream** — hậu kiểm (NLI + confidence) tốn thời gian, chạy sau khi
   token cuối đã tới UI; kết quả tới qua event `verification` và có thể "rút lại" câu
   trả lời (hiện refusal) nếu không đạt.
6. **Tín hiệu máy kiểm được ưu tiên hơn tự khai** — grounding ratio (máy kiểm) được
   thêm làm tín hiệu thứ 6, lấy trọng số từ retriever (bão hòa trên thang RRF) và
   llm_self_eval (model tự chấm); ngưỡng từ chối recalibrate ở **mỗi bậc scale** của
   corpus vì phân bố điểm đổi theo dữ liệu.
7. **Ingest chạy lại được an toàn** — point ID deterministic (uuid5) nên chạy lại không
   tạo point trùng; checkpoint SQLite chỉ để tiết kiệm thời gian embed, không phải cơ
   chế đúng đắn. Cho phép ingest corpus lớn trên Kaggle GPU miễn phí, rớt session thì
   `--resume`.
8. **Lazy-load mọi model nặng** — BGE-M3, reranker, NLI đều chỉ load ở lần dùng đầu;
   dev không trả giá khởi động khi chỉ sửa API.
9. **Hằng số tập trung** — `src/data/constants.py` là nguồn duy nhất cho tên collection
   và bộ lọc ingest; tránh lặp giá trị giữa scripts và src.
