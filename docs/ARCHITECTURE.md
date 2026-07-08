# Kiến trúc mã nguồn — Vietnamese Legal Knowledge Bot

Tài liệu này dành cho người **mới đọc mã nguồn**: nó trả lời "module nào nằm ở đâu,
chịu trách nhiệm gì" và "một câu hỏi của người dùng chạy qua những hàm nào, theo thứ tự
nào". Mọi mô tả đều kèm đường dẫn file/hàm để đối chiếu trực tiếp với code.

Nếu bạn muốn hiểu **tại sao** hệ thống được thiết kế như vậy (bài toán, số đo, roadmap),
đọc kèm [`Danh_Gia_va_Roadmap_Production_2026-07-06.md`](Danh_Gia_va_Roadmap_Production_2026-07-06.md).
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
                                        Qdrant Cloud   BGE-M3        BGE-reranker   LLM (Groq
                                        (vectors)      (embedder)    + mDeBERTa NLI  Qwen3-32B)
```

Mã nguồn chia thành **2 mặt phẳng** độc lập:

- **Offline (ingest)** — `scripts/` + `src/data/`: tải dataset, chunk, embed, index vào
  Qdrant. Chạy một lần (hoặc khi mở rộng corpus).
- **Online (query)** — `src/rag/` + `src/api/` + `src/ui/`: nhận câu hỏi, truy xuất,
  sinh câu trả lời, kiểm chứng, quyết định trả lời hay từ chối. Chạy theo từng request.

Nguyên tắc thiết kế xuyên suốt: **thà từ chối còn hơn trả lời sai** — có 2 cổng từ chối
riêng biệt (mục 3.2 và 3.5).

---

## 2. Bản đồ module

### `src/rag/` — lõi runtime (đọc đầu tiên)

| File | Vai trò |
|---|---|
| `orchestrator.py` | **Trái tim của hệ thống.** `NaiveRAGPipeline` (baseline: dense search → generate) và `AdaptiveRAGPipeline` (routing + verify + confidence). Hai entry point: `query()` (đồng bộ) và `query_stream()` (streaming, dùng cho UI). |
| `citation_lookup.py` | Route tra cứu trực tiếp cho câu hỏi dạng trích dẫn ("Điều 173 BLHS 2015..."): `parse_citation()` bóc Điều/Khoản/Điểm/tên luật/năm bằng regex, `CitationLookup` tra thẳng metadata trong Qdrant thay vì vector search. |
| `confidence.py` | `ConfidenceScorer` — điểm tin cậy 5 tín hiệu có trọng số; `REFUSAL_THRESHOLD = 0.725`; 2 mẫu câu từ chối (`REFUSAL_ANSWER`, `OUT_OF_SCOPE_ANSWER_TEMPLATE`). |
| `verifier.py` | `HallucinationVerifier` — kiểm chứng 3 tầng: structural (Điều được cite có trong context không), grounding (mỗi claim `[n]` trỏ đúng khối nguồn), NLI entailment (mDeBERTa). |

### `src/llm/` — cổng LLM

| File | Vai trò |
|---|---|
| `gateway.py` | `LLMGateway` bọc LiteLLM: chuỗi fallback `groq/qwen3-32b` → `groq/llama-3.3-70b` → `openai/gpt-4o-mini`; xoay vòng nhiều Groq API key khi dính rate-limit (`GROQ_API_KEYS` trong `.env`); `reasoning_effort="none"` để tắt block `<think>` của Qwen3. |
| `prompts.py` | Toàn bộ prompt template: `SYSTEM_PROMPT`, `GENERATION_PROMPT` (yêu cầu cite `[n]`), `COMPLEXITY_CLASSIFIER_PROMPT` (phân loại + scope gate), `QUERY_REWRITE_PROMPT`, `MULTI_QUERY_PROMPT`, `HYDE_PROMPT`, `CONFIDENCE_EVAL_PROMPT`. |

### `src/db/` — vector store

| File | Vai trò |
|---|---|
| `qdrant_client.py` | `LegalVectorStore` bọc Qdrant: tạo collection (dense 1024-dim COSINE + sparse), `upsert_chunks()`, `search_dense()`, `search_hybrid()` (RRF dense+sparse). Trả về dataclass `SearchResult` — kiểu dữ liệu chảy xuyên suốt pipeline. Collection mặc định: `luat_law_ctx` (xem `src/data/constants.py`). |

### `src/data/` — phía ingest

| File / thư mục | Vai trò |
|---|---|
| `constants.py` | Nguồn chân lý duy nhất cho hằng số: `DEFAULT_COLLECTION_NAME = "luat_law_ctx"`, bộ lọc `LEGAL_TYPES`, `MIN_CONTENT_LENGTH`... |
| `chunker.py` + `chunking/` | 5 chiến lược chunking (`article_level`, `hierarchical`, `fixed_size`, `semantic`, `sentence`) cùng trả về dataclass `LegalChunk` (chunk_text + metadata Chương/Điều/Khoản). Factory: `create_chunker()`. |
| `contextualizer.py` | Contextual Retrieval kiểu Anthropic nhưng **rule-based, không tốn LLM call**: ghép header `Luật \| Chương \| Điều \| Khoản` vào chuỗi **đem đi embed** (text hiển thị giữ nguyên). Đây là lý do collection có hậu tố `_ctx`. |
| `embedder.py` | `BGEEmbedder` bọc BGE-M3: một lần encode ra cả dense (1024-dim) lẫn sparse (lexical weights) — không cần BM25 riêng. Có xử lý MPS OOM cho macOS. |
| `metadata_store.py` | Kho JSON lưu metadata enrichment mức document (graph, timeline, amendment). |
| `graph/`, `timeline/`, `amendment/` | Lớp enrichment — xem [`KNOWLEDGE_GRAPH_VERSIONING.md`](KNOWLEDGE_GRAPH_VERSIONING.md). |

### `src/pipeline/` — lớp enrichment lúc chạy

| File | Vai trò |
|---|---|
| `enricher.py` | Chạy 3 module enrichment (graph/timeline/amendment) trên toàn bộ document trong Qdrant, lưu vào `MetadataStore`. |
| `retriever.py` | `EnhancedRetriever` — search có mở rộng ngữ cảnh theo graph (`expand_context=True`) và lọc theo thời điểm hiệu lực (`at_date=`). **Lưu ý:** chỉ được dùng bởi các API endpoint Phase 2 (`/api/search/enhanced`...), pipeline chat chính chưa dùng. |

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
- `scripts/eval.py` — runner 2 tầng: Tier 1 (retrieval: recall@5/@10, MRR — không tốn
  LLM) và Tier 2 (`--answers`: chạy full pipeline, đo refusal accuracy, citation
  accuracy — tốn LLM). Report ghi vào `data/eval_reports/`.
- `scripts/calibrate_threshold.py` — quét ngưỡng từ chối trên report Tier 2 (nguồn gốc
  con số 0.725).
- `scripts/inspect_confidence.py` — soi breakdown 5 tín hiệu của từng câu hỏi (debug
  false-refusal).
- `scripts/ingest.py` — pipeline ingest chính (mục 4); `scripts/download_dataset.py`,
  `scripts/reindex_contextual.py`, `scripts/validate_eval_set.py` — tiện ích kèm theo.

### `tests/`

Pytest cho các phần thuần logic: scope gate, citation lookup, calibration, verifier,
contextualizer, eval metrics. Chạy: `make test`.

**Thứ tự đọc code gợi ý:** `src/rag/orchestrator.py` (nắm luồng chính) →
`src/rag/confidence.py` + `verifier.py` (hiểu cổng từ chối) → `src/rag/citation_lookup.py`
→ `src/db/qdrant_client.py` → `scripts/ingest.py` + `src/data/contextualizer.py` →
phần còn lại khi cần.

---

## 3. Vòng đời một câu hỏi (luồng online)

Đường đi đầy đủ, lấy luồng streaming làm chuẩn (luồng non-streaming `query()` giống hệt,
chỉ khác không phát event trung gian):

```
UI chat.py ──POST /api/chat/stream──▶ api/main.py:chat_stream()
    └──▶ AdaptiveRAGPipeline.query_stream()   (src/rag/orchestrator.py)
```

### 3.1. Route trích dẫn (trước mọi thứ khác)

`parse_citation(question)` (`citation_lookup.py`) chạy **trước cả** classifier. Nếu câu
hỏi chứa "Điều N" (+ tên luật/năm nếu có) → đi route `lookup`: `retrieve_routed()` tra
thẳng metadata Qdrant (lọc document theo alias, tìm chunk **mở đầu** bằng "Điều N.")
thay vì vector search. Lý do: vector search đo được recall@5 = 0.125 trên dạng câu này
(chunk của luật sửa đổi trích dẫn "Điều N" tràn lan làm nhiễu). Cố ý **không** fallback
sang vector search khi lookup rỗng ("Điều 999" không tồn tại → context rỗng → cổng tin
cậy tự từ chối).

### 3.2. Phân loại + cổng phạm vi — 1 LLM call

`_classify_and_scope()` gọi LLM **một lần duy nhất** với `COMPLEXITY_CLASSIFIER_PROMPT`,
trả về đồng thời:

- `SCOPE: IN | OUT` — corpus có trả lời được không (prompt được nhồi `corpus_summary()`
  — danh sách văn bản thực tế trong Qdrant, không hard-code).
- `CLASS: SIMPLE | MEDIUM | COMPLEX` — độ sâu pipeline cần dùng.

Nếu `OUT` → từ chối **ngay, trước retrieval** bằng `OUT_OF_SCOPE_ANSWER_TEMPLATE`
(cổng từ chối thứ nhất). Parse bằng regex và **fail-open**: nếu không parse được thì coi
là `IN` + `MEDIUM` — lỗi classifier chỉ làm chọn sai route chứ không bao giờ chặn nhầm
câu hỏi thật.

### 3.3. Ba route truy xuất

| Route | Các bước (đều trong `orchestrator.py`) |
|---|---|
| `_simple_pipeline` | dense search top-20 → `_dynamic_top_k()` cắt còn 3–15 |
| `_medium_pipeline` | hybrid search (dense + sparse) top-20 → dynamic top-k → `_rerank_results()` (BGE-reranker-v2-m3 cross-encoder) |
| `_complex_pipeline` | rewrite query → multi-query (3 biến thể) → HyDE (văn bản giả định) → hybrid search cho cả 4 query → khử trùng lặp theo `point_id` → dynamic top-k → rerank → `_compress_context()` (LLM lược context) |

`_dynamic_top_k()` co giãn số kết quả theo phân bố điểm: điểm cao tập trung → lấy ít,
điểm thấp/phẳng → lấy nhiều.

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
     phải thực sự chứa nội dung được gán cho nó (chỉ để quan sát/cảnh báo, chưa vào điểm).
   - *Tier 2 NLI*: tách câu trả lời thành claim, chấm ENTAILMENT/CONTRADICTION/NEUTRAL
     bằng mDeBERTa-v3-mnli-xnli.
2. **`ConfidenceScorer.score()`** (`confidence.py`) — 5 tín hiệu có trọng số:

   | Tín hiệu | Trọng số | Nguồn |
   |---|---|---|
   | retriever | 0.25 | max cosine từ search |
   | reranker | 0.20 | max điểm cross-encoder |
   | llm_self_eval | 0.30 | LLM tự chấm 0–100 |
   | context_overlap | 0.15 | trùng lặp từ khóa answer↔context |
   | entailment | 0.10 | tỉ lệ claim được NLI xác nhận |

   Tín hiệu thiếu (ví dụ route simple không có reranker) bị **loại và chuẩn hóa lại
   trọng số** — điểm tối đa luôn là 1.0, không bị "phạt ngầm".
3. Điểm `< 0.725` (`REFUSAL_THRESHOLD`, override bằng env `CONFIDENCE_REFUSAL_THRESHOLD`)
   → thay câu trả lời bằng `REFUSAL_ANSWER`. Lai lịch con số 0.725 được ghi đầy đủ trong
   block comment ngay trên hằng số ở `confidence.py`.

### 3.6. Giao thức streaming UI ↔ backend

`query_stream()` yield chuỗi thường (token) xen kẽ các **marker** tiền tố:
`__STEP__{json}` (bước trung gian), `__CITATIONS__{json}`, `__VERIFICATION__{json}`
(kết quả hậu kiểm, đến sau cùng). `api/main.py:chat_stream()` dịch chúng thành SSE event
`step` / `citations` / `token` / `verification` / `done`; `ui/pages/chat.py` render
tương ứng (step vào expander, citation thành card, verification thành badge tin cậy /
thay câu trả lời bằng refusal nếu bị từ chối).

---

## 4. Pipeline ingest (luồng offline)

Thứ tự chạy (xem `Makefile`):

```
scripts/download_dataset.py      # tải dataset luật VN từ HuggingFace về data/raw/
        │
scripts/ingest.py                # đọc streaming bằng pyarrow (không load hết vào RAM)
        │   1. lọc: legal_type ∈ LEGAL_TYPES, độ dài ≥ MIN_CONTENT_LENGTH
        │   2. chunk: ArticleLevelChunker (mỗi Điều/Khoản một chunk + metadata)
        │   3. contextualize: prepend header "Luật | Chương | Điều | Khoản"
        │      vào bản đem đi embed (build_contextual_text) — text gốc giữ nguyên
        │   4. embed: BGE-M3 → dense + sparse trong 1 lần encode
        │   5. upsert vào Qdrant collection luat_law_ctx
        ▼
Qdrant (payload: chunk_text, document_title, article, chapter, clause, document_id, url...)
```

`scripts/reindex_contextual.py` re-embed một collection sẵn có sang dạng contextual mà
không phải ingest lại từ đầu. Corpus hiện tại: 6 văn bản luật hình sự, 1.670 chunks.

Enrichment (graph/timeline/amendment) là bước offline **riêng**, chạy qua
`POST /api/pipeline/enrich` (hoặc `PipelineEnricher` trực tiếp), ghi vào
`data/metadata/` — không đụng vào Qdrant.

---

## 5. Đánh giá & hiệu chuẩn

- Eval set: `data/eval/eval_set_criminal_v1.json` (41 câu có ground-truth Điều luật) và
  `eval_set_general_v1.json` (29 câu ngoài phạm vi — phải từ chối).
  `scripts/validate_eval_set.py` kiểm tra ground-truth có thực sự tồn tại trong corpus.
- `scripts/eval.py` — Tier 1 đo retrieval (strict = khớp payload, lenient = khớp text;
  khoảng cách giữa 2 số đo chính là độ lệch metadata của chunker), Tier 2 (`--answers`)
  đo hành vi trả lời/từ chối end-to-end.
- `scripts/calibrate_threshold.py` đọc report Tier 2 và quét ngưỡng — quy trình đã chốt
  `REFUSAL_THRESHOLD = 0.725`.
- Số đo baseline hiện tại và lịch sử quyết định: xem
  [`Danh_Gia_va_Roadmap_Production_2026-07-06.md`](Danh_Gia_va_Roadmap_Production_2026-07-06.md).

---

## 6. Chạy dự án

```bash
make install          # pip install -e ".[dev]"
cp .env.example .env  # điền QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEYS
make qdrant           # Qdrant + Redis qua Docker (hoặc dùng Qdrant Cloud qua .env)
make ingest           # chỉ khi cần dựng lại index
make api              # uvicorn src.api.main:app --port 8000
make ui               # streamlit run src/ui/app.py --port 8501
make test             # pytest tests/ -v
make eval             # scripts/eval.py Tier 1
```

Biến môi trường quan trọng (`.env.example` có đủ danh sách): `QDRANT_URL` /
`QDRANT_API_KEY`, `GROQ_API_KEYS` (nhiều key, phẩy ngăn cách — gateway tự xoay khi
rate-limit), `OPENAI_API_KEY` (tùy chọn, chỉ để fallback), `CONFIDENCE_REFUSAL_THRESHOLD`.

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
4. **Fail-open ở classifier, fail-safe ở confidence** — lỗi phân loại chỉ làm chọn sai
   route; còn thiếu bằng chứng thì mặc định từ chối.
5. **Verify sau khi stream** — hậu kiểm (NLI + confidence) tốn thời gian, chạy sau khi
   token cuối đã tới UI; kết quả tới qua event `verification` và có thể "rút lại" câu
   trả lời (hiện refusal) nếu không đạt.
6. **Lazy-load mọi model nặng** — BGE-M3, reranker, NLI đều chỉ load ở lần dùng đầu;
   dev không trả giá khởi động khi chỉ sửa API.
7. **Hằng số tập trung** — `src/data/constants.py` là nguồn duy nhất cho tên collection
   và bộ lọc ingest; tránh lặp giá trị giữa scripts và src.
