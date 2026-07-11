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
                                        Qdrant Cloud   BGE-M3        BGE-reranker   LLM (Groq
                                        (vectors)      (embedder)    + mDeBERTa NLI  Qwen3-32B)
```

Corpus hiện có 2 thế hệ song song trong Qdrant (xem mục 4):

- `luat_law_ctx_v2` — 6 văn bản luật hình sự, 2.242 points (M0.5). Đây vẫn là
  `DEFAULT_COLLECTION_NAME` trong code.
- `luat_law_ctx_v3` — corpus tuyển chọn **100 văn bản (luật + nghị định), 6 lĩnh vực**
  (hình sự, dân sự – tố tụng dân sự, lao động – bảo hiểm, doanh nghiệp – thương mại –
  đầu tư, đất đai, hôn nhân gia đình), 23.727 points (M1). Dùng qua flag
  `--collection` của `scripts/ingest.py` / `scripts/eval.py`; default sẽ chuyển sang
  v3 khi các gate M1 chốt xong.

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
| `citation_lookup.py` | Route tra cứu trực tiếp cho câu hỏi dạng trích dẫn: `parse_citation()` bóc Điều/Khoản/Điểm/tên luật/năm **và số hiệu văn bản** ("59/2020/QH14", "145/2020/NĐ-CP") bằng regex, hiểu viết tắt đa lĩnh vực (BLHS, BLTTHS, BLDS, LDN...); `CitationLookup.resolve_doc()` xếp hạng ứng viên văn bản theo alias/số hiệu/năm; `lookup()` tra thẳng metadata trong Qdrant thay vì vector search. Cũng là nơi sinh `corpus_summary()` (nhóm theo lĩnh vực, nạp vào scope gate) và `corpus_summary_compact()` (1 đoạn ngắn cho câu từ chối out-of-scope). |
| `confidence.py` | `ConfidenceScorer` — điểm tin cậy **6 tín hiệu** có trọng số (bảng ở mục 3.5); `REFUSAL_THRESHOLD = 0.700` (recalibrate M0.3 — lai lịch trong block comment ngay trên hằng số); 2 mẫu câu từ chối (`REFUSAL_ANSWER`, `OUT_OF_SCOPE_ANSWER_TEMPLATE`). |
| `verifier.py` | `HallucinationVerifier` — kiểm chứng 3 tầng: structural (Điều được cite có trong context không), grounding (mỗi claim `[n]` trỏ đúng khối nguồn — **tỉ lệ này là tín hiệu confidence thứ 6**), NLI entailment (mDeBERTa, premise chọn theo từng claim thay vì cắt mù context). |
| `context_blocks.py` | Tiện ích chung cho verifier + confidence: `parse_context_blocks()` tách context thành các khối `[n]`, `select_premise()` chọn đúng khối nguồn làm premise cho NLI/self-eval theo từng claim (fix M0.1 — thay cho `context[:2000]`). |

### `src/llm/` — cổng LLM

| File | Vai trò |
|---|---|
| `gateway.py` | `LLMGateway` bọc LiteLLM: chuỗi fallback `groq/qwen3-32b` → `groq/llama-3.3-70b` → `openai/gpt-4o-mini`; xoay vòng nhiều Groq API key khi dính rate-limit (`GROQ_API_KEYS` trong `.env`); `reasoning_effort="none"` để tắt block `<think>` của Qwen3. |
| `prompts.py` | Toàn bộ prompt template: `SYSTEM_PROMPT` (trợ lý pháp lý **mọi lĩnh vực**), `GENERATION_PROMPT` (yêu cầu cite `[n]`), `COMPLEXITY_CLASSIFIER_PROMPT` (phân loại + scope gate, kèm few-shot đa lĩnh vực — M1.4), `QUERY_REWRITE_PROMPT`, `MULTI_QUERY_PROMPT`, `HYDE_PROMPT`, `CONFIDENCE_EVAL_PROMPT` (self-eval theo rubric 4 tiêu chí, code tự cộng điểm — M0.1). |

### `src/db/` — vector store

| File | Vai trò |
|---|---|
| `qdrant_client.py` | `LegalVectorStore` bọc Qdrant: tạo collection (dense 1024-dim COSINE + sparse, tùy chọn int8 scalar quantization), `upsert_chunks()` với **point ID deterministic** (`uuid5(document_id:chunk_index)` — chạy lại ingest không tạo trùng), `search_dense()`, `search_hybrid()` (RRF dense+sparse). Trả về dataclass `SearchResult` — kiểu dữ liệu chảy xuyên suốt pipeline. |

### `src/data/` — phía ingest

| File / thư mục | Vai trò |
|---|---|
| `constants.py` | Nguồn chân lý duy nhất cho hằng số: `DEFAULT_COLLECTION_NAME = "luat_law_ctx_v2"`, bộ lọc `LEGAL_TYPES`, `MIN_CONTENT_LENGTH`... |
| `chunker.py` + `chunking/` | 5 chiến lược chunking (`article_level`, `hierarchical`, `fixed_size`, `semantic`, `sentence`) cùng trả về dataclass `LegalChunk` (chunk_text + metadata Chương/Điều/Khoản + `parent_chunk_text`). Factory: `create_chunker()`. Ingest dùng `HierarchicalChunker` cho luật thường và `ArticleLevelChunker` cho luật sửa đổi (mục 4). |
| `contextualizer.py` | Contextual Retrieval kiểu Anthropic nhưng **rule-based, không tốn LLM call**: ghép header `Luật \| Chương \| Điều \| Khoản` vào chuỗi **đem đi embed** (text hiển thị giữ nguyên). Đây là lý do collection có hậu tố `_ctx`. |
| `embedder.py` | `BGEEmbedder` bọc BGE-M3: một lần encode ra cả dense (1024-dim) lẫn sparse (lexical weights) — không cần BM25 riêng. Có xử lý MPS OOM cho macOS. |
| `ingest_checkpoint.py` | Checkpoint SQLite theo document (`data/ingest_progress/{collection}.sqlite`) cho `--resume` — tiết kiệm GPU-hour khi rớt session Kaggle, không phải cơ chế đảm bảo đúng đắn (point ID đã deterministic). |
| `metadata_store.py` | Kho JSON lưu metadata enrichment mức document (graph, timeline, amendment). |
| `graph/`, `timeline/`, `amendment/` | Lớp enrichment — xem [`KNOWLEDGE_GRAPH_VERSIONING.md`](KNOWLEDGE_GRAPH_VERSIONING.md). M1.8b sửa `find_replacement` cho đúng ở bậc 100 docs: khớp số hiệu là chuẩn nhất, khớp tên phải đảm bảo văn bản thay thế **mới hơn**, không rõ ngày → không kết luận (12/12 cặp temporal đúng). |

### `src/pipeline/` — lớp enrichment lúc chạy

| File | Vai trò |
|---|---|
| `enricher.py` | Chạy 3 module enrichment (graph/timeline/amendment) trên toàn bộ document trong Qdrant, lưu vào `MetadataStore`. |
| `retriever.py` | `get_phase2_status()` — tra trạng thái temporal/amendment của một (document, Điều); **được pipeline chat chính dùng** làm bộ lọc hiệu lực (mục 3.3). `EnhancedRetriever` (mở rộng ngữ cảnh theo graph, lọc `at_date=`) vẫn chỉ phục vụ các endpoint Phase 2 (`/api/search/enhanced`...). |

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
  `data/eval_reports/`.
- `scripts/calibrate_threshold.py` — quét ngưỡng từ chối trên report Tier 2 (nguồn gốc
  con số 0.700).
- `scripts/inspect_confidence.py` — soi breakdown 6 tín hiệu của từng câu hỏi (debug
  false-refusal).
- `scripts/select_corpus.py` — M1.1: trích 100 văn bản tuyển chọn tay ra
  `corpus_100docs.parquet` (lai lịch chọn lọc:
  [`notes/corpus-100docs-selection-2026-07-10.md`](notes/corpus-100docs-selection-2026-07-10.md)).
- `scripts/ingest.py` — pipeline ingest chính (mục 4); `scripts/download_dataset.py`,
  `scripts/download_all_chunks.py`, `scripts/reindex_contextual.py`,
  `scripts/validate_eval_set.py` — tiện ích kèm theo. Chạy ingest trên Kaggle GPU:
  [`notes/kaggle-ingest-guide.md`](notes/kaggle-ingest-guide.md).

### `tests/`

Pytest cho các phần thuần logic: scope gate, citation lookup (gồm bộ đa lĩnh vực),
prompts đa lĩnh vực, calibration, verifier, context blocks, contextualizer, chunker
(article-level + hierarchical), ingest checkpoint + bộ lọc curated, orchestrator,
retriever Phase-2, version resolver, eval metrics. Chạy: `make test`.

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
   (`is_in_effect=False`). Fail-open: thiếu metadata hoặc không rõ → giữ lại.
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
   → thay câu trả lời bằng `REFUSAL_ANSWER`. Ngưỡng 0.700 recalibrate ở M0.3 sau khi hai
   phân bố answer/refuse lần đầu tách sạch (gap +0.0139); lai lịch đầy đủ trong block
   comment ngay trên hằng số ở `confidence.py`. Margin mỏng → **phải recalibrate ở mỗi
   bậc scale** (gate M1/M2).

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

Collection hiện có: `luat_law_ctx_v3` (100 docs, 23.727 points — corpus M1),
`luat_law_ctx_v2` (6 docs hình sự, 2.242 points — default hiện tại của code, giữ để
so sánh trước/sau), và các collection cũ hơn. `scripts/reindex_contextual.py` re-embed
một collection sẵn có sang dạng contextual mà không phải ingest lại từ đầu.

Enrichment (graph/timeline/amendment) là bước offline **riêng**, chạy qua
`POST /api/pipeline/enrich` (hoặc `PipelineEnricher` trực tiếp), ghi vào
`data/metadata/` — không đụng vào Qdrant. Kết quả được pipeline chat dùng lúc chạy
để lọc văn bản hết hiệu lực (mục 3.3).

---

## 5. Đánh giá & hiệu chuẩn

- Eval set (`data/eval/`): `eval_set_criminal_v1.json` (41 câu hình sự có ground-truth
  Điều luật), `eval_set_sectors_v1.json` (51 câu cho 5 lĩnh vực mới của corpus 100 docs),
  `eval_set_general_v1.json` (41 câu biên/ngoài phạm vi — định nghĩa lại ở M1.6 cho
  corpus 100 docs: 27 câu nay phải **trả lời được**, 14 câu vẫn phải từ chối).
  `scripts/validate_eval_set.py` kiểm tra ground-truth có thực sự tồn tại trong corpus.
- `scripts/eval.py` — Tier 1 đo retrieval trên cả 3 chế độ dense/hybrid/routed
  (strict = khớp payload, lenient = khớp text; khoảng cách giữa 2 số đo chính là độ
  lệch metadata của chunker), kèm bảng recall theo lĩnh vực; Tier 2 (`--answers`) đo
  hành vi trả lời/từ chối end-to-end. Số Tier 1 bộ sectors trên v3: routed recall@5
  0.8922, recall@10 0.9216; citation-lookup và amendment đạt 1.0.
- `scripts/calibrate_threshold.py` đọc report Tier 2 và quét ngưỡng — quy trình đã chốt
  `REFUSAL_THRESHOLD = 0.700` (M0.3) và phải chạy lại ở mỗi bậc scale.
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
4. **Fail-open ở classifier và bộ lọc temporal, fail-safe ở confidence** — lỗi phân
   loại chỉ làm chọn sai route, thiếu metadata hiệu lực thì giữ chunk lại; còn thiếu
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
