# 🏛️ Vietnamese Legal Knowledge Bot — Codebase Documentation

> Cẩm nang đọc code toàn diện: kiến trúc, data flow, từng module, cách debug.
> Dành cho chính bạn sau này và người mới onboard vào project.

---

## 📁 Project Structure

```
vietnamese-legal-bot/
├── .env                          # API keys (GROQ_API_KEY, etc.) — gitignored
├── .env.example                  # Template env vars
├── Dockerfile                    # Multi-stage: api / ui
├── docker-compose.yml            # 4 services: qdrant, redis, api, ui
├── Makefile                      # Shortcuts: make api, make ui, make dev, make ingest-test, etc.
├── pyproject.toml                # Dependencies (langchain-text-splitters, FlagEmbedding, litellm, etc.)
│
├── data/raw/                     # 🚫 gitignored
│   └── legal_docs_merged.parquet # 518K docs, 3.6GB — dataset HuggingFace
│
├── scripts/                      # ⚡ Entry points để chạy từ CLI
│   ├── download_dataset.py       # Tải từ HuggingFace → merge → save parquet
│   ├── explore_data.py           # Scan stats (legal_type, dates) — LOW MEMORY
│   └── ingest.py                 # Full pipeline: filter → chunk → embed → Qdrant
│
├── src/                          # 📦 Code chính
│   ├── data/
│   │   ├── chunker.py            # LegalDocumentChunker: tách văn bản pháp luật
│   │   └── embedder.py           # BGEEmbedder: BGE-M3 (dense 1024-dim + sparse)
│   ├── db/
│   │   └── qdrant_client.py      # LegalVectorStore: CRUD + search (dense/sparse/hybrid)
│   ├── llm/
│   │   ├── gateway.py            # LLMGateway: LiteLLM fallback chain (Groq Qwen3)
│   │   └── prompts.py            # Tất cả prompt templates
│   ├── rag/
│   │   └── orchestrator.py       # NaiveRAGPipeline + AdaptiveRAGPipeline
│   ├── api/
│   │   └── main.py               # FastAPI app (5 endpoints, SSE streaming)
│   └── ui/
│       ├── app.py                # Streamlit entry point (sidebar nav)
│       └── pages/
│           ├── chat.py           # 💬 Chat page (streaming + citations)
│           ├── documents.py      # 📁 Document browser (filter + search)
│           └── eval_dashboard.py # 📊 RAGAS metrics dashboard
│
├── tests/                        # Unit tests (TODO)
└── docs/
    ├── lessons-learned.md        # 8 bài học engineering (bugs + fixes)
    └── CODEBASE.md               # ← You are here
```

---

## 🔄 Data Flow — 2 luồng chính

### Luồng 1: Ingestion (Index dữ liệu vào Qdrant)

```
┌──────────────────┐
│ HuggingFace      │  vohuutridung/vietnamese-legal-documents
│ (518K docs)      │
└────────┬─────────┘
         │ scripts/download_dataset.py
         │ (tải 2 configs: text_data + qa_pairs, merge trên id)
         ▼
┌──────────────────┐
│ legal_docs_merged│  Parquet 3.6GB, 1 row group, 518,255 rows
│ .parquet         │
└────────┬─────────┘
         │ scripts/ingest.py (pyarrow.dataset.to_batches)
         │ ── STREAMING, batch 200 rows (~50MB mỗi batch)
         ▼
┌──────────────────┐
│ filter_row()     │  Loại: legal_type ∈ {Luật, Bộ luật, Nghị định, ...}
│                  │  Thời gian: 2015 ≤ year ≤ 2026
│                  │  Nội dung: len(content) ≥ 500 chars
│                  │  → Còn ~312K docs
└────────┬─────────┘
         ▼
┌──────────────────┐
│ LegalDocument    │  Hierarchical Markdown Chunking:
│ Chunker          │  - MarkdownHeaderTextSplitter (Chương/Điều/Khoản)
│                  │  - Fallback: regex Vietnamese legal structure
│                  │  - Size control: merge <200 chars, split >1000 chars
│                  │  - Each chunk inherits parent metadata
└────────┬─────────┘
         ▼
┌──────────────────┐
│ BGEEmbedder      │  BAAI/bge-m3 (FlagEmbedding):
│ BGE-M3           │  - dense_vecs: np.ndarray (N, 1024) float16
│                  │  - lexical_weights: List[defaultdict[int, float]]
│                  │  - Batch size: 32 (để tránh OOM)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Qdrant           │  Collection: "vietnamese_laws"
│ (Docker)         │  - Dense vector: 1024-dim COSINE
│                  │  - Sparse vector: lexical index
│                  │  - Payload: chunk_text + full metadata
└──────────────────┘
```

### Luồng 2: Query (Hỏi đáp pháp luật)

```
┌──────────────────┐
│ User Question    │  "Thủ tục cấp giấy phép xây dựng là gì?"
└────────┬─────────┘
         │ POST /api/chat hoặc /api/chat/stream
         ▼
┌──────────────────┐
│ FastAPI          │  src/api/main.py
│                  │  Lazy-load BGEEmbedder (nặng ~2GB, load 1 lần)
│                  │  Chọn NaiveRAGPipeline hoặc AdaptiveRAGPipeline
└────────┬─────────┘
         ▼
┌──────────────────────────────────────────────────┐
│ NaiveRAGPipeline.query()                         │
│                                                  │
│  Step 1: embedder.encode_queries([question])     │
│          → dense vector (1024,)                  │
│                                                  │
│  Step 2: vector_store.search_dense(vector, k=5)  │
│          → List[SearchResult]                    │
│                                                  │
│  Step 3: _build_context(results)                 │
│          → format "[1] Title — Điều X\nchunk..."  │
│                                                  │
│  Step 4: LLM.generate(prompt, json_mode=True)    │
│          → Groq Qwen 3 32B (fallback chain)      │
│                                                  │
│  Step 5: _parse_response(json) → RAGResponse     │
│          answer + citations + confidence         │
└──────────────────┬───────────────────────────────┘
                   ▼
┌──────────────────┐
│ ChatResponse     │  JSON: {answer, citations[], confidence, disclaimer,
│ (JSON)           │         latency_ms, pipeline_used}
└──────────────────┘
```

**Streaming variant:**
```
query_stream() → yield "__CITATIONS__[...]" → yield token → yield token → ...
                      ↓
API event_generator() → SSE events:
  event: citations → data: [{doc_title, article, chunk_text, score}, ...]
  event: token     → data: {"token": "..."}
  event: done      → data: {full_answer, citations, confidence, latency_ms}
```

---

## 🧩 Module-by-Module Guide

### 1. `src/data/chunker.py` — LegalDocumentChunker

**Vai trò**: Tách văn bản pháp luật thành chunks nhỏ (target: 200-1000 chars).

**Cấu trúc**:
```
LegalDocumentChunker
├── min_chunk_size=200, max_chunk_size=1000
├── header_splitter: MarkdownHeaderTextSplitter(#, ##, ###, ####)
├── text_splitter: RecursiveCharacterTextSplitter (fallback)
│
├── chunk_document(row: dict) → List[LegalChunk]
│   ├── _split_by_headers(content) → List[dict]
│   │   └── Dùng MarkdownHeaderTextSplitter, track Chapter → Article → Clause
│   ├── _split_by_text(content) → List[dict]
│   │   └── Fallback: regex CHAPTER_PATTERN, ARTICLE_PATTERN, CLAUSE_PATTERN
│   └── _control_chunk_sizes(chunks) → List[dict]
│       ├── Merge chunks < max_chunk_size
│       └── Split chunks > max_chunk_size qua RecursiveCharacterTextSplitter
│
└── LegalChunk (dataclass)
    ├── chunk_text: str
    ├── document_id, document_title, document_number
    ├── legal_type, issuing_authority, issuance_date, url
    ├── chapter, article, clause (metadata từ cấu trúc phân cấp)
    └── chunk_index, parent_chunk_text
```

**Regex patterns** (dùng khi Markdown headers không khớp):
- `CHAPTER_PATTERN`: `CHƯƠNG [IVXLCDM]+[.:)] (.*)`
- `ARTICLE_PATTERN`: `Điều (\d+)[.:)] (.*)`
- `CLAUSE_PATTERN`: `(\d+)\. (.*)`

**Điểm cần lưu ý**:
- Buffer size control: merge nếu tổng ≤ max_chunk_size, flush nếu vượt
- Metadata inheritance: chapter được propagate xuống article và clause
- Hỗ trợ cả Markdown format chuẩn và format văn bản pháp luật thô (regex fallback)

---

### 2. `src/data/embedder.py` — BGEEmbedder

**Vai trò**: Encode text thành dense + sparse vectors dùng BGE-M3.

**Cấu trúc**:
```
BGEEmbedder
├── model: BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cpu")
├── dim: 1024
│
├── encode(texts, batch_size, max_length, return_dense, return_sparse) → Dict
│   └── Returns: {'dense_vecs': np.ndarray, 'lexical_weights': List[defaultdict]}
│       ⚠️  KEY LÀ 'lexical_weights' KHÔNG PHẢI 'sparse_vecs'
│
├── encode_queries(queries) → np.ndarray  (dense only, max_length=512)
│   └── Dùng cho search query — queries ngắn, chỉ cần dense
│
└── encode_documents(documents) → (np.ndarray, List[Dict])
    └── Dùng cho indexing — cần cả dense + lexical_weights
```

**Model specs**:
- **BAAI/bge-m3**: Multilingual (bao gồm tiếng Việt), 1024-dimensional dense + sparse
- **Download**: ~2GB, load 1 lần duy nhất
- **Memory khi load**: ~2.5GB RAM
- **Định dạng lexical_weights**: `defaultdict(int, {token_id: score, ...})`
- **Device**: CPU (laptop) hoặc CUDA

---

### 3. `src/db/qdrant_client.py` — LegalVectorStore

**Vai trò**: Interface với Qdrant vector database.

**Cấu trúc**:
```
LegalVectorStore
├── COLLECTION_NAME: "vietnamese_laws"
├── DENSE_VECTOR_NAME: "dense"
├── SPARSE_VECTOR_NAME: "sparse"
├── VECTOR_DIM: 1024
├── client: QdrantClient
│
├── QUẢN LÝ COLLECTION
│   └── create_collection(recreate=False)
│       ├── Xóa nếu recreate=True
│       ├── Kiểm tra nếu đã tồn tại → skip
│       └── Tạo mới với VectorParams(dense, COSINE) + SparseVectorParams()
│
├── INDEX
│   └── upsert_chunks(texts, dense_vecs, sparse_vecs, metadata, batch_size=100)
│       ├── Tạo PointStruct(id=uuid4, vector={dense, sparse}, payload={...})
│       └── Upsert theo batch 100 points → Qdrant
│
├── SEARCH (3 modes)
│   ├── search_dense(query_vector, top_k=20, filter) → List[SearchResult]
│   │   └── query_points(using="dense")
│   │
│   ├── search_sparse(sparse_vector, top_k=20, filter) → List[SearchResult]
│   │   └── query_points(query=SparseVector, using="sparse")
│   │
│   └── search_hybrid(dense, sparse, top_k=20, fusion="rrf") → List[SearchResult]
│       └── query_points với prefetch=[dense_prefetch, sparse_prefetch]
│           + query=FusionQuery(fusion=Fusion.RRF)
│           → Server-side RRF fusion (Qdrant 1.18+ native)
│
├── DOCUMENT LISTING
│   └── list_documents(legal_type, keyword, page, page_size) → Dict
│       └── Scroll Qdrant, dedup by document_id, paginate
│
├── UTILS
│   ├── _build_filter(conditions) → Filter
│   │   └── {"legal_type": "Luật"} → Filter(must=[FieldCondition(...)])
│   │
│   ├── _to_search_result(ScoredPoint) → SearchResult
│   │   └── Extract payload fields vào dataclass
│   │
│   └── get_collection_info() → Dict
│       └── {name, points_count, indexed_vectors_count}
│
└── SearchResult (dataclass)
    ├── point_id, score, chunk_text
    ├── document_title, document_number, legal_type
    └── article, chapter, url, document_id
```

**Qdrant 1.18.x API notes**:
- `client.search()` → **đã bị xóa**, dùng `client.query_points()`
- `query_vector=(name, vector)` → `query=vector, using=name`
- `vectors_count` → `indexed_vectors_count`
- RRF: từ client-side → server-side native (`models.FusionQuery(fusion=models.Fusion.RRF)`)
- Sparse vector: `qdrant_client.models.SparseVector(indices=[...], values=[...])`

---

### 4. `src/llm/gateway.py` — LLMGateway

**Vai trò**: Unified LLM access qua LiteLLM với auto-fallback.

**Cấu trúc**:
```
LLMGateway
├── MODEL_CHAIN = [
│       "groq/qwen/qwen3-32b",           # Primary: Qwen 3 32B
│       "groq/qwen/qwen3.6-27b",         # Fallback 1: Qwen 3.6 27B
│       "groq/llama-3.3-70b-versatile",  # Fallback 2: Llama 70B
│   ]
├── groq_key = os.getenv("GROQ_API_KEY")
├── openai_key = os.getenv("OPENAI_API_KEY")
│
├── generate(prompt, system_prompt, temperature, max_tokens, json_mode) → str
│   ├── Build messages: [system?, user]
│   ├── Loop MODEL_CHAIN → litellm.completion(**kwargs)
│   ├── _clean_thinking(response) — strip Qwen3 <think> tags
│   └── RuntimeError nếu tất cả models fail
│
├── generate_stream(prompt, system_prompt, ...) → Generator[str]
│   ├── Build messages + stream=True
│   ├── Loop MODEL_CHAIN → yield từng token
│   └── ⚠️ Hiện tại KHÔNG strip <think> tags trong stream mode
│
└── _clean_thinking(text: str) → str
    ├── re.sub(r'<think>[\s\S]*?</think>', '', text)
    └── re.sub(r'<think>[\s\S]*$', '', text)
```

**Fallback chain logic**:
```
Thử groq/qwen/qwen3-32b
  ├── Thành công → return (exit loop)
  └── Fail → Thử groq/qwen/qwen3.6-27b
        ├── Thành công → return
        └── Fail → Thử groq/llama-3.3-70b-versatile
              ├── Thành công → return
              └── Fail → raise RuntimeError("All models failed")
```

**Cấu hình API key**:
```bash
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxx    # Free tier: https://console.groq.com
OPENAI_API_KEY=sk-xxxxxxxxxxxxx   # Optional fallback
```

---

### 5. `src/llm/prompts.py` — Prompt Templates

**Vai trò**: Tất cả prompt templates cho từng bước trong pipeline.

| Template | Dùng trong | Mục đích |
|---|---|---|
| `SYSTEM_PROMPT` | Tất cả generate calls | Định danh: "Trợ lý Pháp lý Việt Nam" + nguyên tắc |
| `GENERATION_PROMPT` | `_build_context` → LLM.generate | Prompt chính: context + question → answer JSON |
| `QUERY_REWRITE_PROMPT` | `AdaptiveRAGPipeline._complex_pipeline` | Viết lại query thành dạng pháp lý chuẩn |
| `MULTI_QUERY_PROMPT` | `AdaptiveRAGPipeline._complex_pipeline` | Tạo 3 biến thể của query |
| `HYDE_PROMPT` | `AdaptiveRAGPipeline._complex_pipeline` | Sinh "hypothetical document" |
| `COMPLEXITY_CLASSIFIER_PROMPT` | `AdaptiveRAGPipeline._classify` | Phân loại SIMPLE / MEDIUM / COMPLEX |

**Expected output format** (JSON mode):
```json
{
  "answer": "Câu trả lời đầy đủ...",
  "citations": [
    {"document_title": "...", "article": "Điều X", "clause": "Khoản Y",
     "excerpt": "...", "relevance": "high|medium"}
  ],
  "confidence": "high|medium|low",
  "disclaimer": "..."
}
```

---

### 6. `src/rag/orchestrator.py` — RAG Pipeline

**Vai trò**: Orchestrate toàn bộ query → search → generate flow.

#### NaiveRAGPipeline (MVP — đang dùng)

```
query(question) → RAGResponse
  ├── embedder.encode_queries(question) → dense vector (1024,)
  ├── vector_store.search_dense(vector, k=5) → List[SearchResult]
  ├── _build_context(results) → str (formatted context)
  ├── llm.generate(prompt, json_mode=True) → JSON string
  ├── _parse_response(json) → RAGResponse
  └── latency tính từ start_time

query_stream(question) → Generator[str]
  ├── Search + build context (giống trên)
  ├── _build_citations(results) → List[Dict]
  ├── yield "__CITATIONS__[{...}]"  ← Marker để API parse
  └── yield từng token từ llm.generate_stream()
```

#### AdaptiveRAGPipeline (v1.0 — chưa enable)

```
query(question) → RAGResponse
  ├── _classify(question) → SIMPLE | MEDIUM | COMPLEX
  │     └── LLM.generate(COMPLEXITY_CLASSIFIER_PROMPT)
  │
  ├── SIMPLE  → _simple_pipeline()
  │     └── Dense search → generate (2 LLM calls total: classify + generate)
  │
  ├── MEDIUM  → _medium_pipeline()
  │     └── Hybrid search (dense + sparse RRF) → generate
  │
  └── COMPLEX → _complex_pipeline()
        ├── Query rewrite
        ├── Multi-query expansion (3 variants)
        ├── HyDE (hypothetical document)
        ├── Hybrid search với 5 queries (rewrite + 2 variants + HyDE)
        ├── Deduplicate + sort by score
        └── Generate với top-5 best results
```

**RAGResponse dataclass**:
```python
answer: str           # Câu trả lời đầy đủ
citations: List[Dict] # Danh sách trích dẫn
confidence: str       # "high" | "medium" | "low"
disclaimer: str       # "Thông tin tham khảo, không phải tư vấn pháp lý"
latency_ms: float     # Thời gian xử lý
pipeline_used: str    # "naive" | "simple" | "medium" | "complex"
```

---

### 7. `src/api/main.py` — FastAPI Backend

**Vai trò**: REST API server — entry point chính của backend.

**Cấu trúc**:
```
FastAPI app (port 8000)
├── Lifespan:
│   ├── Startup:  khởi tạo LegalVectorStore, LLMGateway
│   │              embedder = None (lazy-load vì BGE-M3 nặng 2GB)
│   └── Shutdown: (no-op)
│
├── Middleware: CORS (allow all origins)
│
├── GET  /health              → {"status": "ok", "version": "0.1.0"}
│
├── POST /api/chat            → ChatResponse
│   ├── Input:  {question, session_id?, use_adaptive=false}
│   ├── Logic:  pipeline.query(question) → RAGResponse
│   └── Output: ChatResponse {answer, citations, confidence, latency_ms, pipeline_used}
│
├── POST /api/chat/stream     → SSE (Server-Sent Events)
│   ├── Input:  {question, session_id?, use_adaptive=false}
│   ├── Logic:  pipeline.query_stream(question) → Generator
│   │           Parse "__CITATIONS__" marker → yield citations event
│   │           Stream tokens → yield token event
│   │           Done → yield done event (full_answer + citations + confidence)
│   └── Output: EventSourceResponse
│               event: citations → data: [{doc_title, article, ...}, ...]
│               event: token     → data: {"token": "..."}
│               event: done      → data: {full_answer, citations, confidence, latency_ms}
│
├── GET  /api/documents        → Dict
│   ├── Query:  ?legal_type=Luật&keyword=doanh nghiệp&page=1&page_size=20
│   └── Logic:  store.list_documents() → scroll + dedup + paginate
│
└── GET  /api/stats            → Dict
    └── Logic:  store.get_collection_info() → #points, collection name
```

**Thứ tự startup**:
```
1. from dotenv import load_dotenv; load_dotenv()  ← QUAN TRỌNG: nếu không có dòng này,
                                                     os.getenv() không đọc .env
2. app = FastAPI(lifespan=...)
3. Trong lifespan: LegalVectorStore() + LLMGateway()
4. BGEEmbedder() được lazy-load ở request đầu tiên (tiết kiệm 2GB RAM khi idle)
```

---

### 8. `src/ui/app.py` — Streamlit UI (Entry Point)

**Vai trò**: Multi-page Streamlit app.

```
Streamlit App (port 8501)
├── Sidebar:
│   ├── Radio: 💬 Chat | 📁 Documents | 📊 Evaluation
│   ├── System status: check backend health (GET /health)
│   └── Model info: Qwen 3 32B + BGE-M3
│
├── 💬 Chat      → pages/chat.py     → render_chat_page()
├── 📁 Documents → pages/documents.py → render_documents_page()
└── 📊 Evaluation → pages/eval_dashboard.py → render_eval_page()
```

### 8a. `src/ui/pages/chat.py` — Chat Page

```
render_chat_page()
├── Sidebar settings:
│   ├── Checkbox: Adaptive RAG (v1.0+) — mặc định False
│   ├── Checkbox: Streaming — mặc định True
│   ├── Slider: Temperature (0.0-1.0, default 0.1)
│   └── Suggestions: 5 câu hỏi gợi ý có thể click
│
├── Chat history (st.session_state.messages)
│   ├── Role: user / assistant
│   └── Hiển thị: content + citations (cards) + confidence (badge) + latency
│
├── Chat input (st.chat_input)
│
└── Response handling:
    ├── Non-streaming: _get_response() → POST /api/chat → hiển thị
    └── Streaming: _stream_response() → POST /api/chat/stream → SSE parsing
         ├── event: citations → lưu citations list
         ├── event: token → append + render "▌" cursor
         └── event: done → final render + citations + confidence
```

**SSE parsing logic** (quan trọng):
```python
# SSE format: mỗi event có 2 dòng
# event: citations
# data: [...]
#
# event: token
# data: {"token": "Thủ"}
#
# Code duyệt iter_lines(), track event_type từ dòng "event: " trước đó
# Khi gặp "data: " thì parse JSON và xử lý theo event_type đã lưu
```

### 8b. `src/ui/pages/documents.py` — Document Browser

```
render_documents_page()
├── Filters:
│   ├── Selectbox: Loại văn bản (Tất cả / Luật / Bộ luật / ...)
│   ├── Text input: Từ khóa
│   └── Button: 🔍 Tìm kiếm
│
├── Results: GET /api/documents → data table
│
└── Stats section: GET /api/stats → total chunks + collection name
```

### 8c. `src/ui/pages/eval_dashboard.py` — Evaluation Dashboard

```
render_eval_page()
├── System Status (live từ API):
│   ├── Indexed Chunks
│   ├── Collection Name
│   ├── LLM Model
│   └── API Status
│
├── RAGAS Metrics (targets, chưa có data thật):
│   ├── Faithfulness > 0.80
│   ├── Answer Relevancy > 0.80
│   ├── Context Precision > 0.75
│   └── Context Recall > 0.80
│
├── Pipeline Performance (ước tính):
│   ├── Avg Latency: ~3.0s
│   ├── Search Latency: ~0.1s
│   ├── LLM Latency: ~2.8s
│   └── Cost/Query: $0.00 (Groq free)
│
└── Setup guide: Cách chạy RAGAS eval + GitHub Actions nightly
```

---

### 9. `scripts/ingest.py` — Data Ingestion CLI

**Vai trò**: Full pipeline từ parquet → Qdrant, chạy 1 lần.

```
python scripts/ingest.py --limit 100 --recreate

ingest(data_path, recreate, limit, embed_batch_size, device)
├── ds.dataset(parquet, format="parquet")  ← pyarrow, không phải pandas
├── dataset.to_batches(batch_size=200)     ← STREAMING, ~50MB/batch
│
├── For each batch:
│   ├── batch.to_pandas()
│   ├── For each row:
│   │   ├── filter_row(row_dict)
│   │   │   ├── legal_type check
│   │   │   ├── date check: 2015 ≤ year ≤ 2026
│   │   │   └── content length ≥ 500
│   │   │
│   │   ├── chunker.chunk_document(row_dict) → List[LegalChunk]
│   │   ├── Accumulate vào embed_text_buffer + chunk_buffer
│   │   │
│   │   └── Khi buffer đủ (≥ embed_batch_size * 4):
│   │       ├── embedder.encode_documents(texts) → (dense, sparse)
│   │       └── store.upsert_chunks(texts, dense, sparse, metadata)
│   │
│   └── Progress log mỗi 500 docs
│
└── Final flush: embed + index remaining chunks
```

**Options**:
- `--limit N`: Số documents tối đa (None = toàn bộ)
- `--recreate`: Xóa collection cũ trước khi index
- `--embed-batch-size 32`: Batch size cho embedding
- `--device cpu`: Device cho BGE-M3

---

### 10. Infrastructure Files

#### `docker-compose.yml`
```yaml
services:
  qdrant:    # Vector DB, port 6333 (REST) + 6334 (gRPC)
  redis:     # Cache (optional, chưa dùng trong MVP)
  api:       # FastAPI backend, port 8000
  ui:        # Streamlit frontend, port 8501 (depends_on: api)
```

#### `Dockerfile`
```
Multi-stage build:
  base  → pip install pyproject.toml + python-dotenv
  api   → COPY src/ + HEALTHCHECK curl /health
  ui    → COPY src/ + HEALTHCHECK curl port 8501
```

#### `Makefile`
```
make qdrant      → docker compose up -d qdrant redis
make ingest-test → python scripts/ingest.py --limit 100 --recreate
make api         → uvicorn src.api.main:app --port 8000 --reload
make ui          → streamlit run src/ui/app.py --port 8501
make dev         → api + ui cùng lúc (background)
make test        → pytest tests/ -v
```

---

## 🔧 Cách chạy & debug

### Lần đầu setup
```bash
# 1. Cài dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Copy + điền API keys
cp .env.example .env
# → Thêm GROQ_API_KEY=gsk_xxx vào .env

# 3. Start infrastructure
make qdrant    # Start Qdrant Docker

# 4. Download dataset
make download  # Tải từ HuggingFace (~3.6GB, 15-30 phút)

# 5. Test ingestion với 100 docs
make ingest-test  # ~2 phút trên CPU

# 6. Start backend + frontend
make dev       # API (8000) + UI (8501)
```

### Workflow hàng ngày (Qdrant đã có data)
```bash
# Terminal 1: Backend
make api       # uvicorn reload mode → auto-reload khi code thay đổi

# Terminal 2: Frontend
make ui        # Streamlit auto-reload

# Mở browser: http://localhost:8501
```

### Debug tips
```bash
# Test Qdrant search trực tiếp (không cần API)
.venv/bin/python -c "
from src.data.embedder import BGEEmbedder
from src.db.qdrant_client import LegalVectorStore
e = BGEEmbedder(device='cpu')
s = LegalVectorStore()
q = e.encode_queries(['thủ tục cấp phép xây dựng'])[0]
r = s.search_dense(q.tolist(), top_k=3)
for x in r: print(x.document_title, x.score)
"

# Test LLM gateway trực tiếp
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from src.llm.gateway import LLMGateway
g = LLMGateway()
print(g.generate('Chào bạn, 1+1 bằng mấy?', temperature=0))
"

# Check Qdrant collection size
curl -s http://localhost:6333/collections/vietnamese_laws | python3 -m json.tool

# Test API endpoints
curl -s http://localhost:8000/health
curl -s http://localhost:8000/api/stats
curl -s http://localhost:8000/api/documents?legal_type=Luật&page_size=5 | python3 -m json.tool
curl -s -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Điều kiện thành lập công ty TNHH là gì?"}' | python3 -m json.tool
```

---

## 📊 Các biến môi trường

| Variable | Mục đích | Required? |
|---|---|---|
| `GROQ_API_KEY` | Groq API key (free tier) | ✅ Required cho LLM |
| `OPENAI_API_KEY` | OpenAI fallback key | ❌ Optional |
| `QDRANT_URL` | Qdrant server URL | ❌ Default: localhost:6333 |
| `QDRANT_API_KEY` | Qdrant auth (cloud only) | ❌ Không cần cho local |
| `REDIS_URL` | Redis cache URL | ❌ Chưa dùng |
| `API_URL` | Backend URL (UI gọi) | ❌ Default: localhost:8000 |

---

## 🐛 Known Issues

| Issue | File | Status |
|---|---|---|
| Qwen 3 `<think>` tags trong stream output | `gateway.py:generate_stream()` | Chưa fix — non-streaming đã fix với `_clean_thinking()` |
| SSE streaming chậm hơn non-streaming do parsing overhead | `chat.py`, `api/main.py` | Chấp nhận được cho MVP |
| Chỉ index được 100 docs (MVP) | `scripts/ingest.py` | Để index toàn bộ: `python scripts/ingest.py --recreate` (~vài giờ) |
| Documents page scroll-based (không phải full-text search) | `qdrant_client.py:list_documents()` | OK cho MVP, sau có thể chuyển sang Qdrant full-text |
| Eval dashboard toàn placeholder data | `eval_dashboard.py` | Cần chạy `scripts/eval.py` + RAGAS để có data thật |
| Không có rate limiting | `api/main.py` | Cần thêm trước production |

---

## 📐 Architecture Decisions Log

| Decision | Rationale | Trade-off |
|---|---|---|
| BGE-M3 thay vì text-embedding-3-large | Free, multilingual tốt, có sẵn sparse | Chậm hơn API-based embedding |
| Qdrant thay vì Pinecone/Weaviate | Open-source, self-hosted, hybrid search native | Cần Docker, tự quản lý infra |
| LiteLLM thay vì gọi API trực tiếp | Auto-fallback, unified interface | Thêm 1 dependency |
| pyarrow streaming thay vì pandas | Memory: ~100MB vs 10GB | Code phức tạp hơn 1 chút |
| Qdrant native RRF thay vì client-side | Nhanh hơn, ít code hơn | Lock vào Qdrant ≥ 1.8 |
| Groq free tier thay vì OpenAI | $0 cost cho dev | Model có thể thay đổi bất kỳ lúc nào |
| Naive RAG trước Adaptive RAG | Có baseline để so sánh | MVP ít tính năng hơn |
| Streamlit thay vì React/Next.js | Tốc độ dev, Python-only | UX kém mượt hơn SPA |

---

## 🗺️ Hướng phát triển tiếp theo

Xem chi tiết trong [Vietnamese Legal Knowledge Bot Plan](../../../Vietnamese_Legal_Knowledge_Bot_Plan.md)

- **Week 1-2**: ✅ MVP — Naive RAG (hoàn thành)
- **Week 3-4**: Adaptive RAG (enable `use_adaptive=True`)
- **Week 5-6**: Confidence-based Answering + Hallucination Verification
- **Week 7-8**: Full ingestion (312K docs), RAGAS eval, production deploy
