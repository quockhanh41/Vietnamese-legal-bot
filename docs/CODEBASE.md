# 🏛️ Vietnamese Legal Knowledge Bot — Codebase Documentation

> Cẩm nang đọc code toàn diện: kiến trúc, data flow, từng module, cách debug.
> Dành cho chính bạn sau này và người mới onboard vào project.
>
> **Version**: Phase 2 — Full Legal Intelligence Pipeline
> **Last updated**: 2026-07-04

---

## 📁 Project Structure

```
vietnamese-legal-bot/
├── .env                          # API keys (GROQ_API_KEY, QDRANT_URL, etc.) — gitignored
├── .env.example                  # Template env vars
├── Dockerfile                    # Multi-stage: api / ui
├── docker-compose.yml            # 4 services: qdrant, redis, api, ui
├── Makefile                      # Shortcuts: make api, make ui, make dev, make ingest-test, etc.
├── pyproject.toml                # Dependencies (langchain-text-splitters, FlagEmbedding, litellm, etc.)
│
├── data/                         # 🚫 gitignored
│   ├── raw/
│   │   └── legal_docs_merged.parquet  # 518K docs, 3.6GB — dataset HuggingFace
│   └── metadata/                 # ✨ Phase 2: enriched document metadata (JSON)
│       ├── documents.json        #   Per-document cross-refs, timeline, amendments
│       ├── graph.json            #   Knowledge graph (nodes + edges)
│       ├── timelines.json        #   Effective/expiry dates for all docs
│       ├── amendments.json       #   Amendment instructions (MODIFY/INSERT/REPEAL)
│       └── index.json            #   Summary index
│
├── scripts/                      # ⚡ Entry points để chạy từ CLI
│   ├── download_dataset.py       # Tải từ HuggingFace → merge → save parquet
│   ├── explore_data.py           # Scan stats (legal_type, dates) — LOW MEMORY
│   ├── ingest.py                 # Full pipeline: filter → chunk → embed → Qdrant
│   ├── eval_chunking.py          # Evaluate chunking strategies
│   └── generate_eval_queries.py  # Generate test queries for RAGAS eval
│
├── src/                          # 📦 Code chính
│   ├── data/
│   │   ├── constants.py          # ✨ Single source of truth: LEGAL_TYPES, collection name
│   │   ├── metadata_store.py     # ✨ Phase 2: Document metadata persistence (JSON)
│   │   ├── chunker.py            # LegalDocumentChunker (legacy)
│   │   ├── embedder.py           # BGEEmbedder: BGE-M3 (dense 1024-dim + sparse)
│   │   ├── chunking/             # ✨ Modular chunking strategies
│   │   │   ├── hierarchical.py   #   S3: Chapter → Article → Clause (default)
│   │   │   ├── article_level.py  #   S1: Article-level boundaries
│   │   │   ├── fixed_size.py     #   S2: Fixed-size with overlap
│   │   │   ├── semantic.py       #   S4: Semantic (sentence-boundary aware)
│   │   │   └── sentence.py       #   S5: Sentence-level
│   │   ├── graph/                # ✨ Module 1: Cross-Referencing
│   │   │   ├── ref_extractor.py  #   9 regex patterns, 5 reference types
│   │   │   └── graph_builder.py  #   NetworkX DiGraph, BFS context expansion
│   │   ├── timeline/             # ✨ Module 2: Temporal Versioning
│   │   │   ├── date_extractor.py #   "Hiệu lực thi hành" parser (6 strategies)
│   │   │   └── version_resolver.py # Time-aware query engine + version chains
│   │   └── amendment/            # ✨ Module 3: Amendments Tracking
│   │       ├── amendment_extractor.py  # 20+ regex patterns, 15 action types
│   │       └── amendment_consolidator.py # Apply amendments to base law text
│   ├── pipeline/                 # ✨ Phase 2 Integration Layer
│   │   ├── enricher.py           # Enrichment orchestrator (runs all 3 modules)
│   │   └── retriever.py          # Enhanced retrieval + context expansion + time filter
│   ├── db/
│   │   └── qdrant_client.py      # LegalVectorStore: CRUD + hybrid search + retry
│   ├── llm/
│   │   ├── gateway.py            # LLMGateway: LiteLLM fallback chain (Groq Qwen3)
│   │   └── prompts.py            # Tất cả prompt templates
│   ├── rag/
│   │   └── orchestrator.py       # NaiveRAGPipeline + AdaptiveRAGPipeline
│   ├── api/
│   │   └── main.py               # FastAPI (12 endpoints: chat + Phase 2 pipeline)
│   └── ui/
│       ├── app.py                # Streamlit entry point (sidebar nav)
│       └── pages/
│           ├── chat.py           # 💬 Chat page (streaming + citations)
│           ├── documents.py      # 📁 Document browser (filter + search)
│           └── eval_dashboard.py # 📊 RAGAS metrics dashboard
│
├── tests/                        # Unit tests (TODO)
└── docs/
    ├── lessons-learned.md        # Bài học engineering (bugs + fixes)
    └── CODEBASE.md               # ← You are here
```

---

## 🔄 Data Flow — 3 luồng chính

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
│ filter_row()     │  Loại: legal_type ∈ {Luật} (Phase 2 narrowed)
│                  │  Thời gian: DATE_FILTER_MIN_YEAR/MAX_YEAR (None = all)
│                  │  Nội dung: len(content) ≥ 500 chars
│                  │  → 595 docs (Luật + Bộ luật combined)
└────────┬─────────┘
         ▼
┌──────────────────────────────────────┐
│ Hierarchical Chunking (S3 default):  │
│  - MarkdownHeaderTextSplitter        │
│  - Chương → Mục → Điều → Khoản       │
│  - Size: 500-2000 chars/chunk        │
│  - Metadata inheritance per chunk    │
└────────┬─────────────────────────────┘
         ▼
┌──────────────────┐
│ BGEEmbedder      │  BAAI/bge-m3 (FlagEmbedding):
│ BGE-M3           │  - dense_vecs: np.ndarray (N, 1024) float16
│                  │  - lexical_weights: List[defaultdict[int, float]]
│                  │  - Batch size: 32 (để tránh OOM)
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Qdrant Cloud     │  Collection: "luat_law"
│ (Australia)      │  - Dense vector: 1024-dim COSINE
│                  │  - Sparse vector: lexical index
│                  │  - Payload: chunk_text + full metadata
│                  │  - Retry logic: 3 attempts, exponential backoff
└──────────────────┘
```

### Luồng 2: Phase 2 Enrichment (Làm giàu metadata)

```
┌──────────────────┐
│ Qdrant Cloud     │  Collection: "luat_law"
│ (6/595 docs)     │
└────────┬─────────┘
         │ EnrichmentPipeline.run()
         │ Scroll tất cả points → dedup by document_id
         │ Reconstruct full text từ chunks
         ▼
┌──────────────────────────────────────────────────────────────┐
│                     FOR EACH DOCUMENT                        │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ M1: Cross-Ref   │  │ M2: Temporal    │  │ M3: Amendment│ │
│  │                 │  │                 │  │              │ │
│  │ 9 regex patterns│  │ rfind strategy  │  │ 20+ patterns │ │
│  │ → Reference[]   │  │ → LawTimeline   │  │ → Instruction│ │
│  │ → KnowledgeGraph│  │ → Version chain │  │ → Target map │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                    │                   │         │
└───────────┼────────────────────┼───────────────────┼─────────┘
            │                    │                   │
            ▼                    ▼                   ▼
┌──────────────────────────────────────────────────────────────┐
│                      MetadataStore                            │
│                                                              │
│  data/metadata/                                              │
│  ├── documents.json   ← DocumentMeta (refs, dates, amend)    │
│  ├── graph.json       ← Knowledge graph (nodes + edges)      │
│  ├── timelines.json   ← LawTimeline (effective/expiry dates) │
│  ├── amendments.json  ← AmendmentInstruction list            │
│  └── index.json       ← Summary index                        │
└──────────────────────────────────────────────────────────────┘
```

### Luồng 3: Enhanced Query (Hỏi đáp pháp luật — Phase 2)

```
┌──────────────────┐
│ User Question    │  "Điều 123 Bộ luật Hình sự 2018 quy định thế nào?"
└────────┬─────────┘
         │ POST /api/chat hoặc /api/chat/stream
         │ expand_context=true, expansion_hops=1
         ▼
┌──────────────────┐
│ FastAPI          │  src/api/main.py
│                  │  Lazy-load BGEEmbedder, EnhancedRetriever
│                  │  Chọn NaiveRAGPipeline hoặc AdaptiveRAGPipeline
└────────┬─────────┘
         ▼
┌──────────────────────────────────────────────────────────┐
│ EnhancedRetriever.retrieve()                             │
│                                                          │
│  Step 1: embedder.encode(query) → dense + sparse         │
│                                                          │
│  Step 2: vector_store.search_hybrid(dense, sparse, k=15) │
│          → List[SearchResult]                            │
│                                                          │
│  Step 3: _enrich_result() — merge Phase 2 metadata:      │
│          ├── effective_date từ MetadataStore              │
│          ├── is_amended check (article có bị sửa đổi?)    │
│          └── amending_laws list                           │
│                                                          │
│  Step 4: (optional) _expand_context() — BFS on graph:    │
│          ├── Follow edges: Điều 123 → Điều 456, Điều 789 │
│          └── Add referenced article chunks into context   │
│                                                          │
│  Step 5: NaiveRAGPipeline.query() — generate answer      │
│          → LLM.generate(context+question, json_mode)     │
│                                                          │
│  Step 6: _parse_response(json) → RAGResponse             │
│          answer + citations + confidence + expanded      │
└──────────────────┬───────────────────────────────────────┘
                   ▼
┌──────────────────┐
│ ChatResponse     │  JSON: {answer, citations[], confidence, disclaimer,
│ (JSON)           │         latency_ms, pipeline_used, expanded_results}
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

### 2. `src/data/constants.py` — Shared Configuration ✨

**Vai trò**: Single source of truth cho tất cả shared config.

```python
LEGAL_TYPES: set[str] = {"Luật"}            # Phase 2: narrowed from 6 types
ALL_LEGAL_TYPES: list[str] = [              # For UI dropdowns
    "Luật", "Bộ luật", "Nghị định", "Thông tư", "Nghị quyết", "Quyết định"
]
MIN_CONTENT_LENGTH: int = 500               # Skip short/empty docs
DEFAULT_COLLECTION_NAME: str = "luat_law"   # Phase 2 collection
DATE_FILTER_MIN_YEAR: int | None = None     # No date filter (keep all)
DATE_FILTER_MAX_YEAR: int | None = None
```

**Điểm cần lưu ý**:
- "Bộ luật" documents are classified as "Luật" in the source dataset
- Được import từ 6 files (ingest.py, explore_data.py, download_dataset.py, etc.)
- Python 3.9: cần `from __future__ import annotations` cho `int | None` syntax

---

### 3. `src/data/embedder.py` — BGEEmbedder

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

## 🧩 Phase 2 Modules — Legal Intelligence

### 6a. `src/data/graph/` — Module 1: Cross-Referencing 🔗

**Vai trò**: Trích xuất và xây dựng knowledge graph từ các tham chiếu chéo giữa các điều luật.

**Cấu trúc**:
```
src/data/graph/
├── ref_extractor.py        # ReferenceExtractor
│   ├── 9 compiled regex patterns:
│   │   ├── EXTERNAL_LAW:  "Điều X của Luật Y"
│   │   ├── EXTERNAL_CODE: "Điều X của Bộ luật Y"
│   │   ├── INTERNAL:      "Điều X của Luật/Bộ luật này"
│   │   ├── CLAUSE_DIEU:   "khoản X Điều Y"
│   │   ├── TAI_DIEU_LUAT: "tại Điều X của Luật Y"
│   │   ├── MULTI_DIEU:    "Điều X và Điều Y"
│   │   └── CHAPTER:       "tại Chương X"
│   ├── RefType enum: INTERNAL, EXTERNAL, CLAUSE, MULTI, CHAPTER
│   └── Reference dataclass: source/target doc_id, article, clause, confidence
│
└── graph_builder.py        # LegalKnowledgeGraph
    ├── NetworkX DiGraph — directed edges
    ├── ArticleNode dataclass (node_id format: doc_{id}_dieu_{num})
    ├── add_references(refs) → builds nodes + edges
    ├── get_neighbors(article_id, hops, direction) → BFS traversal
    ├── expand_context(article_ids, hops) → query-time expansion
    ├── most_referenced(top_n) → highest in-degree articles
    ├── find_orphans() → articles with no outgoing refs
    └── to_dict() → JSON-serializable export
```

**Usage**:
```python
from src.data.graph import ReferenceExtractor, LegalKnowledgeGraph

extractor = ReferenceExtractor()
graph = LegalKnowledgeGraph()

refs = extractor.extract(text=content, doc_id=123, law_name="Luật ABC")
graph.add_references(refs)

# Query-time: expand context
expanded = graph.expand_context(["doc_123_dieu_5"], hops=1)
```

**Stats (6 Criminal Code docs)**:
- 409 nodes, 403 edges, 204 internal refs, 23 external refs
- 6 connected components, density: 0.002

---

### 6b. `src/data/timeline/` — Module 2: Temporal Versioning 📅

**Vai trò**: Trích xuất ngày hiệu lực/hết hiệu lực và resolve phiên bản luật theo thời gian.

**Cấu trúc**:
```
src/data/timeline/
├── date_extractor.py       # DateExtractor
│   ├── LawTimeline dataclass: effective_date, expiry_date, article_dates
│   ├── _find_effectiveness_article() — 4 chiến lược:
│   │   ├── 1. rfind("Hiệu lực thi hành") — last occurrence (>70% position)
│   │   ├── 2. Regex "Điều X. Hiệu lực thi hành"
│   │   ├── 3. Amendment pattern "Điều 2. Hiệu lực thi hành"
│   │   └── 4. Any "hiệu lực thi hành" (after preamble)
│   ├── _extract_primary_date() — priority:
│   │   ├── "có hiệu lực thi hành từ ngày DD/MM/YYYY"
│   │   ├── "ngày DD tháng MM năm YYYY"
│   │   └── Any DD/MM/YYYY (validated: 1990-2030)
│   ├── _extract_relative_date() — "sau X ngày kể từ ngày công bố"
│   ├── _extract_expiry() — replaced laws detection
│   └── Fallback: search last 3000 chars for any date
│
└── version_resolver.py     # VersionResolver
    ├── VersionResult: law_name, effective_date, status, replaced_by
    ├── at_time(when, law_name) → laws valid at a point in time
    ├── version_chain(law_name) → all versions sorted by date
    ├── find_replacement(doc_id) → which law replaced this one?
    ├── _classify_status() → "current" | "future" | "expired" | "unknown"
    └── _normalize_name() → strip year/number for fuzzy matching
```

**Version Chain example (BLHS)**:
```
Bộ Luật Hình sự 1999 (1999-12-21)
  → Bộ luật hình sự 2015 (2016-07-01)
  → Luật sửa đổi BLHS 2017 (2018-01-01)
  → Luật sửa đổi BLHS 2025 (2025-06-25) 🟢 current
```

---

### 6c. `src/data/amendment/` — Module 3: Amendments Tracking 🔧

**Vai trò**: Phát hiện "Luật sửa đổi", trích xuất hướng dẫn sửa đổi, áp dụng vào văn bản gốc.

**Cấu trúc**:
```
src/data/amendment/
├── amendment_extractor.py  # AmendmentExtractor
│   ├── AmendmentAction enum (15 types):
│   │   ├── MODIFY: modify_article, modify_clause, modify_point,
│   │   │          modify_multi, modify_multi_article
│   │   ├── INSERT:  insert_article, insert_clause, insert_point
│   │   ├── REPEAL:  repeal_article, repeal_clause, repeal_point
│   │   ├── REPLACE_WORD, BATCH_REPLACE, BATCH_INSERT
│   │   └── UNKNOWN
│   ├── 20+ regex patterns phủ các dạng:
│   │   ├── "Sửa đổi, bổ sung Điều X như sau:"
│   │   ├── "Sửa đổi, bổ sung khoản X Điều Y như sau:"
│   │   ├── "Sửa đổi, bổ sung các khoản X, Y, Z Điều W"
│   │   ├── "Bổ sung điểm X vào sau điểm Y khoản Z Điều W"
│   │   ├── "Bổ sung Điều X vào sau Điều Y"
│   │   ├── "Bãi bỏ Điều X / khoản X Điều Y"
│   │   ├── "Thay thế cụm từ X bằng Y"
│   │   └── "Sửa đổi, bổ sung tên điều, các khoản X, Y"
│   ├── extract_target_laws() — detect base laws from title
│   ├── _split_sections() — handle multi-law amendments
│   ├── _detect_target_in_instruction() — detect target law changes within section
│   └── _parse_sub_instructions() — handle a), b), c) sub-items
│
└── amendment_consolidator.py  # AmendmentConsolidator
    ├── apply(base_text, instructions) → ConsolidationResult
    ├── apply_to_article(article_text, instructions)
    ├── _apply_modify_article/clause/point() — text replacement
    ├── _apply_insert_article/clause/point() — text insertion
    ├── _apply_repeal_article/clause/point() — mark as [Bãi bỏ]
    ├── _apply_replace_word() — inline word substitution
    ├── _find_article_range() — locate article boundaries
    └── extract_article_text() / list_articles()
```

**Known limitation**: Consolidation works best with article-level chunks; full-text reconstruction từ Qdrant chunks có thể bị thiếu article headers.

**Stats (3 amendment laws trên 6 docs)**:
- 2017 amendment: 180 instructions, 149 articles affected
- 2021 amendment: 7 instructions, 7 articles
- 2025 amendment: 46 instructions, 27 articles

---

### 6d. `src/data/metadata_store.py` — Metadata Persistence 💾

**Vai trò**: JSON-backed document metadata store cho toàn bộ Phase 2 enrichment data.

```python
MetadataStore("data/metadata")
├── documents.json   ← Dict[int, DocumentMeta]  (per-doc metadata)
├── graph.json       ← {nodes, edges}            (knowledge graph)
├── timelines.json   ← List[LawTimeline dict]    (temporal data)
├── amendments.json  ← List[Instruction dict]    (amendment data)
└── index.json       ← Summary index

DocumentMeta fields:
  - ref_count, internal_refs, external_refs    (M1)
  - effective_date, expiry_date, is_in_effect  (M2)
  - is_amendment, target_laws, amendment_count (M3)
  - version_chain, graph_node_id, enriched_at
```

**Key operations**:
- `get_document(doc_id)` → DocumentMeta
- `get_effective_laws(at_date)` → laws valid at a date
- `get_article_neighbors(node_id, hops)` → graph traversal
- `get_amendments_for_doc(doc_id)` → amendment instructions
- `build_index()` → summary statistics

---

### 6e. `src/pipeline/` — Integration Layer 🔄

**Vai trò**: Orchestrate Phase 2 enrichment và enhanced retrieval.

```
src/pipeline/
├── enricher.py             # EnrichmentPipeline
│   ├── run() → EnrichmentResult
│   │   ├── _collect_documents() → Dict[int, DocInfo] (scroll Qdrant)
│   │   ├── FOR each document:
│   │   │   ├── M1: extract refs → build graph
│   │   │   ├── M2: extract timeline → version chain
│   │   │   ├── M3: extract amendments (if amendment law)
│   │   │   └── Save DocumentMeta to MetadataStore
│   │   ├── Resolve version chains via VersionResolver
│   │   ├── Save graph, timelines, amendments to MetadataStore
│   │   └── Build index
│   └── get_status() → current enrichment status
│
└── retriever.py            # EnhancedRetriever
    ├── retrieve(query, top_k, at_date, expand_context, hops)
    │   ├── Hybrid search (dense + sparse RRF)
    │   ├── _enrich_result() → merge Phase 2 metadata
    │   ├── Time filter (at_date): skip not-yet-effective laws
    │   └── _expand_context() → BFS graph traversal
    ├── retrieve_with_expansion() → convenience wrapper
    ├── get_article_history(doc_id, article) → amendment timeline
    └── stats → system statistics
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

### 7. `src/api/main.py` — FastAPI Backend (Phase 2)

**Vai trò**: REST API server — 12 endpoints including Phase 2 pipeline.

**Cấu trúc**:
```
FastAPI app (port 8000)
├── Lifespan:
│   ├── Startup:  khởi tạo LegalVectorStore, LLMGateway, MetadataStore
│   │              embedder + enhanced_retriever = None (lazy-load)
│   └── Shutdown: (no-op)
│
├── Middleware: CORS (allow all origins)
│
├── ── Core Endpoints ──
│
├── GET  /health              → {"status": "ok", "version": "0.2.0"}
│
├── POST /api/chat            → ChatResponse (Phase 2 enhanced)
│   ├── Input:  {question, session_id?, use_adaptive, expand_context, expansion_hops}
│   └── Output: ChatResponse {answer, citations, ..., expanded_results}
│
├── POST /api/chat/stream     → SSE (Server-Sent Events)
│   └── Streaming với citations + token + done events
│
├── GET  /api/documents        → Dict (paginated document list)
│
├── ── Phase 2 Pipeline Endpoints ──
│
├── POST /api/pipeline/enrich  → Run Phase 2 enrichment
│   └── Output: {status, docs_processed, graph_nodes, total_amendments, errors, ...}
│
├── GET  /api/pipeline/status  → Enrichment status + index
│
├── POST /api/search/enhanced  → Enhanced search (Phase 2)
│   ├── Query: ?query=...&top_k=5&expand_context=true&expansion_hops=1
│   └── Output: {results: [{...phase2 metadata...}], total}
│
├── GET  /api/article/history  → Article amendment timeline
│   ├── Query: ?doc_id=296661&article=Điều 3
│   └── Output: {is_amended, amending_laws[], version_chain[]}
│
├── GET  /api/graph/stats      → Knowledge graph statistics
│   └── Output: {nodes, edges, node_types{}, edge_types{}}
│
├── GET  /api/graph/neighbors   → Graph neighbors for article
│   └── Query: ?doc_id=296661&article=Điều 3&hops=1
│
├── GET  /api/timeline/effective → Laws effective at a date
│   └── Query: ?at_date=2018-01-01
│
└── GET  /api/stats            → System stats (Phase 2 enhanced)
    └── Output: {indexed_documents, collection_name, phase2: {enriched, graph, ...}}
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

### 9. `scripts/ingest.py` — Data Ingestion CLI (Phase 2)

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
│   │   │   ├── legal_type ∈ {Luật}        ← Phase 2: narrowed from 6 types
│   │   │   ├── date check: configurable (DATE_FILTER_MIN/MAX_YEAR)
│   │   │   └── content length ≥ 500
│   │   │
│   │   ├── HierarchicalChunker (S3 default)
│   │   │   └── Chương → Mục → Điều → Khoản hierarchy
│   │   ├── Accumulate vào embed_text_buffer + chunk_buffer
│   │   │
│   │   └── Khi buffer đủ (≥ embed_batch_size * 4):
│   │       ├── embedder.encode_documents(texts) → (dense, sparse)
│   │       └── store.upsert_chunks(texts, dense, sparse, metadata)
│   │           └── Batch size 25 (Cloud), 3 retries with exponential backoff
│   │
│   └── Progress log mỗi 500 docs
│
└── Final flush: embed + index remaining chunks
```

**Key changes from Phase 1**:
- Legal type filter narrowed to `{"Luật"}` only
- Date filter made optional (None = keep all years for temporal analysis)
- Collection name changed to `luat_law` via `constants.py`
- Qdrant Cloud: timeout=120s, batch_size=25, retry với exponential backoff

**Options**:
- `--limit N`: Số documents tối đa (None = toàn bộ)
- `--recreate`: Xóa collection cũ trước khi index
- `--embed-batch-size 32`: Batch size cho embedding
- `--device cpu`: Device cho BGE-M3
- `--chunk-strategy s3`: Chunking strategy (s1-s5)

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

# Phase 2: Test enrichment pipeline
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from src.db.qdrant_client import LegalVectorStore
from src.data.metadata_store import MetadataStore
from src.pipeline.enricher import EnrichmentPipeline
store = LegalVectorStore(collection_name='luat_law', timeout=120)
meta = MetadataStore()
pipe = EnrichmentPipeline(store, meta)
result = pipe.run()
print(f'Enriched: {result.docs_processed} docs, {result.total_amendments} amendments')
print(f'Graph: {result.graph_nodes} nodes, {result.graph_edges} edges')
"

# Phase 2: Test enhanced search
curl -s -X POST 'http://localhost:8000/api/search/enhanced?query=Điều+123+hình+sự&top_k=5&expand_context=true'

# Phase 2: Test article history
curl -s 'http://localhost:8000/api/article/history?doc_id=296661&article=Điều+3'

# Phase 2: Check enrichment status
curl -s http://localhost:8000/api/pipeline/status | python3 -m json.tool

# Phase 2: Run enrichment via API
curl -s -X POST http://localhost:8000/api/pipeline/enrich | python3 -m json.tool

# View metadata store
cat data/metadata/index.json | python3 -m json.tool
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
| Qwen 3 `<think>` tags trong stream output | `gateway.py:generate_stream()` | Chưa fix |
| SSE streaming chậm hơn non-streaming | `chat.py`, `api/main.py` | Acceptable |
| Qdrant Cloud latency từ Vietnam | `qdrant_client.py` | Mitigated: retry + smaller batches |
| 1/6 docs thiếu effective date (BLHS 1999 format cũ) | `date_extractor.py` | Data limitation |
| NetworkX graph không persist (rebuild mỗi lần) | `graph_builder.py` | Saved as JSON in MetadataStore |
| Consolidation limited by chunked text | `amendment_consolidator.py` | Cần full-text article reconstruction |
| Sub-instructions trong batch ops classified UNKNOWN | `amendment_extractor.py` | Minor — ~19/180 items |
| Eval dashboard toàn placeholder data | `eval_dashboard.py` | Cần RAGAS eval |
| Không có rate limiting | `api/main.py` | Cần trước production |

---

## 📐 Architecture Decisions Log

| Decision | Rationale | Trade-off |
|---|---|---|
| BGE-M3 thay vì text-embedding-3-large | Free, multilingual tốt, có sẵn sparse | Chậm hơn API-based embedding |
| Qdrant Cloud thay vì local Docker | Production-ready, no local RAM usage | Network latency (Vietnam→Australia) |
| Qdrant native RRF thay vì client-side | Nhanh hơn, ít code hơn | Lock vào Qdrant ≥ 1.18 |
| LiteLLM thay vì gọi API trực tiếp | Auto-fallback, unified interface | Thêm 1 dependency |
| pyarrow streaming thay vì pandas | Memory: ~100MB vs 10GB | Code phức tạp hơn 1 chút |
| Groq free tier thay vì OpenAI | $0 cost cho dev | Model có thể thay đổi bất kỳ lúc nào |
| Naive RAG trước Adaptive RAG | Có baseline để so sánh | MVP ít tính năng hơn |
| Streamlit thay vì React/Next.js | Tốc độ dev, Python-only | UX kém mượt hơn SPA |
| ✨ Single collection + metadata filter | Đơn giản, đủ cho 595 docs | Không scale đến hàng triệu docs |
| ✨ JSON metadata store | Human-readable, no DB dependency | Cần migrate nếu >10K docs |
| ✨ NetworkX cho graph (không Neo4j) | In-memory, no infra, đủ cho <100K edges | Không persist, phải rebuild |
| ✨ Regex-based extraction (không LLM) | Nhanh, deterministic, free | Miss edge cases in unstructured text |
| ✨ rfind strategy cho "Hiệu lực" | Bắt đúng article cuối cùng | Old laws without the article format fail (1/6) |
| ✨ Lazy-load embedder trong retriever | Import pipeline không cần FlagEmbedding | Hơi chậm ở lần gọi đầu tiên |

---

## 🗺️ Roadmap

### Phase 1: MVP ✅
- Naive RAG pipeline (dense search → LLM generate)
- Streamlit UI (chat, documents, eval dashboard)
- FastAPI backend (5 endpoints)
- 100 docs indexed

### Phase 2: Legal Intelligence ✅
- ✅ **Module 1**: Cross-Referencing — 409 nodes, 403 edges knowledge graph
- ✅ **Module 2**: Temporal Versioning — effective dates, version chains, time travel queries
- ✅ **Module 3**: Amendments Tracking — 15 action types, 233 instructions extracted
- ✅ **Integration**: Enrichment pipeline + Enhanced retriever + 12 API endpoints
- ✅ Tested on 6-doc Criminal Code cluster (BLHS 1999→2015→2017→2025)

### Phase 3: Scale & Production 📋
- Scale enrichment to all 595 docs (toàn bộ Luật + Bộ luật)
- Article-level re-indexing with amendment-consolidated text
- RAGAS evaluation: compare Phase 2-expanded context vs baseline
- Full ingestion pipeline optimization (parallel embedding)
- Production deployment (Docker Compose / cloud)

### Phase 4: Advanced Features 💡
- LLM-based amendment consolidation (semantic text alignment)
- Real-time legal updates (monitor for new "Luật sửa đổi")
- Cross-law impact analysis: "nếu Điều X thay đổi, điều gì bị ảnh hưởng?"
- Temporal RAG: automatically resolve "at what date" from query context
- Vietnamese legal ontology / taxonomy integration
- GraphQL API for knowledge graph queries
