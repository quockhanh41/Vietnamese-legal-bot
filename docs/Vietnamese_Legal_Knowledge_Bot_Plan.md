# 🏛️ PLAN: Vietnamese Legal Knowledge Bot

> **Academic Portfolio Project** — sau Giai đoạn 4 (RAG Systems Production Grade)
>
> **Status**: Phase 1 ✅ | Phase 2 ✅ | Phase 3+ 📋
>
> **Tagline**: *Chatbot tra cứu văn bản pháp luật Việt Nam — Adaptive RAG + Legal Intelligence (Cross-Refs, Temporal, Amendments)*

---

## 📋 MỤC LỤC

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Phase 2: Legal Intelligence Pipeline](#4-phase-2-legal-intelligence-pipeline) ⭐ IMPLEMENTED
   - 4.1 [Cross-Referencing (Knowledge Graph)](#41-cross-referencing-knowledge-graph)
   - 4.2 [Temporal Versioning (Time-Aware Search)](#42-temporal-versioning-time-aware-search)
   - 4.3 [Amendments Tracking (Merge & Consolidate)](#43-amendments-tracking-merge--consolidate)
   - 4.4 [Pipeline Integration & Enhanced Retrieval](#44-pipeline-integration--enhanced-retrieval)
5. [RAG Pipeline Design](#5-rag-pipeline-design)
   - 5.1 [Phases of Evolution](#51-phases-of-evolution)
   - 5.2 [Adaptive Retrieval (Query Complexity Router)](#52-adaptive-retrieval-query-complexity-router) ⭐ NEW
   - 5.3 [Confidence-based Answering](#53-confidence-based-answering) ⭐ NEW
   - 5.4 [Hallucination Verification (2-tier)](#54-hallucination-verification-2-tier) ⭐ NEW
   - 5.5 [Dynamic Top-K](#55-dynamic-top-k) ⭐ NEW
   - 5.6 [Prompt Templates](#56-prompt-templates)
   - 5.7 [Multi-Hop Reasoning](#57-multi-hop-reasoning)
6. [Tech Stack](#6-tech-stack)
7. [Project Structure](#7-project-structure)
8. [Feature Roadmap](#8-feature-roadmap)
9. [Evaluation Strategy](#9-evaluation-strategy)
10. [UI/UX Design](#10-uiux-design)
11. [Deployment Plan](#11-deployment-plan)
12. [Testing Strategy](#12-testing-strategy)
13. [Timeline & Milestones](#13-timeline--milestones)
14. [Risk & Mitigation](#14-risk--mitigation)
15. [README Template](#15-readme-template)

---

## 1. PROJECT OVERVIEW

### 1.1. Problem Statement

Doanh nghiệp và người dân Việt Nam phải tuân thủ hàng nghìn văn bản pháp luật thuộc nhiều lĩnh vực: dân sự, hình sự, lao động, đất đai, thuế, hành chính... Việc tra cứu thủ công mất nhiều thời gian, dễ bỏ sót điều khoản quan trọng. Một chatbot RAG có thể:
- Trả lời câu hỏi pháp lý **kèm trích dẫn điều khoản chính xác**
- Giảm thời gian tra cứu từ 30 phút xuống <5 giây
- Đảm bảo câu trả lời **có căn cứ** (grounded in legal documents), không hallucinate

### 1.2. Core Value Proposition

| Tiêu chí | Mục tiêu |
|----------|----------|
| **Accuracy** | RAGAS Faithfulness > 0.80 |
| **Latency** | <3s per query (end-to-end, including streaming first token) |
| **Coverage** | 595 văn bản Luật + Bộ luật (toàn bộ, không giới hạn năm) |
| **Language** | Tiếng Việt hoàn toàn |
| **Citations** | Mỗi câu trả lời kèm điều khoản + link văn bản gốc |
| **Legal Intelligence** ⭐ | Cross-references, temporal versioning, amendments tracking |
| **Test Cluster** | 6 docs: Bộ luật Hình sự cluster (1999→2015→2017→2025) |

### 1.3. Dataset

- **Nguồn**: [`vohuutridung/vietnamese-legal-documents`](https://huggingface.co/datasets/vohuutridung/vietnamese-legal-documents) trên Hugging Face
- **Tổng**: 518,255 văn bản (~3.6 GB, Markdown format)
- **Subset sau filter**: 595 văn bản (legal_type = `Luật`, bao gồm cả Bộ luật)
- **Phase 2 test**: 6 documents (Bộ luật Hình sự cluster) — 2,174 vectors in Qdrant
- **Metadata**: `id`, `document_number`, `title`, `url`, `legal_type`, `legal_sectors`, `issuing_authority`, `issuance_date`, `signers`

---

## 2. SYSTEM ARCHITECTURE

### 2.1. High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (Port 8501)                          │
│                                                                           │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────┐   │
│  │   💬 Chat Panel   │  │  📁 Document Browser  │  │  📊 Eval Dashboard │   │
│  │                   │  │                       │  │                    │   │
│  │  • Streaming      │  │  • Filter by type     │  │  • RAGAS scores    │   │
│  │  • Multi-turn     │  │  • Filter by sector   │  │  • Latency charts  │   │
│  │  • Citations      │  │  • Full-text view     │  │  • Query stats     │   │
│  │  • Source links   │  │  • Search highlight   │  │  • Failure modes   │   │
│  └────────┬─────────┘  └──────────┬────────────┘  └─────────┬──────────┘   │
└───────────┼───────────────────────┼──────────────────────────┼─────────────┘
            │                       │                          │
┌───────────▼───────────────────────▼──────────────────────────▼─────────────┐
│                          FASTAPI BACKEND (Port 8000)                        │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────┐    │
│  │ /api/chat/stream│  │ /api/documents  │  │ /api/eval                │    │
│  │ POST (SSE)      │  │ GET (list+filter)│  │ GET (RAGAS dashboard)    │    │
│  │ • question      │  │ • type          │  │ • faithfulness           │    │
│  │ • session_id    │  │ • sector        │  │ • relevancy              │    │
│  │ • history       │  │ • keyword       │  │ • precision/recall       │    │
│  │ → SSE stream    │  │ • page/size     │  │ • latency p50/p95/p99    │    │
│  └────────┬────────┘  │ → JSON          │  │ → JSON                   │    │
│           │           └────────┬────────┘  └───────────┬──────────────┘    │
│           │                    │                       │                    │
│  ┌────────▼────────────────────▼───────────────────────▼──────────────┐    │
│  │                    RAG ORCHESTRATOR (LangChain)                      │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                   QUERY PROCESSING PHASE                       │   │    │
│  │  │                                                               │   │    │
│  │  │  User: "Thành lập công ty TNHH cần những giấy tờ gì?"         │   │    │
│  │  │    │                                                          │   │    │
│  │  │    ▼                                                          │   │    │
│  │  │  ┌────────────────┐                                           │   │    │
│  │  │  │ 1. QUERY REWRITE│  LLM: "Hồ sơ đăng ký thành lập           │   │    │
│  │  │  │   (LLM call)   │  doanh nghiệp TNHH 1 thành viên           │   │    │
│  │  │  └───────┬────────┘  gồm những giấy tờ gì?"                   │   │    │
│  │  │          │                                                    │   │    │
│  │  │          ▼                                                    │   │    │
│  │  │  ┌────────────────┐                                           │   │    │
│  │  │  │ 2. MULTI-QUERY │  → "hồ sơ đăng ký doanh nghiệp TNHH"     │   │    │
│  │  │  │   EXPANSION    │  → "giấy tờ thành lập công ty TNHH"       │   │    │
│  │  │  │   (LLM call)   │  → "thủ tục đăng ký kinh doanh"           │   │    │
│  │  │  └───────┬────────┘                                           │   │    │
│  │  │          │                                                    │   │    │
│  │  │          ▼                                                    │   │    │
│  │  │  ┌────────────────┐                                           │   │    │
│  │  │  │ 3. HYPOTHETICAL │  LLM: "Theo Luật Doanh Nghiệp 2020,      │   │    │
│  │  │  │   DOC (HyDE)   │  hồ sơ đăng ký thành lập công ty          │   │    │
│  │  │  │   (LLM call)   │  TNHH bao gồm: giấy đề nghị,             │   │    │
│  │  │  └───────┬────────┘  điều lệ công ty, danh sách..."           │   │    │
│  │  │          │                                                    │   │    │
│  └─────────────┼────────────────────────────────────────────────────┘   │    │
│                │                                                        │    │
│  ┌─────────────▼────────────────────────────────────────────────────┐   │    │
│  │                   RETRIEVAL PHASE                                  │   │    │
│  │                                                                    │   │    │
│  │  ┌──────────────────────┐    ┌──────────────────────┐             │   │    │
│  │  │ 4a. DENSE SEARCH     │    │ 4b. SPARSE SEARCH    │             │   │    │
│  │  │ (BGE-M3 Embedding)   │    │ (BM25 via Qdrant)    │             │   │    │
│  │  │ → Top-20 results     │    │ → Top-20 results     │             │   │    │
│  │  └──────────┬───────────┘    └──────────┬───────────┘             │   │    │
│  │             │                           │                          │   │    │
│  │             └───────────┬───────────────┘                          │   │    │
│  │                         │                                          │   │    │
│  │                         ▼                                          │   │    │
│  │  ┌──────────────────────────────────────────────┐                 │   │    │
│  │  │ 5. RECIPROCAL RANK FUSION (RRF)               │                 │   │    │
│  │  │ Merge dense + sparse → Top-15 unique results  │                 │   │    │
│  │  └──────────────────────┬───────────────────────┘                 │   │    │
│  │                         │                                          │   │    │
│  │                         ▼                                          │   │    │
│  │  ┌──────────────────────────────────────────────┐                 │   │    │
│  │  │ 6. RE-RANKING (BGE-Reranker v2)              │                 │   │    │
│  │  │ Cross-encoder score từng chunk vs query       │                 │   │    │
│  │  │ → Top-5 results                               │                 │   │    │
│  │  └──────────────────────┬───────────────────────┘                 │   │    │
│  │                         │                                          │   │    │
│  │                         ▼                                          │   │    │
│  │  ┌──────────────────────────────────────────────┐                 │   │    │
│  │  │ 7. CONTEXTUAL COMPRESSION                     │                 │   │    │
│  │  │ LLM: extract only relevant sentences          │                 │   │    │
│  │  │ từ mỗi chunk (giảm noise, tiết kiệm token)    │                 │   │    │
│  │  └──────────────────────┬───────────────────────┘                 │   │    │
│  │                         │                                          │   │    │
│  └─────────────────────────┼──────────────────────────────────────────┘   │    │
│                            │                                              │    │
│  ┌─────────────────────────▼──────────────────────────────────────────┐   │    │
│  │                   GENERATION PHASE                                   │   │    │
│  │                                                                      │   │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │    │
│  │  │ 8. LLM GENERATION (Qwen 2.5 72B / GPT-4o)                     │   │   │    │
│  │  │                                                               │   │   │    │
│  │  │  System Prompt:                                               │   │   │    │
│  │  │  "Bạn là trợ lý pháp lý doanh nghiệp. Trả lời dựa TRÊN       │   │   │    │
│  │  │   tài liệu được cung cấp. Trích dẫn Điều, Khoản cụ thể.      │   │   │    │
│  │  │   Nếu không có thông tin, nói 'Tôi không tìm thấy...'"        │   │   │    │
│  │  │                                                               │   │   │    │
│  │  │  Output format (JSON):                                        │   │   │    │
│  │  │  {                                                            │   │   │    │
│  │  │    "answer": "...",                                           │   │   │    │
│  │  │    "citations": [{"doc_title": "...", "article": "Điều X",    │   │   │    │
│  │  │                    "khoan": "Khoản Y", "excerpt": "...",      │   │   │    │
│  │  │                    "url": "..."}],                             │   │   │    │
│  │  │    "confidence": "high|medium|low"                            │   │   │    │
│  │  │  }                                                            │   │   │    │
│  │  └──────────────────────┬───────────────────────────────────────┘   │   │    │
│  │                         │                                           │   │    │
│  │                         ▼                                           │   │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │    │
│  │  │ 9. POST-PROCESSING + STREAMING                                │   │   │    │
│  │  │  • Validate citations tồn tại trong retrieved docs            │   │   │    │
│  │  │  • Format answer với highlight citations                      │   │   │    │
│  │  │  • Stream từng token qua SSE                                  │   │   │    │
│  │  │  • Log query + response + latency vào audit trail             │   │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │   │    │
│  └─────────────────────────────────────────────────────────────────────┘   │    │
│                                                                            │    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │    │
│  │                   MEMORY SYSTEM (Multi-turn)                          │   │    │
│  │                                                                       │   │    │
│  │  ┌─────────────────────┐    ┌──────────────────────────────────┐     │   │    │
│  │  │ Short-term Memory   │    │ Conversation Buffer              │     │   │    │
│  │  │ (session-level)     │    │ Last N messages (N=10)           │     │   │    │
│  │  │ • conversation_id   │    │ Window-based, oldest dropped      │     │   │    │
│  │  │ • message_history[] │    │ Stored in Redis/In-memory dict   │     │   │    │
│  │  └─────────────────────┘    └──────────────────────────────────┘     │   │    │
│  └─────────────────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                      │
│                                                                           │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐    │
│  │     QDRANT        │  │   LITELLM        │  │   REDIS (optional)   │    │
│  │   (Vector DB)     │  │   (LLM Gateway)  │  │                      │    │
│  │                   │  │                   │  │  • Semantic Cache    │    │
│  │  Collection:      │  │  Router:          │  │  • Session Store     │    │
│  │  legal_docs       │  │  Qwen → GPT-4o   │  │  • Rate Limiting     │    │
│  │                   │  │  mini → GPT-4o   │  │                      │    │
│  │  Index:           │  │                   │  │                      │    │
│  │  • Dense (BGE-M3) │  │  Fallback chain:  │  │                      │    │
│  │  • Sparse (BM25)  │  │  Qwen → GPT-4o   │  │                      │    │
│  │                   │  │  mini → GPT-4o    │  │                      │    │
│  │  Payload:         │  │                   │  │                      │    │
│  │  • chunk_text     │  │  Cost tracking    │  │                      │    │
│  │  • doc_title      │  │  per query         │  │                      │    │
│  │  • article_number │  │                   │  │                      │    │
│  │  • chapter_name   │  │                   │  │                      │    │
│  │  • doc_id         │  │                   │  │                      │    │
│  │  • url            │  │                   │  │                      │    │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     OFFLINE PIPELINE (CI/CD)                               │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                   GITHUB ACTIONS (Nightly)                          │    │
│  │                                                                     │    │
│  │  1. Pull latest eval dataset                                       │    │
│  │  2. Run RAGAS evaluation (100 test questions)                      │    │
│  │  3. Generate eval report (JSON → Markdown table)                   │    │
│  │  4. Update RAGAS badge in README                                   │    │
│  │  5. Alert if faithfulness < 0.75                                   │    │
│  │                                                                     │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │ Metrics tracked:                                              │   │    │
│  │  │ • Faithfulness (target >0.80)                                 │   │    │
│  │  │ • Answer Relevancy (target >0.80)                             │   │    │
│  │  │ • Context Precision (target >0.75)                            │   │    │
│  │  │ • Context Recall (target >0.80)                               │   │    │
│  │  │ • Latency P50/P95/P99 (target <3s P95)                        │   │    │
│  │  │ • Cost per query (target <$0.01)                              │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │                   DATA INDEXING PIPELINE (On-demand)                │    │
│  │                                                                     │    │
│  │  HF Dataset → Filter → Clean → Chunk → Embed → Index → Qdrant     │    │
│  │                                                                     │    │
│  │  Run once initially, then incrementally for new documents.         │    │
│  │  Orchestrated via scripts/ingest.py with Click CLI.               │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2. Request Flow (Sequence Diagram)

```
User          Streamlit        FastAPI       RAG Orchestrator    Qdrant      LLM
 │                │                │                │               │           │
 │  "Thành lập    │                │                │               │           │
 │   công ty TNHH │                │                │               │           │
 │   cần những    │                │                │               │           │
 │   giấy tờ gì?" │                │                │               │           │
 │───────────────>│                │                │               │           │
 │                │  POST /api/    │                │               │           │
 │                │  chat/stream   │                │               │           │
 │                │  {question,    │                │               │           │
 │                │   session_id}  │                │               │           │
 │                │───────────────>│                │               │           │
 │                │                │                │               │           │
 │                │                │  1. Retrieve   │               │           │
 │                │                │  session hist  │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │               │           │
 │                │                │  2. Rewrite     │               │           │
 │                │                │  query          │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │  LLM call     │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──────────────│           │
 │                │                │                │               │           │
 │                │                │  3. Expand      │               │           │
 │                │                │  to multi-query │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │  LLM call     │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──────────────│           │
 │                │                │                │               │           │
 │                │                │  4. HyDE        │               │           │
 │                │                │  (optional)     │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │  LLM call     │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──────────────│           │
 │                │                │                │               │           │
 │                │                │  5. Hybrid      │               │           │
 │                │                │  Search         │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │  Dense search │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──────────────│           │
 │                │                │                │  BM25 search  │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──────────────│           │
 │                │                │                │  RRF merge    │           │
 │                │                │<───────────────│               │           │
 │                │                │                │               │           │
 │                │                │  6. Re-rank     │               │           │
 │                │                │  (BGE-Reranker) │               │           │
 │                │                │───────────────>│               │           │
 │                │                │<───────────────│               │           │
 │                │                │                │               │           │
 │                │                │  7. Compress    │               │           │
 │                │                │  context         │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │  LLM call     │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──────────────│           │
 │                │                │                │               │           │
 │                │                │  8. Generate    │               │           │
 │                │                │  final answer   │               │           │
 │                │                │───────────────>│               │           │
 │                │                │                │  LLM call     │           │
 │                │                │                │  (streaming)  │           │
 │                │                │                │──────────────>│           │
 │                │                │                │<──SSE tokens──│           │
 │                │                │<──SSE stream───│               │           │
 │                │<──SSE stream───│                │               │           │
 │<──SSE stream───│                │                │               │           │
 │                │                │                │               │           │
 │  (Hiển thị      │                │                │               │           │
 │   từng token    │                │                │               │           │
 │   + citations)  │                │                │               │           │
```

---

## 3. DATA PIPELINE

### 3.1. Data Filtering Logic

```python
# scripts/ingest.py — Bộ lọc dataset

FILTERS = {
    "legal_type": [
        "Luật",                    # Laws (highest authority)
        "Bộ luật",                 # Codes (Civil Code, Criminal Code, etc.)
        "Nghị định",               # Government Decrees
        "Thông tư",                # Ministerial Circulars
        "Nghị quyết",              # Resolutions
        "Quyết định",              # Decisions (important ones)
    ],
    # GIỮ TẤT CẢ lĩnh vực pháp luật, không filter sector
    # Điều này giúp bot bao quát toàn bộ hệ thống pháp luật Việt Nam:
    # Dân sự, Hình sự, Lao động, Đất đai, Thuế, Doanh nghiệp,
    # Hành chính, Sở hữu trí tuệ, Hôn nhân gia đình, etc.
    "date_range": ("01/01/2015", "31/12/2026"),
    "min_content_length": 500,     # characters
}

# Output: ~10,000–15,000 documents (đa lĩnh vực)
```

### 3.2. Chunking Strategy Detail

```
Strategy: MarkdownHeaderTextSplitter + Semantic Chunking Hybrid

Step 1: Detect Markdown structure
  Input:  "# CHƯƠNG I\n## Điều 1. ...\n### Khoản 1. ..."
  Split points: H1 (#) → Chapter, H2 (##) → Article, H3 (###) → Clause

Step 2: Chunk với metadata inheritance
  Mỗi chunk mang theo:
  {
    "chunk_text": "Nội dung của Khoản 1, Điều 19...",
    "chapter": "CHƯƠNG II: THÀNH LẬP DOANH NGHIỆP",
    "article": "Điều 19. Hồ sơ đăng ký doanh nghiệp",
    "document_title": "Luật Doanh Nghiệp 2020",
    "document_number": "59/2020/QH14",
    "legal_type": "Luật",
    "issuing_authority": "Quốc hội",
    "issuance_date": "17/06/2020",
    "url": "https://thuvienphapluat.vn/..."
  }

Step 3: Size control
  - Min chunk size: 200 chars (~120 tokens tiếng Việt)
  - Max chunk size: 1000 chars (~600 tokens)
  - Merge small consecutive chunks (ví dụ: các Khoản ngắn trong cùng 1 Điều)
  - Split oversize chunks tại ranh giới câu (dấu chấm xuống dòng)

Step 4: Overlap strategy
  - Parent chunk (Điều) overlap vào child chunk (Khoản)
  - Ví dụ: Khoản 1 của Điều 19 sẽ có overlap là toàn bộ text Điều 19
    → Khi retrieve, model có đủ context của cả Điều lẫn Khoản cụ thể
```

### 3.3. Embedding Pipeline

```python
# Sử dụng BGE-M3 với FlagEmbedding (BAAI)
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Tạo cả dense + sparse vectors trong 1 lần
output = model.encode(
    chunk_text,
    return_dense=True,    # 1024-dim vector
    return_sparse=True,   # lexical weights (thay thế BM25)
)
```

### 3.4. Indexing vào Qdrant

```
Collection: legal_docs
  Vector config:
    - dense (1024-dim, cosine similarity)
    - sparse (BM25-compatible via Qdrant's sparse vector)
  
  Payload (metadata stored alongside vector):
    - chunk_text, chapter, article
    - document_title, document_number
    - legal_type, issuing_authority
    - issuance_date, url
  
  Index type: HNSW (for dense), Inverted Index (for sparse)
```

---

## 4. PHASE 2: LEGAL INTELLIGENCE PIPELINE ⭐ IMPLEMENTED

> **Status**: ✅ Hoàn thành (2026-07-04)
> **Test scope**: 6 documents — Bộ luật Hình sự cluster
> **Code**: `src/data/graph/`, `src/data/timeline/`, `src/data/amendment/`, `src/pipeline/`

Phase 2 adds three specialized modules that transform the system from a generic RAG into a **legal-domain-aware** intelligent retrieval system.

### 4.1. Cross-Referencing (Knowledge Graph) 🔗

**Vấn đề**: Văn bản pháp luật chứa hàng nghìn tham chiếu chéo ("theo Điều X của Luật Y", "tại khoản Z Điều W"). Khi user hỏi về Điều 123, họ cũng cần biết các điều được dẫn chiếu.

**Giải pháp**: Xây dựng NetworkX DiGraph từ tất cả cross-references.

```
ReferenceExtractor (9 regex patterns)
  ├── INTERNAL:  "Điều X của Luật/Bộ luật này"
  ├── EXTERNAL:  "Điều X của Luật Y" / "Điều X của Bộ luật Y"
  ├── CLAUSE:    "khoản X Điều Y"
  ├── MULTI:     "Điều X và Điều Y"
  └── CHAPTER:   "tại Chương X"

LegalKnowledgeGraph
  ├── Nodes: ArticleNode (doc_{id}_dieu_{num})
  ├── Edges: directed, với ref_type + confidence metadata
  ├── expand_context(article_ids, hops=1) → query-time BFS expansion
  └── most_referenced(top_n=10) → highest-authority articles
```

**Kết quả (6 docs)**: 409 nodes, 403 edges, 204 internal refs, 23 external refs, 6 connected components.

**Ứng dụng**: Khi user search "Điều 67", hệ thống tự động expand context để include Điều 68 (được dẫn chiếu bởi Điều 67), tăng recall.

---

### 4.2. Temporal Versioning (Time-Aware Search) 📅

**Vấn đề**: Luật thay đổi theo thời gian. User hỏi "năm 2018 luật quy định thế nào?" cần trả lời đúng phiên bản tại thời điểm đó.

**Giải pháp**: Trích xuất effective/expiry dates + version chain resolution.

```
DateExtractor
  ├── _find_effectiveness_article()
  │   ├── Strategy 1: rfind("Hiệu lực thi hành") — last occurrence (>70% position)
  │   ├── Strategy 2: Regex "Điều X. Hiệu lực thi hành"
  │   ├── Strategy 3: Amendment pattern "Điều 2. Hiệu lực thi hành"
  │   └── Strategy 4: Any "hiệu lực thi hành" after preamble
  ├── _extract_primary_date() — "có hiệu lực từ ngày DD/MM/YYYY" priority
  ├── _extract_relative_date() — "sau X ngày kể từ ngày công bố"
  ├── _extract_expiry() — replaced laws detection
  └── Fallback: search last 3000 chars

VersionResolver
  ├── at_time(when) → laws valid at a point in time
  ├── version_chain(law_name) → chronological version list
  ├── find_replacement(doc_id) → which law replaced this one?
  └── _classify_status() → "current" | "future" | "expired" | "unknown"
```

**Kết quả (6 docs)**: 5/6 có effective date (83%), version chain BLHS: 1999→2015(2016-07-01)→2017 amendment(2018-01-01)→2025 amendment(2025-07-01).

**Ứng dụng**: Query "Điều 123 Bộ luật Hình sự 2018" → tự động filter kết quả từ phiên bản có hiệu lực tại 2018.

---

### 4.3. Amendments Tracking (Merge & Consolidate) 🔧

**Vấn đề**: "Luật sửa đổi, bổ sung" thay đổi nội dung của base law. Khi user hỏi về 1 điều luật, cần biết điều đó đã bị sửa đổi chưa và nội dung mới nhất là gì.

**Giải pháp**: Parse amendment instructions + map amendments → base laws + consolidate text.

```
AmendmentExtractor (20+ regex patterns, 15 action types)
  ├── MODIFY:  modify_article, modify_clause, modify_point,
  │            modify_multi, modify_multi_article
  ├── INSERT:  insert_article, insert_clause, insert_point
  ├── REPEAL:  repeal_article, repeal_clause, repeal_point
  ├── REPLACE: replace_word, batch_replace, batch_insert
  └── UNKNOWN (fallback)

Extraction pipeline:
  1. is_amendment_law(title) — detect "sửa đổi, bổ sung"
  2. extract_target_laws(title, content) — identify base laws
  3. _split_sections() — handle multi-law amendments
  4. _split_into_instructions() — numbered item parsing
  5. _classify_action() — 15 action types via priority-ordered regex
  6. _parse_sub_instructions() — a), b), c) sub-items
  7. _detect_target_in_instruction() — target law changes within section

AmendmentConsolidator
  ├── apply(base_text, instructions) → ConsolidationResult
  ├── _apply_modify_article/clause/point() — text replacement
  ├── _apply_insert_article/clause/point() — text insertion
  ├── _apply_repeal_article/clause/point() — mark as [Bãi bỏ]
  └── extract_article_text() / list_articles()
```

**Kết quả (3 amendment laws)**:
- 2017 amendment: 180 instructions, 149 articles affected
- 2021 amendment: 7 instructions, 7 articles
- 2025 amendment: 46 instructions, 27 articles
- Consolidation tested: 4/5 articles successfully applied

**Ứng dụng**: Query "Điều 3 BLHS" → response includes "⚠ Điều này đã bị sửa đổi bởi Luật sửa đổi BLHS 2017".

---

### 4.4. Pipeline Integration & Enhanced Retrieval 🔄

**Vai trò**: Orchestrate cả 3 modules + tích hợp vào query pipeline.

```
EnrichmentPipeline.run()
  ├── _collect_documents() → scroll Qdrant, reconstruct full text
  ├── FOR each document:
  │   ├── M1: extract refs → build KnowledgeGraph
  │   ├── M2: extract timeline → VersionResolver
  │   ├── M3: extract amendments (if amendment law)
  │   └── Save DocumentMeta to MetadataStore
  ├── Resolve version chains
  └── Build index

MetadataStore (data/metadata/)
  ├── documents.json   ← DocumentMeta (refs, dates, amendments)
  ├── graph.json       ← Knowledge graph (nodes + edges)
  ├── timelines.json   ← LawTimeline data
  ├── amendments.json  ← AmendmentInstruction list
  └── index.json       ← Summary

EnhancedRetriever.retrieve(query, top_k, at_date, expand_context)
  ├── Hybrid search (dense + sparse RRF)
  ├── _enrich_result() → merge Phase 2 metadata
  ├── Time filter: skip not-yet-effective laws at at_date
  └── _expand_context() → BFS graph traversal for cross-references
```

**API Endpoints (12 total)**:
- `POST /api/pipeline/enrich` — run enrichment
- `GET /api/pipeline/status` — enrichment status
- `POST /api/search/enhanced` — enhanced search with context expansion
- `GET /api/article/history` — article amendment timeline
- `GET /api/graph/stats` — knowledge graph statistics
- `GET /api/graph/neighbors` — graph neighbors for an article
- `GET /api/timeline/effective` — laws effective at a date

---

## 5. RAG PIPELINE DESIGN

### 5.1. Phases of Evolution

```
MVP (Week 1-2)          v1.0 (Week 3-4)          v2.0 (Week 7-8)
─────────────────      ──────────────────       ───────────────────
Naive RAG              Adaptive RAG ⭐          Enterprise RAG
                                                
User Question          User Question            User Question
    │                      │                        │
    ▼                      ▼                        ▼
Embed question         ⭐ Complexity            ⭐ Complexity
    │                   Classifier               Classifier
    ▼                   ┌───┴───┐                ┌───┴───┐
Dense Search          SIMPLE  COMPLEX          SIMPLE  COMPLEX
    │                    │       │                │       │
    ▼                    │  Query Rewrite         │  Query Rewrite
Top-5 retrieve           │  Multi-Query Exp.      │  Multi-Query Exp.
    │                    │  HyDE                  │  HyDE
    ▼                    │       │                │       │
Generate                 ▼       ▼                ▼       ▼
    │                 Dense   Hybrid Search    Dense   Hybrid Search
    ▼                 Search  (Dense+BM25)     Search  (Dense+BM25)
Return answer           │       │                │       │
                        ▼       ▼                ▼       ▼
                     ⭐ Dynamic Top-K          ⭐ Dynamic Top-K
                        │       │                │       │
                        ▼       ▼                ▼       ▼
                     Generate  RRF Merge      Generate  RRF Merge
                        │       │                │       │
                        ▼       ▼                ▼       ▼
                     Return    Re-rank         Return    Re-rank
                               │                │       │
                               ▼                ▼       ▼
                           Compress           Compress  Multi-hop Router
                               │                │    (single vs multi-hop)
                               ▼                ▼       │
                           Generate          Generate ┌─┴──┐
                               │                │    Single Multi-hop
                               ▼                ▼      │    Decompose
                           ⭐ Confidence     ⭐ Confidence│    → sub-Q
                           ⭐ Verification   ⭐ Verification│ → retrieve each
                               │                │      │    → aggregate
                               ▼                ▼      ▼    → generate
                           Return + Citations Return + Citations
                           or "I don't know"  or "I don't know"
```

> ⭐ = Kỹ thuật mới được thêm vào plan (sau phân tích nhận xét từ ChatGPT)

### 4.2. Adaptive Retrieval (Query Complexity Router) ⭐

**Vấn đề**: Pipeline cũ chạy TOÀN BỘ các bước (Rewrite → Multi-Query → HyDE → Hybrid → Rerank → Compress) cho mọi query, kể cả câu đơn giản như "Công ty TNHH là gì?". Điều này gây lãng phí 3-5 LLM calls không cần thiết, tăng latency và cost.

**Giải pháp**: Thêm 1 lớp Router phân loại query TRƯỚC khi vào pipeline.

```
Query: "Thành lập công ty TNHH cần giấy tờ gì?"
    │
    ▼
┌─────────────────────────────────┐
│ QUERY COMPLEXITY CLASSIFIER     │  ← 1 LLM call nhẹ (GPT-4o mini)
│                                 │
│ Prompt:                         │
│ "Classify this Vietnamese       │
│  legal query into:              │
│                                 │
│  SIMPLE: definition lookup,     │
│    single concept, yes/no       │
│    question.                    │
│    VD: 'Công ty TNHH là gì?'   │
│                                 │
│  MEDIUM: comparison,            │
│    conditional question,        │
│    multi-aspect single doc.     │
│    VD: 'Điều kiện thành lập    │
│    công ty TNHH 1 TV?'         │
│                                 │
│  COMPLEX: multi-step reasoning, │
│    hypothetical scenario,       │
│    cross-document question.     │
│    VD: 'Muốn thành lập công ty │
│    TNHH 1 TV cần vốn bao nhiêu │
│    và nộp hồ sơ ở đâu?'        │
│                                 │
│  Query: {question}              │
│  Class:"                        │
└──────┬──────┬──────┐
       │      │      │
   SIMPLE  MEDIUM  COMPLEX
   (~70%)  (~20%)  (~10%)
       │      │      │
       ▼      ▼      ▼
```

#### Routing Logic

| Mức độ | Pipeline | LLM Calls | Latency target | Cost target |
|--------|----------|-----------|----------------|-------------|
| **SIMPLE** | Dense Search → Generate | 2 (classify + generate) | <1.5s | <$0.003 |
| **MEDIUM** | Hybrid Search → Rerank → Generate | 2 (classify + generate) | <2.5s | <$0.005 |
| **COMPLEX** | Rewrite → Multi-Query → HyDE → Hybrid → Rerank → Compress → Generate | 6 (classify + 5 pipeline) | <4s | <$0.015 |

#### Lợi ích định lượng (ước tính)

- **70% queries SIMPLE** → giảm từ 5 LLM calls xuống 2 → tiết kiệm 60% cost
- **Latency trung bình**: từ ~3.5s xuống ~1.8s
- **Token usage trung bình**: giảm ~50%

#### Implementation

```python
# src/rag/query_processing.py

class QueryComplexityRouter:
    """Routes queries to appropriate pipeline based on complexity."""
    
    COMPLEXITY_PROMPT = """
    Classify this Vietnamese legal query into SIMPLE, MEDIUM, or COMPLEX.
    
    SIMPLE: definition lookup, single concept question, yes/no.
    MEDIUM: comparison, conditional, multi-aspect but single document.
    COMPLEX: multi-step reasoning, hypothetical, cross-document.
    
    Query: {question}
    Class:"""
    
    def __init__(self, llm_gateway):
        self.llm = llm_gateway  # Use GPT-4o mini for routing
    
    def classify(self, question: str) -> str:
        """Returns 'simple', 'medium', or 'complex'."""
        response = self.llm.generate(
            self.COMPLEXITY_PROMPT.format(question=question),
            max_tokens=10,
            temperature=0  # Deterministic classification
        )
        result = response.strip().lower()
        if result not in ('simple', 'medium', 'complex'):
            return 'medium'  # Safe default
        return result
    
    def route(self, question: str, complexity: str) -> dict:
        """Execute the appropriate pipeline based on complexity."""
        if complexity == 'simple':
            return self._simple_pipeline(question)
        elif complexity == 'medium':
            return self._medium_pipeline(question)
        else:
            return self._complex_pipeline(question)
```

---

### 4.3. Confidence-based Answering ⭐

**Vấn đề**: LLM luôn generate câu trả lời, kể cả khi retrieved context không liên quan hoặc quá yếu. Trong domain pháp lý, **trả lời sai còn nguy hiểm hơn từ chối trả lời**.

**Giải pháp**: Tính Confidence Score từ nhiều tín hiệu, từ chối trả lời khi score < threshold.

```
Generate xong
    │
    ▼
┌──────────────────────────────────────┐
│        CONFIDENCE SCORING            │
│                                      │
│  Input Signals (đều normalized 0-1): │
│                                      │
│  ① Retriever Score (weight: 0.25)   │
│     max cosine similarity trong      │
│     tất cả retrieved chunks          │
│                                      │
│  ② Reranker Score (weight: 0.20)    │
│     max cross-encoder score          │
│                                      │
│  ③ LLM Self-Eval (weight: 0.30)     │
│     LLM tự đánh giá confidence       │
│     trong JSON output                │
│                                      │
│  ④ Context Overlap (weight: 0.15)   │
│     % từ khóa trong answer xuất      │
│     hiện trong retrieved context     │
│                                      │
│  ⑤ Claim Entailment Ratio (0.10)    │
│     % claims được NLI model          │
│     xác nhận là ENTAILMENT           │
│                                      │
│  ─────────────────────────────────   │
│  Weighted Score: Σ(wᵢ × sᵢ)         │
│                                      │
│  Score range: 0.0 - 1.0             │
└──────────┬───────────────────────────┘
           │
     ┌─────┴─────┐
     │           │
   ≥ 0.7      < 0.7
     │           │
     ▼           ▼
  Return      ┌──────────────────────────┐
  answer      │ "Tôi không tìm thấy       │
              │ thông tin chính xác về    │
              │ vấn đề này trong cơ sở    │
              │ dữ liệu. Vui lòng tham    │
              │ khảo ý kiến luật sư       │
              │ để được tư vấn chính xác."│
              │                           │
              │ Kèm theo:                  │
              │ • Confidence score        │
              │ • Gợi ý câu hỏi thay thế  │
              │ • Link đến văn bản liên    │
              │   quan nhất (nếu có)      │
              └──────────────────────────┘
```

#### Edge Cases cho Legal Domain

| Scenario | Score Behavior | Response |
|----------|---------------|----------|
| Context trả về đúng điều luật | 0.85-0.95 → Return | Trả lời + citations |
| Context đúng nhưng thiếu 1 phần | 0.60-0.70 → Refuse | "Tôi tìm thấy thông tin liên quan nhưng chưa đầy đủ để trả lời chính xác..." |
| Context hoàn toàn sai | 0.20-0.40 → Refuse | "Tôi không tìm thấy..." |
| Câu hỏi ngoài phạm vi pháp lý Việt Nam | 0.10-0.30 → Redirect | "Tôi là trợ lý pháp lý Việt Nam, tôi không thể trả lời câu hỏi này..." |
| Context conflict (2 văn bản mâu thuẫn) | Flag đặc biệt | "Có sự khác biệt giữa [Văn bản A] và [Văn bản B]..." |

> 💡 **Tại sao threshold 0.7?** Dựa trên benchmark RAG cho legal domain: với threshold 0.7, precision đạt >90% (90% câu trả lời được return là đúng). Với threshold 0.8, precision cao hơn nhưng từ chối quá nhiều (chỉ return 40% queries). 0.7 là sweet spot.

---

### 4.4. Hallucination Verification (2-tier) ⭐

**Vấn đề**: LLM có thể cite đúng Điều/Khoản nhưng diễn giải SAI ý nghĩa. Hoặc bịa ra con số không có trong context. Plan cũ chỉ check "citation có tồn tại không" → chưa đủ.

**Giải pháp**: 2-tier verification — vừa check structural (citation tồn tại), vừa check semantic (ý nghĩa có khớp context không).

```
Tier 1: STRUCTURAL VALIDATION (đã có)
─────────────────────────────────────
  ✓ "Điều 19" có trong retrieved context không?
  ✓ "Khoản 1" có trong Điều 19 không?
  ✓ Document title đúng không?
  → Nhanh, rule-based, 100% precision

Tier 2: SEMANTIC VERIFICATION (THÊM MỚI) ⭐
─────────────────────────────────────────

  Answer: "Theo Điều 19 Luật Doanh Nghiệp 2020, 
           doanh nghiệp phải nộp 5 loại giấy tờ..."
           
  Context gốc: "Điều 19. Hồ sơ đăng ký bao gồm: 
                1) Giấy đề nghị, 2) Điều lệ, 
                3) Bản sao CMND, 4) Văn bản ủy quyền"

  ┌──────────────────────────────────────┐
  │ NLI VERIFIER                         │
  │ (mDeBERTa-v3-base-mnli-xnli)         │
  │                                      │
  │ For each atomic claim in answer:     │
  │                                      │
  │ Claim 1: "phải nộp 5 loại giấy tờ"   │
  │   Context có 4 loại → CONTRADICTION ✗│
  │                                      │
  │ Claim 2: "Điều 19 quy định về hồ sơ" │
  │   Context nói về hồ sơ → ENTAILMENT ✓│
  │                                      │
  │ Claim 3: "bao gồm giấy đề nghị"      │
  │   Context có "Giấy đề nghị"→ENTAILMENT│
  │                                      │
  │ ──────────────────────────────────── │
  │ Entailment Ratio: 2/3 = 0.67         │
  │ Contradiction Ratio: 1/3 = 0.33 ⚠    │
  └──────────────────┬───────────────────┘
                     │
                     ▼
              ┌──────┴──────┐
              │             │
         Contradiction   No Issue
           > 30%           │
              │             ▼
              ▼          Return
         ┌────────────┐  answer
         │ FLAG +     │
         │ DISCLAIMER │
         │            │
         │ "⚠ Lưu ý:  │
         │  Một số     │
         │  thông tin  │
         │  có thể      │
         │  chưa chính │
         │  xác. Vui   │
         │  lòng kiểm  │
         │  tra văn    │
         │  bản gốc." │
         └────────────┘
```

#### Why NLI Model (not LLM)?

| Tiêu chí | NLI Model (mDeBERTa) | LLM Verification |
|----------|---------------------|------------------|
| **Cost** | Free (local, ~500MB) | ~$0.002/query |
| **Latency** | <50ms | ~500ms |
| **Accuracy** | ~85% trên legal text | ~92% |
| **Trade-off** | Nhanh, rẻ, đủ tốt để flag | Chính xác hơn, dùng cho production |

> Strategy: Dùng NLI model cho development + nightly eval. Optional: thêm LLM verification pass cho production demo (nếu có budget).

---

### 4.5. Dynamic Top-K ⭐

**Vấn đề**: Top-K cố định (retrieve 20 → rerank 5) không tối ưu. Khi query rất khớp (score >0.9), chỉ cần top-3. Khi query mơ hồ (score <0.5), cần top-30 để tránh miss.

**Giải pháp**: Top-K động dựa trên score distribution.

```python
def dynamic_top_k(scores: list[float], stage: str = "retrieval") -> int:
    """
    Calculate adaptive Top-K based on score distribution.
    
    Args:
        scores: List of similarity scores from retrieval/reranking
        stage: "retrieval" or "reranking"
    
    Returns:
        Optimal K value
    """
    if not scores:
        return 5  # default
    
    max_score = max(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0
    
    # Base K by stage
    if stage == "retrieval":
        base_k = 20
        # High max_score → confident, fewer docs needed
        if max_score > 0.85:
            base_k = 8
        elif max_score > 0.70:
            base_k = 15
        elif max_score < 0.50:
            base_k = 30  # Low confidence → cast wider net
    else:  # reranking
        base_k = 5
        if max_score > 0.90:
            base_k = 3
        elif max_score < 0.60:
            base_k = 10
    
    # High variance → ambiguous query → increase K
    if variance > 0.05:
        base_k += 5
    
    return min(base_k, 50)  # Cap at 50
```

**Impact**: 
- Giảm ~30% tokens đưa vào context window (ít chunk hơn khi không cần)
- Tăng recall cho ambiguous queries (nhiều chunk hơn khi cần)
- Gần như 0 cost để implement (thay constant = function)

---

### 4.6. Prompt Templates

<details>
<summary><b>System Prompt (Generation)</b></summary>

```
Bạn là Trợ lý Pháp lý Việt Nam, chuyên tra cứu văn bản pháp luật thuộc mọi lĩnh vực.

NGUYÊN TẮC:
1. CHỈ trả lời dựa trên tài liệu được cung cấp bên dưới.
2. Nếu tài liệu không chứa thông tin để trả lời, hãy nói:
   "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu."
3. MỖI câu trả lời phải trích dẫn chính xác: tên văn bản, Điều, Khoản.
4. Trả lời bằng tiếng Việt, rõ ràng, mạch lạc.

OUTPUT FORMAT (JSON):
{
  "answer": "Câu trả lời đầy đủ...",
  "citations": [
    {
      "document_title": "Luật Doanh Nghiệp 2020",
      "article": "Điều 19",
      "clause": "Khoản 1",
      "excerpt": "Trích đoạn ngắn từ tài liệu gốc",
      "relevance": "high|medium"
    }
  ],
  "confidence": "high|medium|low",
  "disclaimer": "Thông tin trên chỉ mang tính tham khảo..."
}

TÀI LIỆU THAM KHẢO:
{context}
```
</details>

<details>
<summary><b>Query Rewrite Prompt</b></summary>

```
Viết lại câu hỏi sau thành dạng tối ưu cho tìm kiếm văn bản pháp luật.

Quy tắc:
- Giữ nguyên ý nghĩa
- Dùng ngôn ngữ pháp lý chính xác
- Bổ sung từ khóa đồng nghĩa pháp lý
- Bỏ các từ không cần thiết

Câu hỏi gốc: {question}
Câu hỏi đã viết lại:
```
</details>

<details>
<summary><b>Multi-Query Expansion Prompt</b></summary>

```
Tạo 3 phiên bản khác nhau của câu hỏi sau để tìm kiếm
văn bản pháp luật. Mỗi phiên bản nên tập trung vào
1 khía cạnh khác nhau.

Câu hỏi: {question}

3 phiên bản:
1.
2.
3.
```
</details>

### 4.7. Multi-Hop Reasoning Design

```
Flow cho câu hỏi phức tạp:

Q: "Thành lập công ty TNHH 1 TV cần vốn bao nhiêu 
    và nộp hồ sơ ở đâu?"

Step 1 - Complexity Classifier (LLM):
  → Multi-hop: YES
  → Sub-questions:
    1. "Vốn điều lệ tối thiểu thành lập công ty TNHH 1 TV"
    2. "Nơi nộp hồ sơ đăng ký thành lập doanh nghiệp"

Step 2 - Retrieve independently:
  Sub-Q1 → RAG pipeline → Context 1
  Sub-Q2 → RAG pipeline → Context 2

Step 3 - Aggregate & Generate:
  System: "Dưới đây là thông tin từ 2 nguồn..."
  Context 1 + Context 2 → LLM → Unified answer
  Output: "Vốn điều lệ không có yêu cầu tối thiểu 
           (Luật DN 2020, Điều 75). Hồ sơ nộp tại 
           Phòng ĐKKD thuộc Sở KH&ĐT (Điều 19)..."
```

---

## 6. TECH STACK

### 6.1. Core Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| **LLM (Primary)** | Qwen 2.5 72B (Groq/DeepInfra) | latest | Free tier, good Vietnamese, fast |
| **LLM (Fallback)** | GPT-4o mini / GPT-4o | latest | Reliable fallback, superior structured output |
| **LLM Gateway** | LiteLLM | ≥1.40 | Unified API, auto-fallback, cost tracking |
| **Embedding** | BGE-M3 (FlagEmbedding) | latest | Best open-source for Vietnamese |
| **Vector DB** | Qdrant (Docker) | ≥1.9 | Hybrid search (dense + sparse), production-grade |
| **Orchestration** | LangChain | ≥0.3 | RAG chains, streaming, callbacks |
| **Re-ranker** | BGE-Reranker v2 | latest | Cross-encoder, multilingual |
| **API Framework** | FastAPI | ≥0.110 | Async, SSE streaming, auto-docs |
| **Frontend** | Streamlit | ≥1.35 | Fast Python UI, chat components |
| **Container** | Docker + Docker Compose | latest | Reproducible, Render-compatible |
| **CI/CD** | GitHub Actions | - | Nightly RAGAS eval |
| **Cache (optional)** | Redis | ≥7 | Semantic cache + session store |

### 6.2. LLM Fallback Strategy (LiteLLM Config)

```yaml
# litellm_config.yaml
model_list:
  - model_name: qwen-2.5-72b
    litellm_params:
      model: groq/qwen-2.5-72b-32768
      api_key: ${GROQ_API_KEY}
      rpm: 30  # rate limit: 30 requests/minute (free tier)

  - model_name: gpt-4o-mini
    litellm_params:
      model: azure/gpt-4o-mini  # or openai/gpt-4o-mini
      api_key: ${OPENAI_API_KEY}

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: ${OPENAI_API_KEY}

router_settings:
  routing_strategy: "latency-based-routing"
  allowed_fails: 3
  fallbacks:
    - qwen-2.5-72b → gpt-4o-mini
    - gpt-4o-mini → gpt-4o
```

### 6.3. Why NOT X?

| Not Using | Why |
|-----------|-----|
| **LlamaIndex** | LangChain phổ biến hơn trong JD VN, nhiều tài liệu hơn |
| **ChromaDB** | Qdrant có hybrid search built-in, production-ready hơn |
| **Pinecone** | Mất phí, Qdrant self-host miễn phí đủ dùng |
| **Next.js** | Tốn thời gian, Streamlit đủ đẹp cho portfolio |

---

## 6. PROJECT STRUCTURE

```
vietnamese-legal-bot/
│
├── README.md                         # Project overview, badges, demo GIF
├── LICENSE
├── .gitignore
├── .env.example                      # Template for API keys
├── docker-compose.yml                # Qdrant + Redis + App
├── Dockerfile                        # Multi-stage build
├── pyproject.toml                    # Dependencies (uv/pip)
├── Makefile                          # Common commands
│
├── data/                             # (gitignored)
│   ├── raw/                          # HF dataset download cache
│   ├── processed/                    # Filtered + chunked data
│   └── eval/                         # Test questions + ground truth
│       ├── test_questions.json       # 100 curated Q&A pairs
│       └── test_questions_v2.json    # Expanded test set
│
├── scripts/                          # CLI tools
│   ├── ingest.py                     # HF → filter → chunk → embed → Qdrant
│   ├── eval.py                       # RAGAS evaluation runner
│   └── download_dataset.py           # HF dataset downloader
│
├── src/                              # Main application
│   ├── __init__.py
│   │
│   ├── api/                          # FastAPI layer
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py               # POST /api/chat/stream (SSE)
│   │   │   ├── documents.py          # GET /api/documents (list/search)
│   │   │   └── eval.py               # GET /api/eval (dashboard data)
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── logging.py            # Request/response logging
│   │       └── rate_limit.py         # Rate limiting per session
│   │
│   ├── rag/                          # RAG Pipeline core
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # Main orchestrator (ties everything)
│   │   ├── router.py                 # ⭐ Query Complexity Classifier + Adaptive Router
│   │   ├── query_processing.py       # Query rewrite, expansion, HyDE (COMPLEX only)
│   │   ├── retrieval.py              # Hybrid search + RRF + Dynamic Top-K
│   │   ├── reranker.py               # BGE Re-ranker wrapper
│   │   ├── compression.py            # Contextual compression (COMPLEX only)
│   │   ├── generation.py             # LLM generation + structured output
│   │   ├── confidence.py             # ⭐ Multi-signal confidence scoring + refusal gate
│   │   ├── verifier.py               # ⭐ Hallucination verification (2-tier: citation + NLI)
│   │   ├── multi_hop.py              # Multi-hop reasoning router
│   │   └── memory.py                 # Conversation buffer
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py                 # HF dataset loader + filter
│   │   ├── cleaner.py                # Text cleaning (remove headers/footers)
│   │   ├── chunker.py                # Markdown header splitter
│   │   └── embedder.py               # BGE-M3 embedding wrapper
│   │
│   ├── db/                           # Database layer
│   │   ├── __init__.py
│   │   ├── qdrant_client.py          # Qdrant connection + operations
│   │   └── redis_client.py           # Redis cache + session store
│   │
│   ├── llm/                          # LLM utilities
│   │   ├── __init__.py
│   │   ├── gateway.py                # LiteLLM wrapper + fallback
│   │   └── prompts.py                # All prompt templates
│   │
│   └── ui/                           # Streamlit frontend
│       ├── __init__.py
│       ├── app.py                    # Main Streamlit entry
│       ├── pages/
│       │   ├── chat.py               # Chat interface page
│       │   ├── documents.py          # Document browser page
│       │   └── eval_dashboard.py     # RAGAS dashboard page
│       └── components/
│           ├── chat_message.py       # Chat bubble (user/assistant)
│           ├── citation_card.py      # Citation display component
│           └── doc_card.py           # Document preview card
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_data_pipeline.py         # Test chunking, embedding
│   ├── test_retrieval.py             # Test search quality + Dynamic Top-K
│   ├── test_routing.py               # ⭐ Test complexity classifier accuracy
│   ├── test_confidence.py            # ⭐ Test confidence scoring + refusal
│   ├── test_verifier.py              # ⭐ Test NLI hallucination detection
│   ├── test_generation.py            # Test response format
│   ├── test_api.py                   # Test FastAPI endpoints
│   └── conftest.py                   # Pytest fixtures
│
├── .github/                          # CI/CD
│   └── workflows/
│       ├── eval_nightly.yml          # RAGAS eval every night
│       └── deploy.yml                # Deploy to Render on push to main
│
└── docs/                             # Documentation
    ├── architecture.md               # Architecture decisions (ADR)
    ├── api_spec.md                   # API documentation
    ├── eval_reports/                 # Historical RAGAS reports
    └── screenshots/                  # Demo screenshots for README
```

---

## 7. FEATURE ROADMAP

### 7.0. Phase 2: Legal Intelligence ✅ (COMPLETED — 2026-07-04)

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| P1 | **Cross-Referencing** | 9 regex patterns → 409 nodes, 403 edges knowledge graph. Query-time context expansion via BFS. | ✅ |
| P2 | **Temporal Versioning** | Extract effective/expiry dates. Version chains (BLHS: 1999→2015→2017→2025). Time-travel queries. | ✅ |
| P3 | **Amendments Tracking** | 20+ regex patterns, 15 action types, 233 instructions extracted. Amendment → base law mapping. | ✅ |
| P4 | **Metadata Store** | JSON-based persistence: documents.json, graph.json, timelines.json, amendments.json. | ✅ |
| P5 | **Enrichment Pipeline** | Orchestrate M1+M2+M3 → MetadataStore. 6 docs processed in 6.4s, 0 errors. | ✅ |
| P6 | **Enhanced Retriever** | Query-time context expansion + time filtering + amendment status. Lazy-load embedder. | ✅ |
| P7 | **API Endpoints** | 7 new Phase 2 endpoints: enrich, enhanced search, article history, graph stats, neighbors, timeline. | ✅ |
| P8 | **Narrowed Scope** | Filtered to `legal_type = {Luật}` only (595 docs). Single Qdrant collection `luat_law`. | ✅ |

### 7.1. MVP (Week 1–2) — "Chạy được end-to-end" ✅

| # | Feature | Description | Priority |
|---|---------|-------------|----------|
| F1 | **Data Pipeline** | Download HF dataset → filter 10K docs → chunk → embed → index Qdrant | P0 |
| F2 | **Naive RAG Q&A** | Single-turn Q&A: embed question → dense search top-5 → LLM generate | P0 |
| F3 | **Streaming Chat UI** | Chat interface với streaming tokens, input box, send button | P0 |
| F4 | **Basic Citations** | Hiển thị tên văn bản + Điều/Khoản dưới mỗi câu trả lời | P0 |
| F5 | **Dynamic Top-K** ⭐ | Adaptive K dựa trên score distribution (thay constant = function) | P0 |
| F6 | **Docker Compose** | `docker compose up` chạy toàn bộ stack (Qdrant + API + UI) | P0 |

**MVP Success Criteria:**
- [ ] Hỏi "Thành lập công ty TNHH cần giấy tờ gì?" → trả lời đúng + cite Luật Doanh Nghiệp
- [ ] Streaming hoạt động (từng token hiện ra)
- [ ] `docker compose up` → mọi thứ chạy
- [ ] Latency <5s
- [ ] Dynamic Top-K tự động điều chỉnh số lượng docs retrieve

### 7.2. v1.0 (Week 3–4) — "Adaptive RAG"

| # | Feature | Description | Priority |
|---|---------|-------------|----------|
| F7 | **Query Complexity Classifier** ⭐ | LLM nhẹ phân loại query → SIMPLE/MEDIUM/COMPLEX → route pipeline phù hợp | P1 |
| F8 | **Query Rewriting** | LLM rewrites user query → legal language (COMPLEX queries only) | P1 |
| F9 | **Multi-Query Expansion** | Generate 3 query variants → search → merge (COMPLEX queries only) | P1 |
| F10 | **HyDE** | Generate hypothetical legal answer → embed → search (COMPLEX queries only) | P2 |
| F11 | **Hybrid Search** | BM25 (sparse) + Dense (BGE-M3) → Reciprocal Rank Fusion (MEDIUM+ queries) | P1 |
| F12 | **Re-ranking** | BGE-Reranker v2 cross-encode → adaptive Top-K (MEDIUM+ queries) | P1 |
| F13 | **Context Compression** | LLM extracts relevant sentences from chunks (COMPLEX queries only) | P2 |
| F14 | **Citation Highlight** | Bôi đậm điều khoản trong câu trả lời, click để xem full text | P1 |

**v1.0 Success Criteria:**
- [ ] Adaptive pipeline: "Công ty TNHH là gì?" → chỉ 2 LLM calls (classify + generate), latency <1.5s
- [ ] Hybrid search cải thiện recall >15% so với dense-only (đo trên 100 test queries)
- [ ] Re-ranking cải thiện precision@5 >10%
- [ ] RAGAS Faithfulness >0.80
- [ ] Cost trung bình giảm >40% so với full pipeline cho mọi query

### 7.3. v1.5 (Week 5–6) — "Evaluation, Confidence & Verification"

| # | Feature | Description | Priority |
|---|---------|-------------|----------|
| F15 | **RAGAS Eval Pipeline** | 100 test Q&A pairs → automated RAGAS scoring (faithfulness, relevancy, precision, recall) | P1 |
| F16 | **Confidence-based Answering** ⭐ | Multi-signal confidence score → từ chối trả lời khi <0.7 | P1 |
| F17 | **Hallucination Verification** ⭐ | 2-tier: structural citation check + NLI semantic entailment | P1 |
| F18 | **Eval Dashboard** | Streamlit page hiển thị RAGAS scores + confidence distribution + verification stats | P1 |
| F19 | **GitHub Actions Nightly Eval** | Chạy eval mỗi đêm → update badge README → alert nếu <0.75 | P1 |
| F20 | **Multi-turn Conversation** | Conversation buffer (10 messages), context-aware responses | P1 |
| F21 | **Document Browser** | Filter by legal_type + sector, search by keyword, view full text | P2 |
| F22 | **Error Handling** | Graceful fallback khi LLM timeout, hiển thị thông báo thân thiện | P1 |

**v1.5 Success Criteria:**
- [ ] RAGAS eval tự động chạy, badge README cập nhật
- [ ] Confidence <0.7 → bot từ chối trả lời, không hallucinate
- [ ] Hallucination Verifier bắt được >80% contradiction claims
- [ ] Multi-turn: hỏi follow-up "vậy còn thủ tục giải thể thì sao?" → bot hiểu context
- [ ] Document browser tìm kiếm được theo loại + lĩnh vực

### 7.4. v2.0 (Week 7–8) — "Enterprise Features"

| # | Feature | Description | Priority |
|---|---------|-------------|----------|
| F23 | **Multi-hop Reasoning** | Phân loại câu hỏi đơn/phức → decompose → aggregate | P2 |
| F24 | **Audit Trail** | Log toàn bộ query → response → confidence → verification → latency → cost | P2 |
| F25 | **Rate Limiting** | Giới hạn 20 queries/phút/user (bảo vệ LLM API cost) | P2 |
| F26 | **Deploy Render** | Docker deploy lên Render, URL HTTPS public, health check | P1 |
| F27 | **Document Comparison** (optional) | So sánh 2 văn bản cùng lĩnh vực (VD: Luật DN 2014 vs 2020) | P3 |

**v2.0 Success Criteria:**
- [ ] Multi-hop: "Cần vốn bao nhiêu và nộp hồ sơ ở đâu?" → trả lời đủ 2 ý
- [ ] URL public trên Render, gửi cho bạn bè test được
- [ ] Audit trail ghi nhận confidence + verification result cho mọi query
- [ ] Demo video 5 phút hoàn chỉnh

---

## 8. EVALUATION STRATEGY

### 8.1. Test Dataset Construction

```json
// data/eval/test_questions.json — Cấu trúc
[
  {
    "id": "q001",
    "question": "Thành lập công ty TNHH một thành viên cần những giấy tờ gì?",
    "ground_truth": "Theo Điều 19 Luật Doanh Nghiệp 2020, hồ sơ đăng ký thành lập công ty TNHH 1 TV bao gồm: giấy đề nghị đăng ký doanh nghiệp, điều lệ công ty, bản sao giấy tờ pháp lý của chủ sở hữu...",
    "relevant_docs": [12345, 67890],       // Qdrant point IDs
    "type": "single-hop",
    "difficulty": "easy",
    "legal_sector": "doanh nghiệp"
  },
  // ... 100 questions covering:
  // - 5 legal sectors × 10 questions each = 50
  // - 30 additional multi-hop / edge cases
  // - 20 "out of scope" questions (để test "không tìm thấy")
]
```

### 8.2. RAGAS Metrics

| Metric | Target | Công thức / Ý nghĩa |
|--------|--------|---------------------|
| **Faithfulness** | > 0.80 | % claims trong answer có thể suy ra từ context |
| **Answer Relevancy** | > 0.80 | % câu answer liên quan đến question |
| **Context Precision** | > 0.75 | % retrieved chunks thực sự hữu ích |
| **Context Recall** | > 0.80 | % ground-truth context được retrieve |

### 8.3. Custom Metrics

#### RAG Quality Metrics

| Metric | Target | Ý nghĩa |
|--------|--------|---------|
| **Citation Accuracy** | > 0.85 | % citations trỏ đúng Điều/Khoản |
| **Hallucination Rate** | < 10% | % câu trả lời chứa thông tin không có trong context |

#### ⭐ Adaptive & Verification Metrics (NEW)

| Metric | Target | Ý nghĩa |
|--------|--------|---------|
| **Refusal Rate (Confidence <0.7)** | 15-25% | % queries bị từ chối do low confidence. Quá cao (>40%) → pipeline quá保守; quá thấp (<5%) → threshold cần tăng |
| **Safe Refusal Rate** | > 90% | % refused answers ĐÁNG LẼ nên bị hallucinate nếu trả lời (đo trên adversarial test set) |
| **Wrong-but-Answered Rate** | < 5% | % câu trả lời confidence ≥0.7 nhưng thực tế SAI |
| **Verification Precision** | > 85% | % contradictions được NLI verifier bắt chính xác |
| **Verification Recall** | > 75% | % actual contradictions bị NLI verifier phát hiện |
| **Adaptive Routing Accuracy** | > 80% | % queries được classifier phân loại đúng (simple/medium/complex) |

#### Performance Metrics

| Metric | Target | Ý nghĩa |
|--------|--------|---------|
| **Latency P50 (SIMPLE)** | < 1.5s | 50% simple queries hoàn thành trong 1.5s |
| **Latency P95 (COMPLEX)** | < 4s | 95% complex queries hoàn thành trong 4s |
| **Latency P95 (Overall)** | < 3s | 95% tất cả queries hoàn thành trong 3s |
| **Cost per Query (Avg)** | < $0.005 | Trung bình cost LLM API / query (adaptive giúp giảm ~50%) |
| **Cost per Query (P95)** | < $0.015 | 95% queries có cost ≤ $0.015 |
| **LLM Calls per Query (Avg)** | < 3 | Adaptive routing giảm từ 5-6 xuống ~2.5 trung bình |
| **System Refusal Rate** | < 3% | % queries bị từ chối do LỖI HỆ THỐNG (≠ confidence refusal) |

### 8.4. GitHub Actions Workflow

```yaml
# .github/workflows/eval_nightly.yml
name: Nightly RAGAS Evaluation

on:
  schedule:
    - cron: '0 2 * * *'  # 2:00 AM UTC daily
  workflow_dispatch:       # Manual trigger

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[eval]"
      
      - name: Run RAGAS Evaluation
        run: python scripts/eval.py --output data/eval_reports/
        env:
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      
      - name: Update Eval Badge
        uses: schneegans/dynamic-badges-action@v1
        with:
          auth: ${{ secrets.GIST_SECRET }}
          gistID: ${{ secrets.GIST_ID }}
          filename: ragas_faithfulness.json
          label: RAGAS Faithfulness
          message: ${{ env.FAITHFULNESS_SCORE }}
          color: ${{ env.FAITHFULNESS_COLOR }}
      
      - name: Alert if degraded
        if: env.FAITHFULNESS_SCORE < 0.75
        run: |
          echo "⚠️ RAGAS Faithfulness dropped below 0.75!"
          # Add Slack/Email notification here
```

---

## 9. UI/UX DESIGN

### 9.1. Page Structure (Streamlit)

```
┌─────────────────────────────────────────────────────────┐
│  🏛️ Enterprise Legal Knowledge Bot        [Sidebar]     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─ Sidebar ───────────────────────────────────────┐   │
│  │                                                   │   │
│  │  📁 Document Browser                              │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │ Filter by Type:  [Luật ▾]                │    │   │
│  │  │ Filter by Sector: [Doanh nghiệp ▾]       │    │   │
│  │  │ Search: [________________] [🔍]          │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  │                                                   │   │
│  │  📋 Results:                                      │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │ 📄 Luật Doanh Nghiệp 2020                │    │   │
│  │  │    59/2020/QH14 · 17/06/2020             │    │   │
│  │  │    [Xem chi tiết]                         │    │   │
│  │  │                                           │    │   │
│  │  │ 📄 Nghị định 01/2021/NĐ-CP               │    │   │
│  │  │    Về đăng ký doanh nghiệp               │    │   │
│  │  │    [Xem chi tiết]                         │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  │                                                   │   │
│  │  ─────────────────────────────────────────────    │   │
│  │  📊 Eval Dashboard   [Go →]                       │   │
│  │  ℹ️ About             [Go →]                       │   │
│  └───────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─ Main Chat Area ─────────────────────────────────┐   │
│  │                                                   │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │ 🤖 Assistant:                             │    │   │
│  │  │                                           │    │   │
│  │  │ Để thành lập công ty TNHH một thành viên, │    │   │
│  │  │ bạn cần chuẩn bị các giấy tờ sau:         │    │   │
│  │  │                                           │    │   │
│  │  │ 1. Giấy đề nghị đăng ký doanh nghiệp      │    │   │
│  │  │ 2. Điều lệ công ty                        │    │   │
│  │  │ 3. Bản sao giấy tờ pháp lý cá nhân        │    │   │
│  │  │    của chủ sở hữu                         │    │   │
│  │  │                                           │    │   │
│  │  │ 📎 Trích dẫn:                              │    │   │
│  │  │ ┌────────────────────────────────────┐    │    │   │
│  │  │ │ 📄 Luật Doanh Nghiệp 2020           │    │    │   │
│  │  │ │    Điều 19, Khoản 1                 │    │    │   │
│  │  │ │    "Hồ sơ đăng ký doanh nghiệp..."  │    │    │   │
│  │  │ │    [🔗 Xem văn bản gốc]              │    │    │   │
│  │  │ └────────────────────────────────────┘    │    │   │
│  │  │                                           │    │   │
│  │  │ Độ tin cậy: 🟢 Cao                        │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  │                                                   │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │ 👤 User: Thành lập công ty TNHH cần       │    │   │
│  │  │    những giấy tờ gì?                      │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  │                                                   │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │ [💬 Nhập câu hỏi...________________] [➤] │    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  │                                                   │   │
│  └───────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 9.2. Key UI Interactions

1. **Streaming Response**: Từng token hiện ra real-time (không đợi response đầy đủ)
2. **Citation Click**: Click vào citation → scroll đến văn bản đầy đủ trong sidebar
3. **Source Link**: Mở link `thuvienphapluat.vn` trong tab mới
4. **Confidence Indicator**: 🟢 Cao / 🟡 Trung bình / 🔴 Thấp
5. **Multi-turn Context**: Câu hỏi follow-up tự động kèm history

---

## 10. DEPLOYMENT PLAN

### 10.1. Render Deployment Architecture

```
┌─────────────────────────────────────────────┐
│                 RENDER.COM                    │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │  Web Service: vietnamese-legal-bot   │    │
│  │                                      │    │
│  │  Docker Container:                   │    │
│  │  ┌────────────────────────────────┐  │    │
│  │  │  Port 8000: FastAPI (internal) │  │    │
│  │  │  Port 8501: Streamlit (public) │  │    │
│  │  │  Port 6333: Qdrant (internal)  │  │    │
│  │  └────────────────────────────────┘  │    │
│  │                                      │    │
│  │  Env Variables (Render Dashboard):   │    │
│  │  • GROQ_API_KEY                      │    │
│  │  • OPENAI_API_KEY (fallback)         │    │
│  │  • QDRANT_URL=localhost:6333         │    │
│  │  • REDIS_URL=localhost:6379          │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  Plan: Starter ($7/mo) — 512MB RAM, 1 vCPU   │
│  URL: https://vietnamese-legal-bot.onrender.com │
└─────────────────────────────────────────────┘
```

### 10.2. Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: api
    ports:
      - "8000:8000"
    environment:
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - qdrant
      - redis
    volumes:
      - ./data:/app/data

  ui:
    build:
      context: .
      dockerfile: Dockerfile
      target: ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

volumes:
  qdrant_data:
  redis_data:
```

### 10.3. Dockerfile (Multi-stage)

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# --- API Stage ---
FROM base AS api
COPY src/ ./src/
COPY scripts/ ./scripts/
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- UI Stage ---
FROM base AS ui
COPY src/ ./src/
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 10.4. CI/CD Deploy

```yaml
# .github/workflows/deploy.yml
name: Deploy to Render

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Render
        uses: renderinc/deploy-action@v1
        with:
          serviceId: ${{ secrets.RENDER_SERVICE_ID }}
          apiKey: ${{ secrets.RENDER_API_KEY }}
```

---

## 11. TESTING STRATEGY

### 11.1. Test Pyramid

```
        ╱───────╲
       ╱   E2E   ╲         5 tests: Full flow (Q → A)
      ╱───────────╲
     ╱ Integration ╲       15 tests: API endpoints, Qdrant connection
    ╱───────────────╲
   ╱   Unit Tests    ╲     30+ tests: Individual components
  ╱───────────────────╲
```

### 11.2. Test Categories

| Category | Scope | Tool | Count |
|----------|-------|------|-------|
| **Unit: Data Pipeline** | Chunking, cleaning, embedding | pytest | 10+ |
| **Unit: RAG Components** | Query processing, retrieval, generation | pytest + mocks | 15+ |
| **Integration: API** | FastAPI endpoints, SSE streaming | pytest + httpx | 10+ |
| **Integration: Qdrant** | Search quality, indexing | pytest | 5+ |
| **E2E** | Full query → answer flow | manual + script | 5 scenarios |
| **RAGAS Eval** | Faithfulness, relevancy, precision/recall | RAGAS + GitHub Actions | 100 queries |

### 11.3. Key Test Scenarios

```python
# tests/test_retrieval.py — Example test
def test_hybrid_search_outperforms_dense():
    """Hybrid search (dense + BM25) should have higher recall than dense-only."""
    query = "điều kiện thành lập doanh nghiệp tư nhân"
    
    dense_results = dense_search(query, top_k=20)
    hybrid_results = hybrid_search(query, top_k=20)
    
    dense_recall = calculate_recall(dense_results, ground_truth_ids)
    hybrid_recall = calculate_recall(hybrid_results, ground_truth_ids)
    
    assert hybrid_recall > dense_recall, \
        f"Hybrid recall ({hybrid_recall:.2f}) should exceed dense ({dense_recall:.2f})"
```

---

## 12. TIMELINE & MILESTONES

### 12.1. Actual Progress (vs Planned)

```
✅ PHASE 1: MVP (hoàn thành)
  ├── Data pipeline: download → filter → chunk → embed → Qdrant
  ├── Naive RAG: dense search → LLM generate
  ├── FastAPI + SSE streaming
  ├── Streamlit UI (chat, documents, eval dashboard)
  ├── Adaptive RAG (query classifier + routing)
  └── Docker Compose setup

✅ PHASE 2: Legal Intelligence (hoàn thành — 2026-07-04)
  ├── M1: Cross-Referencing — 409 nodes, 403 edges knowledge graph
  ├── M2: Temporal Versioning — effective dates, version chains
  ├── M3: Amendments Tracking — 15 action types, 233 instructions
  ├── Integration: Enrichment pipeline + Enhanced retriever
  ├── Metadata Store: JSON persistence (5 files)
  └── API: 12 endpoints (7 new Phase 2 endpoints)
  Tested on 6-doc Criminal Code cluster: BLHS 1999→2015→2017→2025

📋 PHASE 3: Scale & Production (planned)
  ├── Scale enrichment to all 595 docs
  ├── Article-level re-indexing with consolidated text
  ├── RAGAS evaluation (Phase 2 vs baseline)
  ├── Parallel embedding pipeline optimization
  └── Production deployment

💡 PHASE 4: Advanced Features (future)
  ├── LLM-based amendment consolidation
  ├── Cross-law impact analysis
  ├── Temporal RAG (auto-detect query date)
  └── Vietnamese legal ontology integration
```

### 12.2. Milestone Checklist

| Milestone | Status | Deliverable |
|-----------|--------|-------------|
| 🟢 MVP Live | ✅ | Dense search + LLM generate + Docker + UI |
| 🔵 Adaptive RAG | ✅ | Query classifier (simple/medium/complex) + hybrid search |
| 🟣 Phase 2: Cross-Referencing | ✅ | 409 nodes, 403 edges, BFS context expansion |
| 🟣 Phase 2: Temporal Versioning | ✅ | 6/6 effective dates, version chains, time-travel |
| 🟣 Phase 2: Amendments Tracking | ✅ | 233 instructions, 15 actions, amendment→base mapping |
| 🟣 Phase 2: Integration | ✅ | Enrichment pipeline, enhanced retriever, 12 API endpoints |
| 🟠 Scale to 595 docs | 📋 | Full ingestion + enrichment pipeline |
| 🟠 Production Deploy | 📋 | Render/Docker, public URL, demo video |

---

## 13. RISK & MITIGATION

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Qwen 2.5 hallucinate citations** | High | Medium | Fallback to GPT-4o; validate citations post-generation; add guardrails |
| **BGE-M3 poor on legal Vietnamese** | High | Low | Test early (Day 3-4); fallback to Voyage-multilingual-2 |
| **Groq rate limit (30 RPM)** | Medium | Medium | LiteLLM auto-fallback; implement client-side rate limiting; cache frequent queries |
| **Qdrant OOM trên Render (512MB)** | High | Medium | Limit to 15K docs, giảm vector dim nếu cần; upgrade Render plan nếu phải |
| **RAGAS score không đạt 0.80** | Medium | Medium | Iterate chunking strategy; thử custom legal parser (Option A); cải thiện prompt |
| **⭐ Query Classifier sai complexity** | Medium | Medium | Dùng GPT-4o mini (deterministic, temp=0); fallback về MEDIUM pipeline nếu uncertainty; log & monitor classification accuracy |
| **⭐ Confidence threshold quá cao/thấp** | Medium | Medium | Bắt đầu với 0.7, monitor Wrong-but-Answered Rate và Safe Refusal Rate trong nightly eval → điều chỉnh threshold |
| **⭐ NLI Verifier miss contradictions** | Low | Medium | Bắt đầu với mDeBERTa; nếu recall <70% → upgrade lên LLM-based verification cho production demo |
| **Adaptive pipeline quá phức tạp để debug** | Medium | Low | Mỗi route có test riêng; log complexity classification + route taken cho mọi query |
| **Không kịp timeline** | Medium | Medium | Scope cut: bỏ Document Comparison, giảm Multi-hop scope; giữ Adaptive + Confidence + Verification là tối thiểu cho v1.5 |
| **Streamlit performance kém** | Low | Low | Streamlit 1.35+ có caching tốt; nếu chậm → tách API + UI riêng |

---

## 14. README TEMPLATE

```markdown
# 🏛️ Vietnamese Legal Knowledge Bot

[![RAGAS Faithfulness](https://img.shields.io/endpoint?url=...)](https://github.com/...)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deploy: Render](https://img.shields.io/badge/deploy-Render-46E3B7.svg)](https://render.com)

> **Adaptive RAG chatbot tra cứu văn bản pháp luật Việt Nam toàn diện — từ Dân sự, Hình sự đến Lao động, Đất đai**
> 
> 🔗 **Live Demo**: https://vietnamese-legal-bot.onrender.com
> 
> 🎥 **Demo Video**: [YouTube](https://youtu.be/...)

## 📖 Overview

Vietnamese Legal Knowledge Bot là chatbot tra cứu văn bản pháp luật 
Việt Nam toàn diện, bao quát mọi lĩnh vực: Dân sự, Hình sự, Lao động, 
Đất đai, Thuế, Doanh nghiệp, Hành chính... Hệ thống sử dụng 
**Adaptive RAG** với query routing, hybrid search, confidence scoring, và 
hallucination verification để trả lời câu hỏi kèm trích dẫn điều khoản cụ thể — 
và **từ chối trả lời khi không đủ bằng chứng**.

### Key Features

- 🔀 **Adaptive Retrieval**: Query Complexity Classifier tự động chọn pipeline (SIMPLE/MEDIUM/COMPLEX) → giảm 50% cost và latency
- 🔍 **Hybrid Search**: BM25 + Dense Vector (BGE-M3) → RRF fusion cho MEDIUM+ queries
- 🛡️ **Confidence-based Answering**: Multi-signal confidence score → từ chối trả lời khi <0.7 (quan trọng cho legal domain)
- ✅ **Hallucination Verification (2-tier)**: Citation validation + NLI semantic entailment check
- 📎 **Citation-Backed Answers**: Mỗi câu trả lời kèm Điều, Khoản, link văn bản gốc
- 🧠 **Multi-hop Reasoning**: Xử lý câu hỏi phức tạp cần thông tin từ nhiều văn bản
- 📊 **Automated Evaluation**: RAGAS + custom metrics (confidence, verification) chạy nightly qua GitHub Actions
- 💬 **Multi-turn Chat**: Context-aware conversation với memory
- ⚡ **Streaming**: Real-time token streaming qua SSE
- 🐳 **Docker**: One-command deploy với Docker Compose

### Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM (Primary) | Qwen 2.5 72B (Groq) |
| LLM (Router) | GPT-4o mini (classifier) |
| LLM (Fallback) | GPT-4o |
| LLM Gateway | LiteLLM (auto-fallback) |
| Embedding | BGE-M3 (BAAI) |
| Vector DB | Qdrant (hybrid search: dense + sparse) |
| Re-ranker | BGE-Reranker v2 |
| NLI Verifier | mDeBERTa-v3 (XNLI) |
| Orchestration | LangChain |
| API | FastAPI + SSE streaming |
| UI | Streamlit |
| Eval | RAGAS + GitHub Actions |
| Deploy | Docker + Render |

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/vietnamese-legal-bot.git
cd vietnamese-legal-bot

# 2. Setup environment
cp .env.example .env
# Edit .env: thêm GROQ_API_KEY, OPENAI_API_KEY (optional)

# 3. Index data (lần đầu, ~15 phút)
docker compose run --rm api python scripts/ingest.py

# 4. Start
docker compose up

# 5. Open http://localhost:8501
```

## 📊 Evaluation

RAGAS metrics (updated nightly via GitHub Actions):

| Metric | Score | Target |
|--------|-------|--------|
| Faithfulness | 0.84 ✅ | >0.80 |
| Answer Relevancy | 0.82 ✅ | >0.80 |
| Context Precision | 0.78 ✅ | >0.75 |
| Context Recall | 0.83 ✅ | >0.80 |

## 🏗️ Architecture

```
User → Streamlit → FastAPI → Adaptive RAG Pipeline
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
               SIMPLE       MEDIUM       COMPLEX
               (70%)        (20%)        (10%)
                    │           │           │
                    ▼           ▼           ▼
              Dense Search  Hybrid    Full Pipeline
                    │      Search    (Rewrite→Multi-Q
                    │        │        →HyDE→Hybrid
                    │        ▼        →Rerank→Compress)
                    │     Rerank          │
                    │        │           │
                    └────────┼───────────┘
                             │
                             ▼
                        Generation
                             │
                             ▼
                    ┌─────────────────┐
                    │ CONFIDENCE GATE  │
                    │   (≥0.7 → OK)   │
                    └────┬───────┬────┘
                         │       │
                       OK    REFUSE
                         │       │
                         ▼       ▼
                    ┌────────┐ "Tôi không
                    │VERIFY  │  tìm thấy..."
                    │(NLI)   │
                    └───┬────┘
                        │
                        ▼
                   Return Answer
                   + Citations
                   + Confidence

Infrastructure:
  Qdrant (Vector + BM25)   LiteLLM Gateway    mDeBERTa (NLI)
  Redis (Cache/Session)    GitHub Actions     Render (Deploy)
```

See [docs/architecture.md](docs/architecture.md) for detailed ADR.

## 📁 Project Structure

```
vietnamese-legal-bot/
├── src/
│   ├── api/          # FastAPI backend
│   ├── rag/          # RAG pipeline
│   ├── data/         # Data processing
│   ├── db/           # Qdrant + Redis
│   ├── llm/          # LLM gateway + prompts
│   └── ui/           # Streamlit frontend
├── scripts/          # CLI tools (ingest, eval)
├── tests/            # Test suite
├── data/eval/        # Evaluation dataset
└── docs/             # Documentation
```

## 🔮 Future Work

- [ ] Authentication (JWT + RBAC) — [design doc](docs/auth_design.md)
- [ ] Support PDF/Word upload (user-provided documents)
- [ ] Fine-tuned Vietnamese legal embedding model
- [ ] Multi-tenant support (isolated knowledge bases per company)

## 📝 License

MIT © [Your Name] — 2025
```

---

## ✅ KẾ HOẠCH HÀNH ĐỘNG (NEXT STEPS)

### Đã hoàn thành

1. ✅ **Phase 1 MVP**: Naive RAG + Adaptive RAG + Streamlit UI + FastAPI
2. ✅ **Phase 2 Legal Intelligence**: Cross-Referencing + Temporal Versioning + Amendments Tracking
3. ✅ **Pipeline Integration**: Enrichment orchestrator + Enhanced retriever + 12 API endpoints
4. ✅ **Documentation**: CODEBASE.md (1,144 lines) + Plan (this file)

### Tiếp theo

1. **Scale enrichment**: Chạy pipeline trên toàn bộ 595 docs → Qdrant
2. **Article-level re-indexing**: Merge amendment-consolidated text vào chunks
3. **RAGAS evaluation**: Đo lường impact của Phase 2 context expansion
4. **Production deploy**: Docker Compose / Render với full dataset
5. **Advanced features**: Temporal RAG, cross-law impact analysis, LLM-based consolidation

### Quick Start (hiện tại)

```bash
# Test enrichment pipeline trên 6 docs
python -c "
from src.db.qdrant_client import LegalVectorStore
from src.pipeline.enricher import EnrichmentPipeline
store = LegalVectorStore(collection_name='luat_law', timeout=120)
pipeline = EnrichmentPipeline(store)
result = pipeline.run()
print(f'Enriched {result.docs_processed} docs in {result.elapsed_seconds:.1f}s')
"

# Start API với Phase 2 endpoints
uvicorn src.api.main:app --port 8000 --reload

# Xem metadata
cat data/metadata/index.json | python3 -m json.tool
```

---

> **Plan này là bản sống (living document).** Phase 1 + 2 đã hoàn thành đúng hướng: từ generic RAG → legal-domain-aware intelligent retrieval. Phase 3 sẽ scale lên toàn bộ 595 docs và production deployment.
