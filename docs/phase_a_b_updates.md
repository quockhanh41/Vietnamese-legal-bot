# Tổng kết Phase A & Phase B

## ✅ Phase A hoàn tất — Tóm tắt các thay đổi

### 1. 🐛 Fix bug `split("\\n")` → `split("\n")`
- **File:** `src/rag/orchestrator.py` (dòng 311, 436)
- Bug trong `query_stream()` — dùng `\\n` (literal `\n`) thay vì `\n` (newline). Chỉ ảnh hưởng COMPLEX path khi parse multi-query expansion.

### 2. 📊 Dynamic Top-K (F5)
- **File:** `src/rag/orchestrator.py` — method `_dynamic_top_k()` (dòng 213-236)
```python
@staticmethod
def _dynamic_top_k(results, min_k=3, max_k=20) -> int:
```
- Normalize scores về 0-1 (hoạt động với cả cosine similarity lẫn RRF score)
- Nhiều kết quả >0.7 → confident → giữ ít hơn (≈min_k)
- Ít kết quả >0.7 → ambiguous → giữ nhiều hơn (≈max_k)
- Áp dụng cho cả 3 pipelines: SIMPLE (search 20→dynamic), MEDIUM (search 20→dynamic), COMPLEX (search 40 raw→dedup→dynamic)

### 3. 🎯 Re-ranking (F12) — BGE-Reranker v2
- **File:** `src/rag/orchestrator.py` — `_get_reranker()` + `_rerank_results()` (dòng 197-211, 356-374)
- Lazy load: Chỉ tải FlagReranker khi lần đầu MEDIUM/COMPLEX được gọi
- Graceful fallback: Nếu model chưa được cache → log warning, skip reranking, giữ nguyên kết quả
- Dùng cho MEDIUM và COMPLEX pipelines (cả non-streaming lẫn streaming)

### 4. 📦 Context Compression (F13) — COMPLEX only
- **File:** `src/rag/orchestrator.py` — `_compress_context()` (dòng 378-398)
- LLM-extract: giữ lại chỉ các câu liên quan đến query
- Prompt tiếng Việt: "Trích xuất CHỈ những câu có liên quan trực tiếp"
- Fallback: nếu LLM trả về <50 chars → giữ nguyên context
- Chỉ chạy ở COMPLEX pipeline (tiết kiệm token cho câu hỏi khó)

### 5. 🖍️ Citation Highlight (F14)
- **File:** `src/ui/pages/chat.py` — function `_highlight_legal_refs()` (dòng 17-39)
- Bôi đậm các tham chiếu: Điều 123, Khoản 1, Luật Doanh Nghiệp 2020, Bộ luật Hình sự, Nghị định 01/2021, Thông tư 01/2024...
- Áp dụng ở cả 3 chỗ: streaming token, final answer, và history replay

### Bảng so sánh: Trước vs Sau

| Tính năng | Trước | Sau |
| :--- | :--- | :--- |
| **Top-K** | Hardcoded (5/10) | Dynamic (3-20 tùy score) |
| **Re-ranking** | ❌ Không có | BGE-Reranker v2 (lazy load) |
| **Context Compression** | ❌ Không có | LLM extract (COMPLEX only) |
| **Citation Highlight** | ❌ Không có | Bold inline trong answer |
| **Bug `\\n`** | `split("\\n")` | `split("\n")` ✅ |

---

## ✅ Phase B hoàn tất — Tóm tắt tất cả các thay đổi

### 📁 File mới

| File | Dung lượng | Chức năng |
| :--- | :--- | :--- |
| `src/rag/verifier.py` | 9.1 KB | HallucinationVerifier — Tier 1 (citation check) + Tier 2 (NLI) + claim splitter |
| `src/rag/confidence.py` | 8.9 KB | ConfidenceScorer — 5-signal weighted scoring + refusal gate |

### 📝 File đã sửa

| File | Thay đổi |
| :--- | :--- |
| `src/llm/prompts.py` | Thêm `CONFIDENCE_EVAL_PROMPT` (LLM self-eval) |
| `src/rag/orchestrator.py` | +2 imports, +3 fields in `RAGResponse`, +`_post_process()`, +`_enrich_response()`, tích hợp vào cả 3 pipelines |
| `src/api/main.py` | `ChatResponse` thêm `confidence_score` + `refused` |
| `src/ui/pages/chat.py` | Thêm `_render_confidence_score()` + verification warnings + refusal notice |

### 🔄 Data Flow — Post-Generation Pipeline

```text
Generate answer (Markdown)
    ↓
_post_process():
    ├── F17 Verifier:
    │   ├── Tier 1: mỗi citation → "Điều X" có trong context? → citation_validity
    │   ├── Tier 2: split answer → claims → NLI per claim → entailment_ratio
    │   └── Output: VerificationResult(passed, warnings, contradiction_ratio)
    │
    ├── F16 ConfidenceScorer:
    │   ├── ① Retriever (0.25): max search score
    │   ├── ② Reranker (0.20): max cross-encoder score (or 0)
    │   ├── ③ LLM Self-Eval (0.30): LLM rates own answer 0-100
    │   ├── ④ Context Overlap (0.15): keyword overlap answer↔context
    │   ├── ⑤ Entailment Ratio (0.10): from verifier
    │   └── weighted_score = Σ(w_i × s_i)
    │
    └── Refusal Gate:
        ├── score ≥ 0.70 → return answer + confidence + verification
        └── score < 0.70 → REFUSAL_ANSWER + refused=True
```

### 🏗️ Architecture Decisions

| Decision | Rationale |
| :--- | :--- |
| Lazy load cả 2 module | Giống reranker — không tăng startup time |
| NLI skip trong streaming | CPU inference ~300ms/claim × 3-5 claims = 1-2s — skip để giữ streaming mượt |
| mDeBERTa-v3-base-mnli-xnli | Multilingual, hỗ trợ tiếng Việt, ~800MB |
| Threshold 0.7 | Theo plan §4.3: sweet spot giữa precision và refusal rate |
| Fallback nếu NLI không có | Chỉ dùng 4 signals còn lại, không crash |
