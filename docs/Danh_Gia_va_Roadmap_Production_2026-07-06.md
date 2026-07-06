# 📋 Đánh giá Plan & Codebase — Roadmap tiến tới Production

> **Ngày đánh giá**: 2026-07-06
> **Phạm vi**: `docs/Vietnamese_Legal_Knowledge_Bot_Plan.md` + toàn bộ source code (`src/`, `scripts/`, cấu trúc dự án)
> **Mục tiêu**: Xác định khoảng cách so với production và các kỹ thuật SOTA mà legal RAG hiện đại đang áp dụng

---

## TL;DR

Plan viết rất tốt — kiến trúc Adaptive RAG + Confidence Gate + Hallucination Verification là đúng hướng cho legal domain. Nhưng codebase hiện tại có **khoảng cách lớn giữa "có code" và "code chạy đúng"**:

- Có ít nhất **6 bug nghiêm trọng** làm các tính năng cốt lõi (routing, NLI verification, citation check, confidence) **thực chất không hoạt động dù nhìn như đang hoạt động**.
- Đường **streaming** (đường UI dùng chính) **bỏ qua hoàn toàn** confidence + verification, hiển thị độ tin cậy giả.
- Các trụ cột production (tests, CI, eval dataset, observability, memory, README) đều **rỗng**.
- Điểm sáng lớn nhất: hạ tầng Phase 2 (graph tham chiếu chéo, timeline, amendments) đã build khá công phu — nhưng **chưa được nối vào pipeline chat**, nên chưa tạo ra giá trị.

---

## 1. Bug nghiêm trọng — sửa TRƯỚC khi làm gì khác

### 1a. NLI verifier không bao giờ nhìn thấy context ⚠️ (bug lớn nhất)

**Vị trí**: `src/rag/verifier.py:228-235`

Code dùng `pipeline("zero-shot-classification")` với `sequences=claim` và `candidate_labels=["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]`. Cách gọi này so claim với các câu "The context says: ENTAILMENT." — **retrieved context không hề được truyền vào model**. Kết quả entailment/contradiction hiện tại là vô nghĩa (số rác).

**Hệ quả**: Cả verification (Tier 2) lẫn tín hiệu entailment trong confidence score (10% trọng số) đều dựa trên dữ liệu vô nghĩa.

**Fix**: Dùng model NLI dạng `text-classification` với cặp `(premise=context_chunk, hypothesis=claim)` — đúng cách mDeBERTa-v3-mnli-xnli được huấn luyện. Hoặc tối thiểu `zero_shot(sequences=context, hypothesis_template=claim)`.

### 1b. Complexity classifier gần như luôn trả về "medium"

**Vị trí**: `src/rag/orchestrator.py:256-263` (và cùng lỗi ở `src/rag/confidence.py:188`)

Gọi `max_tokens=10` trên `groq/qwen/qwen3-32b` — là **model thinking**: 10 token đầu tiên bị nuốt vào `<think>`, `_clean_thinking` strip phần think dở dang → chuỗi rỗng → fallback `"medium"`.

**Hệ quả**:
- **Adaptive routing — tính năng trung tâm của plan — hiện không route gì cả** (mọi query đi pipeline MEDIUM).
- LLM self-eval trong confidence (30% trọng số) luôn trả về 0.5 mặc định.

**Fix**: Với Groq Qwen3, set `reasoning_format: "hidden"` (hoặc `reasoning_effort: "none"`), hoặc dùng model non-thinking cho các call phân loại/self-eval.

### 1c. Tier-1 citation check là vòng lặp tự chứng minh (vacuous)

**Vị trí**: `src/rag/orchestrator.py:150-162` (build citations) + `src/rag/verifier.py:105-134` (check)

Citations được build từ chính retrieval results, rồi verifier kiểm tra "Điều X có trong context không" — trong khi context cũng build từ chính các results đó. **Check này luôn pass, không bắt được gì.**

**Fix**: Check đúng phải là — parse các trích dẫn trong **answer text do LLM sinh ra**, rồi đối chiếu với context. (Xem thêm mục 3.2: grounded citations `[n]` giải quyết triệt để hơn.)

### 1d. Streaming bỏ qua toàn bộ confidence + verification ⚠️ (nguy hiểm nhất về mặt sản phẩm)

**Vị trí**: `src/rag/orchestrator.py:472-565` (`query_stream`) + `src/api/main.py:210-218`

Đường stream (UI dùng) không chạy `_post_process`, và event `done` hardcode `confidence = "high" if len(citations) >= 2` — tức là **UI đang hiển thị độ tin cậy giả**. Trong legal domain đây là điều nguy hiểm nhất.

**Fix**: Sau khi stream xong tokens, chạy verify + confidence trên full answer rồi emit event `verification` cuối cùng (UI hiển thị badge sau khi stream xong). Kiến trúc này cũng giải quyết mâu thuẫn latency (xem mục 5).

### 1e. Reranker signal luôn = 0, retriever signal luôn ≈ 1.0

**Vị trí**: `src/rag/orchestrator.py:290` + `src/rag/confidence.py:165-167`

- `rerank_scores=None` được truyền cứng → 20% trọng số confidence luôn mất → điểm tối đa thực tế là 0.80 trong khi ngưỡng refuse là 0.70 → **hệ thống bias nặng về từ chối** (hoặc buộc phải hạ threshold mù quáng).
- Hack rescale RRF `max_s * 60` khiến retriever signal ≈ 1.0 cho mọi query hybrid (RRF top score luôn ≈ 1/60).

**Tổng kết**: Trong 5 tín hiệu confidence thì **3 cái đang chết hoặc bão hòa** (reranker=0, self-eval=0.5 cứng do bug 1b, retriever≈1.0). Cần fix từng tín hiệu rồi mới calibrate threshold bằng dữ liệu thật.

### 1f. Fallback chain không như plan + blocking event loop

**Vị trí**: `src/llm/gateway.py:23-27` + `src/api/main.py:147`

- Cả 3 model trong chain đều Groq (không có GPT-4o fallback như plan), và `groq/qwen/qwen3.6-27b` **không phải model tồn tại**. Nếu Groq down/rate-limit → toàn hệ thống chết.
- Endpoint `async def` của FastAPI gọi pipeline **sync** → **block event loop**, một request chậm làm nghẽn tất cả request khác.

**Fix**: Thêm fallback cross-provider thật; đổi endpoint thành `def` (FastAPI tự đưa vào threadpool) hoặc dùng `run_in_executor`.

---

## 2. Gap so với production (plan hứa nhưng chưa có)

| # | Gap | Chi tiết |
|---|-----|----------|
| 1 | **Tests + CI rỗng** | `tests/` chỉ có `__init__.py`, `.github/workflows/` rỗng. Plan có test pyramid + nightly RAGAS eval → hiện 0 test, 0 CI. Với RAG, quan trọng nhất là **eval dataset vàng + regression gate** — chưa có `data/eval/test_questions.json`. |
| 2 | **Không có memory/multi-turn** | `session_id` được nhận (`api/main.py:33`) rồi vứt đi. Redis chạy trong docker-compose nhưng không dòng code nào dùng. |
| 3 | **Không có observability** | Không structured logging, không tracing, không cost tracking. Đang debug bằng `print()`. Pipeline 5-6 LLM call mà không trace từng stage thì không tune được. |
| 4 | **Thiếu README, LICENSE, .env.example** | Plan có template sẵn — với portfolio project đây là mặt tiền. |
| 5 | **Security** | CORS `*`, không rate limit, `/api/pipeline/enrich` (`api/main.py:283`) cho phép bất kỳ ai trigger job nặng không cần auth. |
| 6 | **Phase 2 không được dùng trong chat** | `EnhancedRetriever` (graph expansion, time filter, amendment status — code tốt nhất repo) chỉ được gọi từ `/api/search/enhanced`. Pipeline chat trong `orchestrator.py` vẫn gọi thẳng `vector_store`. |

**Kỹ thuật nhỏ nên dọn**:
- `sys.path.insert` hack (`orchestrator.py:16`)
- Code stream/non-stream duplicate ~150 dòng trong orchestrator
- `_build_context` cắt chunk còn 500 ký tự — điều luật thường dài hơn, mất nội dung
- `_expand_context` trong retriever loop O(N) qua toàn bộ documents cho mỗi result
- `context = self._build_context(results)` bị tính 2 lần trong mỗi pipeline

---

## 3. Kỹ thuật SOTA legal RAG nên áp dụng (xếp theo ROI)

### Tier 1 — tác động lớn, vừa sức

**3.1. Contextual Retrieval (Anthropic, 2024)**
Trước khi embed mỗi chunk, dùng LLM rẻ sinh 1-2 câu ngữ cảnh ("Đoạn này thuộc Điều 19 Luật Doanh nghiệp 2020, quy định về hồ sơ đăng ký...") và prepend vào chunk. Giảm retrieval failure ~35-49% theo benchmark Anthropic. Với văn bản luật có cấu trúc Chương/Điều/Khoản, làm được **phiên bản rule-based gần như miễn phí** từ metadata sẵn có.

**3.2. Grounded citations kiểu "quote-then-answer"**
Đánh số context block `[1]..[k]`, buộc LLM cite `[n]` inline + trích nguyên văn span. Citation lúc đó **verify được bằng string-match** (span phải xuất hiện verbatim trong chunk `[n]`) — thay thế luôn Tier-1 check đang vô dụng (bug 1c), rẻ hơn và chính xác hơn NLI. Đây là cách các sản phẩm legal AI thương mại (Harvey, Thomson Reuters CoCounsel) neo trích dẫn.

**3.3. Citation-lookup route**
Query dạng "Điều 51 Luật Hôn nhân gia đình" không nên đi qua vector search — parse regex → lookup trực tiếp bằng metadata filter trong Qdrant. Router nên có nhánh `LOOKUP` bên cạnh SIMPLE/MEDIUM/COMPLEX. Đây là loại query phổ biến nhất trong legal và cũng là loại vector search hay fail nhất.

**3.4. Parent-document / small-to-big retrieval**
Embed và search ở mức Khoản (chính xác), nhưng đưa vào context cả Điều cha (đủ ngữ cảnh). Plan §3.2 mô tả đúng ý bằng "overlap" nhưng cách đúng là lưu `parent_id` và fetch parent lúc retrieve, không duplicate text.

**3.5. Eval trên benchmark Việt Nam có sẵn**
Dùng **ALQAC** (Automated Legal Question Answering Competition — task retrieval + QA trên luật Việt Nam, có ground truth) và **Zalo Legal Text Retrieval** làm bộ đo retrieval (recall@k, MRR, nDCG) trước khi đo RAGAS. **Đo retrieval tách khỏi generation** — hiện không biết lỗi nằm ở tầng nào.

### Tier 2 — nâng cấp có chọn lọc

**3.6. Nối Phase 2 vào pipeline chat**
- Temporal filter ("luật còn hiệu lực tại ngày hỏi") nên là **filter mặc định trong Qdrant payload**, không phải post-filter.
- Amendment status ("Điều này đã bị sửa đổi bởi...") nên xuất hiện trong context đưa cho LLM và trong citation card.
- Đây là điểm khác biệt thật sự của legal RAG so với RAG thường — đã build 80%, chỉ thiếu 20% wiring.

**3.7. Corrective RAG / Self-RAG nhẹ**
Thay vì chỉ refuse khi confidence thấp, thêm 1 vòng: nếu reranker score thấp → rewrite query và retrieve lại 1 lần trước khi bỏ cuộc. Rẻ và tăng answer rate đáng kể.

**3.8. Fine-tune embedding cho legal tiếng Việt**
BGE-M3 zero-shot ổn, nhưng SOTA cho tiếng Việt là fine-tune bi-encoder trên cặp (câu hỏi, điều luật) — data ALQAC/Zalo có sẵn, hoặc sinh synthetic queries từ chính corpus. **Đo trước bằng benchmark 3.5 để biết có đáng không.**

**3.9. Observability bằng Langfuse hoặc Arize Phoenix** (self-host, free)
Trace từng stage, latency, token, cost per query. Là thứ biến project từ "demo" thành "production-grade", và là **điều kiện tiên quyết để tune threshold 0.7 bằng dữ liệu thật** thay vì niềm tin.

**3.10. Semantic cache trên Redis**
Cho query lặp — Redis đã có trong compose, chỉ cần dùng.

### KHÔNG khuyên làm bây giờ

- **GraphRAG full** (community detection kiểu Microsoft) — overkill khi graph tham chiếu hiện có đã đủ.
- **ColBERT/late-interaction** — BGE-M3 + reranker đủ tốt.
- **Agentic multi-hop phức tạp** — làm sau khi single-hop đo được tử tế.

---

## 4. Thứ tự hành động đề xuất

| Tuần | Việc | Chi tiết |
|------|------|----------|
| **1** | **Sửa cho đúng cái đã có** | Fix 6 bug ở mục 1 (NLI, classifier max_tokens, streaming verification, reranker signal, fallback chain, blocking event loop). **Không thêm feature nào trước khi xong.** |
| **2** | **Đo được** | Build eval set (50-100 câu từ ALQAC + tự soạn), viết `scripts/eval.py` đo retrieval metrics riêng + RAGAS riêng, đưa vào GitHub Actions. Từ đây mọi thay đổi đều có số. |
| **3-4** | **Nâng chất lượng theo số đo** | Contextual retrieval, grounded citations `[n]`, citation-lookup route, parent-document retrieval, nối temporal/amendment vào chat pipeline. |
| **5+** | **Production polish** | Langfuse, Redis cache + session memory, rate limit + auth cho endpoint admin, README + demo. |

---

## 5. Lưu ý cập nhật Plan

1. **Mâu thuẫn latency vs confidence gate**: Mục tiêu latency (<1.5s SIMPLE) mâu thuẫn với confidence gate cần thêm 1 LLM call + NLI trên CPU sau generation. Với kiến trúc **streaming + verify-sau-stream** (mục 1d) mâu thuẫn tự giải quyết — plan nên cập nhật flow đó.

2. **Render 512MB không khả thi**: Mục tiêu chạy BGE-M3 + reranker + NLI + Qdrant trên Render 512MB là không thể (riêng BGE-M3 đã ~2GB RAM). Thực tế: nhắm HF Spaces/máy có GPU nhỏ, hoặc chuyển embedding sang API (ví dụ Cohere embed-multilingual) khi deploy.

3. **Cấu trúc code lệch plan**: Plan định nghĩa `src/rag/{router,query_processing,retrieval,reranker,compression,generation,multi_hop,memory}.py` nhưng thực tế là orchestrator monolith 564 dòng + `src/pipeline/`. Nên tách dần theo plan khi refactor (đặc biệt khi dedupe code stream/non-stream).
