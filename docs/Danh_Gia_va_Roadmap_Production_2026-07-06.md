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

### ✅ Trạng thái fix (cập nhật 2026-07-06, cùng ngày)

Cả 6 bug đã được fix (sub-agent Sonnet 5 thực hiện, Fable 5 verify bằng live test với Groq + model NLI thật):

| Bug | Trạng thái | Ghi chú verify |
|-----|-----------|----------------|
| 1a NLI | ✅ Fixed | Test thật với mDeBERTa: answer bịa → 67% contradiction + flag "Điều 99"; answer đúng → pass, 3/5 ENTAILMENT |
| 1b Classifier | ✅ Fixed (có chỉnh) | Sub-agent dùng `reasoning_format="hidden"` → **vẫn rỗng** (model vẫn đốt token budget để think). Fable đổi sang `reasoning_effort="none"` (Groq-native, tắt hẳn thinking) → live test trả về `SIMPLE`/`MEDIUM` đúng; self-eval trả về số |
| 1c Citation check | ✅ Fixed | Giờ parse "Điều n" từ answer text, đối chiếu context — bắt được citation bịa |
| 1d Streaming verification | ✅ Fixed (có chỉnh) | Event SSE `verification` mới với score thật; `done` dùng giá trị thật. Fable phát hiện thêm: raw stream chứa `<think>` bị đưa vào NLI → đã strip trước khi verify |
| 1e Confidence signals | ✅ Fixed | Rerank scores được truyền thật; weights renormalize khi thiếu tín hiệu (max đạt 1.0); RRF dùng margin-over-median thay vì `*60`. E2E: answer tốt + phân bố RRF thực tế → 0.808 (không refuse), có rerank → 0.830 |
| 1f Fallback + event loop | ✅ Fixed | Chain: qwen3-32b → llama-3.3-70b → gpt-4o-mini (skip khi thiếu key); `/api/chat` sync def; stream dùng `iterate_in_threadpool` |

**Việc còn lại liên quan**: (i) calibrate threshold 0.7 + công thức margin RRF bằng eval data thật (tuần 2 roadmap); (ii) UI chưa render event `verification`/`refusal_answer` trong chế độ stream (hiện bỏ qua an toàn); (iii) NLI hay trả NEUTRAL với claim ngắn tiếng Việt → entailment_ratio thiên thấp, cân nhắc khi calibrate.

### ✅ Trạng thái Tuần 2 (cập nhật 2026-07-07): eval foundation + UI verification đã xong

**Đã có** (Sonnet 5 implement, Fable 5 verify):
- 2 eval set: `data/eval/eval_set_criminal_v1.json` (41 câu, ground truth đối chiếu corpus thật) + `eval_set_general_v1.json` (29 câu, vai trò kép refuse-now/answer-later)
- `scripts/validate_eval_set.py` (schema + grounding check), `scripts/eval.py` (Tier 1 retrieval không tốn LLM: recall@5/10 + MRR, dense vs hybrid, strict vs lenient; Tier 2 `--answers`: refusal accuracy + citation accuracy + phân bố confidence cho calibration), `src/eval/metrics.py` + 27 unit tests đầu tiên của repo, `.github/workflows/eval.yml` (nightly, chỉ Tier 1)
- UI đã render event SSE `verification`: confidence thật, warnings, refusal answer (câu gốc trong expander)

**📉 Kết quả đo baseline đầu tiên (14 câu, hybrid, strict):**

| Loại câu hỏi | recall@5 | MRR | Kết luận |
|---|---|---|---|
| single-hop (câu hỏi tự nhiên) | **0.83** | 0.72 | Retrieval hoạt động tốt |
| citation-lookup ("Điều X quy định gì?") | **0.125** | 0.03 | Gần như thất bại hoàn toàn |

Phát hiện kèm theo: (i) hybrid == dense trên bộ này — BGE-M3 sparse không cứu được citation-lookup vì chunk luật sửa đổi chứa rất nhiều số "Điều N" gây nhiễu; (ii) lenient (match theo text) = 0.0 < strict — tức **chunk_text thường thiếu header "Điều N."** (bắt đầu giữa điều luật); (iii) Tier 2 smoke: crim-001 bị false-refuse ở confidence 0.673 (ngưỡng 0.70).

**→ Ưu tiên Tuần 3-4 giờ có số chứng minh:** (1) ~~citation-lookup route (mục 3.3)~~ ✅ ĐÃ XONG 2026-07-07; (2) ~~contextual retrieval (mục 3.1)~~ ✅ ĐÃ XONG 2026-07-07; (3) calibrate ngưỡng 0.70 — 🟡 TOOLING XONG, chờ dữ liệu (xem dưới). ~~Cải tiến nhỏ cho eval: lenient thành strict∨text~~ ✅ đã sửa.

**✅ Out-of-scope detection — HOÀN THÀNH 2026-07-07** — SCOPE gate gộp vào chính LLM call của complexity classifier (1 call làm 2 việc, không thêm latency/cost): prompt mới `COMPLEXITY_CLASSIFIER_PROMPT` nhận `{corpus_summary}` sinh động từ alias index (`CitationLookup.corpus_summary()` — corpus lớn lên không phải sửa prompt), trả `SCOPE: IN|OUT` + `CLASS:`. Nuance được encode bằng 4 few-shot: câu hỏi **trách nhiệm hình sự** cho hành vi thuộc lĩnh vực khác (đất đai, thuế...) vẫn IN scope; câu **thủ tục hành chính/chuyên ngành** là OUT. Câu OUT → refuse ngay TRƯỚC retrieval (`pipeline_used="out-of-scope"`, conf 0.0, ~200-650ms, template liệt kê đúng các văn bản hiện có); parse hỏng → fail-open về pipeline thường (không bao giờ refuse nhầm vì lỗi parse). Cả `query()` lẫn `query_stream()` (UI nhận `__VERIFICATION__` với `reason: "out_of_scope"`). Citation-lookup route vẫn chạy trước gate (Điều N có thật ⇒ in-scope by construction). 15 tests mới (`tests/test_scope_gate.py`, 82 total).

Kết quả (Tier 2, reports `20260707T015439Z` general + `20260707T020605Z` criminal, 0 lỗi):

| Metric | Trước (chỉ threshold 0.725) | Sau (scope gate) |
|---|---|---|
| General refusal accuracy | 23/29 | **28/29** (0.9655) |
| 4 outlier "điểm cao vượt ngưỡng" (gen-203 0.8455...) | lọt hết | **chặn hết** tại gate, conf 0.0 |
| Criminal false-refusal | 5/34 | **4/34** (0 câu answer bị gate chặn nhầm) |
| Criminal refusal accuracy | 6/7 | **7/7** |
| Citation accuracy | 0.711 | **0.8529** |

Còn lại: (i) **gen-802 leak** (tranh chấp đất đai dân sự → classifier cho IN vì dính chữ "đất đai" + nuance hình sự, answered 0.7625) — đúng vùng biên của nuance, chấp nhận được ở corpus 6 docs, theo dõi khi mở rộng; (ii) 4 false-refusal còn lại (crim-109/201/202/302, điểm 0.67-0.70) là việc của confidence gate chứ không phải scope gate — trùng việc "soi per-signal crim-302" đã ghi ở dưới. Fable verify độc lập: đọc diff, chạy lại 82 tests, đối chiếu từng row trong 2 report, live-test format output classifier trên Groq (model lặp nhãn `SCOPE:` đúng định dạng cả 2 chiều IN/OUT).

**✅ Calibrate ngưỡng confidence — HOÀN THÀNH 2026-07-07 (lần 2, đủ dữ liệu): REFUSAL_THRESHOLD 0.70 → 0.725**
- Sau khi thêm GROQ_API_KEYS rotation (nhiều key, xoay khi rate-limit — `gateway.py`), Tier 2 full-run hoàn thành **70/70 câu, 0 lỗi** (34 answer + 36 refuse).
- Sweep Youden-optimal: **t=0.725** (refusal recall 0.806→**0.889**, false-refusal giữ nguyên 0.147, Youden 0.742). 0.725 nằm giữa khe hở giữa refuse cao nhất dưới biên (0.7202) và answer thấp nhất trên biên (0.7303) — margin ~0.005 mỗi phía → **coi là tạm thời, recalibrate khi eval set lớn lên**. Override bằng env `CONFIDENCE_REFUSAL_THRESHOLD`.
- Baseline chất lượng tại 0.70: refusal accuracy criminal 6/7, general 23/29; citation accuracy 0.711.
- **Vấn đề còn lại lộ ra từ calibration** (không giải quyết được bằng threshold): 4 câu refuse có điểm cao vượt mọi ngưỡng hợp lý — gen-203 (0.8455!), gen-202, gen-301, gen-702 — đều là câu đất đai/dân sự/BHXH retrieval trả về chunk hình sự "có vẻ liên quan". Đây là bài toán **out-of-scope detection** (nhận diện câu hỏi ngoài phạm vi corpus TRƯỚC khi trả lời), không phải bài toán ngưỡng — ✅ ĐÃ GIẢI QUYẾT 2026-07-07, xem block phía trên. Outlier phía ngược: crim-302 (answer, 0.435) — câu amendment 2025, đáng soi per-signal. Ứng viên việc tiếp theo.

**🟡 Ghi chú lần chạy 1 (bị Groq daily cap chặn — đã giải quyết bằng key rotation)**
- `scripts/calibrate_threshold.py` (sweep t∈[0.30,0.90], Youden index, histogram, outliers) + 18 tests (70 total pass).
- Run Tier 2 bị **Groq daily token cap** chặn: criminal xong 31/41 (10 lỗi = 24.4%, vượt gate 20%), general chỉ chạy được ~8/29 rồi bị hủy. Chỉ có **1 câu refuse** hoàn thành → theo đúng decision rule, KHÔNG đổi ngưỡng trên dữ liệu này.
- **Tín hiệu sơ bộ từ dữ liệu partial** (chưa hành động): 0.70 có vẻ hơi cao — tại t∈[0.575, 0.675] false-refusal giảm từ 10% xuống 3.3% mà vẫn giữ refusal recall 1.0 (n=1). Outliers cần soi per-signal ở run sau: crim-108 (0.5597), crim-116 (0.6797), crim-202 (0.6979) — các câu answer bị điểm thấp.
- **Việc cần làm để chốt**: sau khi Groq TPD reset (~24h) hoặc thêm OPENAI_API_KEY thật vào .env (hiện là placeholder rỗng → fallback cross-provider bị skip), chạy lại: `caffeinate -i .venv/bin/python scripts/eval.py --set criminal --answers --sleep 3` và `--set general --answers`, rồi `scripts/calibrate_threshold.py`. Lưu ý chi phí thực đo được: ~3.5k token/câu qua adaptive pipeline (generate + self-eval + verify).

**✅ Contextual Retrieval rule-based (2026-07-07)** — `src/data/contextualizer.py`: prepend header metadata (`Bộ luật hình sự 2015 (Luật 100/2015/QH13) | Chương | Điều N. tiêu đề | Khoản n`) vào text TRƯỚC KHI EMBED (chunk_text hiển thị giữ nguyên; không tốn LLM call nào); re-embed 1670 chunks vào collection `luat_law_ctx` (script `scripts/reindex_contextual.py`, collection cũ giữ để so sánh); `scripts/ingest.py` áp dụng cho ingest tương lai. `DEFAULT_COLLECTION_NAME` đã chuyển sang `luat_law_ctx`. Kèm bug fix quan trọng: `embedder.py` truyền `device=` bị FlagEmbedding ≥1.3 âm thầm bỏ qua (đúng tên là `devices=`) → model tự chạy MPS và OOM 17GB với chunk dài.

Kết quả (34 câu, strict, mode routed = production path):

| Metric | luat_law (cũ) | luat_law_ctx (mới) |
|---|---|---|
| recall@5 | 0.794 | **0.922** |
| recall@10 | 0.922 | **0.976** |
| MRR | 0.703 | **0.863** |
| single-hop r@5 | 0.882 | **1.0** |
| amendment r@5 | 0.467 | **0.80** |
| comparison r@5 | 0.25 | **0.75** |

Riêng hybrid thuần (không route): citation-lookup r@5 0.125 → 0.625 — chứng tỏ contextual embedding tự nó cũng cứu được phần lớn citation query (route lookup vẫn cần để đạt 1.0). Lưu ý nhỏ: multi-hop r@5 giảm 0.583→0.417 trên mẫu n=2 (r@10 và MRR không đổi) — nhiễu mẫu nhỏ, theo dõi khi mở rộng eval set.

**✅ Citation-lookup route (2026-07-07)** — `src/rag/citation_lookup.py`: parse "Điều N + tên luật + năm" → tra trực tiếp metadata trong Qdrant, tích hợp vào AdaptiveRAGPipeline (route "lookup" đi trước classifier, tiết kiệm 1 LLM call; Điều không tồn tại → KHÔNG fallback vector search, để refusal gate xử lý). Kết quả đo trên toàn bộ 34 câu (mode "routed" trong Tier 1):

| Metric (strict) | hybrid | routed |
|---|---|---|
| recall@5 tổng | 0.588 | **0.794** |
| recall@10 tổng | 0.701 | **0.922** |
| MRR tổng | 0.476 | **0.701** |
| citation-lookup recall@5 | 0.125 | **1.0** (MRR 1.0) |
| single-hop recall@5 | 0.882 | 0.882 (không tụt) |
| amendment recall@10 | 0.833 | **0.933** |

Đã verify: câu hỏi tự nhiên không bị parser hijack (route field ghi nhận); "Điều 999" → 0 hits; "Điều 173" không kèm tên luật → trả về đúng cả 3 bộ luật (ambiguity xử lý đúng). Lưu ý nhỏ khi đọc MRR: Qdrant tie-break không ổn định với các hit đồng điểm RRF → MRR dao động nhẹ giữa các lần chạy. TODO scale: thêm payload index `article_number` lúc ingest khi corpus lớn.

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

---

## 6. Checklist scale corpus: 6 docs → toàn bộ dataset (~10-15K docs sau filter)

> Viết 2026-07-07. Corpus hiện tại: 6 docs / 1670 chunks. Full corpus ước tính **500K–2M chunks** — nhiều quyết định đúng ở quy mô nhỏ sẽ sai ở quy mô lớn. Checklist chia theo giai đoạn, có gate đo lường giữa các bước.

### 6.0. Nguyên tắc: scale theo bậc thang, không nhảy cóc

`6 docs → ~100 docs (thêm 1-2 lĩnh vực: doanh nghiệp, lao động) → ~1.000 docs → toàn bộ`. Sau MỖI bậc: chạy `scripts/eval.py` Tier 1 + validate lại eval set. Nếu recall tụt >5 điểm so với bậc trước → dừng, tìm nguyên nhân trước khi lên tiếp. Bậc ~100 docs là bậc quan trọng nhất — nó kích hoạt `eval_set_general_v1` làm answer-test thật.

### 6.1. BẮT BUỘC trước khi ingest lại (blocker — sai là phải re-embed từ đầu)

| # | Việc | Vị trí | Lý do |
|---|------|--------|-------|
| 1 | **Point ID deterministic** — thay `uuid.uuid4()` bằng hash ổn định (vd `uuid5(NAMESPACE, f"{document_id}:{chunk_index}")`) | `src/db/qdrant_client.py:113` | uuid4 → re-run ingest tạo DUPLICATE point. Với 2M chunks, ingest phải resume được sau crash → upsert phải idempotent. Đây là bug tiềm ẩn ngay cả ở 6 docs. |
| 2 | **Checkpoint/resume cho ingest** — ghi tiến độ (doc_id đã xong) ra file/SQLite, flag `--resume` | `scripts/ingest.py` | Embedding 2M chunks chạy nhiều giờ/ngày — chắc chắn sẽ bị gián đoạn (bài học: reindex 1670 chunks đã chết 2 lần). |
| 3 | **Fix chunker metadata misalignment TRƯỚC** — chunk gắn nhãn "Điều 168" nhưng text là Điều 167 (đã phát hiện ở Tuần 2) | `src/data/chunker.py`, `chunking/` | Lỗi này nhân theo corpus. Đo bằng gap strict-vs-lenient trong Tier 1 sau khi fix; phải ≈ 0 trên 6 docs trước khi ingest thêm. |
| 4 | **Ghi `article_number` (số hóa) vào payload lúc ingest** + tạo **payload index** cho `document_id`, `article_number`, `legal_type`, `issuance_date` | `scripts/ingest.py`, `qdrant_client.py` | Citation lookup hiện scroll toàn collection client-side (TODO đã ghi trong `citation_lookup.py`) — chết ở 100K+ chunks. Index xong thì chuyển sang server-side filter. |
| 5 | **Embedding compute**: local CPU không khả thi (~2.5s/8 chunks → 2M chunks ≈ 7 ngày). Chạy trên Colab/Kaggle GPU T4 (script standalone đọc parquet → upsert thẳng Qdrant Cloud) hoặc thuê GPU spot | `scripts/reindex_contextual.py` làm mẫu | Máy local chỉ dùng cho query-time. |
| 6 | **Qdrant dung lượng**: 2M chunks × 1024d fp32 ≈ 8GB — vượt xa free tier 1GB. Bật **scalar quantization int8** (giảm ~4x) + vectors on-disk, hoặc nâng plan trả phí. Quyết định TRƯỚC khi ingest vì đổi sau là re-index | Qdrant collection config | Tính cả sparse vectors + payload (~vài GB text). |

### 6.2. Code phải sửa khi corpus lớn (không blocker ingest, nhưng blocker chất lượng/latency)

| # | Việc | Vị trí | Lý do |
|---|------|--------|-------|
| 7 | **Citation lookup → server-side** (sau khi có index ở #4): filter `document_id + article_number` thay vì scroll cache | `src/rag/citation_lookup.py` | Cache client-side = tải cả collection vào RAM. |
| 8 | **Doc-alias resolution nâng cấp**: hiện match substring title — với hàng nghìn luật trùng tên ("Luật sửa đổi...", luật cùng tên khác năm) cần: chuẩn hóa số hiệu văn bản (59/2020/QH14) làm khóa chính, bảng alias tên viết tắt mở rộng (BLDS, LDN, LĐĐ...), disambiguation theo năm/hiệu lực | `citation_lookup.py` `_DOC_HINT_PATTERNS`, `resolve_doc` | Pattern hiện tại hardcode cho hình sự (BLHS/TTHS). |
| 9 | **Metadata store JSON → SQLite/Postgres**: `documents.json`/`graph.json`/`timelines.json` load toàn bộ vào RAM; enrichment + `EnhancedRetriever._enrich_result` loop O(N) qua all documents cho MỖI result | `src/data/metadata_store.py`, `src/pipeline/retriever.py:228` | O(N) × 15K docs × mỗi query = không dùng được. |
| 10 | **Filter `is_in_effect` mặc định trong search**: corpus 6 docs toàn văn bản còn hiệu lực; full corpus chứa nhiều văn bản HẾT hiệu lực → mặc định chỉ search văn bản còn hiệu lực (payload filter), point-in-time là option | `qdrant_client.py` search + payload lúc ingest | Trả lời theo luật hết hiệu lực là lỗi nghiêm trọng nhất của legal bot. |
| 11 | **Reranker thành bottleneck**: CPU cross-encode ~15 chunks/query đã chậm; corpus lớn → candidate pool lớn hơn. Chọn: GPU serving, giảm candidates (top-10), hoặc rerank API (Cohere rerank-multilingual) | `orchestrator.py _rerank_results` | Đo latency P95 ở bậc 1K docs để quyết. |
| 12 | **Complexity classifier + prompts cập nhật đa lĩnh vực**: prompt hiện thiên ví dụ doanh nghiệp/hình sự; mở rộng `LEGAL_TYPES` (Nghị định, Thông tư) kéo theo hierarchy văn bản (luật > nghị định > thông tư) trong ranking/context | `src/llm/prompts.py`, `constants.py` | Thông tư hướng dẫn thường là câu trả lời thực tế cho câu hỏi thủ tục. |
| 13 | **Enrichment pipeline (graph/timeline/amendment) chạy batch + incremental**: hiện `/api/pipeline/enrich` chạy đồng bộ 1 phát — với 15K docs phải là job offline có checkpoint, và chỉ enrich docs mới | `src/pipeline/enricher.py`, `api/main.py` | Endpoint sync sẽ timeout; cũng cần auth (đã ghi ở mục 2). |

### 6.3. Eval & calibration phải làm lại (kết quả cũ không chuyển được)

| # | Việc | Chi tiết |
|---|------|----------|
| 14 | **Kích hoạt `eval_set_general_v1` làm answer-test**: khi `required_laws` của câu nào đã ingest → flip refuse→answer. PHẢI chạy `scripts/validate_eval_set.py` mở rộng: verify `ground_truth_verified: false` với corpus thật (đối chiếu điều luật, điền `expected_articles` doc_id) trước khi tin số đo |
| 15 | **Re-calibrate REFUSAL_THRESHOLD**: 0.725 calibrated trên corpus hình sự, margin chỉ ~0.005 — phân bố score chắc chắn đổi khi corpus lớn (nhiều distractor gần hơn → score cao hơn cho câu sai). Chạy lại Tier 2 + `scripts/calibrate_threshold.py` ở mỗi bậc scale |
| 16 | **Mở rộng eval set theo lĩnh vực mới**: mỗi lĩnh vực ingest thêm cần ≥10 câu grounded (theo mẫu criminal set); giữ nhóm out-of-scope (câu ngoài pháp luật + Điều không tồn tại) vì vẫn cần đo refusal |
| 17 | **Out-of-scope detection đổi bản chất**: corpus đa lĩnh vực → ít câu "ngoài corpus" hơn, nhưng vấn đề "retrieval trả về chunk có-vẻ-liên-quan cho câu ngoài phạm vi" (gen-203 conf 0.845) vẫn còn với câu hành chính/quốc tế/tư vấn cá nhân — làm ở bậc 100 docs khi còn đo được |
| 18 | **CI eval**: Tier 1 với eval set lớn hơn vẫn OK (không LLM), nhưng thêm timeout + chạy theo shard nếu >200 câu; giữ Tier 2 ngoài CI |

### 6.4. Infra/vận hành

| # | Việc | Chi tiết |
|---|------|----------|
| 19 | **RAM query-time**: BGE-M3 (~2GB) + reranker (~2GB) + NLI (~1GB) — máy serve cần ≥8GB hoặc tách embedding/rerank thành service riêng (hoặc API hóa) — khớp với lưu ý Render 512MB ở mục 5.2 |
| 20 | **Semantic cache (Redis) trở nên đáng làm**: corpus lớn → nhiều user → nhiều query lặp; cache theo embedding similarity của query |
| 21 | **Observability trước khi scale user**: Langfuse trace per-stage — không có nó thì không debug được pipeline 6 bước trên corpus lớn |
| 22 | **Backup/version collection**: đặt tên collection theo version (`luat_law_ctx_v2_YYYYMM`), giữ collection cũ đến khi eval xác nhận bậc mới ≥ bậc cũ (pattern đã dùng khi chuyển luat_law → luat_law_ctx) |

### Thứ tự đề xuất

1. **Trước tiên (trên 6 docs hiện tại)**: #1 point ID, #3 fix chunker (đo strict-lenient gap ≈ 0), #4 payload index + article_number → re-ingest 6 docs lần cuối để có nền sạch.
2. **Bậc ~100 docs**: #2 checkpoint, #5 GPU embedding, #8 alias, #10 is_in_effect, #14-16 eval, #17 out-of-scope.
3. **Bậc ~1K docs**: #6 quantization, #7 lookup server-side, #9 SQLite, #11 đo reranker, #15 re-calibrate.
4. **Toàn bộ**: #12, #13, #18-22 + re-calibrate lần cuối.
