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

**✅ Soi per-signal 4 câu false-refusal (crim-109/201/202/302) — HOÀN THÀNH 2026-07-07** — chi tiết đầy đủ tại `docs/notes/crim-false-refusal-diagnosis-2026-07-07.md`; công cụ tái sử dụng: `scripts/inspect_confidence.py` (chạy pipeline thật, dump đủ 5 signal + retrieval vs ground truth + NLI labels cho ID bất kỳ). Tìm ra **1 bug thật trong entailment signal**: mọi answer đều kết thúc bằng section bắt buộc "💡 Câu hỏi liên quan" (3 câu hỏi gợi ý), và `_split_claims()` đưa cả header + 3 câu hỏi này vào NLI như thể là claims pháp lý → nhãn nhiễu (có cả ENTAILMENT/CONTRADICTION vô nghĩa trên chuỗi header) làm méo `entailment_ratio` của MỌI câu. Fix: dừng split khi gặp dòng header đứng riêng (regex chỉ match standalone header, không match cụm "câu hỏi liên quan đến..." giữa câu văn thật — Fable siết lại trong lúc verify). 4 tests mới (`tests/test_verifier.py`, 86 total).

Full eval criminal sau fix (report `20260707T052327Z`, 0 lỗi): **false-refusal 4/34 → 2/34** (crim-109 & crim-302 hết bị refuse nhầm), refusal accuracy giữ 7/7, citation accuracy 0.8529 → **0.8824**, refuse-side max confidence **giảm** 0.7055 → 0.5319 (fix gỡ cả inflation giả ở câu nên-refuse → tách 2 phía xa hơn, là debiasing chứ không phải boost điểm). crim-201/202 vẫn refuse (0.694/0.670) — signal đang báo trung thực (multi-hop shaky + self-eval variance), không phải bug.

**✅ Grounded citations `[n]` — HOÀN THÀNH 2026-07-07 (roadmap §3.2, "quote-then-answer")** — mỗi ý trong `### 📋 Chi tiết` giờ phải gắn số khối nguồn `[n]` trỏ vào khối context được đánh số (khối đã có sẵn trong `_build_context`). (1) Prompt: `SYSTEM_PROMPT` thêm rule 6 + `GENERATION_PROMPT` yêu cầu `[n]` cuối mỗi bullet, coexist với cite "Điều X, Khoản Y", loại trừ trailer "Câu hỏi liên quan". (2) Verifier Tier-1.5 mới trong `verifier.py`: `_verify_grounding()` parse khối từ context (split `\n\n---\n\n`, match `^\[(\d+)\]`), với mỗi claim check `[n] ∈ [1,k]` (bắt block-ref bịa → `invalid_block_refs`) và siết bug 1c — "Điều X" phải nằm trong **đúng khối `[n]` được cite** chứ không phải đâu đó trong context (`ungrounded_article_citations`); thêm `grounding_ratio`. **Không** đổi `passed` (chỉ signal quan sát) và **không** đụng REFUSAL_THRESHOLD/DEFAULT_WEIGHTS → calibration 0.725 nguyên vẹn. 5 test mới (`tests/test_verifier.py`, 9 verifier / 91 suite pass). Fable verify độc lập: đọc diff, chạy lại test, live-test qua `query_stream` trỏ đúng `luat_law_ctx` (crim-202 answer, conf 0.7575, model phát `[2][7]`/`[6][9]`... đều hợp lệ, `invalid_block_refs=[]`, `ungrounded_articles=[]`; chèn bogus `[115]` → detector fire đúng). **2 lưu ý:** (i) `grounding_ratio` nhiễu — dòng tiêu đề mục ("Phương thức thực hiện:") và câu "Trả lời ngắn" bị `_split_claims` tính là claim không có `[n]` → ratio ~0.55 dù mọi bullet nội dung đều grounded; observability-only nên không sao, nhưng phải siết `_split_claims` trước khi đưa ratio vào confidence hay hiển thị. (ii) `SYSTEM_PROMPT` có placeholder `{context}` truyền literal (không format) — context thật chỉ vào qua `GENERATION_PROMPT`; wart có sẵn từ trước, không ảnh hưởng feature (live test xác nhận model vẫn cite khối đúng). Follow-up: đưa grounding thành signal confidence có trọng số (cần recalibrate threshold), verbatim-span quote để verify chặt hơn, ~~wire `[n]` vào citation card UI~~ ✅ xong (xem block dưới).

**✅ Hoàn thiện grounded citations (UI + ratio) — HOÀN THÀNH 2026-07-07** — hai phần: **(A) wire `[n]` ↔ citation card**: `_build_citations` (orchestrator) giờ tag `block_index` (enumerate 1-based khớp `_build_context`) và trả **tất cả** khối context (không cắt top-3 nữa) — trước đây `[7]` model cite trỏ vào card không tồn tại. UI `chat.py` `_render_citations(answer_text, citations)` parse `[n]` từ answer, hiện card được cite lên đầu (`**[n]** 📘 title — Điều`), phần còn lại gộp expander "Xem tất cả nguồn"; `_highlight_legal_refs` bôi đậm `[n]`. Bonus: fix bug dead-field có sẵn (card cũ đọc `excerpt`/`clause`/`relevance` chưa từng set → luôn hiện "..."/"cao"; giờ đọc `chunk_text`/`legal_type`/`score`/`url` thật) + `_render_answer_extras` khiến card hiện ngay lượt hiện tại thay vì trễ 1 rerun. **(B) siết grounding_ratio**: thêm `_groundable_claims` (selector RIÊNG, chỉ bullet mục "Chi tiết", bỏ summary "Trả lời ngắn" + dòng label `**...**:`), `_verify_grounding` dùng nó thay `_split_claims`. **`_split_claims` bất biến** (vẫn feed NLI → confidence weight 0.10 → threshold 0.725 nguyên vẹn; có test pin `_split_claims` chống drift). 3 test mới (12 verifier / 94 suite pass). Fable verify độc lập: đọc diff, chạy test, kiểm `_split_claims` bất biến (4 claims như cũ) vs `_groundable_claims` (2 bullet), live-run trên `luat_law_ctx` (query người <18 tuổi, answer conf 0.779: grounding **BEFORE 0.583 → AFTER 0.875**, mọi `[n]`=1,2,3,4,5,14 resolve về card), và thực thi hàm render UI qua mock `st` (card `[n]` lên đầu + expander phần còn lại, không exception). Follow-up còn lại: grounding thành confidence signal có trọng số, verbatim-span quote.

**✅ Fix default collection lệch — HOÀN THÀNH 2026-07-07:** `LegalVectorStore()` từng mặc định `collection_name="vietnamese_laws"` (qdrant_client.py:55) — collection đó giờ chứa 3441 điểm VB HÀNH CHÍNH/thuế/thủ tục (có cả VB 2026), KHÔNG phải corpus hình sự. Corpus hình sự ở `luat_law_ctx` (1670)/`luat_law` (1670). App (`api/main.py`) + eval scripts đều truyền collection tường minh (`DEFAULT_COLLECTION_NAME`="luat_law_ctx") nên KHÔNG dính bug; chỉ code gọi `LegalVectorStore()` bare mới lấy nhầm corpus (không chỗ nào trong repo làm vậy — chỉ script ad-hoc). User xác nhận default nên trỏ lại `luat_law_ctx` → đổi constructor default thành `= DEFAULT_COLLECTION_NAME` (import từ `src.data.constants`, không hardcode nữa để không lệch lại). Verify: bare `LegalVectorStore()` → luat_law_ctx (1670 điểm), query hình sự answer conf 0.768, 91 test pass. Collection `vietnamese_laws` giữ nguyên trong Qdrant (không xoá) — nếu là corpus mở rộng có chủ đích thì xử lý theo roadmap §6 khi mở rộng eval set.

**✅ Recalibrate threshold sau khi reranker sống — HOÀN THÀNH 2026-07-07** — re-run Tier-2 FRESH cả hai set với reranker thật (reports `20260707T110621Z` criminal 34 answer+7 refuse, `20260707T111111Z` general 29 refuse; 0 lỗi, 0 null-score), calibrate qua `scripts/calibrate_threshold.py` (report `20260707T111144Z_calibration`). **Kết luận: GIỮ 0.725** (chỉ đổi comment trong `src/rag/confidence.py`, không đổi giá trị). Lý do: grid Youden-optimal là t=0.675 (youden 0.9428) nhưng 0.675 **làm rò gen-802** — câu out-of-scope duy nhất lọt qua scope gate (đất đai dân sự, score 0.7045) → refusal_recall tụt 1.0→0.9722. Với legal bot, trả lời câu ngoài phạm vi tệ hơn từ chối nhầm bảo thủ, nên giữ threshold trong dải leak-safe (t > 0.7045); 0.725 là grid point thấp nhất vẫn refuse gen-802 (refusal_recall **1.0**, false_refusal **0.0588** = crim-201+crim-203, youden 0.9412 — chỉ kém optimum 0.0016). Phân bố chồng lấn (max refuse 0.7045 > min answer crim-203 0.5698, gap −0.1347): không threshold nào tách sạch. Margin mới: khe sạch gen-802 (0.7045) → crim-104 (0.7291) = 0.0246; 0.725 cách gen-802 **+0.0205** (đệm dày phía rò — an toàn) và crim-104 **−0.0041** (mỏng phía false-refuse benign). Reranker KHÔNG làm sập phân bố answer (mean 0.83); chỉ đẩy crim-203 (câu so sánh BLHS 1999 vs 2015) xuống ~0.53-0.57 — **đã verify độc lập: retrieval thành công (Điều 123 Tội giết người có ở cosine 0.833) nhưng cross-encoder xếp thấp vì câu so sánh cần tổng hợp 2+ điều, max single-chunk relevance 0.2015; cộng self-eval draw 0.40 + NLI NEUTRAL trên claim so sánh tiếng Việt ngắn (entailment 0.0)**. crim-203 là follow-up per-signal (reranker-on-comparison / self-eval / NLI-VN), KHÔNG phải vấn đề threshold — cứu nó cần t<0.57 sẽ rò mọi câu refuse. Verify (Fable): tự re-run calibrate reproduce y hệt số của agent, 34 test logic pass, module load 0.725. Vẫn tạm thời ở corpus 6-doc — recalibrate lại khi eval set lớn.

**✅ Fix reranker dependency — HOÀN THÀNH 2026-07-07 (ngay sau diagnosis)** — pin `transformers>=4.44.2,<5.0.0` trong pyproject.toml (cài 4.57.6, bản 4.x mới nhất; lower bound theo đúng ràng buộc của FlagEmbedding 1.4.0). FlagReranker chạy lại bình thường (verify sống: normalize=True trả 0.986 cho cặp liên quan vs 0.020 không liên quan), embedder không ảnh hưởng, 86 tests pass. Full eval sau fix (report `20260707T054623Z`, 0 lỗi): aggregate trung hòa (false-refusal giữ 2/34, refusal accuracy 7/7) nhưng **crim-201 & crim-202 — 2 câu false-refusal còn sót — giờ answer đúng nhờ reranker** (0.694→0.738, 0.670→0.775, citation crim-202 0→1.0); đổi lại crim-203 (medium, 0.906→0.532 — reranker thật kéo xuống, đáng soi) và crim-304 (lookup, 0.725 sát nút — variance thuần) refuse mới. ⚠️ **Hệ quả cần xử lý: threshold 0.725 được calibrate khi reranker signal luôn vắng mặt** — giờ signal 0.20-weight có số thật, phân bố điểm đổi → recalibrate threshold trên data mới là việc tiếp theo hợp lý (gộp với mở rộng eval set).

**2 follow-up còn lại từ diagnosis** (chưa fix, có chủ đích):
1. NLI premise bị cắt `context[:2000]` — điều luật đúng có thể bị cắt khỏi premise khi không nằm trong ~3-4 chunk đầu (crim-109). Cần quyết định thiết kế (per-claim premise selection).
2. Self-eval ceiling effect: 0.85–1.00 ở 8/8 quan sát, kể cả khi verifier gắn cờ contradiction — signal weight 0.30 nhưng ít khả năng phân biệt. Cân nhắc khi recalibrate.

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

### ✅ Trạng thái M0 — Plan Giai Đoạn Cuối (cập nhật 2026-07-10)

Giai đoạn cuối chạy theo [`Plan_Giai_Doan_Cuoi_2026-07-10.md`](Plan_Giai_Doan_Cuoi_2026-07-10.md). Tiến độ M0:

- **✅ M0.4a (commit `5dd8f43`)**: point ID deterministic `uuid5(document_id:chunk_index)` + fallback content-hash; payload `article_number` (keyword, giữ hậu tố "123a", parse bằng `artnum()` dùng chung); payload index 5 field trong `create_collection`. 20 tests mới.
- **✅ M0.4b (commit `8c202a1`)**: chunker misalignment — chẩn đoán tại [`notes/chunker-misalignment-diagnosis-2026-07-10.md`](notes/chunker-misalignment-diagnosis-2026-07-10.md): 126/1670 chunk production nhãn Điều SAI (merge xuyên ranh giới Điều trong `_control_chunk_sizes`) và **178 Điều MẤT HẲN khỏi index** (flush drop text ngắn). Fix HierarchicalChunker (merge cùng-Điều-mới-gộp, giữ Điều ngắn, parent_text two-pass đồng nhất) + ArticleLevelChunker (Fable follow-up: "Điều 3. Hiệu lực thi hành" 2025 từng bị drop). Gate script `scripts/verify_chunker_alignment.py`: WRONG_LABEL 126→0, Điều mất 178→0, HEADER_OK 193→1288. Chunks 1670→2242.
- **✅ M0.5 (commits `4d2c9e7`, `9a17f17`)**: collection mới **`luat_law_ctx_v2`** (2.242 points) — int8 scalar quantization `always_ram` + fp32 dense on-disk (xác nhận sống trên Qdrant Cloud), ingest có `--collection`/`--quantized`. Gate Tier-1 (report `20260710T080938Z`): recall@5 routed **0.9216** (baseline 0.922, per-type giống hệt baseline ở mọi loại câu), **lenient == strict** (gap về 0 — nhãn khớp text, đúng gate M0). MRR 0.863→0.805 là xáo trộn hạng top-10 (RRF tie-break + nhiễu quantization), recall không đổi — theo dõi ở M0.3. `DEFAULT_COLLECTION_NAME` → `luat_law_ctx_v2`; `luat_law_ctx`/`luat_law` giữ để so sánh.
- **🔄 M0.1 đang làm** (Sonnet): NLI premise per-claim (thay `context[:2000]`, verifier.py) + self-eval rubric (ceiling 0.85-1.00, confidence.py/prompts.py).
- **⏭ Còn lại**: M0.2 (grounding thành weighted signal) → M0.3 (Tier-2 full + recalibrate threshold — một lần duy nhất cuối M0).
- ⚠️ Phát hiện ngoài lề: nightly eval CI fail `Connection refused` — secrets `QDRANT_URL`/`QDRANT_API_KEY` chưa set trong repo GitHub (workflow wiring đúng); cần user set bằng `gh secret set`.

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

**3.4. Parent-document / small-to-big retrieval** — ✅ ĐÃ LÀM 2026-07-08
Embed và search ở mức Khoản (chính xác), nhưng đưa vào context cả Điều cha (đủ ngữ cảnh). Plan §3.2 mô tả đúng ý bằng "overlap" nhưng cách đúng là lưu `parent_id` và fetch parent lúc retrieve, không duplicate text.

Triển khai: `parent_chunk_text` đã tồn tại sẵn trong payload (tính lúc chunk, `HierarchicalChunker`) nhưng chưa từng được đọc ra — `SearchResult`/`EnhancedSearchResult` không có field này, và `citation_lookup.py._point_to_search_result` build `SearchResult` thủ công (bỏ sót field). Đã thêm field + `orchestrator._expand_to_parent()`: swap `chunk_text` → `parent_chunk_text`, dedupe các Khoản cùng Điều thành 1 block (test trực tiếp: 3 Khoản của Điều 173 → 1 block đầy đủ 3 khoản thay vì 3 block trùng lặp/cụt). `_build_context` cap 500→2000 ký tự/block (bug đã ghi ở mục 2).

**Bug phát hiện khi làm**: `ArticleLevelChunker` (dùng cho 3 văn bản sửa đổi) set `parent_chunk_text = content.strip()` — TOÀN BỘ document (tới 267K ký tự) thay vì chỉ Điều đó — vì mỗi chunk ở strategy này ĐÃ là 1 Điều trọn vẹn, không có "parent" nào lớn hơn. Đã fix code (`parent_chunk_text = art["text"]`) + patch payload 23 điểm đã ingest trên Qdrant Cloud (chỉ sửa payload, không re-embed) + thêm guard `_PARENT_TEXT_MAX_LEN=3000` trong `_expand_to_parent` (fallback về chunk_text nếu parent bất thường lớn) phòng dữ liệu xấu tương tự trong tương lai.

**3.5. Eval trên benchmark Việt Nam có sẵn**
Dùng **ALQAC** (Automated Legal Question Answering Competition — task retrieval + QA trên luật Việt Nam, có ground truth) và **Zalo Legal Text Retrieval** làm bộ đo retrieval (recall@k, MRR, nDCG) trước khi đo RAGAS. **Đo retrieval tách khỏi generation** — hiện không biết lỗi nằm ở tầng nào.

### Tier 2 — nâng cấp có chọn lọc

**3.6. Nối Phase 2 vào pipeline chat** — ✅ ĐÃ LÀM 2026-07-08
- Temporal filter ("luật còn hiệu lực tại ngày hỏi") nên là **filter mặc định trong Qdrant payload**, không phải post-filter.
- Amendment status ("Điều này đã bị sửa đổi bởi...") nên xuất hiện trong context đưa cho LLM và trong citation card.
- Đây là điểm khác biệt thật sự của legal RAG so với RAG thường — đã build 80%, chỉ thiếu 20% wiring.

Triển khai: `EnhancedRetriever._enrich_result`'s amendment/temporal lookup tách thành hàm dùng chung `get_phase2_status()` (`src/pipeline/retriever.py`), gọi từ TẤT CẢ route của orchestrator (naive, lookup, simple, medium, complex, cả streaming lẫn non-streaming). Hai phần:
1. **Default temporal filter** (`_filter_in_effect`, fail-open — thiếu metadata hoặc `is_in_effect=None` thì GIỮ, chỉ loại khi biết chắc `False`): áp dụng trên broad pool trước dynamic-top-k/rerank.
2. **Amendment badge**: `_build_context` chèn dòng `⚠️ Đã được sửa đổi bởi: ...` vào block bị sửa đổi; `_build_citations` thêm field `is_amended`/`amending_laws`; UI (`chat.py._render_citation_card`) hiện badge tương ứng.

**Bug phát hiện khi làm — filter trên vốn sẽ là no-op nếu không sửa cái này**: `enricher.py` chưa bao giờ cross-link "luật A thay thế luật B" sang `is_in_effect=False` cho B — chỉ tự ghi nhận từ chính văn bản của nó. Kết quả: BLHS 1999 (doc 46056, đã bị thay thế bởi BLHS 2015 từ 2016-07-01) vẫn mang `is_in_effect=True` trong `data/metadata/documents.json`. Code `VersionResolver.find_replacement()` để làm chính việc này đã tồn tại sẵn nhưng chưa từng được gọi trong `enricher.run()`. Đã nối vào (dùng version chain sẵn có ở Step 3) + chạy lại enrichment local — `currently_in_effect` 6→5, xác nhận qua diff chỉ đổi đúng 1 doc (46056).

**Verify sống**: truy vấn dense "Tội giết người bị xử phạt như thế nào?" — raw top-20 có 7/20 kết quả từ BLHS 1999 (luật đã hết hiệu lực); sau filter còn 0/13. Trước khi có fix này, bot có thể đã trích dẫn điều luật từ bộ luật 1999 như thể còn hiệu lực mà không ai biết.

**Lưu ý chưa làm** (ngoài phạm vi hôm nay): filter vẫn là post-filter (client-side sau khi lấy `is_in_effect` từ `MetadataStore`), CHƯA phải payload filter trong Qdrant như mục 6.2 #10 đề xuất cho corpus lớn — ở quy mô 6 docs post-filter đủ nhanh, để dành khi scale.

**Cảnh báo hiệu ứng phụ lên confidence — CẦN recalibrate**: `_build_context` giờ trả về text dài hơn nhiều (full Điều cha thay vì 500 ký tự) — chuỗi này được `context_overlap` signal (15%) và `llm_self_eval` (qua `context[:2000]`, 30%) tiêu thụ trực tiếp. Chưa chạy lại Tier-2 eval + `calibrate_threshold.py` sau thay đổi này — ngưỡng 0.725 (calibrate lần gần nhất 2026-07-07) có thể không còn margin an toàn như đã đo. Khuyến nghị: recalibrate trước khi coi đây là "xong" cho production.

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
