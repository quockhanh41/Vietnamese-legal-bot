# Chiến lược chọn model LLM theo từng bước trong query path

Cập nhật: 2026-07-19. Tài liệu này giải thích **model nào được dùng ở bước nào** trong
một lượt query, và **vì sao** — bổ sung cho [`ARCHITECTURE.md`](ARCHITECTURE.md) (mục
"`src/llm/` — cổng LLM") vốn chỉ mô tả cơ chế gateway/fallback chung, không đi sâu vào
lựa chọn model theo từng bước.

Liên quan: `src/llm/gateway.py` (`LLMGateway`, gồm `generate_with_model()` — mục 5),
`src/rag/orchestrator.py` (`MODEL_CLASSIFY`/`MODEL_QUERY_TRANSFORM`/`MODEL_SIMPLE`/
`MODEL_COMPLEX`), `src/rag/confidence.py` (`MODEL_EVAL`), `.env` (`LLM_MODEL_*` override).

---

## 1. Vì sao mỗi bước có chain riêng thay vì 1 `MODEL_CHAIN` chung

`LLMGateway.MODEL_CHAIN` (trong `gateway.py`) là chain **mặc định**, chỉ dùng khi một
lệnh gọi không truyền `model=`. Trong query path thật, 5 bước LLM dùng 5 chain riêng,
mỗi chain override được qua biến môi trường cùng tên (`_get_model_chain(env_var,
default)` — đọc `LLM_MODEL_*`, phân tách bởi dấu phẩy, rơi về default Python nếu biến
môi trường không đặt).

Lý do tách thay vì dùng chung một chain:

1. **Các bước khác hẳn nhau về giá trị/rủi ro khi model yếu hoặc sai.** Phân loại
   scope/complexity là fail-open (sai → mặc định an toàn, không chặn nhầm câu thật);
   nhưng self-eval quyết định refuse/answer trực tiếp, và query-rewrite/HyDE định hình
   recall của route khó nhất. Dùng chung một model cho mọi việc này nghĩa là hoặc trả
   giá quá cao cho việc rẻ, hoặc trả giá quá thấp cho việc quan trọng.
2. **Groq tính quota theo tổ chức (per-organization TPD), không theo model — nhưng mỗi
   key trong `GROQ_API_KEYS`/`GEMINI_API_KEY`/`OPENROUTER_API_KEY` thuộc một tài khoản
   khác nhau** (xác nhận 2026-07-19 — sửa lại giả định cũ trong `gateway.py` rằng key
   Groq share chung 1 org/bucket). Rotation vì vậy nhân capacity theo số key ở **cả 3
   provider**, không riêng Gemini/OpenRouter — rủi ro cạn quota cùng ngày nhẹ hơn nhiều
   so với giả định "1 bucket 100K/ngày dùng chung" ban đầu. **Vì vậy quyết định ban đầu
   (đặt Gemini làm primary cho `MODEL_QUERY_TRANSFORM` để né tải Groq) đã bị đảo lại
   (xem mục 2/3)** — latency (Groq ~0.25s vs Gemini ~1-2s cross-region, ×3 lệnh tuần tự)
   giờ quan trọng hơn một rủi ro quota đã yếu đi. Rủi ro còn lại không phải "cạn quota"
   mà là **Groq-outage diện rộng** (dịch vụ lỗi, không phải 1 tài khoản) — hiếm hơn và
   được fallback sang Gemini xử lý, chấp nhận đánh đổi latency ở phần lớn thời gian để
   đổi lấy khả năng phục hồi ở phần hiếm khi Groq lỗi.

---

## 2. Bảng 5 chain

| Chain (env override) | Default (primary → fallback) | Dùng ở đâu | Lý do |
|---|---|---|---|
| `MODEL_CLASSIFY`<br>(`LLM_MODEL_CLASSIFY`) | `groq/llama-3.1-8b-instant` → `gemini/gemini-3.5-flash` | `orchestrator.py: _classify_and_scope()` — 1 lệnh gọi duy nhất, output ngắn (`SCOPE:`/`CLASS:`/`COMPARE:`) | Task rẻ và **fail-open theo thiết kế**: parse lỗi → mặc định `IN` + `MEDIUM`, không bao giờ chặn nhầm câu hỏi thật. Không cần trả giá cho model mạnh ở một quyết định mà lỗi phân loại chỉ đổi route chứ không đổi tính đúng-sai của câu trả lời. |
| `MODEL_QUERY_TRANSFORM`<br>(`LLM_MODEL_QUERY_TRANSFORM`) | `groq/llama-3.3-70b-versatile` → `gemini/gemini-3.5-flash` | `complex_retrieve()` + bản inline trong `query_stream()`: query rewrite, multi-query (3 biến thể), HyDE | **Tách khỏi `MODEL_CLASSIFY`** (trước đó dùng chung, cùng model 8B) vì đây là bước **sinh** câu hỏi phụ định hình recall của `_complex_pipeline` — chất lượng kém ở đây kéo tụt đúng route khó nhất. Primary **Groq** (không phải Gemini — đã đổi lại 2026-07-19 sau khi biết mỗi key Groq là 1 tài khoản riêng, xem mục 1 điểm 2): nhanh hơn nhiều (~0.25s vs ~1-2s cross-region) cho 3 lệnh tuần tự, và là model non-thinking nên không cần lệ thuộc `reasoning_effort` để ổn định. Gemini giữ làm fallback cho trường hợp Groq lỗi diện rộng. |
| `MODEL_SIMPLE`<br>(`LLM_MODEL_SIMPLE`) | `groq/llama-3.3-70b-versatile` → `gemini/gemini-3.5-flash` → `openrouter/llama-3.3-70b-instruct:free` | `_simple_pipeline` — sinh câu trả lời cho câu hỏi dễ | Route `simple` chiếm khối lượng traffic lớn nhất và rẻ nhất để trả lời đúng — ưu tiên provider nhanh/ổn định nhất (Groq, ~0.25s, so với ~1-2s cross-region của Gemini) thay vì để dư năng lực Groq ở đúng tier ít cần tốc độ nhất. |
| `MODEL_COMPLEX`<br>(`LLM_MODEL_COMPLEX`) | `groq/openai/gpt-oss-120b` → `gemini/gemini-3.5-flash` → `openrouter/llama-3.3-70b-instruct:free` | `_medium_pipeline` + `_complex_pipeline` — sinh câu trả lời cuối cho câu medium/complex | Câu khó xứng đáng model mạnh nhất đã live-verify miễn phí (gpt-oss-120b, 120B tham số). Lựa chọn cũ `gemini-3.5-pro` **không tồn tại** (404 thật từ Google), và quan trọng hơn: **mọi model "pro" của Gemini free tier có quota = 0** (xác nhận qua lỗi 429 `limit: 0` trực tiếp từ Google, không phải rate-limit tạm thời) — flash là tier free duy nhất Gemini cấp, không có "pro" free nào để thay. |
| `MODEL_EVAL`<br>(`LLM_MODEL_EVAL`) | `groq/openai/gpt-oss-120b` → `gemini/gemini-3.5-flash` → `openrouter/llama-3.3-70b-instruct:free` | `confidence.py: ConfidenceScorer._llm_self_eval()` — chấm rubric 4 tiêu chí (CĂN CỨ/TRÍCH DẪN/SUY DIỄN/ĐẦY ĐỦ), tín hiệu **nặng nhất** (0.25/1.00) của cổng từ chối | Nâng từ `llama-3.3-70b-versatile` lên `gpt-oss-120b`. Đây là tín hiệu an toàn nhất trong 6 tín hiệu confidence — quyết định refuse/answer — và lịch sử calibration của dự án đã ghi nhận nhiễu run-to-run ±0.1-0.2 ở tín hiệu này. Chấm 4 tiêu chí đối chiếu claim↔context không dễ hơn việc sinh câu trả lời, nên không có lý do dùng model yếu hơn `MODEL_COMPLEX` cho việc chấm điểm. **Vì `MODEL_COMPLEX` và `MODEL_EVAL` cùng dùng `gpt-oss-120b` làm primary, model này bị loại động khỏi candidate list khi chính nó vừa generate ra câu trả lời đang được chấm — xem mục 5.** |
| *(ngoài 5 chain trên)*<br>NLI | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` — model **local**, không qua LLM gateway/API | `verifier.py: HallucinationVerifier._verify_semantic()` (Tier 2) | Native NLI classifier chuyên việc premise/hypothesis — không tốn quota/latency mạng, không lệ thuộc bias của model sinh câu trả lời (self-grading). Thiết kế này giữ nguyên, không nằm trong đợt tái cấu trúc model chain lần này — đã đúng ngay từ đầu. |

---

## 3. Số lệnh gọi Groq mỗi câu hỏi

Nếu mọi primary đều thành công (không rơi vào fallback), **hiện tại** (sau khi đổi
`MODEL_QUERY_TRANSFORM` về Groq primary, mục 2):

| Route | Tổng lệnh LLM | Trong đó rơi vào Groq |
|---|---|---|
| simple | classify (1) + generate (1) + eval (1) = 3 | 3/3 |
| medium | classify (1) + generate (1) + eval (1) = 3 | 3/3 |
| complex | classify (1) + rewrite/multi-query/HyDE (3) + generate (1) + eval (1) = 6 | 6/6 |

Mọi route giờ đều Groq-primary toàn phần. Đây là quyết định có chủ đích sau khi cân nhắc
lại (2026-07-19): ban đầu route complex được thiết kế 3/6 (rải 3 lệnh query-transform
sang Gemini) để né dồn tải Groq, dựa trên giả định "6 key Groq share chung 1 bucket
100K/ngày". Sau khi xác nhận mỗi key thực ra là 1 tài khoản riêng (mục 1, điểm 2), rủi ro
cạn quota không còn đủ lớn để đánh đổi latency (~3-6s chậm hơn cho 3 lệnh tuần tự trước
retrieval) — nên đổi lại về Groq-primary cho nhanh, chấp nhận rủi ro còn lại
(Groq-outage diện rộng, hiếm hơn nhiều so với cạn quota) được fallback sang Gemini xử lý
khi xảy ra.

---

## 4. Bug thật phát hiện và sửa trong lúc tái cấu trúc (2026-07-19)

Các phát hiện này là lý do trực tiếp cho vài chi tiết trong bảng ở mục 2 — không chỉ là
"chọn model nào" mà còn "vì sao cách gọi phải đi kèm tham số cụ thể".

1. **`_clean_thinking(None)` crash.** Model thinking (Gemini 3.x, Groq gpt-oss) trả
   `content=None` khi đốt hết `max_tokens` để "suy nghĩ" nội bộ mà không có
   `reasoning_effort` ép tắt. `gateway.py::_clean_thinking()` gọi `re.sub()` trực tiếp
   trên `text` — nếu `text is None` thì crash `TypeError`, bị nuốt thành lỗi
   "non-rate-limit" và khiến gateway bỏ qua **toàn bộ các key còn lại** của model đó
   (chỉ thử 1 key) rồi nhảy sang model kế tiếp trong chain, che giấu một lỗi có thể sửa
   được (chỉ cần tăng `max_tokens` hoặc ép `reasoning_effort`) thành "model này chết".
   **Sửa:** `_clean_thinking()` trả `""` khi `text is None` thay vì crash.

2. **Groq gpt-oss từ chối `reasoning_effort="none"`.** Groq chỉ chấp nhận
   `low`/`medium`/`high` cho gpt-oss, không có `none` như Qwen3. Đồng thời
   `_is_thinking_model()` cũ không nhận diện gpt-oss là thinking-model nên không forward
   `reasoning_effort` cho nó — kết quả: gpt-oss dùng effort mặc định của Groq, đốt hết
   `max_tokens=150` (dùng trong `confidence.py`'s eval call) và trả rỗng **3/3 lần** khi
   live-test. **Sửa:** `_is_thinking_model()` nhận diện thêm gpt-oss;
   `_normalize_reasoning_effort()` map `"none"` → `"low"` riêng cho gpt-oss. Sau fix:
   2/3 lần đủ 4 tiêu chí ở `max_tokens=150`, 6/6 lần ổn định ở `max_tokens=250` (đã tăng
   giá trị này trong `confidence.py`).

3. **Gemini (khi đó đang là primary của `MODEL_QUERY_TRANSFORM`) thỉnh thoảng trả rỗng**
   cùng lý do (thinking model, không ép effort). **Sửa:** thêm `reasoning_effort="none"`
   vào cả 6 lệnh gọi rewrite/multi-query/HyDE (2 nơi: `complex_retrieve()` và bản inline
   trong `query_stream()`). Live-verify: 1/4 lần rỗng khi không ép → 4/4 ổn định khi ép.
   Fix này vẫn cần giữ dù Gemini đã đổi sang fallback (mục 2/3) — bug áp dụng cho bất kỳ
   lúc nào Gemini được gọi ở bước này, kể cả khi Groq lỗi và fallback rơi xuống Gemini.

4. **Model đã chết/không tồn tại, phát hiện qua live-test chứ không phải đọc code:**
   - `groq/qwen/qwen3-32b` (primary cũ của `MODEL_CHAIN`) — Groq đã ngừng phục vụ, 404.
   - `gemini/gemini-3.5-pro` — không tồn tại dưới tên này; kể cả đổi sang tên đúng
     (`gemini-pro-latest`, `gemini-3-pro-preview`...) vẫn 429 `limit: 0` vì **free tier
     Gemini không cấp quota cho bất kỳ model "pro" nào**.
   - `openrouter/deepseek/deepseek-r1:free` — OpenRouter đã khai tử, trả lỗi hướng dẫn
     dùng slug trả phí thay thế.

---

## 5. Cross-eval: eval không được chấm bằng chính model vừa generate (2026-07-19)

**Vấn đề:** `MODEL_COMPLEX` và `MODEL_EVAL` (mục 2) cùng có `groq/openai/gpt-oss-120b`
làm primary. Với câu medium/complex — route mà self-eval quan trọng nhất — nếu cả 2
primary đều thành công thì **cùng một model vừa sinh ra câu trả lời, vừa chấm điểm câu
trả lời đó**, đúng kiểu self-preference bias (model có xu hướng chấm cao cho phong cách
suy luận của chính nó) mà tín hiệu `llm_self_eval` (0.25 trọng số) muốn tránh.

**Giải pháp — loại trừ động (không phải đổi thứ tự chain tĩnh):** thay vì chọn cứng một
model "luôn luôn khác" cho `MODEL_EVAL` (đánh đổi lấy chất lượng thấp hơn ở phần lớn
trường hợp), hệ thống giờ theo dõi **model nào thực sự vừa generate thành công** và loại
đúng model đó khỏi danh sách candidate của eval tại runtime — đúng ở mọi route, kể cả khi
generate rơi vào fallback (không cứng nhắc theo giả định "primary luôn thành công").

Cơ chế (`src/llm/gateway.py`, `src/rag/confidence.py`, `src/rag/orchestrator.py`):

- `LLMGateway.generate_with_model()` (mới, cạnh `generate()` không đổi) trả về
  `(text, model_used)` — cả 2 dùng chung `_generate_impl()` để không trùng lặp logic
  retry/fallback.
- `generate_stream()` nhận thêm tham số `used_model: Optional[dict]` — generator không
  thể vừa `yield` vừa `return` giá trị cho một `for` loop thường, nên dùng **out-param**
  (dict rỗng truyền vào, được điền `{"model": ...}` ngay khi chunk đầu tiên stream thành
  công) thay vì lưu state trên `self`.
- **Quan trọng — an toàn concurrency:** `LLMGateway` là **1 instance dùng chung cho mọi
  request** (`app.state.llm` trong `src/api/main.py`, xử lý concurrent qua threadpool).
  Không thể lưu "model vừa dùng" như thuộc tính mutable trên gateway (race condition
  giữa các request chạy song song) — thông tin này luôn được truyền qua **return
  value / out-param của từng lệnh gọi**, không qua state chung.
- `ConfidenceScorer.score(..., generation_model=...)` → `_llm_self_eval(...,
  exclude_model=...)`: lọc `MODEL_EVAL` bỏ đúng `exclude_model` trước khi gọi; nếu lọc
  xong danh sách rỗng thì dùng lại `MODEL_EVAL` gốc (không để eval không còn model nào).
- Cả 5 nơi sinh câu trả lời cuối cùng (lookup route, `_simple_pipeline`,
  `_medium_pipeline`, `_complex_pipeline`, `query_stream()`) đều đổi sang
  `generate_with_model()`/`used_model=` và truyền `generation_model=` xuống
  `_post_process()`.

Live-verify: generate với `MODEL_COMPLEX` → xác nhận trả về đúng `groq/openai/gpt-oss-120b`
là model thực dùng; gọi `_llm_self_eval(..., exclude_model="groq/openai/gpt-oss-120b")` →
xác nhận lệnh eval thực tế chỉ còn thử `["groq/llama-3.3-70b-versatile",
"gemini/gemini-3.5-flash"]`, không còn `gpt-oss-120b`.

**Giới hạn đã biết:** loại trừ theo **đúng chuỗi tên model**, không theo "họ" model — vd
`groq/llama-3.3-70b-versatile` (dùng ở `MODEL_SIMPLE`) và
`openrouter/meta-llama/llama-3.3-70b-instruct:free` (fallback cuối của `MODEL_EVAL`) là
cùng họ Llama 3.3 70B nhưng khác chuỗi tên nên KHÔNG bị loại trừ lẫn nhau. Chấp nhận đơn
giản hoá này — định nghĩa "cùng họ" là chủ quan/mờ, và trường hợp cả hai đều rơi vào
fallback cùng lúc là hiếm.

---

## 6. Cách verify

- `test_models.py` (repo root) — test kết nối + tham số thật (`max_tokens`,
  `reasoning_effort`) của cả 5 chain, khớp đúng cách mỗi chain được gọi trong
  `orchestrator.py`/`confidence.py`. Dùng tham số khác production (vd `max_tokens` quá
  nhỏ) có thể báo model "chết" giả trong khi thực ra chỉ là thinking model bị cắt cụt —
  xem mục 4.
- `scripts/check_llm_keys.py` — kiểm từng cặp (model, key) sống hay không, bỏ qua chuỗi
  fallback của gateway để lỗi không bị che giấu.
