# Hợp nhất đường stream và non-stream — thiết kế

**Ngày:** 2026-08-18
**Trạng thái:** chờ review
**Nguồn:** tổng kiểm tra codebase 2026-08-18 (mục #1, #2, #3, #6 của báo cáo audit)

---

## 1. Vấn đề nền

`AdaptiveRAGPipeline` có hai đường đi song song, mỗi đường tự cài lại toàn bộ
logic định tuyến:

- `query()` (`src/rag/orchestrator.py:405`) → `_lookup_pipeline` / `_simple_pipeline` /
  `_medium_pipeline` / `_complex_pipeline`
- `query_stream()` (`:955`) → ~150 dòng sao chép inline, xen kẽ các event `__STEP__`

Comment tại `:702` đã ghi nhận sự trùng lặp này là cố ý ("query_stream keeps its
own inline copy because it interleaves `__STEP__` progress events").

**Điều chưa được ghi nhận, và là lý do thật sự phải sửa:**

| | Đường nào |
|---|---|
| Mọi số đo của dự án (B0, calibration, citation accuracy) | `query()` — `scripts/eval.py:364`, docstring `:15` ghi rõ *"non-streaming path"* |
| Mọi lưu lượng người dùng thật | `query_stream()` — `src/api/main.py:352`, `src/ui/app_gradio.py:112` |

Hai tập hợp này **rời nhau**. Hệ quả: bất kỳ chỗ nào hai đường lệch nhau đều vô
hình với mọi gate mà dự án có. Đó không phải rủi ro giả định — nó đã xảy ra ở #3.

## 2. Ba lỗi trong phạm vi

### #3 — `RERANK_MAX_K` chỉ có tác dụng ở 1/4 điểm gọi

| Điểm gọi | Route | Giá trị |
|---|---|---|
| `orchestrator.py:662` | `_medium_pipeline` (non-stream) | `RERANK_MAX_K` |
| `orchestrator.py:1036` | `query_stream` medium | cứng `15` |
| `orchestrator.py:728` | `complex_retrieve` | cứng `20` |
| `orchestrator.py:1091` | `query_stream` complex | cứng `20` |

Knob này thêm ở M5 để hạ P95 reranker trên CPU HF Space (M2.6 đo P95 **7.58s**
@pool 100). Đặt `RERANK_MAX_K=8` hiện **không đổi một byte nào** trên đường
streaming — tức đúng đường đang chạy trên Space. Thêm `:668` `results[:15]` là
magic number trùng vai trò.

### #2 — Cạn quota trên đường stream: chuỗi lỗi bị cache và ghi vào lịch sử

`src/llm/gateway.py:325` — khi mọi model trong chain thất bại, `generate_stream`
**yield một chuỗi text** thay vì raise:

```python
yield "[Lỗi] Không thể kết nối đến LLM. Vui lòng thử lại sau."
```

Vì không có exception nào được ném, dây chuyền ở `src/api/main.py` diễn ra như
với một câu trả lời hợp lệ:

1. `except Exception` (`:386`) không chạy → `is_exhaustion` không chạy → cờ
   `degraded` không bật → người dùng **không** thấy `DEGRADED_MSG`
2. `:406` `store.append_turn(...)` ghi chuỗi lỗi vào lịch sử hội thoại → đầu độc
   `condense_query` ở mọi lượt sau trong phiên
3. `:408` `cache.set(...)` lưu chuỗi lỗi vào semantic cache → phát lại cho câu
   hỏi tương tự về sau

`src/rag/degraded.py:14` chỉ nhận diện `RuntimeError("All models failed")` — tức
chỉ tín hiệu từ `_generate_impl` (đường non-stream). Đường stream hở hoàn toàn.
Lỗi tương tự ở `gateway.py:319` (đứt giữa chừng).

*Yếu tố may rủi hiện có:* nếu `_post_process` chấm chuỗi lỗi dưới ngưỡng 0.700
thì `refused=True` và cache bị bỏ qua. Đó là ngẫu nhiên, không phải thiết kế.

### #1 — Streamlit mặc định chạy `NaiveRAGPipeline`

`src/ui/pages/chat.py:116` → `st.checkbox("Adaptive RAG (v1.0+)", value=False)`

`NaiveRAGPipeline` không có: scope gate, confidence gate/refusal (ngưỡng 0.700
hiệu chuẩn suốt M0→M3), verifier, reranker, citation-lookup route, cross-ref
expansion (M3.5), comparison⊗supersession (M3.4). `_parse_response:326` trả
thẳng `confidence="unverified"`.

Cộng với §1: người dùng Streamlit mặc định nhận `NaiveRAGPipeline.query_stream()`
— vừa không adaptive, vừa không phải đường nào từng được đo.

## 3. Quyết định đã chốt với người dùng

1. **Non-stream là chuẩn.** Nơi nào hai đường lệch, `query()` thắng — vì đó là
   đường mọi eval/calibration đã chạy qua. Rủi ro dồn về UI, không dồn vào gate.
2. **Phạm vi gồm cả ba lỗi** (#6/#3 hợp nhất, #2 gateway, #1 mặc định adaptive)
   — cả ba nằm trên cùng một đường đi, test một lần, đo một lần.
3. **Phương án A** (generator dùng chung), loại B và C — xem §5.

## 4. Diện tích thay đổi hành vi thực tế

Đối chiếu từng điểm lệch ở **cấu hình mặc định**:

| Điểm lệch | Mặc định | Đổi hành vi? |
|---|---|---|
| medium `max_k` | `RERANK_MAX_K = getenv(…, "15")` = 15, stream cứng 15 | ❌ chỉ lệch khi env var được set |
| medium `[:15]` sau rerank | `top_k` đã ≤ 15 | ❌ no-op |
| simple `max_k` | 15 / 15 | ❌ |
| complex `max_k` | 20 / 20 | ❌ |
| out-of-scope | `_out_of_scope_response()` vs yield inline | ❌ nội dung tương đương |
| strip `<think>` | `_parse_response:316` vs `re.sub:1140` | ❌ cùng regex |
| **complex: context vào `_post_process`** | non-stream **chưa nén** (`:760` ghi đè `:747`), stream **đã nén** | ✅ **khác biệt thật duy nhất** |

**Kết luận:** hợp nhất theo chuẩn non-stream chỉ đổi đúng một thứ quan sát được —
route complex sẽ verify/chấm confidence trên context chưa nén. Chỉ cần đo lại
phần đó, không phải toàn bộ B0.

Việc giữ hành vi non-stream ở đây còn đúng về mặt lý: context nén là output LLM,
lossy — verify trên đó có thể báo oan "không có căn cứ" khi chính bước nén đã
cắt mất câu chống lưng. Ở bản hợp nhất, hành vi này phải **cố ý và có tên**
(biến riêng), không còn là tác dụng phụ của một dòng `_build_context` thừa.

## 5. Thiết kế — phương án A: generator dùng chung

### Hình dạng

```python
@dataclass
class RouteResult:
    results: List[SearchResult]
    annotations: Dict[str, Dict]
    rerank_scores: Optional[List[float]]

def _route(self, question) -> Generator[str, None, Tuple[str, bool, bool]]:
    """yield các event __STEP__; return (complexity, in_scope, is_comparison)."""

def _retrieve(self, question, complexity, is_comparison
              ) -> Generator[str, None, RouteResult]:
    """yield các event __STEP__; return RouteResult."""
```

Cả hai là generator có giá trị trả về (PEP 380).

`_route` gộp cả hai nhánh vào đầu hiện nay:

- Câu hỏi parse được thành trích dẫn (`parse_citation` khác `None`) → yield event
  `Citation Lookup`, `return ("lookup", True, False)` — **không** gọi classifier,
  đúng như cả `query():415` lẫn `query_stream():966` hiện nay
- Ngược lại → `_classify_and_scope`, yield event `Classification`,
  `return (complexity, in_scope, is_comparison)`

Trường hợp `in_scope=False` **không** vào `_retrieve`: caller xử lý riêng
(`query()` gọi `_out_of_scope_response()`, `query_stream()` yield câu trả lời +
`__VERIFICATION__`) vì hai bên trả về hai kiểu khác nhau và không có retrieval
nào chạy. Đây là chỗ duy nhất còn phân nhánh theo stream/non-stream.

`_retrieve` với `complexity == "lookup"` tự lấy `self._get_citation_lookup()`,
không cần caller truyền vào.

- `query_stream()` dùng `result = yield from self._retrieve(...)` → event tiến
  trình vẫn phát **real-time**, đúng như hiện nay
- `query()` dùng helper `_drain()` bỏ qua event, chi phí bằng 0:

```python
def _drain(gen):
    """Chạy hết một generator có giá trị trả về, bỏ qua mọi event."""
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
```

### Vì sao là generator chứ không phải callback

Một callback `emit(...)` không thể `yield` — nên các event `__STEP__` sẽ phải
gom vào list và phát sau khi `_retrieve` trả về. Retrieval chính là phần chậm
nhất, nên gom lại đồng nghĩa mọi bước tiến trình nhảy ra cùng lúc ở cuối: mất
đúng giá trị mà `__STEP__` tồn tại để tạo ra. `yield from` giữ nguyên UX.

### Chữ ký công khai giữ nguyên

`query()` và `query_stream()` không đổi signature. `main.py`, `app_gradio.py`,
`chat_turn.py`, `scripts/eval.py` không phải sửa.

### Sau khi hợp nhất

- `query()` = `_drain(_route)` → `_drain(_retrieve)` → build context → generate →
  `_post_process` → `RAGResponse`
- `query_stream()` = `yield from _route` → `yield from _retrieve` → build context →
  `generate_stream` → `_post_process` → `__VERIFICATION__`

Bốn method `_simple_pipeline` / `_medium_pipeline` / `_complex_pipeline` /
`_lookup_pipeline` co lại thành phần "sinh + hoàn thiện" dùng chung; toàn bộ
phần truy hồi chuyển vào `_retrieve`. `complex_retrieve()` (public, dùng bởi
eval harness M3.5) giữ nguyên như một wrapper mỏng quanh `_retrieve`.

### Dọn kèm (thuộc phạm vi vì cùng dòng code)

- Bỏ cặp `_build_context`/`_build_citations` gọi lần hai ở cả 4 route (#7)
- `_post_process:564` nhận `annotations` (#8)
- Bỏ `[:15]` cứng ở `:668`, dùng `RERANK_MAX_K` (#3)
- Bỏ `import statistics` thừa (`:13`) và `import re` cục bộ (`:315`)

### Phương án bị loại

**B — `query()` gọi lại `query_stream()` rồi gom token.** Dedup tối đa, nhưng
đường non-stream mất `generate_with_model` (tên model thật, cần để
`ConfidenceScorer` không tự chấm bài mình), và eval sẽ chạy qua generator — phá
đúng quyết định "non-stream là chuẩn" ở §3.1.

**C — class per-route (`SimpleRoute`/`MediumRoute`/…).** Sạch hơn về OO nhưng
`orchestrator.py` đang là hub: `condense.py`, `semantic_cache.py`, `chat_turn.py`
đều import ngược từ nó. Diff lớn, rủi ro import churn, và chỉ có 4 route ổn định
— vi phạm YAGNI (CLAUDE.md §2). Để ngỏ cho lần sau nếu số route tăng.

## 6. Thiết kế — #2: gateway báo lỗi bằng exception

`generate_stream` đổi từ "yield chuỗi lỗi" sang raise:

- **Cạn toàn bộ chain:** `raise RuntimeError(f"All models failed. Last error: {last_error}")`
  — **cùng dạng tín hiệu** với `_generate_impl:231`, nên `is_exhaustion()`
  (`degraded.py:14`) nhận diện được ngay, không phải sửa file đó.
- **Đứt giữa chừng** (đã có token tới client): exception riêng
  `StreamInterrupted`, vì ngữ nghĩa khác — không thể retry, không thể rút lại
  token đã gửi. Caller phát thông báo gián đoạn rồi **bỏ qua cache + history**.

### Vì sao chỉ cần sửa một file

`src/api/main.py:386-396` và `src/ui/app_gradio.py:129-135` **đã có sẵn** nhánh
`is_exhaustion(e)` → phát `DEGRADED_MSG`, `return` trước khi persist/cache. Các
nhánh này hiện là code chết vì không có gì raise. `src/ui/pages/chat.py:322` đã
đọc cờ `degraded` từ event `done`. Toàn mạch đã nối sẵn từ đầu tới cuối — sửa
gateway là nó sống lại.

`StreamInterrupted` cần thêm nhánh xử lý mới ở cả hai front-end (hiện chưa có).

## 7. Thiết kế — #1: adaptive làm mặc định

- `src/ui/pages/chat.py:116` — `value=False` → `value=True`
- `src/api/main.py:43` — `use_adaptive: bool = False` → `True`

**Đánh đổi phải nói rõ:** adaptive tốn thêm ≥1 LLM call mỗi câu (classify), tối
đa +4 ở route complex (rewrite, multi-query, HyDE, compress), cộng ~60s nạp
reranker ở lần đầu. Lý do `False` ban đầu nhiều khả năng chính là tiết kiệm
quota Groq.

**Lập luận ngược lại, và là lý do chốt True:** `app_gradio.py:72` — bản demo
public đang sống trên HF Space — chạy adaptive **vô điều kiện**. Chi phí đó đang
được trả rồi. Để Streamlit mặc định naive nghĩa là hai front-end cùng repo trả
lời với hai chất lượng khác nhau, và cái mặc định là cái chưa từng được đo.

Đổi `main.py:43` là thay đổi hợp đồng API. Kiểm tra: `test_api_chat_memory.py:86`
và `test_answer_format_fixes.py:124` đều truyền `use_adaptive` tường minh nên
không vỡ; `scripts/smoke_test.py:43` vốn đã mặc định `True`.

## 8. Lưới an toàn — test parity

Gốc rễ khiến ba lỗi này sống sót qua 533 test là **không test nào so hai đường
với nhau**. Không có phần này thì #3 mọc lại nguyên vẹn ở lần sửa kế tiếp.

| # | Test | Chặn được gì | Trên code hiện tại |
|---|---|---|---|
| 1 | Với mỗi route, `query()` và `query_stream()` gọi `_dynamic_top_k` cùng `max_k` | Lệch tham số pool giữa hai đường | 🟢 **xanh** — ở mặc định hai bên trùng giá trị (§4) |
| 2 | Set `RERANK_MAX_K=8` → **cả hai** đường đổi theo | #3 tái phát | 🔴 **đỏ** — chỉ non-stream đổi |
| 3 | Gateway cạn chain → raise, và `is_exhaustion()` bắt được | #2 tái phát | 🔴 **đỏ** — hiện yield string, không raise |
| 4 | Stream cạn quota → `cache.set` **không** được gọi, `store.append_turn` **không** ghi chuỗi lỗi | Đầu độc cache/history | 🔴 **đỏ** |
| 5 | `query()` và `query_stream()` trả cùng tập `point_id` cho cùng input (retrieval + LLM mock) | Mọi lệch định tuyến còn lại | 🟢 **xanh** — điểm lệch duy nhất (§4) không chạm `point_id` |

Hai loại test khác vai, đừng lẫn:

- **Test 2, 3, 4 là test tái hiện lỗi** — phải **đỏ trước, xanh sau**. Nếu chúng
  xanh ngay từ đầu thì test viết sai, không phải lỗi đã hết.
- **Test 1 và 5 là chốt hãm cho chính đợt refactor** — phải **xanh cả trước lẫn
  sau**. Chúng không tố cáo lỗi nào đang có; việc của chúng là bắt lỗi mà thao
  tác dời 150 dòng code có thể vô tình tạo ra. Chạy chúng trên `main` trước khi
  đụng bất cứ gì, để biết chắc chúng thật sự đang kiểm cái mình tưởng.

Test 5 là cái quan trọng nhất về lâu dài: nó biến "hai đường phải giống nhau" từ
một quy ước bằng miệng thành một khẳng định máy kiểm được.

## 9. Rủi ro và việc đo lại

| Rủi ro | Mức | Xử lý |
|---|---|---|
| Route complex đổi context đưa vào verify (§4) | Trung bình | Đo lại Tier-2 route complex trước/sau; gate giữ `rr ≥ 0.7619`, `fr ≤ 0.0972` |
| Adaptive mặc định làm cạn quota Groq nhanh hơn | Trung bình | Đã có key rotation đa tài khoản; `is_exhaustion` sau fix #2 sẽ degrade tử tế thay vì cache lỗi |
| `yield from` làm vỡ context OTel của Langfuse trên đường stream | Thấp | Đường stream **vốn đã** vỡ context qua `iterate_in_threadpool` (M4.3 ghi nhận, rủi ro #1) — không tệ thêm |
| Refactor 150 dòng lõi làm vỡ hành vi im lặng | Trung bình | Test 1 + 5 chạy xanh trên `main` trước khi đụng code, rồi phải giữ xanh sau |

**Thứ tự bắt buộc:**

1. Viết test 1 và 5 → chạy trên `main`, phải **xanh**. Đây là ảnh chụp hành vi
   hiện tại; nếu chúng đỏ thì hiểu biết về code trong spec này sai ở đâu đó và
   phải dừng lại xem xét trước khi refactor.
2. Viết test 2, 3, 4 → phải **đỏ**. Nếu xanh thì test sai.
3. Refactor + sửa gateway + đổi mặc định. Đích: cả 5 xanh, 533 test cũ vẫn xanh.
4. Đo lại Tier-2 route complex (§9 hàng 1).

## 10. Ngoài phạm vi

Các mục sau đã xuất hiện trong audit nhưng **không** thuộc spec này:

- #4 test cho `AmendmentConsolidator` (741 dòng, 0 test) — task riêng, ưu tiên kế tiếp
- #5 thêm job pytest vào CI — task riêng, rẻ, nên làm ngay sau
- #9–#14 (rò message exception, endpoint không rate-limit, race lazy-load, cache
  không TTL, `get_phase2_status` O(N), ngữ nghĩa `is_in_effect`) — thật nhưng
  chưa cắn ai ở quy mô demo hiện tại
- #15 circular import `chunker.py` ↔ `chunking/` — độc lập hoàn toàn
- #16–#17 docs drift + `.env.example` — thuộc gate M6
- Mọi mục 🟢 — dọn một lượt sau cùng
