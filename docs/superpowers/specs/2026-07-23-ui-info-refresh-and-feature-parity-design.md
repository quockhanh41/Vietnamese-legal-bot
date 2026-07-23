# UI info refresh + feature parity (Documents/Đồ thị/Hiệu lực) trên cả 2 UI

> Ngày: 2026-07-23 · Trạng thái: design đã duyệt, chờ implementation plan.

## Vì sao

Sau pivot M5 (Gradio+ZeroGPU là live public demo, Streamlit là self-hosted
Docker path), rà lại 2 UI phát hiện:

1. **Thông tin sai** trong Streamlit: sidebar (`app.py`) và Eval dashboard
   (`eval_dashboard.py`) hiển thị `"Model: Qwen 3 32B (Groq)"` — model này đã
   bị Groq retire (404 từ 2026-07-19, xem comment `gateway.py:31-34`). Model
   thật đang chạy đầu chuỗi fallback là `groq/llama-3.3-70b-versatile`.
   `"Embedding: BGE-M3 (1024-dim)"` cũng sai — production dùng embedder lai
   (`DEFAULT_COLLECTION_NAME = "luat_law_ctx_v4_aiteamvn"`): dense từ
   **AITeamVN/Vietnamese_Embedding**, sparse từ **BGE-M3**. Reranker
   (**AITeamVN/Vietnamese_Reranker**) không được hiển thị ở đâu cả.
2. **1 bug thật** trong Documents page: `documents.py:44` đọc key
   `total_documents` nhưng `/api/documents` trả về key `total` — ô "Tìm thấy
   N văn bản" luôn hiện 0 dù có kết quả.
3. **3 endpoint backend có thật nhưng chưa có UI nào**: `/api/timeline/
   effective`, `/api/article/history`, `/api/graph/stats` — tra cứu luật còn
   hiệu lực theo ngày, lịch sử sửa đổi 1 điều, thống kê đồ thị trích dẫn
   chéo. Đều đã chạy thật qua `MetadataStore`/`EnhancedRetriever`.
4. **Gradio Space (live public demo) chỉ có 1 ô chat** — không có Documents,
   không có thông tin hệ thống, không có tra cứu nâng cao. Streamlit (self-
   hosted) có nhưng Gradio thì không, và Gradio giờ là bản người dùng thật sẽ
   thấy trước tiên.

## Quyết định khung

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| UI mục tiêu | Cả Streamlit lẫn Gradio | User chọn "Cả hai" — Streamlit đang có info sai cần sửa; Gradio là live demo đang thiếu tính năng. |
| Phạm vi "cập nhật thông tin" | Chỉ chuỗi hiển thị trong UI, không đụng README/ARCHITECTURE.md/LLM_MODEL_STRATEGY.md | User chọn hẹp phạm vi — các doc kiến trúc để dịp khác. |
| Nguồn sự thật cho tên LLM model hiển thị | Đọc `LLMGateway.MODEL_CHAIN[0]` (class attribute, không cần instantiate gateway) thay vì hardcode lại | Đây chính xác là nguyên nhân bug lần này (2 chỗ hardcode, không ai sửa khi model đổi) — đọc từ nguồn duy nhất triệt tiêu lớp lỗi này. |
| Nguồn sự thật cho Embedding/Reranker | Hardcode chuỗi đã sửa đúng (không có constant trung tâm sẵn) | Đây là quyết định hiếm khi đổi (lần gần nhất là quyết định GO của M3, một sự kiện lớn có track record riêng) — không đáng thêm constant mới chỉ để tránh 1 lần sửa tay. |
| Kiến trúc thêm tính năng vào Gradio | `gr.Blocks()` + `gr.Tabs()` bọc quanh `ChatInterface` hiện có | Gradio hỗ trợ nhúng `ChatInterface` vào trong 1 tab của `Blocks`; giữ nguyên `handle_turn`/`@spaces.GPU` không đổi. |
| Documents/Đồ thị trên Gradio gọi thế nào | Gọi thẳng `vector_store`/`metadata_store` in-process (đã có sẵn ở module scope `app_gradio.py`) | Đúng pattern M5 pivot đã chọn (no HTTP hop); các thao tác này thuần CPU/SQLite/Qdrant — không cần `@spaces.GPU`, không tốn quota ZeroGPU. |
| Trang mới ở Streamlit gọi thế nào | Qua HTTP tới FastAPI (`requests.get(f"{API_BASE}/api/...")`) | Đúng pattern hiện có của `chat.py`/`documents.py`/`eval_dashboard.py` — Streamlit luôn nói chuyện qua API, không import RAG modules trực tiếp. |
| Trực quan hoá đồ thị | Không vẽ graph (không network-x-vis/d3) — chỉ bảng/text | YAGNI: không ai yêu cầu visualization, endpoint trả JSON phẳng là đủ để hiện bảng. |
| Format tên model hiển thị | Hiện nguyên `LLMGateway.MODEL_CHAIN[0]` (vd `"groq/llama-3.3-70b-versatile"`), không map sang tên thân thiện | Thêm bảng map provider/model→tên đẹp tái tạo đúng lớp lỗi đang sửa (phải nhớ cập nhật map mỗi lần đổi model) — id gốc + caption rõ ràng đã đủ minh bạch cho project kỹ thuật. Đã cân nhắc theo góp ý review (`nhan-xet.md` mục B) và quyết định giữ nguyên. |
| Chọn văn bản trong dropdown (Khối B / doc_id) | `format_func`/`(label, value)` tuple — không parse chuỗi hiển thị | Tránh lớp lỗi "tách int từ string label" hoàn toàn thay vì tách rồi validate; API sẵn có của cả Streamlit lẫn Gradio hỗ trợ trực tiếp. |

## Kiến trúc / luồng dữ liệu

```
Streamlit (self-hosted, qua HTTP → FastAPI)          Gradio Space (live, in-process)
┌────────────────────────────────┐                    ┌────────────────────────────────┐
│ app.py (sidebar)                │                    │ app_gradio.py                   │
│  Model  ← LLMGateway.MODEL_CHAIN[0]                  │  gr.Blocks()                    │
│  Embedding/Reranker ← hardcode  │                    │   ├─ Tab Chat: ChatInterface    │
│                                  │                    │   │   (không đổi, handle_turn)  │
│ pages/documents.py               │                    │   ├─ Tab Documents              │
│  fix key `total`                │                    │   │   vector_store.list_documents()
│  fix error message               │                    │   ├─ Tab Thống kê                │
│                                  │                    │   │   MODEL_CHAIN[0] + hardcode  │
│ pages/graph.py (MỚI)             │                    │   │   embed/rerank + get_collection_info()
│  ├─ hiệu lực theo ngày           │                    │   └─ Tab Đồ thị & Hiệu lực (MỚI) │
│  │   GET /api/timeline/effective │                    │       metadata_store.get_effective_laws()
│  ├─ lịch sử sửa đổi điều luật    │                    │       retriever.get_article_history()
│  │   GET /api/article/history    │                    │       metadata_store.get_graph()
│  └─ thống kê đồ thị              │                    │                                  │
│      GET /api/graph/stats        │                    └────────────────────────────────┘
│                                  │
│ pages/eval_dashboard.py          │
│  fix "Qwen 3 32B" → MODEL_CHAIN[0]│
└────────────────────────────────┘
```

## Thành phần chi tiết

### 1. `src/ui/app.py` — sửa sidebar (không đổi cấu trúc nav ngoài việc thêm 1 mục)

```python
from src.llm.gateway import LLMGateway
...
st.sidebar.caption(f"Model: {LLMGateway.MODEL_CHAIN[0]} (auto-fallback)")
st.sidebar.caption("Embedding: AITeamVN/Vietnamese_Embedding (dense) + BGE-M3 (sparse)")
st.sidebar.caption("Reranker: AITeamVN/Vietnamese_Reranker")
```

Thêm mục nav thứ 4: `"🕸️ Đồ thị & Hiệu lực"` → `render_graph_page()`.

### 2. `src/ui/pages/documents.py` — fix bug + error message

- `data.get('total_documents', 0)` → `data.get('total', 0)`.
- Đổi `except Exception as e: st.warning(f"Chức năng đang được phát triển. {e}")`
  → `st.error(f"Không thể kết nối đến backend: {e}")` (đúng bản chất lỗi —
  đây là exception kết nối/timeout, không phải tính năng chưa xong; nhánh
  "không có kết quả" đã được xử lý trong khối `try` không qua exception).

### 3. `src/ui/pages/eval_dashboard.py` — sửa 2 chỗ hardcode

- `cols[2].metric("🔍 LLM Model", "Qwen 3 32B (Groq)")` →
  `cols[2].metric("🔍 LLM Model", LLMGateway.MODEL_CHAIN[0])`.
- Caption `"Based on local testing with Qwen 3 32B on Groq."` → thay tên
  model bằng `LLMGateway.MODEL_CHAIN[0]`, giữ nguyên phần "estimated/local
  testing" (không đổi số liệu latency — ngoài phạm vi).

### 4. `src/ui/pages/graph.py` (MỚI) — trang "Đồ thị & Hiệu lực"

Theo đúng pattern các page khác (`API_BASE = os.environ.get("API_URL", ...)`,
gọi `requests.get`, `st.spinner`/`st.error` khi lỗi). 3 khối, độc lập nhau:

**Khối A — Luật có hiệu lực tại 1 ngày**
- `st.date_input` (mặc định hôm nay) → `GET /api/timeline/effective?at_date=YYYY-MM-DD`
- Hiện bảng (`law_name`, `effective_date`) + `count`.

**Khối B — Lịch sử sửa đổi 1 điều luật**
- `st.text_input` từ khoá tên văn bản → `GET /api/documents?keyword=...` để
  lấy danh sách văn bản khớp (tái dùng logic có sẵn ở `documents.py`, không
  gọi thẳng vector store) → `st.selectbox(options=documents,
  format_func=lambda d: d["title"])` — **truyền thẳng list dict làm options,
  không parse chuỗi hiển thị**; giá trị trả về là dict gốc nên đọc
  `selected["document_id"]` trực tiếp, không cần tách số ra khỏi label kiểu
  `"101 - Luật Đất đai 2024"` → `st.text_input` số điều (placeholder "Điều
  5") → nút tra cứu → `GET /api/article/history?doc_id=...&article=...` →
  hiện kết quả JSON dạng bảng/markdown (timeline hiệu lực + danh sách luật
  sửa đổi nếu có).
- Lỗi `{"error": "Document not found", ...}` từ endpoint → `st.warning`.

**Khối C — Thống kê đồ thị trích dẫn chéo**
- `GET /api/graph/stats` → hiện tổng nodes/edges + bảng theo `node_types`/
  `edge_types` (2 cột `st.columns`, mỗi cột 1 bảng nhỏ).

Mỗi khối bọc riêng `try/except` — lỗi 1 khối không chặn 2 khối còn lại (nhất
quán với cách `documents.py` tách khối Filters/Results khỏi khối Stats).

### 5. `src/ui/app_gradio.py` — chuyển `ChatInterface` trần → `Blocks` + `Tabs`

Cấu trúc mới, thay đoạn cuối file (từ `demo = gr.ChatInterface(...)` trở
xuống); phần module-scope loading models (dòng 47-70) và `handle_turn` giữ
nguyên 100%:

```python
chat_interface = gr.ChatInterface(
    fn=handle_turn, type="messages",
    title="Vietnamese Legal Knowledge Bot",
    description=(...),  # giữ nguyên
)

def search_documents(legal_type, keyword):
    """Gọi thẳng vector_store — thuần CPU/Qdrant, không cần @spaces.GPU."""
    lt = None if legal_type == "Tất cả" else legal_type
    result = vector_store.list_documents(legal_type=lt, keyword=keyword or None,
                                          page=1, page_size=50)
    df = pd.DataFrame(result["documents"]).rename(columns={
        "title": "Tên văn bản", "legal_type": "Loại văn bản",
        "issuance_date": "Ngày ban hành",
    })
    return f"Tìm thấy {result['total']} văn bản", df

def document_choices(keyword):
    """(label, value) tuples cho gr.Dropdown — value = document_id gốc (int),
    không parse chuỗi hiển thị (cùng lý do st.selectbox format_func ở graph.py)."""
    result = vector_store.list_documents(keyword=keyword or None, page=1, page_size=50)
    return [(d["title"], d["document_id"]) for d in result["documents"]]

def system_info():
    info = vector_store.get_collection_info()
    return (
        f"**Model:** {LLMGateway.MODEL_CHAIN[0]} (auto-fallback)\n\n"
        f"**Embedding:** AITeamVN/Vietnamese_Embedding (dense) + BGE-M3 (sparse)\n\n"
        f"**Reranker:** AITeamVN/Vietnamese_Reranker\n\n"
        f"**Collection:** {info['name']} ({info['points_count']:,} chunks)"
    )

def effective_laws(at_date_str):
    """at_date_str: chuỗi YYYY-MM-DD từ gr.Textbox tự do (mặc định hôm nay).
    Không dùng st.date_input tương đương ở Gradio -> input có thể sai định
    dạng, khác Streamlit (calendar widget luôn hợp lệ) nên cần try/except ở
    đây mà graph.py phía Streamlit không cần."""
    try:
        at_date = date.fromisoformat(at_date_str) if at_date_str else None
    except ValueError:
        return f"Định dạng ngày không hợp lệ: '{at_date_str}' (cần YYYY-MM-DD)"
    laws = metadata_store.get_effective_laws(at_date)
    return pd.DataFrame([{"Luật": l["law_name"], "Hiệu lực từ": l["effective_date"]} for l in laws])

def article_history(doc_id, article):
    """doc_id: int, lấy từ gr.Dropdown value (xem document_choices) — không
    phải parse từ label hiển thị."""
    return retriever.get_article_history(int(doc_id), article)  # dict → gr.JSON

def graph_stats():
    graph = metadata_store.get_graph()
    ...  # đếm node_types/edge_types như main.py:/api/graph/stats

with gr.Blocks(title="Vietnamese Legal Knowledge Bot") as demo:
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chat_interface.render()
        with gr.Tab("📁 Documents"):
            # legal_type dropdown (ALL_LEGAL_TYPES), keyword textbox, nút tìm kiếm, dataframe
        with gr.Tab("📊 Thống kê"):
            gr.Markdown(system_info())
        with gr.Tab("🕸️ Đồ thị & Hiệu lực"):
            # 3 khối tương tự graph.py: effective_laws(textbox ngày);
            # gr.Dropdown(choices=document_choices(keyword)) + article textbox -> article_history;
            # graph_stats() hiển thị 1 lần khi mở tab (không cần input)
```

Import thêm ở đầu file: `pandas as pd`, `from datetime import date`,
`from src.data.constants import ALL_LEGAL_TYPES`, `from src.pipeline.retriever
import EnhancedRetriever`. `retriever = EnhancedRetriever(vector_store=
vector_store, metadata_store=metadata_store, embedder=embedder)` khởi tạo
**1 lần ở module scope**, cạnh `pipeline`/`citation_lookup` hiện có (dòng
47-70) — không tạo lại đối tượng mỗi lần bấm nút.

**Ràng buộc quan trọng (giữ nguyên từ M5 pivot):** không thêm bất kỳ lệnh gọi
model/GPU nào ngoài `handle_turn` đã có `@spaces.GPU`. Mọi hàm Documents/
Thống kê/Đồ thị chỉ chạm `vector_store`, `metadata_store`, `retriever` — các
đối tượng này không giữ tensor trên GPU, an toàn gọi ngoài `@spaces.GPU`.

## Xử lý lỗi

- Streamlit: theo đúng pattern hiện có mỗi trang — bọc `try/except`, lỗi kết
  nối → `st.error`/`st.warning` với thông báo tiếng Việt rõ ràng, không để
  exception rơi ra ngoài làm crash trang.
- Gradio: các hàm Documents/Thống kê/Đồ thị chạy in-process, không có
  network hop nên khả năng lỗi thấp hơn nhiều so với call Groq/embedding;
  vẫn bọc `try/except` trả về thông báo lỗi dạng string/DataFrame rỗng thay
  vì để Gradio hiện traceback thô cho người dùng cuối.
- `article_history`: input rỗng hoặc doc_id không tồn tại → hiển thị
  `{"error": ...}` đã có sẵn từ `get_article_history`, không cần logic mới.
- `effective_laws` (chỉ nhánh Gradio): `gr.Textbox` tự do cho ngày có thể
  nhận chuỗi sai định dạng (vd "abc") → `date.fromisoformat` raise
  `ValueError` → bọc try/except, trả thông báo lỗi rõ ràng thay vì crash.
  **Không áp dụng cho Streamlit** — `st.date_input` là calendar widget,
  luôn trả về `datetime.date` hợp lệ, không có đường nào để chuỗi sai định
  dạng lọt tới `.isoformat()`.

## Testing

- **Streamlit fixes (Phần 1-3, 5)**: unit test cho phần logic thuần
  (`backend_state`, style tương tự `chat.py` đã có) không áp dụng trực tiếp
  vì đây là các hàm `render_*` gọi `st.*` — theo đúng convention hiện tại của
  repo, các trang Streamlit **không có unit test riêng** (không thấy test
  file cho `documents.py`/`eval_dashboard.py`/`app.py` trong `tests/`); xác
  minh bằng chạy `streamlit run src/ui/app.py` thủ công + `make api` chạy
  song song, tự tay kiểm tra 4 nav item.
- **`gradio_helpers.py`-style pure functions mới** (nếu tách logic
  `search_documents`/`system_info`/`graph_stats` thành hàm thuần không đụng
  model — theo đúng lý do `gradio_helpers.py` tồn tại, "no model/network/GPU
  imports on purpose ... can be unit-tested"): viết unit test cho phần build
  DataFrame/format string, mock `vector_store`/`metadata_store`.
- **`make test`** (suite pytest hiện có, 522 pass tính đến M5 revert) phải
  vẫn pass — không đổi hành vi RAG core.
- Xác minh thủ công (không tự động hoá được — cần chạy `app_gradio.py`, cần
  GPU/HF runtime thật cho ZeroGPU, không chạy được local đầy đủ): mở
  `demo.launch()` local, click qua 4 tab, xác nhận Documents/Thống kê/Đồ thị
  không kích hoạt `@spaces.GPU` (Chat tab vẫn hoạt động bình thường).

## Không làm (ngoài phạm vi, theo lựa chọn của user)

- Không sửa README.md, ARCHITECTURE.md, LLM_MODEL_STRATEGY.md.
- Không đổi số liệu latency ước tính trong `eval_dashboard.py`.
- Không thêm trực quan hoá đồ thị (network graph visual).
- Không đụng `handle_turn`, `@spaces.GPU`, session/history logic của Gradio.
- Không thêm tính năng mới ngoài Documents/Thống kê/Đồ thị & Hiệu lực (không
  mở rộng sang admin enrichment endpoints — các endpoint đó cần `ADMIN_KEY`,
  không phù hợp UI công khai).
