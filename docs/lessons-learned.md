# 📝 Lessons Learned

> Ghi chép các vấn đề gặp phải trong quá trình phát triển và cách giải quyết.
> Mỗi bài học là một evidence của engineering maturity khi phỏng vấn.

---

## LL-01: Tràn RAM khi xử lý dataset lớn (518K docs, 10GB)

**Ngày**: 2026-07-02  
**Impact**: 🔴 Treo máy — Antigravity process chiếm 35GB RAM, hệ thống không phản hồi  
**Phase**: MVP — Data Pipeline

### Vấn đề

Khi test data pipeline, chạy 3 tiến trình Python song song, mỗi tiến trình dùng `pd.read_parquet()` để load toàn bộ dataset 518K rows vào pandas DataFrame.

```
Tiến trình 1: test chunker     → pd.read_parquet() → ~10GB RAM
Tiến trình 2: test chunker     → pd.read_parquet() → ~10GB RAM
Tiến trình 3: test embedder    → pd.read_parquet() → ~10GB RAM
Hệ thống + Antigravity                               → ~5GB RAM
─────────────────────────────────────────────────────────────
TOTAL                                               → ~35GB RAM → TREO MÁY
```

### Root Cause

1. **`pd.read_parquet()` không hỗ trợ streaming** — luôn load toàn bộ file vào RAM
2. **Parquet file chỉ có 1 row group** — không thể dùng `read_row_group()` để chia nhỏ
3. **Background tasks trong Claude Code** — mỗi lần chạy test là 1 tiến trình Python mới, không share memory
4. **Không giới hạn số lượng tiến trình** — chạy 3 tiến trình đồng thời mà không kiểm tra RAM còn trống

### Giải pháp

#### ✅ Đã áp dụng: Streaming với `pyarrow.dataset`

```python
# ❌ TRƯỚC: Load toàn bộ vào RAM
df = pd.read_parquet("legal_docs_merged.parquet")  # 10GB

# ✅ SAU: Stream từng batch nhỏ
import pyarrow.dataset as ds
dataset = ds.dataset("legal_docs_merged.parquet", format="parquet")
for batch in dataset.to_batches(batch_size=200):   # ~50MB/batch
    df = batch.to_pandas()
    # Xử lý từng batch...
```

**Kết quả**: Memory usage giảm từ 10GB → ~50-200MB, có thể chạy trên laptop 8GB RAM.

#### ✅ Đã áp dụng: Script exploration riêng biệt

Tách `explore_data.py` (chỉ scan statistics, không build pipeline) khỏi `ingest.py` (full pipeline). Exploration dùng `pyarrow` scan từng cột riêng để lấy counts mà không cần load toàn bộ data.

#### ✅ Đã áp dụng: Filter sớm (push-down)

Thay vì load toàn bộ rồi filter trong pandas, filter được áp dụng ngay khi đọc từng batch:

```python
def filter_row(row: dict) -> bool:
    """Filter trên từng row riêng lẻ — O(1) memory."""
    if row.get("legal_type") not in LEGAL_TYPES:
        return False
    if not row.get("content") or len(row["content"]) < 500:
        return False
    # ...
    return True
```

### Nguyên tắc rút ra

| # | Nguyên tắc | Áp dụng |
|---|-----------|---------|
| 1 | **Không bao giờ load toàn bộ dataset vào RAM** nếu dataset > 50% RAM khả dụng | Dùng streaming/batching |
| 2 | **Chỉ chạy 1 tiến trình data-intensive tại 1 thời điểm** | Tránh background tasks chồng lấn |
| 3 | **Test với `--limit N` trước khi chạy full pipeline** | `--limit 100` → verify → `--limit 5000` → full |
| 4 | **Monitor memory usage trong quá trình chạy dài** | Dùng `tqdm` + log memory mỗi N batches |
| 5 | **Biết rõ cấu trúc file parquet trước khi đọc** | Kiểm tra `num_row_groups`, schema, compression |

### Cách trình bày trong phỏng vấn

> *"Khi xây dựng data pipeline cho 518K văn bản pháp luật (~3.6GB parquet), tôi gặp vấn đề tràn RAM vì pandas load toàn bộ dataset. Tôi đã chuyển sang dùng pyarrow.dataset streaming API, giảm memory footprint từ 10GB xuống ~100MB. Điều này cho phép pipeline chạy được trên laptop cá nhân mà không cần cloud VM. Tôi cũng implement cơ chế filter sớm (push-down filtering) để loại bỏ documents không liên quan ngay khi đọc, trước khi vào các bước nặng như chunking và embedding."*

### Prevention cho các phase sau

- [ ] **BGE-M3 model load**: Model ~2GB, chỉ load 1 lần, share giữa các requests
- [ ] **Embedding batch**: Giới hạn batch size = 32 để tránh OOM trên GPU/CPU
- [ ] **Qdrant upsert batch**: Max 100 points/batch để tránh gRPC message quá lớn
- [ ] **Streamlit caching**: Dùng `@st.cache_resource` cho heavy objects (embedder, vector store)
- [ ] **Production deployment**: Render free tier có 512MB RAM → cần tối ưu memory từ đầu

---

## LL-02: Dependency version conflicts với langchain ecosystem

**Ngày**: 2026-07-02  
**Impact**: 🟡 ModuleNotFoundError — `langchain.text_splitter` không tồn tại trong phiên bản mới  
**Phase**: MVP — Data Pipeline

### Vấn đề

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
# ModuleNotFoundError: No module named 'langchain.text_splitter'
```

### Root Cause

LangChain >= 0.3 đã tách `text_splitter` thành package riêng `langchain-text-splitters`.

### Giải pháp

```python
# Sửa import
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Thêm vào pyproject.toml
"langchain-text-splitters>=0.3.0",
```

### Nguyên tắc

- Luôn check breaking changes trong CHANGELOG khi upgrade LangChain
- Prefer `langchain-text-splitters`, `langchain-core` thay vì import từ `langchain` monolith
- Pin version trong production, không dùng `>=` một cách mù quáng

---

## LL-03: FlagEmbedding BGE-M3 trả về `lexical_weights` thay vì `sparse_vecs`

**Ngày**: 2026-07-02
**Impact**: 🟡 KeyError — pipeline crash khi encode documents
**Phase**: MVP — Data Pipeline

### Vấn đề

```python
# Code gốc trong embedder.py
dense, sparse = embedder.encode_documents(texts)
# KeyError: 'sparse_vecs'
```

### Root Cause

Documentation HuggingFace và các blog hướng dẫn dùng key `sparse_vecs`, nhưng FlagEmbedding library thực tế trả về:

```python
>>> output.keys()
['dense_vecs', 'lexical_weights', 'colbert_vecs']
```

BGE-M3 dùng cơ chế **token-level lexical importance scoring**, nên key chính xác là `lexical_weights`. Đây là tình huống điển hình của "docs không sync với code".

### Giải pháp

```python
# ❌ Trước
return output["dense_vecs"], output["sparse_vecs"]

# ✅ Sau
return output["dense_vecs"], output["lexical_weights"]
```

### Nguyên tắc rút ra

- **Test API với 1 sample input trước khi code toàn bộ pipeline** — `output.keys()` mất 10 giây nhưng tiết kiệm hàng giờ debug
- **Đừng tin documentation 100%** — library open-source thay đổi nhanh hơn docs
- **Tên field trong library ≠ tên trong paper** — lexical_weights vs sparse_vecs là 1 ví dụ

### Cách trình bày trong phỏng vấn

> *"Khi integrate BGE-M3 embedding model, tôi gặp KeyError vì documentation ghi key là `sparse_vecs` nhưng library thực tế trả về `lexical_weights`. Tôi đã test trực tiếp API với 1 sample để xác nhận schema thực tế. Đây là bài học về việc không nên mù quáng tin documentation—luôn verify bằng code."*

---

## LL-04: Qdrant 1.18.x thay thế `search()` bằng `query_points()` — breaking API change

**Ngày**: 2026-07-02
**Impact**: 🔴 AttributeError — toàn bộ 3 search methods (dense, sparse, hybrid) bị hỏng
**Phase**: MVP — Retrieval Pipeline

### Vấn đề

```python
# Code gốc dùng Qdrant SDK cũ (tutorials phổ biến trên mạng)
results = self.client.search(
    collection_name="vietnamese_laws",
    query_vector=("dense", query_vector),
    limit=10,
)
# AttributeError: 'QdrantClient' object has no attribute 'search'
```

### Root Cause

Qdrant 1.18.x (phiên bản latest trên Docker Hub) đã refactor toàn bộ search API:

| Cũ (≤ 1.7) | Mới (≥ 1.8) |
|---|---|
| `client.search(...)` | `client.query_points(...)` |
| `query_vector=(name, vector)` | `query=vector, using='dense'` |
| RRF tự implement client-side | RRF native server-side qua `FusionQuery` |
| `search_groups(...)` | `query_points_groups(...)` |

Tutorials và blog posts trên mạng đều dùng API cũ → code theo thì vỡ.

### Giải pháp

```python
# ❌ Cũ (Qdrant < 1.8)
results = self.client.search(
    collection_name="vietnamese_laws",
    query_vector=("dense", query_vector),
    limit=10,
)

# ✅ Mới (Qdrant 1.18.x)
response = self.client.query_points(
    collection_name="vietnamese_laws",
    query=query_vector,
    using="dense",
    limit=10,
    with_payload=True,
)
results = response.points  # List[ScoredPoint]
```

**Hybrid search cũng thay đổi** — từ 2 lần search + RRF client-side → 1 lần `query_points` với `prefetch` + `FusionQuery` server-side:

```python
# ✅ Hybrid với Qdrant 1.18.x native RRF
response = self.client.query_points(
    collection_name="vietnamese_laws",
    prefetch=[
        models.Prefetch(query=models.NearestQuery(nearest=dense_vec), using='dense', limit=10),
        models.Prefetch(query=SparseVector(indices=..., values=...), using='sparse', limit=10),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=5,
    with_payload=True,
)
```

### Nguyên tắc rút ra

- **Pin version infrastructure components** — `qdrant/qdrant:v1.7.4` thay vì `latest`; `qdrant-client>=1.7,<2.0`
- **Check `dir(client)` trước khi code** — 30 giây inspect API surface tránh hàng giờ debug
- **Đọc Migration Guide trước khi upgrade** — mỗi major version đều có breaking changes document
- **Server-side fusion > client-side** — RRF native nhanh hơn, ít code hơn, ít bug hơn

### Cách trình bày trong phỏng vấn

> *"Khi deploy Qdrant cho RAG pipeline, tôi gặp tình huống API search bị xóa hoàn toàn trong version mới. Thay vì downgrade, tôi đã migrate toàn bộ search layer sang `query_points` API mới, đồng thời tận dụng được server-side RRF fusion — vừa fix bug vừa cải thiện performance. Bài học là luôn pin version cho infrastructure components và đọc migration guide trước khi upgrade."*

---

## LL-05: `FusionQuery.prefetch` không tồn tại — nhầm nesting giữa query và prefetch

**Ngày**: 2026-07-02
**Impact**: 🟡 ValidationError — hybrid search không hoạt động dù đã sửa LL-04
**Phase**: MVP — Retrieval Pipeline

### Vấn đề

```python
# ❌ Sai: nhét prefetch vào trong FusionQuery
fusion = models.FusionQuery(
    fusion=models.Fusion.RRF,
    prefetch=[  # ← ValidationError: Extra inputs are not permitted
        models.Prefetch(query=..., using='dense'),
        models.Prefetch(query=..., using='sparse'),
    ],
)
```

### Root Cause

Nhầm lẫn giữa 2 level của API:
- **`FusionQuery`** = chỉ là **fusion strategy** (`RRF`) — 1 field duy nhất
- **`prefetch`** = **top-level parameter** của `query_points()`, định nghĩa các nhánh tìm kiếm song song

Pydantic model `FusionQuery` có `model_config = ConfigDict(extra='forbid')` nên field lạ bị reject ngay lập tức.

### Giải pháp

```python
# ✅ Đúng: prefetch ở top-level query_points(), FusionQuery chỉ chứa strategy
result = c.query_points(
    collection_name="vietnamese_laws",
    prefetch=[                                          # ← top-level param
        models.Prefetch(query=..., using='dense', limit=10),
        models.Prefetch(query=..., using='sparse', limit=10),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),  # ← chỉ strategy
    limit=5,
)
```

### Nguyên tắc rút ra

- **`inspect.signature()` là công cụ debug số 1** — cho biết chính xác function/method nhận những param gì
- **Pydantic `extra='forbid'`** là default trong nhiều library mới → field sai tên là crash ngay
- **Đọc API theo cấu trúc phẳng** — prefetch và query là sibling params, không phải parent-child

### Cách trình bày trong phỏng vấn

> *"Khi làm việc với API mới, tôi dùng `inspect.signature()` để kiểm tra chính xác function signature thay vì đoán. Điều này giúp tôi phát hiện `prefetch` là top-level param của `query_points`, không phải field của `FusionQuery` — tiết kiệm được nhiều thời gian debug kiểu 'tại sao nó không nhận field này?'."*

---

## LL-06: `CollectionInfo.vectors_count` bị đổi thành `indexed_vectors_count`

**Ngày**: 2026-07-02
**Impact**: 🟢 AttributeError — chỉ ảnh hưởng stats display, không blocking pipeline
**Phase**: MVP — Retrieval Pipeline

### Vấn đề

```python
info = client.get_collection("vietnamese_laws")
print(info.vectors_count)  # AttributeError
```

### Root Cause

Qdrant 1.18.x restructured `CollectionInfo`:
- `vectors_count` → `indexed_vectors_count` (vì collection giờ có thể có nhiều named vectors: dense + sparse)

### Giải pháp

```python
# ❌ Trước
return {"vectors_count": info.vectors_count}

# ✅ Sau
return {
    "points_count": info.points_count,
    "indexed_vectors_count": info.indexed_vectors_count,
}
```

### Nguyên tắc rút ra

- **Stats/display code nên fail gracefully** — không nên crash toàn bộ pipeline vì thiếu 1 trường metadata
- **Dùng `getattr()` với fallback** cho code cần tương thích nhiều version: `getattr(info, 'vectors_count', None) or info.points_count`

---

## LL-07: Model Groq `qwen-2.5-72b-32768` không còn tồn tại — dynamic model discovery

**Ngày**: 2026-07-02
**Impact**: 🔴 NotFoundError — LLM gateway fallback hết 3 models đều fail
**Phase**: MVP — LLM Integration

### Vấn đề

```python
MODEL_CHAIN = [
    "groq/qwen-2.5-72b-32768",   # ← NotFoundError: model does not exist
    "openai/gpt-4o-mini",         # ← không có OpenAI key → fail
    "openai/gpt-4o",              # ← không có OpenAI key → fail
]
# → Tất cả models đều fail → RuntimeError
```

### Root Cause

1. **Groq đã xóa `qwen-2.5-72b`** và thay bằng Qwen 3 series
2. **Model chain phụ thuộc vào OpenAI fallback** — nhưng user chỉ có Groq key (free)
3. **Hardcode model names** thay vì discovery động từ API

### Giải pháp

**Bước 1**: Dùng Groq REST API để list models đang available:

```python
import requests
resp = requests.get('https://api.groq.com/openai/v1/models',
    headers={'Authorization': f'Bearer {api_key}'})
for m in resp.json()['data']:
    print(m['id'])
# Output: qwen/qwen3-32b, qwen/qwen3.6-27b, llama-3.3-70b-versatile, ...
```

**Bước 2**: Cập nhật model chain với các model thực tế:

```python
# ✅ Tất cả đều free qua Groq — không cần OpenAI key
MODEL_CHAIN = [
    "groq/qwen/qwen3-32b",           # Primary: Qwen 3 32B, Vietnamese tốt nhất
    "groq/qwen/qwen3.6-27b",         # Fallback 1: Qwen 3.6 27B
    "groq/llama-3.3-70b-versatile",  # Fallback 2: Llama 70B
]
```

**Lưu ý**: Qwen 3 models có thinking mode (`<think>...</think>` tag trong output). Cần cleanup hoặc chấp nhận trong prompt engineering.

### Nguyên tắc rút ra

- **Không hardcode model names** — với cloud provider, models thay đổi hàng tháng
- **Test model availability bằng REST API** trước khi code — 1 dòng `curl`/`requests.get` xác nhận
- **Fallback chain nên độc lập provider** — nếu tất cả models cùng 1 provider, 1 lỗi auth là chết hết
- **Có ít nhất 1 model free trong chain** để đảm bảo dev/test không bị blocker bởi billing

### Cách trình bày trong phỏng vấn

> *"Khi build LLM gateway với LiteLLM, tôi phát hiện model list trên Groq đã thay đổi hoàn toàn so với documentation. Thay vì hardcode, tôi đã viết dynamic model discovery — gọi REST API của provider để lấy danh sách models thực tế, chọn model phù hợp nhất cho tiếng Việt. Điều này giúp hệ thống không bị vỡ khi provider thay đổi model catalog."*

---

## LL-08: `.env` không được load tự động — FastAPI cần `python-dotenv`

**Ngày**: 2026-07-02
**Impact**: 🟡 API server khởi động thành công nhưng LLM gateway luôn fail vì thiếu API key
**Phase**: MVP — API Server

### Vấn đề

User thêm `GROQ_API_KEY` vào `.env` file, server restart, nhưng LLM gateway vẫn báo:

```
OpenAIException - Missing credentials. Please pass an api_key
```

### Root Cause

Python **không tự động load `.env` files**. Cần gọi `load_dotenv()` từ thư viện `python-dotenv`:

```python
# ❌ Không có dòng này → os.getenv("GROQ_API_KEY") trả về None
from dotenv import load_dotenv
load_dotenv()  # Phải gọi explicit, lý tưởng nhất là ở entry point (main.py)
```

Nhiều framework như Django auto-load `.env`, nhưng FastAPI/Starlette thì không.

### Giải pháp

```python
# Thêm vào đầu src/api/main.py
from dotenv import load_dotenv
load_dotenv()  # Load .env trước khi bất kỳ import nào dùng os.getenv()
```

### Nguyên tắc rút ra

- **Luôn thêm `load_dotenv()` vào entry point** — `main.py`, `cli.py`, hoặc `__init__.py` của app
- **Test env vars ngay sau khi load** — thêm assert hoặc warning nếu key quan trọng bị thiếu
- **Đừng assume framework auto-loads `.env`** — mỗi framework có cơ chế khác nhau
- **Backup: dùng `os.getenv("KEY", "default")`** cho mọi biến môi trường để có fallback rõ ràng

### Cách trình bày trong phỏng vấn

> *"Một bug tinh tế tôi gặp là API server khởi động thành công nhưng mọi LLM call đều fail vì thiếu API key. Hóa ra FastAPI không auto-load `.env` như Django. Tôi đã thêm `python-dotenv` vào entry point và viết health check verify các biến môi trường quan trọng ngay khi startup. Đây là loại bug 'nó chạy mà không chạy' rất dễ bỏ qua nếu không có monitoring tốt."*

---

## LL-09: `import os` ở cuối file gây `NameError` tại runtime

**Ngày**: 2026-07-02
**Impact**: 🔴 NameError — Streamlit page crash ngay khi load
**Phase**: MVP — UI (Day 9-14)

### Vấn đề

```python
# src/ui/pages/chat.py (trước fix)
API_BASE = os.environ.get("API_URL", "http://localhost:8000")  # Dòng 14 — NameError!

# ... 160 dòng code ...

import os  # Dòng 178 ← QUÁ MUỘN!
```

### Root Cause

1. **Thói quen "import ở cuối file"** — có thể do copy-paste hoặc auto-format tool xếp sai
2. **Không test từng page riêng lẻ** — nếu chưa click vào Chat page, bug này ẩn hoàn toàn
3. **Streamlit import dynamics** — `app.py` import `chat.py` qua `from src.ui.pages.chat import render_chat_page`, lỗi xảy ra ngay lúc import, không phải lúc gọi hàm

### Giải pháp

```python
# ✅ Sau fix: tất cả imports ở đầu file
import json
import os        # ← ĐÃ CHUYỂN LÊN ĐẦU
import re
import time
from typing import List, Dict

import streamlit as st
import requests

API_BASE = os.environ.get("API_URL", "http://localhost:8000")  # ✅ Hoạt động
```

### Nguyên tắc rút ra

- **Tất cả `import` phải ở đầu file** — PEP 8 quy định rõ, không có ngoại lệ
- **Test import từng module ngay sau khi viết** — `python -c "from src.ui.pages.chat import render_chat_page"` mất 2 giây
- **Linter sẽ bắt lỗi này ngay** — `ruff check` hoặc `pylint` sẽ cảnh báo `E402: module level import not at top of file`
- **Luôn click test UI sau khi code** — bug này ẩn hoàn toàn cho đến khi user click vào page đó

### Cách trình bày trong phỏng vấn

> *"Một bug tưởng nhỏ nhưng gây crash toàn bộ UI: `import os` để ở cuối file trong khi `os.environ.get` được gọi ở đầu file. Bug này chỉ lộ khi user click vào page — trước đó mọi thứ trông bình thường. Từ đó tôi luôn chạy linter trước khi commit và test import từng module riêng lẻ."*

---

## LL-10: Qwen 3 reasoning tokens leak vào output — model behavior thay đổi khi upgrade

**Ngày**: 2026-07-02
**Impact**: 🟡 Output nhiễu — streaming trả về chuỗi suy nghĩ nội bộ của model
**Phase**: MVP — LLM Integration (Day 9-14)

### Vấn đề

Khi migrate từ Qwen 2.5 lên Qwen 3 models, output chứa reasoning tokens:

```
<think>
Here's a thinking process:
1. Analyze the user's question...
2. The legal basis is...
</think>

Thủ tục cấp giấy phép xây dựng được quy định tại...
```

### Root Cause

1. **Qwen 3 là reasoning model** — sinh `<think>` tokens trước answer, không thể tắt qua API parameter
2. **Model upgrade = behavior change** — khi chuyển từ qwen2.5-72b lên qwen3-32b, output format thay đổi hoàn toàn

### Giải pháp

**Non-streaming** — regex post-processing sau khi nhận full response:

```python
def _clean_thinking(self, text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<think>[\s\S]*$', '', text, flags=re.IGNORECASE)
    return text.strip()
```

**Streaming** — cần stateful buffer filter (đã thiết kế, chưa implement):

```python
# Ý tưởng: track trạng thái in_thinking, buffer token vào buffer
# Chỉ yield khi gặp </think> hoặc buffer không nằm trong <think> block
in_thinking = False
for token in stream:
    if "<think>" in token: in_thinking = True; continue
    if "</think>" in token: in_thinking = False; continue
    if not in_thinking: yield token
```

### Nguyên tắc rút ra

- **Test actual model output trước khi integrate** — gọi 1 query đơn giản để biết output format
- **Reasoning models cần output post-processing** — không giống completion models thông thường
- **Streaming và non-streaming cần filter khác nhau** — regex cho non-streaming, state machine cho streaming
- **Chuẩn bị fallback model khác loại** — Llama 3.3 không có `<think>` tags, có thể dùng nếu Qwen 3 gây vấn đề

### Cách trình bày trong phỏng vấn

> *"Khi migrate LLM từ Qwen 2.5 lên Qwen 3, tôi phát hiện model mới là reasoning model — nó sinh ra reasoning tokens trong tag `<think>` trước khi trả lời. Với non-streaming, tôi dùng regex để strip. Với streaming, cần stateful buffer filter để không hiển thị suy nghĩ nội bộ cho user. Đây là bài học về việc model upgrade không chỉ là đổi tên — behavior có thể thay đổi hoàn toàn."*

---

## LL-11: SSE streaming parsing giả định `event:` và `data:` trên 2 dòng riêng biệt

**Ngày**: 2026-07-02
**Impact**: 🟡 Mất citations/data — streaming UI bỏ lỡ metadata quan trọng
**Phase**: MVP — API + UI Integration (Day 9-14)

### Vấn đề

Code SSE parsing ban đầu trong `chat.py` chỉ xử lý `data:` line, bỏ qua `event:` line:

```python
# ❌ Trước: chỉ parse data, không track event type
for line in resp.iter_lines():
    if line.startswith("data: "):
        data = json.loads(line[6:])
        if "token" in data:
            full_answer += data["token"]
        elif "full_answer" in data:
            # done event
```

Hậu quả: citations event bị parse nhầm thành token (vì data field không có key "token"), dẫn đến mất citations.

### Root Cause

1. **SSE protocol có 2 fields: `event:` và `data:`** — cần track event type từ `event:` line trước đó
2. **Code gốc assume tất cả events đều là token** — không phân biệt `event: citations` vs `event: token` vs `event: done`
3. **Không test streaming với sample output thật** — chỉ test với happy path "có token"

### Giải pháp

```python
# ✅ Sau fix: track event_type từ 'event:' line
event_type = "token"  # default
for line in resp.iter_lines():
    line_str = line.decode("utf-8")
    
    if line_str.startswith("event: "):
        event_type = line_str[7:]  # Lưu event type
        continue
    
    if line_str.startswith("data: "):
        data = json.loads(line_str[6:])
        
        if event_type == "citations":
            citations = data           # List[Dict]
        elif event_type == "token":
            full_answer += data["token"]
        elif event_type == "done":
            full_answer = data["full_answer"]
            confidence = data["confidence"]
            latency_ms = data["latency_ms"]
```

### Nguyên tắc rút ra

- **SSE parsing cần state machine** — `event:` line set state, `data:` line xử lý theo state
- **Test streaming với nhiều event types** — không chỉ test `token`, phải test cả `citations` và `done`
- **Default event type = "message"** theo SSE spec — nhưng nên explicit set default cho safety
- **Log unknown event types** — giúp phát hiện protocol mismatch sớm

### Cách trình bày trong phỏng vấn

> *"Khi implement SSE streaming cho RAG chatbot, tôi gặp bug mất citations vì parser chỉ xử lý `data:` line mà không track `event:` type trước đó. SSE protocol dùng `event:` để phân biệt loại message — bỏ qua nó là mất context. Tôi đã refactor parser thành state machine: `event:` set type, `data:` xử lý theo type. Bài học là phải hiểu protocol trước khi parse."*

---

## 📊 Template cho các bài học sau

```markdown
## LL-XX: [Tiêu đề ngắn gọn]

**Ngày**: YYYY-MM-DD
**Impact**: 🔴/🟡/🟢 (Cao/Trung bình/Thấp)
**Phase**: [Phase trong plan]

### Vấn đề
[Mô tả ngắn gọn — cái gì đã xảy ra?]

### Root Cause
[Tại sao nó xảy ra? Phân tích kỹ thuật]

### Giải pháp
[Cách đã fix — kèm code before/after nếu có]

### Nguyên tắc rút ra
- [Nguyên tắc 1]
- [Nguyên tắc 2]

### Cách trình bày trong phỏng vấn
> "[Quote ngắn gọn để dùng khi được hỏi]"
```
