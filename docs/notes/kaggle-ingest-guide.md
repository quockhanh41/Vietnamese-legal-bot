# Chạy `scripts/ingest.py` trên Kaggle GPU (M1.2)

Mục tiêu: embed + upsert `corpus_100docs.parquet` lên Qdrant Cloud dùng GPU
miễn phí của Kaggle (T4 x2, ~30h/tuần) thay vì CPU máy local (chậm với
BGE-M3). Colab tương tự — ghi chú khác biệt ở cuối file.

Điểm quan trọng nhất: **rớt session (Kaggle hết giờ quota, mất kết nối) →
chạy lại y nguyên lệnh với thêm `--resume`**. Point ID trong Qdrant là
deterministic (`uuid5(document_id:chunk_index)`, xem `compute_point_id` ở
`src/db/qdrant_client.py`) nên chạy lại **không có** `--resume` cũng an
toàn (không tạo trùng), nhưng sẽ tốn thời gian embed lại từ đầu. Checkpoint
SQLite (`src/data/ingest_checkpoint.py`) chỉ tồn tại để tiết kiệm GPU-hour,
không phải để đảm bảo đúng đắn.

## Cell 1 — Clone repo

```python
!git clone https://github.com/<your-username>/vietnamese-legal-bot.git
%cd vietnamese-legal-bot
```

Nếu repo private, dùng Kaggle Secrets cho token GitHub (xem Cell 3 cho cách
đọc Secrets) rồi:
```python
import os
token = os.environ["GITHUB_TOKEN"]
!git clone https://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@github.com/<your-username>/vietnamese-legal-bot.git
```

## Cell 2 — Cài dependencies tối thiểu

Kaggle notebook GPU đã có sẵn torch (CUDA-matched) — **không** cài lại
torch, sẽ dễ vỡ bản CUDA đang có. `requirements-ingest.txt` ở repo root chỉ
liệt kê các package thực sự cần cho `scripts/ingest.py` (không có
streamlit/fastapi/langfuse/reranker...):

```python
%cd vietnamese-legal-bot
!pip install -q -r requirements-ingest.txt
```

Kiểm tra GPU sẵn sàng:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

## Cell 3 — Set QDRANT_URL / QDRANT_API_KEY qua Kaggle Secrets

**Không hardcode credentials vào notebook.** Add secrets qua
`Add-ons → Secrets` trong giao diện Kaggle (tên gợi ý: `QDRANT_URL`,
`QDRANT_API_KEY`), rồi đọc vào biến môi trường — `scripts/ingest.py` đã
gọi `load_dotenv()` + đọc `os.getenv("QDRANT_URL")` /
`os.getenv("QDRANT_API_KEY")` bên trong `LegalVectorStore.__init__`
(`src/db/qdrant_client.py`), nên chỉ cần set biến môi trường là đủ, không
cần sửa code:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["QDRANT_URL"] = secrets.get_secret("QDRANT_URL")
os.environ["QDRANT_API_KEY"] = secrets.get_secret("QDRANT_API_KEY")
```

## Cell 4 — Attach corpus như Kaggle Dataset

Upload `data/raw/corpus_100docs.parquet` một lần thành một Kaggle Dataset
riêng (Kaggle notebook đính kèm dataset ở `/kaggle/input/...` — không cần
re-upload mỗi session, và không phụ thuộc `git clone` mang theo file dữ
liệu lớn / gitignored). Sau khi attach dataset vào notebook (nút
`+ Add Input` bên phải), đường dẫn sẽ có dạng:

```python
DATA_PATH = "/kaggle/input/<your-dataset-slug>/corpus_100docs.parquet"
import os
assert os.path.exists(DATA_PATH), "Dataset chưa được attach vào notebook"
```

## Cell 5 — Chạy ingest

```python
!python scripts/ingest.py \
    --data-path "{DATA_PATH}" \
    --collection <ten_collection> \
    --device cuda \
    --embed-batch-size 128 \
    --resume
```

Ghi chú:
- `--collection <ten_collection>`: đặt tên rõ ràng (vd. `luat_law_ctx_v3`)
  — khớp với collection bạn định dùng ở app. Lần chạy **đầu tiên** không
  cần `--recreate` nếu collection chưa tồn tại (script tự tạo); chỉ dùng
  `--recreate` khi cố ý xoá sạch làm lại từ đầu (và nó cũng xoá luôn
  checkpoint cũ của collection đó — xem `src/data/ingest_checkpoint.py`).
- `--device cuda`: bắt buộc để BGE-M3 chạy trên GPU thay vì CPU.
- `--embed-batch-size 128`: tăng so với mặc định CPU (32) vì GPU rảnh VRAM
  hơn — hạ xuống nếu gặp CUDA OOM (thử 64, 32).
- `--resume`: **luôn thêm** khi chạy lại một collection đã ingest dở —
  script skip các document đã đánh dấu xong trong checkpoint SQLite, in
  ra số document bị skip. Bỏ qua khi chạy hoàn toàn lần đầu (không hại gì
  nếu vẫn để `--resume` từ đầu — checkpoint rỗng thì không skip gì cả).
- Corpus curated (`corpus_100docs.parquet` có cột `sector`) được
  `scripts/ingest.py` tự động nhận diện và chuyển sang "curated mode":
  bỏ filter theo `legal_type`/ngày (giữ nguyên 29 văn bản Nghị định trong
  corpus), chỉ còn guard nội dung tối thiểu — và **in cảnh báo** nếu có
  document nào bị loại (không rơi rớt âm thầm). Nhãn `sector` cũng được
  gắn vào payload từng chunk trong Qdrant.

### Nếu session bị rớt giữa chừng

Chạy lại **đúng lệnh Cell 5 ở trên, y nguyên** (giữ `--resume`, KHÔNG thêm
`--recreate`). Script sẽ:
1. Mở lại checkpoint SQLite tại `data/ingest_progress/<ten_collection>.sqlite`
   (nằm trong container Kaggle — mất khi session kết thúc, xem lưu ý bên
   dưới).
2. In ra số document đã có sẵn trong checkpoint, sẽ bị skip.
3. Tiếp tục embed + upsert phần còn lại.

**Lưu ý quan trọng**: `data/ingest_progress/` nằm trong filesystem tạm
thời của Kaggle container — nếu session bị kill hoàn toàn (không phải
"Restart & Run"), thư mục này biến mất cùng checkpoint. Trong trường hợp
đó, `--resume` sẽ không skip được gì — nhưng vẫn AN TOÀN để chạy lại nhờ
point ID deterministic (chỉ tốn thời gian embed lại, không tạo trùng dữ
liệu trong Qdrant). Nếu muốn checkpoint sống sót qua nhiều session, lưu
`data/ingest_progress/` như một Kaggle Dataset output và tải lại ở đầu mỗi
session (tùy chọn nâng cao, không bắt buộc cho corpus 100 docs).

## Cell 6 — Verify

```python
from src.db.qdrant_client import LegalVectorStore

store = LegalVectorStore(collection_name="<ten_collection>")
info = store.get_collection_info()
print(f"Collection: {info['name']}")
print(f"Points:     {info['points_count']:,}")
```

Đối chiếu với số dòng "Documents" / "Chunks" mà `scripts/ingest.py` in ra ở
cuối lần chạy cuối cùng (tổng qua các lần resume) — số points nên khớp với
tổng số chunks đã ingest.

## Khác biệt trên Google Colab

- Không có `kaggle_secrets` / `UserSecretsClient`. Dùng Colab Secrets
  (biểu tượng chìa khoá bên trái) + `google.colab.userdata`:
  ```python
  from google.colab import userdata
  import os
  os.environ["QDRANT_URL"] = userdata.get("QDRANT_URL")
  os.environ["QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY")
  ```
- Không có cơ chế "attach Dataset" — upload `corpus_100docs.parquet` qua
  Google Drive (`from google.colab import drive; drive.mount(...)`) rồi
  trỏ `--data-path` vào đường dẫn trong Drive, hoặc upload trực tiếp qua
  panel Files (mất khi runtime restart, cần re-upload).
- Colab free tier đổi GPU T4/khác tuỳ phiên, timeout ngắn hơn Kaggle — quy
  trình `--resume` ở trên áp dụng y hệt.
