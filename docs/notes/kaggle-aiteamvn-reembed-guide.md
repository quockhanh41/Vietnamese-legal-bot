# Re-embed 600-doc với AITeamVN (mixed-hybrid) trên Kaggle GPU

Guide tự-chứa để dựng collection **`luat_law_ctx_v4_aiteamvn`** trên Qdrant Cloud:
embed corpus 600-doc bằng **`MixedHybridEmbedder`** (dense = `AITeamVN/Vietnamese_Embedding`,
sparse = `BAAI/bge-m3`) trên GPU miễn phí của Kaggle (T4 x2), rồi so với mốc M2.

- Spec: `docs/superpowers/specs/2026-07-15-aiteamvn-internal-comparison-design.md`
- Code chạy ở branch **`m3-aiteamvn-comparison`** (chưa merge vào main).
- Sau khi Kaggle xong, phần còn lại (backfill `is_in_effect` + Tier-1 + baseline + Tier-2)
  chạy **local** — xem `docs/notes/aiteamvn-internal-comparison.md`.

**Rớt session → chạy lại y nguyên Cell 5, thêm `--resume`.** Point ID deterministic
(`uuid5(document_id:chunk_index)`) nên chạy lại không tạo trùng; checkpoint SQLite chỉ để
tiết kiệm GPU-hour, không phải để đảm bảo đúng đắn.

---

## Chuẩn bị notebook (làm 1 lần)

1. Tạo notebook mới trên Kaggle, **Settings → Accelerator = GPU T4 x2**.
2. **Settings → Internet = On** (bắt buộc: cần cho `git clone`, tải model HF, và Qdrant Cloud).
3. **Add-ons → Secrets**: thêm `QDRANT_URL` và `QDRANT_API_KEY` (trỏ đúng Qdrant Cloud
   đang chứa `luat_law_ctx_v4`). Nếu repo private, thêm `GITHUB_TOKEN`.
4. **+ Add Input**: attach dataset chứa `corpus_600docs.parquet` (upload 1 lần thành
   Kaggle Dataset riêng — xem Cell 4).

---

## Cell 1 — Clone repo (branch `m3-aiteamvn-comparison`)

```python
!git clone -b m3-aiteamvn-comparison https://github.com/quockhanh41/Vietnamese-legal-bot.git
%cd Vietnamese-legal-bot
```

Nếu repo private, đọc token từ Secrets rồi clone:
```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone -b m3-aiteamvn-comparison https://{token}@github.com/quockhanh41/Vietnamese-legal-bot.git
%cd Vietnamese-legal-bot
```

Xác nhận đúng branch + có file mixed embedder:
```python
!git rev-parse --abbrev-ref HEAD          # -> m3-aiteamvn-comparison
!test -f src/data/mixed_embedder.py && echo "mixed_embedder OK"
```

> **Nếu bạn đã clone repo từ trước**: `!git pull` để lấy bản vá bộ nhớ mới của
> `MixedHybridEmbedder` (chia 2 GPU + fp16 + cap seq 2048 — chống CUDA OOM).
> Kiểm tra có bản vá: `!grep -q "cuda:1" src/data/mixed_embedder.py && echo PATCHED`.

## Cell 2 — Cài dependencies

Kaggle GPU đã có sẵn torch (CUDA-matched) — **KHÔNG** cài lại torch.
`requirements-ingest.txt` kéo `FlagEmbedding`, và FlagEmbedding **tự kéo
`sentence-transformers`** (mà `MixedHybridEmbedder` cần).

```python
!pip install -q -r requirements-ingest.txt
```

Kiểm tra GPU + 2 model loader import được (fallback cài ST nếu thiếu):
```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
try:
    from sentence_transformers import SentenceTransformer
    from FlagEmbedding import BGEM3FlagModel
    print("embedder imports OK")
except ImportError:
    !pip install -q sentence-transformers
    from sentence_transformers import SentenceTransformer
    print("installed sentence-transformers")
```

## Cell 3 — Set QDRANT_URL / QDRANT_API_KEY qua Secrets

**Không hardcode credentials.** `scripts/ingest.py` gọi `load_dotenv()` + đọc
`os.getenv("QDRANT_URL")` / `os.getenv("QDRANT_API_KEY")` trong
`LegalVectorStore.__init__` — chỉ cần set biến môi trường:

```python
from kaggle_secrets import UserSecretsClient
import os
secrets = UserSecretsClient()
os.environ["QDRANT_URL"] = secrets.get_secret("QDRANT_URL")
os.environ["QDRANT_API_KEY"] = secrets.get_secret("QDRANT_API_KEY")
```

## Cell 4 — Trỏ tới `corpus_600docs.parquet`

Sau khi attach dataset (nút `+ Add Input`), đường dẫn có dạng `/kaggle/input/<slug>/...`:
```python
import os
DATA_PATH = "/kaggle/input/<your-dataset-slug>/corpus_600docs.parquet"
assert os.path.exists(DATA_PATH), "Dataset chưa attach vào notebook"
print("data:", DATA_PATH)
```

> ⚠️ **KHÔNG chạy cell xoá collection cũ.** Collection `luat_law_ctx_v4` (mốc M2) và
> v3/v2 (baseline) phải giữ nguyên. Ta chỉ TẠO MỚI `luat_law_ctx_v4_aiteamvn`.

## Cell 5 — Chạy ingest (mixed embedder → collection mới)

```python
!python scripts/ingest.py \
    --data-path "{DATA_PATH}" \
    --collection luat_law_ctx_v4_aiteamvn \
    --embedder mixed \
    --embed-model AITeamVN/Vietnamese_Embedding \
    --quantized \
    --device cuda \
    --embed-batch-size 128 \
    --resume
```

Ghi chú từng cờ:
- `--collection luat_law_ctx_v4_aiteamvn`: **collection MỚI** (song song, không đụng v4).
  Lần chạy đầu script tự tạo — không cần `--recreate`.
- `--embedder mixed` + `--embed-model AITeamVN/Vietnamese_Embedding`: dense từ AITeamVN
  (SentenceTransformer, 1024-dim, normalize, **fp16**, cap seq 2048), sparse từ `bge-m3`.
  `MixedHybridEmbedder` **tự chia 2 GPU** khi có T4 x2: dense → `cuda:0`, bge-m3 → `cuda:1`
  (2 model ~560M không vừa 1 T4 16GB nếu chung GPU).
- `--quantized`: **BẮT BUỘC ngay lần đầu** — int8 + fp32 on-disk chỉ áp lúc TẠO collection
  (khớp config v4 → apples-to-apples). Quên rồi thêm sau là phải `--recreate`.
- `--device cuda`: bắt buộc (code tự phân bổ cuda:0/cuda:1). **Cần Accelerator = T4 x2** để
  chia GPU; nếu chỉ 1 GPU, cả 2 model dồn 1 card → phải hạ batch mạnh.
- `--embed-batch-size 32`: đã đủ thoáng với bản vá (T4 x2). Score attention/lớp ≈
  `batch × 16 × 2048² × 2B`; batch 32 ≈ 4.3GB/lớp, vừa GPU0 sau khi tách bge-m3 sang GPU1.
  **CUDA OOM → hạ 16 rồi 8**; nếu chỉ 1 GPU, bắt đầu từ 8.
- `--resume`: luôn để — chạy lại sau khi rớt session sẽ skip document đã xong (checkpoint
  `data/ingest_progress/luat_law_ctx_v4_aiteamvn.sqlite`).
- Corpus curated (`corpus_600docs.parquet` có cột `sector`) → ingest tự nhận "curated mode":
  giữ mọi Nghị định/Thông tư, gắn `sector` vào payload, chỉ guard nội dung tối thiểu.

**Nếu rớt giữa chừng**: chạy lại **đúng lệnh trên, y nguyên** (giữ `--resume`, KHÔNG
`--recreate`). Checkpoint sống trong container Kaggle — nếu session bị kill hẳn thì mất,
lúc đó `--resume` không skip được gì nhưng vẫn AN TOÀN chạy lại (point ID deterministic,
chỉ tốn thời gian embed lại).

## Cell 6 — Verify

```python
from src.db.qdrant_client import LegalVectorStore
info = LegalVectorStore(collection_name="luat_law_ctx_v4_aiteamvn").get_collection_info()
print("Collection:", info["name"])
print("Points:    ", f"{info['points_count']:,}")   # kỳ vọng ~60,958
```

Đối chiếu với dòng "Chunks" mà `ingest.py` in ở cuối (tổng qua các lần resume) — số points
nên khớp tổng chunks.

## ❌ KHÔNG chạy enrichment trên Kaggle

Bỏ qua bước enrichment/`run_enrichment.py`. Kaggle chỉ có JSON seed **v3 (100 docs)** →
enrichment ở đây sẽ ra metadata SAI cho corpus 600-doc. `is_in_effect` được backfill
**local** (nơi `store.db` có đủ 605 docs) bằng `scripts/backfill_is_in_effect.py` — đây là
việc phía local, cần cho parity routed + Tier-2 với v4.

## Xong — báo lại gì

Chỉ cần gửi lại **số points ở Cell 6** (kỳ vọng ~60,958). Sau đó, phần local (không cần GPU):
1. `backfill_is_in_effect.py --collection luat_law_ctx_v4_aiteamvn` (parity)
2. Tier-1 (mixed) + re-đo baseline v4 @1/@3
3. Tier-2 full-swap (`--reranker AITeamVN/Vietnamese_Reranker`, tốn Groq)
4. Điền bảng + verdict ở `docs/notes/aiteamvn-internal-comparison.md`.

---

## Khác biệt trên Google Colab

- Secrets: dùng `google.colab.userdata` thay `kaggle_secrets`:
  ```python
  from google.colab import userdata
  import os
  os.environ["QDRANT_URL"] = userdata.get("QDRANT_URL")
  os.environ["QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY")
  ```
- Không có "Add Input": upload `corpus_600docs.parquet` qua Drive
  (`from google.colab import drive; drive.mount('/content/drive')`) rồi trỏ `--data-path`.
- GPU T4 tuỳ phiên, timeout ngắn hơn — quy trình `--resume` áp dụng y hệt.
