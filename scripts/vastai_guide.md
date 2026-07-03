# Chạy Ingestion trên vast.ai GPU

## Bước 1: Thuê GPU instance trên vast.ai

1. Vào [vast.ai](https://vast.ai/console/create/)
2. Chọn instance:
   - **GPU**: RTX 3080/3090/4070 hoặc A10 (chỉ cần 1 GPU, VRAM ≥ 8GB)
   - **RAM**: ≥ 16GB
   - **Disk**: ≥ 50GB (data file ~10GB, model ~2GB, vectors ~5GB)
   - **Price**: ~$0.20-0.50/h — tìm instance rẻ
   - **Image**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` (hoặc bất kỳ PyTorch 2.x + CUDA image nào)
3. Sau khi instance running, copy SSH command

## Bước 2: Upload code + data lên instance

```bash
# Từ máy local của bạn:
# Upload project (không gửi .venv và data để tiết kiệm băng thông)
rsync -avz --exclude '.venv' --exclude 'data/raw/*.parquet' \
  ./ user@instance_ip:/root/vietnamese-legal-bot/

# Upload riêng data file (10GB, nén lại sẽ nhanh hơn)
scp data/raw/legal_docs_merged.parquet user@instance_ip:/root/vietnamese-legal-bot/data/raw/

# Upload .env (có chứa Qdrant Cloud API key)
scp .env user@instance_ip:/root/vietnamese-legal-bot/
```

> **Mẹo**: Nếu instance của bạn có public URL, có thể tải data trực tiếp từ URL thay vì upload.

## Bước 3: SSH vào instance và setup

```bash
ssh user@instance_ip

cd /root/vietnamese-legal-bot

# Chạy script setup tự động
bash scripts/vastai_setup.sh

# Hoặc làm thủ công:
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pyarrow python-dotenv click tqdm FlagEmbedding
```

## Bước 4: Chạy ingestion

```bash
source .venv/bin/activate

# Test trước với 500 docs (đảm bảo mọi thứ hoạt động):
python scripts/ingest.py --limit 500 --device cuda

# Nếu test OK, chạy full (312K docs, ~4-6 tiếng với GPU):
python scripts/ingest.py --recreate --device cuda

# Nếu bị gián đoạn, resume (KHÔNG dùng --recreate):
python scripts/ingest.py --device cuda
```

## Bước 5: Verify kết quả

```bash
source .venv/bin/activate
python -c "
from dotenv import load_dotenv
load_dotenv()
from src.db.qdrant_client import LegalVectorStore
store = LegalVectorStore()
info = store.get_collection_info()
print(f'Collection: {info[\"name\"]}')
print(f'Vectors: {info[\"points_count\"]:,}')
"
```

## Monitoring

Trong quá trình chạy, script sẽ in progress mỗi 500 docs:
```
  500 docs → 7,234 chunks
  1000 docs → 14,891 chunks
  ...
```

## Troubleshooting

| Vấn đề | Fix |
|---|---|
| Out of VRAM | Giảm `--embed-batch-size 16` hoặc `8` |
| Connection timeout tới Qdrant Cloud | Retry tự động, không cần làm gì |
| Hết disk | Instance vast.ai thường có 50-100GB, đủ cho data + model |
