#!/bin/bash
# Setup script for running ingestion on vast.ai GPU instance
# Usage: bash scripts/vastai_setup.sh

set -e

echo "============================================"
echo "Setting up Vietnamese Legal Bot on vast.ai"
echo "============================================"

# 1. Install system dependencies
echo "[1/5] Installing system packages..."
apt-get update -qq && apt-get install -y -qq python3-pip python3-venv git curl 2>/dev/null || true

# 2. Create virtual env
echo "[2/5] Setting up Python venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# 3. Install project dependencies
echo "[3/5] Installing Python dependencies..."
pip install -e . -q
pip install pyarrow python-dotenv click tqdm FlagEmbedding -q

# 4. Verify GPU
echo "[4/5] Checking GPU..."
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# 5. Check data file
echo "[5/5] Verifying data file..."
if [ -f "data/raw/legal_docs_merged.parquet" ]; then
    SIZE=$(du -h data/raw/legal_docs_merged.parquet | cut -f1)
    echo "✅ Data file found: $SIZE"
else
    echo "⚠️  WARNING: data/raw/legal_docs_merged.parquet NOT FOUND!"
    echo "   Upload it first: scp data/raw/legal_docs_merged.parquet user@instance:/path/to/project/data/raw/"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Before running, verify your .env has Qdrant Cloud credentials:"
echo "  cat .env | grep QDRANT"
echo ""
echo "Run test first (50 docs):"
echo "  source .venv/bin/activate"
echo "  python scripts/ingest.py --limit 50 --device cuda"
echo ""
echo "Run full ingestion:"
echo "  source .venv/bin/activate"
echo "  python scripts/ingest.py --recreate --device cuda"
echo ""
echo "For batch/resume mode (in case of interruption):"
echo "  python scripts/ingest.py --device cuda  # resumes, no --recreate"
echo "============================================"
