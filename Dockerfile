FROM python:3.11-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && pip install python-dotenv

# --- API Stage ---
FROM base AS api
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env* ./
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- UI Stage ---
FROM base AS ui
COPY src/ ./src/
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8501 || exit 1
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# --- Combined Space Stage (HF Docker Space: Streamlit public :8501, API internal :8000) ---
FROM base AS space
# Pre-fetch models at BUILD time so cold start only loads from disk (no ~5GB re-download).
RUN python -c "from FlagEmbedding import BGEM3FlagModel, FlagReranker; \
    BGEM3FlagModel('BAAI/bge-m3', use_fp16=True); \
    FlagReranker('AITeamVN/Vietnamese_Reranker', use_fp16=True)" || echo "model pre-fetch skipped"
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('AITeamVN/Vietnamese_Embedding')" || echo "embedder pre-fetch skipped"
RUN python -c "from transformers import pipeline; \
    pipeline('text-classification', model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli', device=-1)" || echo "NLI pre-fetch skipped"
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/metadata/store.db ./data/metadata/store.db
COPY .env* ./
COPY start.sh ./start.sh
RUN chmod +x start.sh
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=5 CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["bash", "start.sh"]
