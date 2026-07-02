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
COPY .env .env 2>/dev/null || true
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- UI Stage ---
FROM base AS ui
COPY src/ ./src/
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl -f http://localhost:8501 || exit 1
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
