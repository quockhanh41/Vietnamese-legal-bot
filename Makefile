.PHONY: help install download ingest qdrant api ui dev clean test eval

help:
	@echo "Vietnamese Legal Knowledge Bot"
	@echo ""
	@echo "Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make download   - Download dataset from HuggingFace"
	@echo "  make qdrant     - Start Qdrant + Redis (Docker)"
	@echo "  make ingest     - Run full data ingestion pipeline"
	@echo "  make api        - Start FastAPI backend"
	@echo "  make ui         - Start Streamlit frontend"
	@echo "  make dev        - Start API + UI together"
	@echo "  make test       - Run tests"
	@echo "  make eval       - Run RAGAS evaluation"
	@echo "  make clean      - Clean temporary files"

install:
	pip install -e ".[dev]"

download:
	python scripts/download_dataset.py

qdrant:
	docker compose up -d qdrant redis

ingest:
	python scripts/ingest.py

ingest-test:
	python scripts/ingest.py --limit 100 --recreate

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run src/ui/app.py --server.port 8501

dev:
	@echo "Starting API on port 8000 and UI on port 8501..."
	@uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
	@sleep 2
	@streamlit run src/ui/app.py --server.port 8501

test:
	pytest tests/ -v

eval:
	python scripts/eval.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache
