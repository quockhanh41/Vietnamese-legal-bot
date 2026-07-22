#!/usr/bin/env bash
# M5: run FastAPI (internal :8000) + Streamlit (public :8501) in one container.
# If either process exits, kill the other so the HF Space restarts cleanly.
set -euo pipefail

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

export API_URL="http://localhost:8000"
streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

trap 'kill "$API_PID" "$UI_PID" 2>/dev/null || true' EXIT
wait -n "$API_PID" "$UI_PID"   # PID-arg form needs bash>=5.1 (python:3.11-slim ships bash 5.x)
