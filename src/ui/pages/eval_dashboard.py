"""
Streamlit Evaluation Dashboard page — RAGAS metrics (v1.5).
"""

import os
import streamlit as st
import requests

API_BASE = os.environ.get("API_URL", "http://localhost:8000")


def render_eval_page():
    st.title("📊 Evaluation Dashboard")
    st.caption("System health & performance overview.")

    # --- Live System Status ---
    st.subheader("🖥️ System Status")
    try:
        stats_resp = requests.get(f"{API_BASE}/api/stats", timeout=5)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            cols = st.columns(4)
            cols[0].metric("📄 Indexed Chunks", f"{stats.get('indexed_documents', 0):,}")
            cols[1].metric("🗂️ Collection", stats.get("collection_name", "N/A"))
            cols[2].metric("🔍 LLM Model", "Qwen 3 32B (Groq)")
            cols[3].metric("⚡ API Status", "🟢 Online")
        else:
            st.warning("Backend API not reachable")
    except Exception:
        st.error("🔴 Backend Offline — start with `make api`")

    # --- RAGAS Scores (placeholder until nightly eval runs) ---
    st.divider()
    st.subheader("🎯 RAGAS Metrics (target)")
    st.caption("These are target thresholds. Actual metrics will populate after running `scripts/eval.py`.")
    cols = st.columns(4)
    metrics = [
        ("Faithfulness", ">0.80", "LLM answer matches retrieved docs"),
        ("Answer Relevancy", ">0.80", "Answer directly addresses the question"),
        ("Context Precision", ">0.75", "Retrieved chunks are relevant"),
        ("Context Recall", ">0.80", "All needed information is retrieved"),
    ]
    for col, (name, target, desc) in zip(cols, metrics):
        col.metric(name, target, delta=desc)

    # --- Pipeline Performance ---
    st.divider()
    st.subheader("📈 Pipeline Performance (estimated)")
    st.caption("Based on local testing with Qwen 3 32B on Groq.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Latency", "3.0s", delta="embed+search+llm")
    col2.metric("Search Latency", "~0.1s", delta="Qdrant hybrid RRF")
    col3.metric("LLM Latency", "~2.8s", delta="Qwen 3 32B streaming")
    col4.metric("Cost/Query", "$0.00", delta="Groq free tier")

    # --- Evaluation Setup Guide ---
    st.divider()
    st.subheader("🔧 How to Run Evaluation")
    st.markdown("""
    1. **Create eval dataset**: Generate 50+ question-answer pairs for testing
    2. **Run RAGAS eval**: `make eval` — measures faithfulness, relevancy, precision, recall
    3. **GitHub Actions nightly**: Configure `.github/workflows/eval_nightly.yml` for automated daily evals
    4. **View trends**: Historical charts appear here after 3+ eval runs

    See `docs/lessons-learned.md` for engineering practices applied during development.
    """)

    # --- Placeholder for charts ---
    st.info("📊 Historical charts will appear here after nightly eval pipeline runs "
            "(requires `.github/workflows/eval_nightly.yml`).")
