"""
Streamlit Evaluation Dashboard page — RAGAS metrics + Chunking Evaluation (v1.6).
"""

import json
import os
import streamlit as st
import requests
import pandas as pd

from src.llm.gateway import LLMGateway

API_BASE = os.environ.get("API_URL", "http://localhost:8000")


def _load_eval_results():
    """Load chunking evaluation results, preferring round 2 over round 1."""
    for path in ["data/eval/round2_results.json", "data/eval/round1_results.json"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _color_cell(val, best_val, worst_val, higher_is_better=True):
    """Return CSS color for a metric cell relative to best/worst."""
    if best_val == worst_val or best_val == 0:
        return ""
    if higher_is_better:
        ratio = (val - worst_val) / (best_val - worst_val)
    else:
        ratio = (best_val - val) / (best_val - worst_val)
    if ratio > 0.8:
        return "background-color: #d4edda"  # green
    elif ratio < 0.2:
        return "background-color: #f8d7da"  # red
    elif ratio < 0.5:
        return "background-color: #fff3cd"  # yellow
    return ""


def _render_chunking_eval():
    """Render the chunking strategy evaluation section."""
    st.divider()
    st.subheader("🧪 Chunking Strategy Evaluation")
    st.caption("Compare different chunking strategies using RAGAS metrics.")

    results = _load_eval_results()

    if results is None or not results.get("results"):
        st.info(
            "📋 No evaluation results yet. Run the evaluation pipeline:\n\n"
            "```bash\n"
            "# Step 1: Generate test queries\n"
            "python scripts/generate_eval_queries.py --docs 20\n\n"
            "# Step 2: Run round 1 evaluation\n"
            "python scripts/eval_chunking.py --round 1 --docs 20\n\n"
            "# Step 3: Run round 2 with best params\n"
            "python scripts/eval_chunking.py --round 2 --docs 100\n"
            "```"
        )
        return

    eval_data = results["results"]
    round_num = results.get("round", "?")
    num_docs = results.get("num_docs", "?")
    num_queries = results.get("num_queries", "?")

    st.caption(f"Round {round_num} — {num_docs} documents, {num_queries} test queries")

    # Build comparison table
    metric_cols = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    ]

    # Find best/worst for each metric column for color coding
    best_vals = {}
    worst_vals = {}
    for col in metric_cols:
        vals = [r.get(col, 0) for r in eval_data if r.get(col, 0) > 0]
        best_vals[col] = max(vals) if vals else 1.0
        worst_vals[col] = min(vals) if vals else 0.0

    # Build DataFrame
    rows = []
    for r in sorted(eval_data, key=lambda x: x.get("faithfulness", 0), reverse=True):
        rows.append({
            "Strategy": r.get("label", f"{r['strategy']} {r['params']}"),
            "Faithfulness": f"{r.get('faithfulness', 0):.3f}",
            "Ctx Precision": f"{r.get('context_precision', 0):.3f}",
            "Ctx Recall": f"{r.get('context_recall', 0):.3f}",
            "Ans Relevancy": f"{r.get('answer_relevancy', 0):.3f}",
            "# Chunks": r.get("num_chunks", 0),
            "Emb Time (s)": f"{r.get('embedding_time_s', 0):.1f}",
            "Search (ms)": f"{r.get('avg_search_latency_ms', 0):.1f}",
        })

    df = pd.DataFrame(rows)

    # Style the dataframe
    def style_row(row):
        styles = [""] * len(row)
        # Color faith, precision, recall, relevancy columns
        for i, col_name in enumerate(df.columns):
            if col_name == "Faithfulness":
                val = float(row[col_name])
                styles[i] = _color_cell(val, best_vals["faithfulness"], worst_vals["faithfulness"])
            elif col_name == "Ctx Precision":
                val = float(row[col_name])
                styles[i] = _color_cell(val, best_vals["context_precision"], worst_vals["context_precision"])
            elif col_name == "Ctx Recall":
                val = float(row[col_name])
                styles[i] = _color_cell(val, best_vals["context_recall"], worst_vals["context_recall"])
            elif col_name == "Ans Relevancy":
                val = float(row[col_name])
                styles[i] = _color_cell(val, best_vals["answer_relevancy"], worst_vals["answer_relevancy"])
        return styles

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Best strategy recommendation
    if eval_data:
        best = max(eval_data, key=lambda r: r.get("faithfulness", 0))
        st.success(
            f"🏆 **Best Strategy:** {best.get('label', 'N/A')}\n\n"
            f"- Faithfulness: **{best.get('faithfulness', 0):.3f}**\n"
            f"- Context Precision: **{best.get('context_precision', 0):.3f}**\n"
            f"- Context Recall: **{best.get('context_recall', 0):.3f}**\n"
            f"- {best.get('num_chunks', '?')} chunks, {best.get('embedding_time_s', 0):.1f}s embedding"
        )


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
            cols[2].metric("🔍 LLM Model", LLMGateway.MODEL_CHAIN[0])
            cols[3].metric("⚡ API Status", "🟢 Online")
        else:
            st.warning("Backend API not reachable")
    except Exception:
        st.error("🔴 Backend Offline — start with `make api`")

    # --- RAGAS Metrics (live or target) ---
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

    # --- Chunking Strategy Evaluation (NEW) ---
    _render_chunking_eval()

    # --- Pipeline Performance ---
    st.divider()
    st.subheader("📈 Pipeline Performance (estimated)")
    st.caption(f"Based on local testing with {LLMGateway.MODEL_CHAIN[0]}.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Latency", "3.0s", delta="embed+search+llm")
    col2.metric("Search Latency", "~0.1s", delta="Qdrant hybrid RRF")
    col3.metric("LLM Latency", "~2.8s", delta=f"{LLMGateway.MODEL_CHAIN[0]} streaming")
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
