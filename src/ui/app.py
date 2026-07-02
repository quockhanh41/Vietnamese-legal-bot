"""
Streamlit UI for Vietnamese Legal Knowledge Bot.

Pages:
- Chat: Main Q&A interface with streaming
- Documents: Browse/search legal documents
- Eval: RAGAS evaluation dashboard (v1.5)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import streamlit as st

st.set_page_config(
    page_title="Vietnamese Legal Knowledge Bot",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Navigation ---
st.sidebar.title("🏛️ Vietnamese Legal Bot")

page = st.sidebar.radio(
    "Điều hướng",
    ["💬 Chat", "📁 Documents", "📊 Evaluation"],
)

st.sidebar.divider()

# System status
st.sidebar.subheader("System Status")
try:
    import httpx
    resp = httpx.get("http://localhost:8000/health", timeout=2)
    if resp.status_code == 200:
        st.sidebar.success("🟢 Backend Online")
    else:
        st.sidebar.warning("🟡 Backend Unstable")
except Exception:
    st.sidebar.error("🔴 Backend Offline")

st.sidebar.caption("API: http://localhost:8000")
st.sidebar.caption("Model: Qwen 3 32B (Groq)")
st.sidebar.caption("Embedding: BGE-M3 (1024-dim)")

# --- Pages ---
if page == "💬 Chat":
    # Import and run chat page
    from src.ui.pages.chat import render_chat_page
    render_chat_page()

elif page == "📁 Documents":
    from src.ui.pages.documents import render_documents_page
    render_documents_page()

elif page == "📊 Evaluation":
    from src.ui.pages.eval_dashboard import render_eval_page
    render_eval_page()
