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
API_BASE = os.environ.get("API_URL", "http://localhost:8000")
try:
    import httpx
    health_ok = httpx.get(f"{API_BASE}/health", timeout=2).status_code == 200
    ready_ok = False
    if health_ok:
        rj = httpx.get(f"{API_BASE}/ready", timeout=2).json()
        ready_ok = bool(rj.get("ready"))
    from src.ui.pages.chat import backend_state
    state = backend_state(ready_ok, health_ok)
    if state == "ready":
        st.sidebar.success("🟢 Backend sẵn sàng")
    elif state == "warming":
        st.sidebar.warning("🟡 Đang khởi động (cold start ~1–2 phút, đang nạp model)")
    else:
        st.sidebar.error("🔴 Backend offline")
except Exception:
    st.sidebar.error("🔴 Backend offline")

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
