"""
Streamlit Chat Page — Main Q&A interface with streaming.
"""

import json
import os
import re
import time
from typing import List, Dict

import streamlit as st
import requests


API_BASE = os.environ.get("API_URL", "http://localhost:8000")


def render_chat_page():
    st.title("💬 Tra cứu Pháp luật Việt Nam")
    st.caption("Hỏi bất kỳ câu hỏi nào về pháp luật Việt Nam — bot sẽ trả lời kèm trích dẫn điều khoản cụ thể.")

    # --- Sidebar Settings ---
    with st.sidebar:
        st.subheader("⚙️ Settings")
        use_adaptive = st.checkbox("Adaptive RAG (v1.0+)", value=False,
                                   help="Bật query routing: SIMPLE/MEDIUM/COMPLEX")
        use_streaming = st.checkbox("Streaming", value=True,
                                    help="Hiển thị câu trả lời từng token")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

        st.divider()
        st.subheader("💡 Gợi ý câu hỏi")
        suggestions = [
            "Tội lừa đảo chiếm đoạt tài sản bị xử phạt thế nào?",
            "Người dưới 18 tuổi phạm tội thì bị xử lý ra sao?",
            "Có được bảo lãnh cho người bị bắt tạm giam không?",
            "Tội cố ý gây thương tích theo luật năm 2015 quy định gì?",
            "Những trường hợp nào được miễn trách nhiệm hình sự?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s[:20]}"):
                st.session_state.pending_question = s

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display Chat History ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                _render_citations(msg["citations"])
            if msg.get("confidence"):
                _render_confidence(msg["confidence"])
            if msg.get("latency_ms"):
                st.caption(f"⏱️ {msg['latency_ms']:.0f}ms | Pipeline: {msg.get('pipeline_used', 'naive')}")
            if msg.get("role") == "assistant":
                st.caption("⚠️ *Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.*")

    # --- Chat Input ---
    question = st.chat_input("💬 Nhập câu hỏi pháp lý của bạn...")

    # Handle suggestion click
    if "pending_question" in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        del st.session_state.pending_question

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Get bot response
        with st.chat_message("assistant"):
            if use_streaming:
                response_data = _stream_response(question, use_adaptive)
            else:
                response_data = _get_response(question, use_adaptive)

            # Store in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["answer"],
                "citations": response_data.get("citations", []),
                "confidence": response_data.get("confidence", "medium"),
                "latency_ms": response_data.get("latency_ms", 0),
                "pipeline_used": response_data.get("pipeline_used", "naive"),
            })


def _get_response(question: str, use_adaptive: bool) -> dict:
    """Get non-streaming response from API."""
    try:
        resp = requests.post(
            f"{API_BASE}/api/chat",
            json={"question": question, "use_adaptive": use_adaptive},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"answer": f"Lỗi: {resp.status_code}", "citations": [], "confidence": "low"}
    except Exception as e:
        return {"answer": f"Không thể kết nối đến backend: {e}", "citations": [], "confidence": "low"}


def _stream_response(question: str, use_adaptive: bool) -> dict:
    """Stream response from API and return full data."""
    answer_placeholder = st.empty()
    full_answer = ""
    citations = []
    confidence = "medium"
    latency_ms = 0
    pipeline_used = "naive"

    try:
        with requests.post(
            f"{API_BASE}/api/chat/stream",
            json={"question": question, "use_adaptive": use_adaptive},
            stream=True,
            timeout=60,
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")

                # Parse SSE event type
                if line_str.startswith("event: "):
                    event_type = line_str[7:]
                    continue

                if line_str.startswith("data: "):
                    try:
                        data = json.loads(line_str[6:])
                    except json.JSONDecodeError:
                        continue

                    if event_type == "citations":
                        citations = data
                    elif event_type == "token" and "token" in data:
                        full_answer += data["token"]
                        # Dynamically hide <think> blocks while streaming
                        display_text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', full_answer, flags=re.IGNORECASE)
                        answer_placeholder.markdown(display_text + "▌")
                    elif event_type == "done":
                        full_answer = data.get("full_answer", full_answer)
                        citations = data.get("citations", citations)
                        confidence = data.get("confidence", "medium")
                        latency_ms = data.get("latency_ms", 0)
                        
                        # Clean thinking tags from final answer
                        full_answer = re.sub(r'<think>[\s\S]*?</think>', '', full_answer, flags=re.IGNORECASE)
                        full_answer = full_answer.strip()
                        answer_placeholder.markdown(full_answer)

        return {
            "answer": full_answer,
            "citations": citations,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "pipeline_used": pipeline_used,
        }
    except Exception as e:
        error_msg = f"Lỗi streaming: {e}"
        answer_placeholder.error(error_msg)
        return {"answer": error_msg, "citations": [], "confidence": "low"}


def _render_citations(citations: List[Dict]):
    """Render citation cards."""
    if not citations:
        return
    st.divider()
    st.markdown("### 📚 Căn cứ pháp lý")
    for i, cite in enumerate(citations, 1):
        title = cite.get('document_title', 'Văn bản pháp luật')
        article = cite.get('article', '')
        clause = cite.get('clause', '')
        header = f"📘 **{title}**"
        if article:
            header += f" — {article}"
        if clause:
            header += f", {clause}"
            
        with st.expander(header):
            st.markdown(f"*{cite.get('excerpt', '...')}*")
            st.caption(f"Độ liên quan: {cite.get('relevance', 'cao')}")


def _render_confidence(confidence: str):
    """Render confidence indicator."""
    colors = {
        "high": ("🟢", "Cao", "green"),
        "medium": ("🟡", "Trung bình", "orange"),
        "low": ("🔴", "Thấp", "red"),
    }
    icon, text, color = colors.get(confidence, ("⚪", "Không xác định", "gray"))
    st.markdown(f"**{icon} Độ tin cậy:** :{color}[**{text}**]")
