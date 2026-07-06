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


def _highlight_legal_refs(text: str) -> str:
    """Bôi đậm các tham chiếu điều khoản pháp lý trong câu trả lời."""
    # Highlight patterns in order: longer phrases first to avoid overlap
    patterns = [
        (r'(Điều\s+\d+\s*và\s+Điều\s+\d+)', r'**\1**'),
        (r'(Điều\s+\d+\s*,\s*Điều\s+\d+)', r'**\1**'),
        (r'(Điều\s+\d+)', r'**\1**'),
        (r'(Khoản\s+\d+)', r'**\1**'),
        (r'(khoản\s+\d+)', r'**\1**'),
        (r'(Chương\s+[IVXLCDM]+\d*)', r'**\1**'),
        (r'(Bộ\s+luật\s+[^"“\n,.;:()]+?)(?:\s|$|[,.;:()])', r'**\1**'),
        (r'(Luật\s+[^"“\n,.;:()]+?\s+\d+)', r'**\1**'),
        (r'(Luật\s+[^"“\n,.;:()]+?)(?:\s|$|[,.;:()])', r'**\1**'),
        (r'(Nghị\s+định\s+\d+)', r'**\1**'),
        (r'(Thông\s+tư\s+\d+)', r'**\1**'),
    ]
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


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
            if msg.get("steps"):
                with st.expander("🔍 Pipeline Steps (Debug)", expanded=False):
                    lines = []
                    for s in msg["steps"]:
                        lines.append(f"**{s['step']}**:")
                        if len(s['result']) > 150 or "\n" in s['result']:
                            lines.append(f"```text\n{s['result']}\n```")
                        else:
                            lines.append(f"> {s['result']}")
                        lines.append("---")
                    st.markdown("\n".join(lines))
            st.markdown(_highlight_legal_refs(msg["content"]))
            if msg.get("citations"):
                _render_citations(msg["citations"])
            if msg.get("confidence"):
                _render_confidence(msg["confidence"])
            # Phase B: confidence score badge
            conf_score = msg.get("confidence_score")
            if conf_score is not None:
                _render_confidence_score(conf_score)
            # Phase B: verification warnings
            if msg.get("verification_warnings"):
                for w in msg["verification_warnings"]:
                    st.warning(w)
            # Phase B: refused notice
            if msg.get("refused"):
                st.info("ℹ️ Bot không đủ tự tin để trả lời câu hỏi này một cách chính xác.")
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
                "steps": response_data.get("steps", []),
                "citations": response_data.get("citations", []),
                "confidence": response_data.get("confidence", "medium"),
                "confidence_score": response_data.get("confidence_score", 0.0),
                "refused": response_data.get("refused", False),
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
    steps_expander = st.expander("🔍 Pipeline Steps (Debug)", expanded=True)
    steps_placeholder = steps_expander.empty()
    
    answer_placeholder = st.empty()
    full_answer = ""
    citations = []
    confidence = "medium"
    latency_ms = 0
    pipeline_used = "naive"
    steps = []

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

                    if event_type == "step":
                        steps.append(data)
                        lines = []
                        for s in steps:
                            lines.append(f"**{s['step']}**:")
                            if len(s['result']) > 150 or "\n" in s['result']:
                                lines.append(f"```text\n{s['result']}\n```")
                            else:
                                lines.append(f"> {s['result']}")
                            lines.append("---")
                        steps_placeholder.markdown("\n".join(lines))
                    elif event_type == "citations":
                        citations = data
                    elif event_type == "token" and "token" in data:
                        full_answer += data["token"]
                        # Dynamically hide <think> blocks while streaming
                        display_text = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', full_answer, flags=re.IGNORECASE)
                        answer_placeholder.markdown(_highlight_legal_refs(display_text) + "▌")
                    elif event_type == "done":
                        full_answer = data.get("full_answer", full_answer)
                        citations = data.get("citations", citations)
                        confidence = data.get("confidence", "medium")
                        latency_ms = data.get("latency_ms", 0)
                        
                        # Clean thinking tags from final answer
                        full_answer = re.sub(r'<think>[\s\S]*?</think>', '', full_answer, flags=re.IGNORECASE)
                        full_answer = full_answer.strip()
                        answer_placeholder.markdown(_highlight_legal_refs(full_answer))

        return {
            "answer": full_answer,
            "steps": steps,
            "citations": citations,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "pipeline_used": pipeline_used,
        }
    except Exception as e:
        error_msg = f"Lỗi streaming: {e}"
        answer_placeholder.error(error_msg)
        return {"answer": error_msg, "steps": steps, "citations": [], "confidence": "low"}


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

def _render_confidence_score(score: float):
    """Render confidence score as colored badge (Phase B)."""
    if score >= 0.80:
        icon, label, color = "🟢", f"{score:.0%}", "green"
    elif score >= 0.50:
        icon, label, color = "🟡", f"{score:.0%}", "orange"
    elif score >= 0.30:
        icon, label, color = "🟠", f"{score:.0%}", "orange"
    else:
        icon, label, color = "🔴", f"{score:.0%}", "red"
    st.markdown(f"**{icon} Confidence Score:** :{color}[**{label}**]")
