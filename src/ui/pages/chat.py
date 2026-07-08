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
        # Grounded block-citation markers, e.g. "[2]" or "[2][3]" at the end
        # of a "Chi tiết" bullet — bold them so they read as a distinct
        # footnote-style marker rather than blending into the sentence.
        (r'(\[\d{1,3}\])', r'**\1**'),
    ]
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _cited_block_indices(answer_text: str) -> set:
    """Parse `[n]` block-citation markers out of an answer's raw text
    (before `_highlight_legal_refs` bolds them)."""
    if not answer_text:
        return set()
    return {int(n) for n in re.findall(r'\[(\d+)\]', answer_text)}


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
            # Phase B: original (pre-refusal) answer, kept available for inspection
            if msg.get("original_answer"):
                with st.expander("Xem câu trả lời gốc (độ tin cậy thấp)", expanded=False):
                    st.markdown(_highlight_legal_refs(msg["original_answer"]))
            if msg.get("role") == "assistant":
                _render_answer_extras(msg["content"], msg)
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

            # Render citations/confidence/warnings for THIS turn immediately —
            # the history loop above already ran before this message existed
            # in session_state, so without this the current turn's cards
            # would only appear after the *next* rerun.
            _render_answer_extras(response_data["answer"], response_data)
            st.caption("⚠️ *Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.*")

            # Store in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["answer"],
                "steps": response_data.get("steps", []),
                "citations": response_data.get("citations", []),
                "confidence": response_data.get("confidence", "medium"),
                "confidence_score": response_data.get("confidence_score", 0.0),
                "refused": response_data.get("refused", False),
                "original_answer": response_data.get("original_answer"),
                "warnings": response_data.get("warnings", []),
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
    confidence_score = 0.0
    refused = False
    warnings = []
    refusal_answer = None
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
                    elif event_type == "verification":
                        # Only emitted by the adaptive pipeline, after the full
                        # answer has streamed (hallucination check + confidence
                        # gate). Never arrives for the naive pipeline.
                        confidence = data.get("confidence", confidence)
                        confidence_score = data.get("confidence_score", confidence_score)
                        refused = data.get("refused", False)
                        warnings = data.get("warnings", [])
                        refusal_answer = data.get("refusal_answer")
                    elif event_type == "done":
                        full_answer = data.get("full_answer", full_answer)
                        citations = data.get("citations", citations)
                        # "done" carries the same real values when a "verification"
                        # event preceded it; otherwise (naive pipeline) these are
                        # the backend defaults, so fall back to what we already have.
                        confidence = data.get("confidence", confidence)
                        confidence_score = data.get("confidence_score", confidence_score)
                        latency_ms = data.get("latency_ms", 0)

                        # Clean thinking tags from final answer
                        full_answer = re.sub(r'<think>[\s\S]*?</think>', '', full_answer, flags=re.IGNORECASE)
                        full_answer = full_answer.strip()

                        if refused and refusal_answer:
                            # Confidence gate refused: show the refusal message as the
                            # main answer, keep the original streamed answer inspectable.
                            answer_placeholder.markdown(_highlight_legal_refs(refusal_answer))
                            with st.expander("Xem câu trả lời gốc (độ tin cậy thấp)", expanded=False):
                                st.markdown(_highlight_legal_refs(full_answer))
                        else:
                            answer_placeholder.markdown(_highlight_legal_refs(full_answer))

        return {
            "answer": refusal_answer if (refused and refusal_answer) else full_answer,
            "original_answer": full_answer if (refused and refusal_answer) else None,
            "steps": steps,
            "citations": citations,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "refused": refused,
            "warnings": warnings,
            "latency_ms": latency_ms,
            "pipeline_used": pipeline_used,
        }
    except Exception as e:
        error_msg = f"Lỗi streaming: {e}"
        answer_placeholder.error(error_msg)
        return {"answer": error_msg, "steps": steps, "citations": [], "confidence": "low"}


def _render_citation_card(cite: Dict):
    """Render a single citation as an expander, labeled with its block_index
    (the same 1-based number the answer's `[n]` markers refer to) so the
    card is unambiguously identifiable from the answer text."""
    idx = cite.get('block_index')
    title = cite.get('document_title', 'Văn bản pháp luật')
    article = cite.get('article', '')
    header = f"**[{idx}]** 📘 **{title}**" if idx is not None else f"📘 **{title}**"
    if article:
        header += f" — {article}"

    with st.expander(header):
        if cite.get('is_amended'):
            laws = ", ".join(cite.get('amending_laws') or [])
            st.warning(f"⚠️ Điều này đã được sửa đổi bởi: {laws}" if laws else "⚠️ Điều này đã được sửa đổi.")
        excerpt = cite.get('chunk_text') or '...'
        st.markdown(f"*{excerpt}*")
        meta_bits = []
        if cite.get('legal_type'):
            meta_bits.append(cite['legal_type'])
        if cite.get('score') is not None:
            meta_bits.append(f"score={cite['score']}")
        if meta_bits:
            st.caption(" · ".join(meta_bits))
        if cite.get('url'):
            st.caption(f"🔗 {cite['url']}")


def _render_citations(answer_text: str, citations: List[Dict]):
    """Render citation cards.

    Every card is tagged with `block_index` — the same 1-based number the
    answer's `[n]` markers refer to (see orchestrator._build_citations /
    _build_context). To avoid a wall of up to ~15-20 cards, only the blocks
    ACTUALLY cited in `answer_text` are shown up front; the rest are tucked
    under a collapsed "Xem tất cả nguồn" expander. If the answer carries no
    `[n]` markers at all (refusal template, old-format answers), fall back
    to showing everything flat.
    """
    if not citations:
        return

    cited_indices = _cited_block_indices(answer_text)
    by_index = {c.get('block_index'): c for c in citations if c.get('block_index') is not None}

    cited = [by_index[n] for n in sorted(cited_indices) if n in by_index]
    others = [c for c in citations if c.get('block_index') not in cited_indices]

    if not cited:
        # Nothing parsed — degrade gracefully to the old flat rendering.
        cited, others = citations, []

    st.divider()
    st.markdown("### 📚 Căn cứ pháp lý")
    for cite in cited:
        _render_citation_card(cite)

    if others:
        with st.expander(f"Xem tất cả nguồn ({len(others)} khối khác)"):
            for cite in others:
                _render_citation_card(cite)


def _render_answer_extras(answer_text: str, data: dict):
    """Render citations/confidence/warnings/latency for one assistant turn.

    Shared between the history-replay loop (`msg` dict) and the just-answered
    turn (`response_data` dict) — both use the same key names for these
    fields, so one function keeps the two render paths from drifting apart.
    """
    if data.get("citations"):
        _render_citations(answer_text, data["citations"])
    if data.get("confidence"):
        _render_confidence(data["confidence"])
    conf_score = data.get("confidence_score")
    if conf_score is not None:
        _render_confidence_score(conf_score)
    if data.get("warnings"):
        st.warning("\n".join(data["warnings"]))
    if data.get("refused"):
        st.info("ℹ️ Bot không đủ tự tin để trả lời câu hỏi này một cách chính xác.")
    if data.get("latency_ms"):
        st.caption(f"⏱️ {data['latency_ms']:.0f}ms | Pipeline: {data.get('pipeline_used', 'naive')}")


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
