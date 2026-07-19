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
    """Bôi đậm các marker trích khối [n] trong câu trả lời.

    Chỉ xử lý [n] — SYSTEM_PROMPT (rule 4) đã yêu cầu model tự bôi đậm thuật
    ngữ pháp lý, và lớp re-bold từng có ở đây match lồng vào text đã bold sẵn
    rồi nuốt dấu câu sau tên luật ("Luật Đất đai 2024," → "**Luật Đất**đai
    2024**"). Lookaround bỏ qua marker đã nằm trong "**...**" nên hàm
    idempotent trên text model đã bold.
    """
    return re.sub(r'(?<!\*)(\[\d{1,3}\])(?!\*)', r'**\1**', text)


def _cited_block_indices(answer_text: str) -> set:
    """Parse `[n]` block-citation markers out of an answer's raw text
    (before `_highlight_legal_refs` bolds them)."""
    if not answer_text:
        return set()
    return {int(n) for n in re.findall(r'\[(\d+)\]', answer_text)}


# Refusal-style answers come in 3 shapes: REFUSAL_ANSWER (confidence gate),
# OUT_OF_SCOPE_ANSWER_TEMPLATE (scope gate), and the model's own SYSTEM_PROMPT
# rule-2 refusal line — all share one of these phrases and carry no [n].
_REFUSAL_MARKERS = ("không tìm thấy", "ngoài phạm vi dữ liệu")


def _is_refusal_text(answer_text: str) -> bool:
    """Câu trả lời kiểu từ chối (template gate hoặc model tự từ chối).

    Câu trả lời có marker [n] luôn là câu trả lời thực — cụm "không tìm thấy"
    bên trong một bullet có dẫn nguồn không tính là từ chối.
    """
    if not answer_text or _cited_block_indices(answer_text):
        return False
    lowered = answer_text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


_RELATED_HEADER = re.compile(r'^\W*câu\s*hỏi\s*liên\s*quan\W*$', re.IGNORECASE)
_MD_HEADING = re.compile(r'^#{1,6}\s')


def _split_related_questions(answer_text: str):
    """Tách mục "Câu hỏi liên quan" khỏi markdown câu trả lời.

    Returns (main_text, questions): main_text là câu trả lời đã bỏ mục này
    (mục được render riêng thành button qua `_render_related_questions`),
    questions là list câu hỏi gợi ý. Không có mục → (text nguyên vẹn, []).
    """
    if not answer_text:
        return answer_text, []

    lines = answer_text.split('\n')
    start = None
    for i, line in enumerate(lines):
        stripped = re.sub(r'#+\s*', '', line).strip()
        if stripped and _RELATED_HEADER.match(stripped):
            start = i
            break
    if start is None:
        return answer_text, []

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _MD_HEADING.match(lines[j].strip()):
            end = j
            break

    questions = []
    for line in lines[start + 1:end]:
        s = line.strip()
        if s.startswith('- ') or s.startswith('* '):
            questions.append(s[2:].strip())

    main_text = '\n'.join(lines[:start] + lines[end:]).rstrip()
    return main_text, questions


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
            "Đồng phạm là gì? Có những loại người đồng phạm nào?",
            "Điều kiện để được tha tù trước thời hạn có điều kiện là gì?",
            "Bộ luật Hình sự 1999 và 2015 khác nhau thế nào về tội cố ý gây thương tích?",
            "Tình huống: A (15 tuổi) rủ B (17 tuổi) đi trộm cắp tài sản trị giá 5 triệu đồng. Trách nhiệm hình sự của mỗi người như thế nào?",
            "Đi xe máy vượt đèn đỏ thì bị phạt bao nhiêu tiền?",
        ]
        for s in suggestions:
            if st.button(s, key=f"sug_{s[:20]}"):
                st.session_state.pending_question = s

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display Chat History ---
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            # "Câu hỏi liên quan" is stripped from the displayed markdown —
            # _render_answer_extras renders it as clickable buttons instead.
            display_text, _ = _split_related_questions(msg["content"])
            st.markdown(_highlight_legal_refs(display_text))
            # Phase B: original (pre-refusal) answer, kept available for inspection
            if msg.get("original_answer"):
                with st.expander("Xem câu trả lời gốc (độ tin cậy thấp)", expanded=False):
                    st.markdown(_highlight_legal_refs(msg["original_answer"]))
            if msg.get("role") == "assistant":
                _render_answer_extras(msg["content"], msg, key_ns=f"m{msg_idx}")
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
                # _stream_response renders the answer itself (via its
                # placeholder); the non-streaming path must render here or
                # the answer only appears after the next rerun.
                display_text, _ = _split_related_questions(response_data["answer"])
                st.markdown(_highlight_legal_refs(display_text))

            # Render citations/confidence/warnings for THIS turn immediately —
            # the history loop above already ran before this message existed
            # in session_state, so without this the current turn's cards
            # would only appear after the *next* rerun. key_ns is the index
            # this assistant message gets when appended below, so its
            # related-question buttons keep the same widget keys on the
            # history re-render (a changed key would swallow the click).
            _render_answer_extras(response_data["answer"], response_data,
                                  key_ns=f"m{len(st.session_state.messages)}")
            st.caption("⚠️ *Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.*")

            # Store in history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_data["answer"],
                "citations": response_data.get("citations", []),
                "confidence": response_data.get("confidence", "unverified"),
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
    answer_placeholder = st.empty()
    full_answer = ""
    citations = []
    confidence = "unverified"
    confidence_score = 0.0
    refused = False
    warnings = []
    refusal_answer = None
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

                    if event_type == "step":
                        # Pipeline-step debug events still arrive over SSE but
                        # are no longer displayed — consume and ignore.
                        pass
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
                            # Final render drops the "Câu hỏi liên quan" section —
                            # it's rendered as buttons by _render_answer_extras.
                            display_text, _ = _split_related_questions(full_answer)
                            answer_placeholder.markdown(_highlight_legal_refs(display_text))

        return {
            "answer": refusal_answer if (refused and refusal_answer) else full_answer,
            "original_answer": full_answer if (refused and refusal_answer) else None,
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
        return {"answer": error_msg, "citations": [], "confidence": "low"}


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


def _render_citations(answer_text: str, citations: List[Dict], refused: bool = False):
    """Render citation cards.

    Every card is tagged with `block_index` — the same 1-based number the
    answer's `[n]` markers refer to (see orchestrator._build_citations /
    _build_context). To avoid a wall of up to ~15-20 cards, only the blocks
    ACTUALLY cited in `answer_text` are shown up front; the rest are tucked
    under a collapsed "Xem tất cả nguồn" expander. If the answer carries no
    `[n]` markers at all (refusal template, old-format answers), fall back
    to showing everything flat.

    `refused` (or a refusal-style answer text): a refusal followed by a
    prominent "📚 Căn cứ pháp lý" wall is a visual contradiction — the
    retrieved sources are kept inspectable but tucked into one collapsed
    expander instead.
    """
    if not citations:
        return

    if refused or _is_refusal_text(answer_text):
        st.divider()
        with st.expander(f"Nguồn đã truy xuất ({len(citations)} khối — không đủ căn cứ để trả lời)"):
            for cite in citations:
                _render_citation_card(cite)
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


def _render_related_questions(answer_text: str, key_ns: str):
    """Render mục "Câu hỏi liên quan" thành các button bấm được — bấm sẽ gửi
    câu hỏi đó như một lượt chat mới (cùng cơ chế `pending_question` với các
    gợi ý ở sidebar). `key_ns` phải ổn định giữa render lượt-hiện-tại và
    render lại từ history (dùng index của message) để click không bị nuốt
    giữa hai lần rerun."""
    _, questions = _split_related_questions(answer_text)
    if not questions:
        return
    st.markdown("**💡 Câu hỏi liên quan:**")
    for i, q in enumerate(questions):
        if st.button(q, key=f"rel_{key_ns}_{i}"):
            st.session_state.pending_question = q


def _render_answer_extras(answer_text: str, data: dict, key_ns: str = "cur"):
    """Render related-questions/citations/confidence/warnings/latency for one
    assistant turn.

    Shared between the history-replay loop (`msg` dict) and the just-answered
    turn (`response_data` dict) — both use the same key names for these
    fields, so one function keeps the two render paths from drifting apart.
    """
    _render_related_questions(answer_text, key_ns)
    refused = bool(data.get("refused")) or _is_refusal_text(answer_text)
    if data.get("citations"):
        _render_citations(answer_text, data["citations"], refused=refused)
    if data.get("confidence"):
        _render_confidence(data["confidence"])
    # Truthy check (not `is not None`): a 0.0 score means "no signal ran"
    # (naive pipeline), which must not render as an alarming red 0% badge.
    conf_score = data.get("confidence_score")
    if conf_score:
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
        # Naive pipeline: no verification/confidence scoring ran at all —
        # an honest "unverified" instead of a fabricated level.
        "unverified": ("⚪", "Chưa xác minh", "gray"),
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
