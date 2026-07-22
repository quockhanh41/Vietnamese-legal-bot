"""
Tests for the answer-format presentation fixes (2026-07-19 review):

1. UI `_highlight_legal_refs` must stop mangling markdown — the old
   sequential regexes nested bold markers inside model-bolded text and ate
   the delimiter after law names ("Luật Đất đai 2024," → "**Luật Đất**đai 2024**").
   New behavior: only bold `[n]` block-citation markers, idempotently.
2. Naive pipeline must not fabricate a confidence label ("high" whenever
   len(citations) >= 2 — i.e. nearly always): it reports "unverified",
   because no verification/confidence scoring ran on that path.
3. Refusal-style answers (gate templates AND the model's own prompt-rule-2
   refusal line) are detected UI-side so citation cards are not rendered
   as a prominent "Căn cứ pháp lý" wall under a refusal.
4. The "### 💡 Câu hỏi liên quan" trailer is parsed out of the answer
   markdown so the UI can strip it from the displayed text and render the
   questions as clickable buttons.
5. Citation-card excerpt cap raised 200 → 2000 chars (same cap
   `_build_context` uses) so the card shows enough source text to be
   checked against the answer.

Run: .venv/bin/python -m pytest tests/test_answer_format_fixes.py -q
"""

import json
import os
import re
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.ui.pages.chat as chat
from src.db.qdrant_client import SearchResult
from src.rag.confidence import REFUSAL_ANSWER, OUT_OF_SCOPE_ANSWER_TEMPLATE
from src.rag.orchestrator import NaiveRAGPipeline


def _sr(point_id, chunk_text="chunk"):
    return SearchResult(
        point_id=point_id,
        score=0.9,
        chunk_text=chunk_text,
        document_title="Bộ luật hình sự 2015",
        document_number="100/2015/QH13",
        legal_type="Bộ luật",
        article="Điều 17",
        document_id=1,
    )


def _make_pipeline() -> NaiveRAGPipeline:
    return NaiveRAGPipeline(
        vector_store=MagicMock(), embedder=MagicMock(), llm_gateway=MagicMock(),
    )


SAMPLE_ANSWER = (
    "### 💬 Trả lời ngắn\n"
    "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm.\n\n"
    "### 📋 Chi tiết\n"
    "- Đồng phạm là trường hợp có hai người trở lên (Điều 17 BLHS 2015) [1]\n"
    '  > "có hai người trở lên cố ý cùng thực hiện tội phạm"\n\n'
    "### 💡 Câu hỏi liên quan\n"
    "- Phạm tội có tổ chức là gì?\n"
    "- Người xúi giục bị xử lý thế nào?\n"
    "- Đồng phạm có được giảm nhẹ trách nhiệm hình sự không?\n"
)


# ── 1. _highlight_legal_refs: no more markdown mangling ─────────────────────

def test_highlight_preserves_law_names_and_punctuation():
    text = "Theo Luật Đất đai 2024, người sử dụng đất có quyền chuyển nhượng [2]."
    out = chat._highlight_legal_refs(text)
    assert "Luật Đất đai 2024," in out  # no eaten comma, no stray asterisks
    assert "**[2]**" in out


def test_highlight_no_nested_bold_in_law_names():
    text = "Căn cứ Bộ luật Hình sự, Điều 17 quy định về đồng phạm."
    out = chat._highlight_legal_refs(text)
    assert "Bộ luật Hình sự, Điều 17" in out


def test_highlight_idempotent_on_model_bolded_text():
    text = 'Quy định tại **Điều 17** BLHS **[2]**.'
    assert chat._highlight_legal_refs(text) == text


def test_highlight_bolds_adjacent_block_refs():
    out = chat._highlight_legal_refs("- Nội dung... (Điều 5) [2][3]")
    assert "**[2]****[3]**" in out


# ── 2. Naive pipeline confidence is honest ──────────────────────────────────

def test_naive_parse_response_reports_unverified():
    pipeline = _make_pipeline()
    citations = [{"block_index": i} for i in range(1, 4)]
    resp = pipeline._parse_response("### 💬 Trả lời ngắn\nNội dung.", citations)
    assert resp.confidence == "unverified"


def test_stream_done_event_reports_unverified_for_naive(monkeypatch):
    """Naive pipeline never emits __VERIFICATION__, so the SSE `done` event
    fallback label must be "unverified", not a fabricated "medium"."""
    from fastapi.testclient import TestClient
    import src.api.main as main

    fake = MagicMock()
    fake.query_stream = lambda q: iter(["xin ", "chào"])
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: fake)

    # M4: chat_stream now reads session/cache/llm off app.state (normally set
    # once in `lifespan`); this test calls the raw ASGI app without running
    # lifespan, so stub them the same way tests/test_api_wiring.py does.
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    client = TestClient(main.app)
    resp = client.post("/api/chat/stream", json={"question": "hi", "use_adaptive": False})
    assert resp.status_code == 200
    m = re.search(r"event: done\r?\ndata: (.*)", resp.text)
    assert m, f"no done event in body: {resp.text!r}"
    assert json.loads(m.group(1))["confidence"] == "unverified"


def test_render_confidence_has_unverified_label(monkeypatch):
    captured = []
    monkeypatch.setattr(chat.st, "markdown", lambda s: captured.append(s))
    chat._render_confidence("unverified")
    assert any("Chưa xác minh" in s for s in captured)


def test_zero_confidence_score_badge_not_rendered(monkeypatch):
    """A 0.0 score means 'no signal' (naive pipeline), not 'red 0%'."""
    monkeypatch.setattr(chat, "st", MagicMock())
    chat._render_answer_extras("câu trả lời", {"confidence_score": 0.0})
    calls = " ".join(str(c) for c in chat.st.mock_calls)
    assert "Confidence Score" not in calls


# ── 3. Refusal detection so citation cards aren't a wall under a refusal ────

def test_is_refusal_detects_gate_templates_and_model_refusal():
    assert chat._is_refusal_text(REFUSAL_ANSWER)
    oos = OUT_OF_SCOPE_ANSWER_TEMPLATE.format(corpus_summary="- Bộ luật Hình sự 2015")
    assert chat._is_refusal_text(oos)
    assert chat._is_refusal_text(
        "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu."
    )


def test_is_refusal_false_for_grounded_answers():
    assert not chat._is_refusal_text(SAMPLE_ANSWER)
    # "không tìm thấy" inside a cited bullet is answer content, not a refusal
    text = "### 📋 Chi tiết\n- Nếu không tìm thấy tài sản, cơ quan thi hành án đình chỉ (Điều 44) [2]\n"
    assert not chat._is_refusal_text(text)


def test_refused_citations_tucked_into_collapsed_expander(monkeypatch):
    monkeypatch.setattr(chat, "st", MagicMock())
    cites = [{"block_index": 1, "document_title": "Bộ luật Hình sự 2015"}]
    chat._render_citations(REFUSAL_ANSWER, cites, refused=True)
    calls = " ".join(str(c) for c in chat.st.mock_calls)
    assert "Căn cứ pháp lý" not in calls
    assert "Nguồn đã truy xuất" in calls


# ── 4. Related-questions parsing (buttons) ──────────────────────────────────

def test_split_related_questions_extracts_and_strips():
    main_text, questions = chat._split_related_questions(SAMPLE_ANSWER)
    assert questions == [
        "Phạm tội có tổ chức là gì?",
        "Người xúi giục bị xử lý thế nào?",
        "Đồng phạm có được giảm nhẹ trách nhiệm hình sự không?",
    ]
    assert "Câu hỏi liên quan" not in main_text
    assert "Chi tiết" in main_text  # rest of the answer untouched


def test_split_related_questions_tolerates_header_variants():
    text = "Nội dung chính.\n\n## Câu hỏi liên quan\n- Một?\n- Hai?\n"
    main_text, questions = chat._split_related_questions(text)
    assert questions == ["Một?", "Hai?"]
    assert main_text.strip() == "Nội dung chính."


def test_split_related_questions_absent_section():
    text = "### 💬 Trả lời ngắn\nNội dung."
    main_text, questions = chat._split_related_questions(text)
    assert questions == []
    assert main_text == text


# ── 5. Citation-card excerpt matches the context cap ────────────────────────

def test_citation_excerpt_cap_matches_context_cap():
    pipeline = _make_pipeline()
    results = [_sr("a", chunk_text="x" * 3000)]
    cites = pipeline._build_citations(results)
    assert len(cites[0]["chunk_text"]) == 2000


# ── 6. Page-level rendering (streamlit AppTest) ─────────────────────────────

def _page_script():
    # AppTest.from_function runs the function SOURCE as a standalone script,
    # so the import has to live inside; sys.modules returns the same module
    # object the test monkeypatched.
    import src.ui.pages.chat as chat
    chat.render_chat_page()


def test_non_streaming_renders_answer_immediately(monkeypatch):
    """Non-streaming path must display the answer in the SAME run — before
    this fix it was only visible after the next rerun (via history replay)."""
    from streamlit.testing.v1 import AppTest
    from unittest.mock import MagicMock

    monkeypatch.setattr(
        chat, "_get_response",
        lambda q, a: {"answer": "NỘI DUNG TRẢ LỜI KIỂM CHỨNG", "citations": [],
                      "confidence": "unverified", "confidence_score": 0.0},
    )
    # M5: monkeypatch httpx so the /ready gate doesn't block this test
    import httpx
    _real_get = httpx.get
    def _fake_get(url, **kwargs):
        if "/ready" in str(url):
            m = MagicMock(); m.json.return_value = {"ready": True}; return m
        return _real_get(url, **kwargs)
    monkeypatch.setattr(httpx, "get", _fake_get)

    at = AppTest.from_function(_page_script)
    at.run()
    # checkbox[0] = Adaptive RAG, checkbox[1] = Streaming (default True)
    at.checkbox[1].uncheck().run()
    at.chat_input[0].set_value("Đồng phạm là gì?").run()
    assert any("NỘI DUNG TRẢ LỜI KIỂM CHỨNG" in (m.value or "") for m in at.markdown)


def test_pipeline_steps_debug_expander_removed():
    """The 🔍 Pipeline Steps (Debug) expander is gone — a history message
    carrying old `steps` data must not render step content anymore."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_page_script)
    at.session_state["messages"] = [
        {"role": "user", "content": "câu hỏi"},
        {"role": "assistant", "content": "câu trả lời", "citations": [],
         "steps": [{"step": "Routing", "result": "SIMPLE"}]},
    ]
    at.run()
    rendered = " ".join((m.value or "") for m in at.markdown)
    assert "Routing" not in rendered
    assert "câu trả lời" in rendered
