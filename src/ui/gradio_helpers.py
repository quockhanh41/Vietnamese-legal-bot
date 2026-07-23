"""Pure helpers for the Gradio+ZeroGPU chat app (M5 pivot). No model/network/
GPU imports on purpose — src/ui/app_gradio.py's module scope loads heavy
models and can't be unit-tested (see spec), but this file can be."""
from __future__ import annotations

from typing import Dict, List


def history_to_turns(history: List[Dict]) -> List[Dict]:
    """Gradio `type="messages"` history (flat list of {"role","content"})
    -> the [{"question","answer"}, ...] shape src.rag.condense.condense_query
    expects. Pairs consecutive user/assistant messages in order; a trailing
    unanswered user message (no assistant reply yet in history) is dropped
    -- it's the current turn's own question, passed separately."""
    turns: List[Dict] = []
    pending_question = None
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pending_question = content
        elif role == "assistant" and pending_question is not None:
            turns.append({"question": pending_question, "answer": content})
            pending_question = None
    return turns


def format_citations_markdown(citations: List[Dict]) -> str:
    """Render citation dicts (shape: orchestrator._build_citations) as a
    Markdown block appended after the streamed answer. [] -> ""."""
    if not citations:
        return ""
    lines = ["", "**Căn cứ pháp lý:**"]
    for c in citations:
        title = c.get("document_title") or ""
        article = c.get("article") or ""
        label = f"{article}, {title}" if article else title
        lines.append(f"{c.get('block_index', '?')}. {label}")
    return "\n".join(lines)


def is_quota_exceeded(e: Exception) -> bool:
    """Best-effort detection of ZeroGPU's daily-quota error, distinct from
    src.rag.degraded.is_exhaustion (LLM-provider budget, not GPU quota).
    Matched by substring: the exact exception class isn't pinned locally (no
    local GPU to raise it against) -- verify against real Space logs after
    deploy (Task 9) and tighten this if the live error text differs."""
    text = str(e).lower()
    return "quota" in text and "gpu" in text
