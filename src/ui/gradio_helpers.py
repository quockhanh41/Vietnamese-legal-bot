"""Pure helpers for the Gradio+ZeroGPU chat app (M5 pivot). No model/network/
GPU imports on purpose — src/ui/app_gradio.py's module scope loads heavy
models and can't be unit-tested (see spec), but this file can be."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd


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


def document_choices(documents: List[Dict]) -> List[Tuple[str, int]]:
    """documents: rows from LegalVectorStore.list_documents()["documents"] (each
    has "title"/"document_id") -> (label, value) tuples for gr.Dropdown choices,
    so the widget's value is the real document_id, never a parsed string."""
    return [(d["title"], d["document_id"]) for d in documents]


def rename_document_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vietnamese column labels for the Documents tab table -- mirrors the
    st.dataframe column_config used in src/ui/pages/documents.py."""
    return df.rename(columns={
        "title": "Tên văn bản",
        "legal_type": "Loại văn bản",
        "issuance_date": "Ngày ban hành",
    })


def format_system_info(model_id: str, collection_name: str, points_count: int) -> str:
    """Model/embedding/reranker/collection summary shown on the Thống kê tab."""
    return (
        f"**Model:** {model_id} (auto-fallback)\n\n"
        f"**Embedding:** AITeamVN/Vietnamese_Embedding (dense) + BGE-M3 (sparse)\n\n"
        f"**Reranker:** AITeamVN/Vietnamese_Reranker\n\n"
        f"**Collection:** {collection_name} ({points_count:,} chunks)"
    )


def parse_effective_date(at_date_str: str) -> Optional[date]:
    """'' -> None (caller's store call defaults to today); malformed string ->
    raises ValueError (caller catches and shows a friendly message). Only
    needed on the Gradio path -- Streamlit's st.date_input is a calendar
    widget and can't produce a malformed string."""
    return date.fromisoformat(at_date_str) if at_date_str else None


def format_effective_laws(laws: List[Dict]) -> pd.DataFrame:
    """laws: rows from MetadataStore.get_effective_laws() (each has
    "law_name"/"effective_date") -> display DataFrame for the Đồ thị & Hiệu lực tab."""
    return pd.DataFrame([
        {"Luật": law["law_name"], "Hiệu lực từ": law["effective_date"]}
        for law in laws
    ])


def graph_type_counts(nodes: List[Dict], edges: List[Dict]) -> Dict:
    """Tally node/edge counts by type -- mirrors src/api/main.py's graph_stats()
    endpoint logic, duplicated here because the Gradio path reads
    MetadataStore.get_graph() in-process instead of calling that HTTP endpoint."""
    node_types: Dict[str, int] = {}
    for n in nodes:
        nt = n.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1
    edge_types: Dict[str, int] = {}
    for e in edges:
        et = e.get("ref_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1
    return {
        "nodes": len(nodes), "edges": len(edges),
        "node_types": node_types, "edge_types": edge_types,
    }
