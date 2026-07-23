# UI Info Refresh + Feature Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix stale model/embedding info shown in the Streamlit UI, fix a real key-mismatch bug in the Documents page, and bring the Gradio ZeroGPU Space (the live public demo) up to feature parity with Streamlit's Documents/Evaluation pages plus a new "Đồ thị & Hiệu lực" (graph/effective-law) view on both UIs.

**Architecture:** Streamlit pages keep talking to the FastAPI backend over HTTP (existing pattern in `chat.py`/`documents.py`/`eval_dashboard.py`); the Gradio Space keeps calling `vector_store`/`metadata_store`/`retriever` in-process (existing M5 pattern, no HTTP hop, no `@spaces.GPU` on non-chat tabs). A new `gradio_helpers.py` layer of pure, unit-tested functions sits between the Gradio Blocks UI and the in-process singletons, mirroring why that file already exists.

**Tech Stack:** Streamlit (`requests` to FastAPI), Gradio `Blocks`/`Tabs` (in-process calls), pandas, pytest.

Spec: [`docs/superpowers/specs/2026-07-23-ui-info-refresh-and-feature-parity-design.md`](../specs/2026-07-23-ui-info-refresh-and-feature-parity-design.md)

## Global Constraints

- Do not modify `README.md`, `docs/ARCHITECTURE.md`, or `docs/LLM_MODEL_STRATEGY.md` — scope is UI display strings only.
- Do not change the latency numbers in `eval_dashboard.py` (`"3.0s"`, `"~0.1s"`, `"~2.8s"`, `"$0.00"`) — only the model-name references.
- No graph visualization library (no networkx-vis/d3) — tables/markdown only.
- Do not touch `handle_turn`, `@spaces.GPU`, or Gradio's built-in chat history/session logic in `app_gradio.py`.
- Do not add UI for the admin enrichment endpoints (`/api/pipeline/enrich`, `/api/pipeline/status`) — they require `ADMIN_KEY` and are not meant for a public UI.
- Display the LLM model name by reading `LLMGateway.MODEL_CHAIN[0]` directly (a class attribute — no need to instantiate `LLMGateway`); never hardcode a model name string, and never add a formatting/mapping table to prettify it.
- Any document picker (Streamlit `selectbox` or Gradio `Dropdown`) must resolve `document_id` via the widget's native label/value mechanism (`format_func` in Streamlit, `(label, value)` tuples in Gradio) — never by parsing an `int` out of a display string.
- Date-parsing `try/except ValueError` is only needed on the Gradio path (`gr.Textbox` free text). Streamlit's `st.date_input` is a calendar widget and cannot produce a malformed string — do not add defensive parsing there.

---

### Task 1: Pure helper functions for the Gradio tabs (`gradio_helpers.py`)

**Files:**
- Modify: `src/ui/gradio_helpers.py`
- Test: `tests/test_gradio_helpers.py`

**Interfaces:**
- Produces (consumed by Task 6):
  - `document_choices(documents: List[Dict]) -> List[Tuple[str, int]]`
  - `rename_document_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `format_system_info(model_id: str, collection_name: str, points_count: int) -> str`
  - `parse_effective_date(at_date_str: str) -> Optional[date]` (raises `ValueError` on malformed non-empty input)
  - `format_effective_laws(laws: List[Dict]) -> pd.DataFrame`
  - `graph_type_counts(nodes: List[Dict], edges: List[Dict]) -> Dict`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gradio_helpers.py` (add these imports to the existing `from src.ui.gradio_helpers import (...)` block at the top of the file, alongside `format_citations_markdown`, `history_to_turns`, `is_quota_exceeded`):

```python
    document_choices,
    format_effective_laws,
    format_system_info,
    graph_type_counts,
    parse_effective_date,
    rename_document_columns,
```

Add near the top of the file, below the existing import block:

```python
from datetime import date

import pandas as pd
import pytest
```

Append these test functions at the end of the file:

```python
def test_document_choices_returns_label_value_tuples():
    documents = [
        {"document_id": 101, "title": "Luật Đất đai 2024"},
        {"document_id": 205, "title": "Bộ luật Hình sự 2015"},
    ]
    assert document_choices(documents) == [
        ("Luật Đất đai 2024", 101),
        ("Bộ luật Hình sự 2015", 205),
    ]


def test_document_choices_empty():
    assert document_choices([]) == []


def test_rename_document_columns():
    df = pd.DataFrame([{"title": "Luật X", "legal_type": "Luật", "issuance_date": "2024-01-01"}])
    renamed = rename_document_columns(df)
    assert list(renamed.columns) == ["Tên văn bản", "Loại văn bản", "Ngày ban hành"]


def test_format_system_info_includes_all_fields():
    info = format_system_info("groq/llama-3.3-70b-versatile", "luat_law_ctx_v4_aiteamvn", 60958)
    assert "groq/llama-3.3-70b-versatile" in info
    assert "AITeamVN/Vietnamese_Embedding" in info
    assert "AITeamVN/Vietnamese_Reranker" in info
    assert "luat_law_ctx_v4_aiteamvn" in info
    assert "60,958" in info


def test_parse_effective_date_empty_string_returns_none():
    assert parse_effective_date("") is None


def test_parse_effective_date_valid_string():
    assert parse_effective_date("2026-07-23") == date(2026, 7, 23)


def test_parse_effective_date_malformed_raises_value_error():
    with pytest.raises(ValueError):
        parse_effective_date("not-a-date")


def test_format_effective_laws_builds_dataframe():
    laws = [{"law_name": "Luật Đất đai 2024", "effective_date": "2025-01-01"}]
    df = format_effective_laws(laws)
    assert list(df.columns) == ["Luật", "Hiệu lực từ"]
    assert df.iloc[0]["Luật"] == "Luật Đất đai 2024"


def test_format_effective_laws_empty():
    df = format_effective_laws([])
    assert len(df) == 0


def test_graph_type_counts_tallies_by_type():
    nodes = [{"node_type": "article"}, {"node_type": "article"}, {"node_type": "law"}]
    edges = [{"ref_type": "amends"}, {"ref_type": "refers_to"}]
    counts = graph_type_counts(nodes, edges)
    assert counts["nodes"] == 3
    assert counts["edges"] == 2
    assert counts["node_types"] == {"article": 2, "law": 1}
    assert counts["edge_types"] == {"amends": 1, "refers_to": 1}


def test_graph_type_counts_empty():
    counts = graph_type_counts([], [])
    assert counts == {"nodes": 0, "edges": 0, "node_types": {}, "edge_types": {}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_gradio_helpers.py -v`
Expected: `ImportError` (the 6 new names don't exist yet in `src/ui/gradio_helpers.py`).

- [ ] **Step 3: Implement the 6 functions**

Add to the top of `src/ui/gradio_helpers.py`, right after the existing `from typing import Dict, List` line — change that line and add two more:

```python
from typing import Dict, List, Optional, Tuple

from datetime import date

import pandas as pd
```

Append these functions at the end of the file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_gradio_helpers.py -v`
Expected: all tests PASS (existing 6 + 12 new = 18 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ui/gradio_helpers.py tests/test_gradio_helpers.py
git commit -m "feat(ui): pure helpers for Gradio Documents/Stats/Graph tabs"
```

---

### Task 2: Fix Documents page bug (`documents.py`)

**Files:**
- Modify: `src/ui/pages/documents.py:44`, `src/ui/pages/documents.py:57`

**Interfaces:** none (leaf page, no other task depends on this file).

- [ ] **Step 1: Fix the key mismatch**

In `src/ui/pages/documents.py`, change line 44:

```python
                    st.success(f"Tìm thấy {data.get('total_documents', 0)} văn bản")
```

to:

```python
                    st.success(f"Tìm thấy {data.get('total', 0)} văn bản")
```

(`/api/documents` returns key `total`, not `total_documents` — see `src/api/main.py:432` `list_documents()` returning `result` from `LegalVectorStore.list_documents()`, and `src/db/qdrant_client.py:563-568` which builds `{"total": total, ...}`. The old key always defaulted to 0, so the success banner showed "Tìm thấy 0 văn bản" even on a real hit.)

- [ ] **Step 2: Fix the misleading error message**

Change line 57:

```python
            except Exception as e:
                st.warning(f"Chức năng đang được phát triển. {e}")
```

to:

```python
            except Exception as e:
                st.error(f"Không thể kết nối đến backend: {e}")
```

(This `except` only ever fires on a connection/timeout failure — the "no results" case is already handled inside the `try` via the 200-response branch without raising. "Chức năng đang được phát triển" mislabels a real connectivity error as an unfinished feature.)

- [ ] **Step 3: Verify manually**

Run in one terminal: `make api`
Run in a second terminal: `make ui`
In the browser, open the Documents page, leave all filters blank, click "🔍 Tìm kiếm".
Expected: the success banner shows the real document count (not 0), matching the row count in the table below it.
Stop the API process (Ctrl-C in the first terminal) and click "🔍 Tìm kiếm" again.
Expected: a red `st.error` box reading "Không thể kết nối đến backend: ...", not "Chức năng đang được phát triển".

- [ ] **Step 4: Commit**

```bash
git add src/ui/pages/documents.py
git commit -m "fix(ui): documents page total-count key mismatch + error message"
```

---

### Task 3: Fix stale model name in Eval dashboard (`eval_dashboard.py`)

**Files:**
- Modify: `src/ui/pages/eval_dashboard.py:9` (import), `:147`, `:174`, `:179`

**Interfaces:** none.

- [ ] **Step 1: Add the import**

In `src/ui/pages/eval_dashboard.py`, after line 9 (`API_BASE = os.environ.get("API_URL", "http://localhost:8000")`), add:

```python
from src.llm.gateway import LLMGateway
```

- [ ] **Step 2: Replace the 3 hardcoded "Qwen 3 32B (Groq)" references**

Change line 147:

```python
            cols[2].metric("🔍 LLM Model", "Qwen 3 32B (Groq)")
```

to:

```python
            cols[2].metric("🔍 LLM Model", LLMGateway.MODEL_CHAIN[0])
```

Change line 174:

```python
    st.caption("Based on local testing with Qwen 3 32B on Groq.")
```

to:

```python
    st.caption(f"Based on local testing with {LLMGateway.MODEL_CHAIN[0]}.")
```

Change line 179:

```python
    col3.metric("LLM Latency", "~2.8s", delta="Qwen 3 32B streaming")
```

to:

```python
    col3.metric("LLM Latency", "~2.8s", delta=f"{LLMGateway.MODEL_CHAIN[0]} streaming")
```

(The `"3.0s"`/`"~0.1s"`/`"~2.8s"`/`"$0.00"` numeric estimates are unchanged — only the retired model name is fixed.)

- [ ] **Step 3: Verify manually**

With `make api` and `make ui` running, open the Evaluation page.
Expected: "🔍 LLM Model" metric and the two captions below the Pipeline Performance section all show `groq/llama-3.3-70b-versatile` (or whatever `LLMGateway.MODEL_CHAIN[0]` currently is) — no occurrence of "Qwen 3 32B" remains on the page.

- [ ] **Step 4: Commit**

```bash
git add src/ui/pages/eval_dashboard.py
git commit -m "fix(ui): eval dashboard shows retired Qwen 3 32B instead of the live model"
```

---

### Task 4: New Streamlit page — "Đồ thị & Hiệu lực" (`graph.py`)

**Files:**
- Create: `src/ui/pages/graph.py`

**Interfaces:**
- Produces (consumed by Task 5): `render_graph_page() -> None`
- Consumes: `GET /api/timeline/effective`, `GET /api/documents`, `GET /api/article/history`, `GET /api/graph/stats` (all already implemented in `src/api/main.py`, no backend change).

- [ ] **Step 1: Create the page**

Create `src/ui/pages/graph.py`:

```python
"""
Streamlit Graph & Effective-Law Page — laws in effect on a given date,
per-article amendment history, and cross-reference graph stats.
"""

from datetime import date

import os
import streamlit as st
import requests

API_BASE = os.environ.get("API_URL", "http://localhost:8000")


def render_graph_page():
    st.title("🕸️ Đồ thị & Hiệu lực")
    st.caption("Tra cứu luật còn hiệu lực theo ngày, lịch sử sửa đổi điều luật, "
               "và thống kê đồ thị trích dẫn chéo.")

    _render_effective_laws()
    st.divider()
    _render_article_history()
    st.divider()
    _render_graph_stats()


def _render_effective_laws():
    st.subheader("📅 Luật có hiệu lực tại 1 ngày")
    at_date = st.date_input("Chọn ngày", value=date.today())
    if st.button("🔍 Tra cứu hiệu lực"):
        try:
            resp = requests.get(f"{API_BASE}/api/timeline/effective",
                                 params={"at_date": at_date.isoformat()}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"{data.get('count', 0)} luật có hiệu lực tại {data.get('at_date')}")
                if data.get("laws"):
                    st.dataframe(data["laws"], use_container_width=True,
                                 column_config={"law_name": "Tên luật",
                                                "effective_date": "Ngày hiệu lực"})
            else:
                st.error(f"Lỗi API: {resp.status_code}")
        except Exception as e:
            st.error(f"Không thể kết nối đến backend: {e}")


def _render_article_history():
    st.subheader("📜 Lịch sử sửa đổi 1 điều luật")
    keyword = st.text_input("Tìm văn bản theo từ khoá", key="graph_doc_keyword")
    documents = []
    if keyword:
        try:
            resp = requests.get(f"{API_BASE}/api/documents",
                                 params={"keyword": keyword, "page_size": 20}, timeout=10)
            if resp.status_code == 200:
                documents = resp.json().get("documents", [])
            else:
                st.error(f"Lỗi API: {resp.status_code}")
        except Exception as e:
            st.error(f"Không thể kết nối đến backend: {e}")

    if documents:
        selected = st.selectbox("Chọn văn bản", options=documents,
                                 format_func=lambda d: d["title"])
        article = st.text_input("Số điều", placeholder="Điều 5")
        if st.button("🔍 Tra cứu lịch sử") and article:
            try:
                resp = requests.get(f"{API_BASE}/api/article/history",
                                     params={"doc_id": selected["document_id"],
                                             "article": article}, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("error"):
                        st.warning(data["error"])
                    else:
                        st.json(data)
                else:
                    st.error(f"Lỗi API: {resp.status_code}")
            except Exception as e:
                st.error(f"Không thể kết nối đến backend: {e}")
    elif keyword:
        st.info("Không tìm thấy văn bản khớp từ khoá.")


def _render_graph_stats():
    st.subheader("📊 Thống kê đồ thị trích dẫn chéo")
    try:
        resp = requests.get(f"{API_BASE}/api/graph/stats", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            col1, col2 = st.columns(2)
            col1.metric("Tổng số node", data.get("nodes", 0))
            col1.write(data.get("node_types", {}))
            col2.metric("Tổng số edge", data.get("edges", 0))
            col2.write(data.get("edge_types", {}))
        else:
            st.error(f"Lỗi API: {resp.status_code}")
    except Exception as e:
        st.error(f"Không thể kết nối đến backend: {e}")
```

- [ ] **Step 2: Sanity-check the module imports cleanly**

Run: `.venv/bin/python -c "import src.ui.pages.graph"`
Expected: no output, exit code 0 (this only checks for syntax/import errors — `render_graph_page()` itself needs a running Streamlit session and is verified in Task 5's manual check).

- [ ] **Step 3: Commit**

```bash
git add src/ui/pages/graph.py
git commit -m "feat(ui): Đồ thị & Hiệu lực page (effective-law lookup, article history, graph stats)"
```

---

### Task 5: Wire `app.py` — sidebar info fix + nav item for the new page

**Files:**
- Modify: `src/ui/app.py:14` (import), `:28` (nav list), `:54-56` (sidebar captions), append new `elif` branch after line 70.

**Interfaces:**
- Consumes: `render_graph_page` from Task 4 (`src/ui/pages/graph.py`).

- [ ] **Step 1: Add the import**

In `src/ui/app.py`, after line 14 (`import streamlit as st`), add:

```python

from src.llm.gateway import LLMGateway
```

- [ ] **Step 2: Add the 4th nav item**

Change line 28:

```python
    ["💬 Chat", "📁 Documents", "📊 Evaluation"],
```

to:

```python
    ["💬 Chat", "📁 Documents", "🕸️ Đồ thị & Hiệu lực", "📊 Evaluation"],
```

- [ ] **Step 3: Fix the sidebar Model/Embedding captions, add Reranker**

Change lines 55-56:

```python
st.sidebar.caption("Model: Qwen 3 32B (Groq)")
st.sidebar.caption("Embedding: BGE-M3 (1024-dim)")
```

to:

```python
st.sidebar.caption(f"Model: {LLMGateway.MODEL_CHAIN[0]} (auto-fallback)")
st.sidebar.caption("Embedding: AITeamVN/Vietnamese_Embedding (dense) + BGE-M3 (sparse)")
st.sidebar.caption("Reranker: AITeamVN/Vietnamese_Reranker")
```

- [ ] **Step 4: Route the new nav item**

After the existing block:

```python
elif page == "📁 Documents":
    from src.ui.pages.documents import render_documents_page
    render_documents_page()
```

insert:

```python

elif page == "🕸️ Đồ thị & Hiệu lực":
    from src.ui.pages.graph import render_graph_page
    render_graph_page()
```

(Keep the existing `elif page == "📊 Evaluation":` branch below this, unchanged.)

- [ ] **Step 5: Verify manually**

Run in one terminal: `make api`
Run in a second terminal: `make ui`
In the browser sidebar, confirm exactly 4 nav items in this order: 💬 Chat, 📁 Documents, 🕸️ Đồ thị & Hiệu lực, 📊 Evaluation.
Confirm the sidebar shows 3 caption lines: `Model: groq/llama-3.3-70b-versatile (auto-fallback)`, `Embedding: AITeamVN/Vietnamese_Embedding (dense) + BGE-M3 (sparse)`, `Reranker: AITeamVN/Vietnamese_Reranker`.
Click "🕸️ Đồ thị & Hiệu lực": confirm the "Luật có hiệu lực tại 1 ngày" section defaults to today's date and, after clicking "🔍 Tra cứu hiệu lực", shows a result (count + table, or an explicit error if the backend enrichment index is empty — either is acceptable, a silent crash is not).
Type a keyword (e.g. a word from a known document title) into "Tìm văn bản theo từ khoá": confirm a dropdown of matching titles appears, selecting one plus entering an article number and clicking "🔍 Tra cứu lịch sử" shows either a JSON result or a "Document not found"-style warning.
Confirm "📊 Thống kê đồ thị trích dẫn chéo" shows node/edge counts without raising an unhandled exception in the terminal running `make api`.

- [ ] **Step 6: Commit**

```bash
git add src/ui/app.py
git commit -m "feat(ui): wire Đồ thị & Hiệu lực nav item, fix sidebar model/embedding info"
```

---

### Task 6: Gradio Space — `ChatInterface` → `Blocks` + `Tabs` (Documents/Thống kê/Đồ thị)

**Files:**
- Modify: `src/ui/app_gradio.py` (imports near top; module-scope singletons around line 52; replace lines 138-149 — the `demo = gr.ChatInterface(...)` block and `if __name__ == "__main__":` block — with the new `Blocks`/`Tabs` structure).

**Interfaces:**
- Consumes: `document_choices`, `rename_document_columns`, `format_system_info`, `parse_effective_date`, `format_effective_laws`, `graph_type_counts` from Task 1 (`src/ui/gradio_helpers.py`).
- Consumes: `EnhancedRetriever(vector_store, metadata_store, embedder)` and `.get_article_history(doc_id: int, article: str) -> Dict` (`src/pipeline/retriever.py`, already exists, unmodified).
- Consumes: `ALL_LEGAL_TYPES: list[str]` (`src/data/constants.py`, already exists).

This file's module scope eagerly loads real models (AITeamVN embedder, reranker, LLM gateway) and connects to Qdrant — per its own header comment, it "can't be unit-tested" locally, which is exactly why `gradio_helpers.py` exists as the tested surface (Task 1 already covers the pure logic this task wires up). This task's own verification is therefore a syntax check + a live-Space smoke check, not a local pytest run — do not claim a local functional test passed if it wasn't actually run against the live Space; if the live-Space check can't be completed in this session, say so explicitly rather than asserting success.

- [ ] **Step 1: Add imports**

In `src/ui/app_gradio.py`, change:

```python
import json

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from FlagEmbedding import FlagReranker

from src.data.constants import DEFAULT_COLLECTION_NAME
from src.data.metadata_store import MetadataStore
from src.data.mixed_embedder import build_embedder
from src.db.qdrant_client import LegalVectorStore
from src.llm.gateway import LLMGateway
from src.rag.citation_lookup import CitationLookup
from src.rag.condense import condense_query
from src.rag.degraded import DEGRADED_MSG, is_exhaustion
from src.rag.orchestrator import AdaptiveRAGPipeline, RAGResponse
from src.rag.semantic_cache import SemanticCache
from src.ui.gradio_helpers import (
    format_citations_markdown,
    history_to_turns,
    is_quota_exceeded,
)
```

to:

```python
import json

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import pandas as pd
from FlagEmbedding import FlagReranker

from src.data.constants import ALL_LEGAL_TYPES, DEFAULT_COLLECTION_NAME
from src.data.metadata_store import MetadataStore
from src.data.mixed_embedder import build_embedder
from src.db.qdrant_client import LegalVectorStore
from src.llm.gateway import LLMGateway
from src.pipeline.retriever import EnhancedRetriever
from src.rag.citation_lookup import CitationLookup
from src.rag.condense import condense_query
from src.rag.degraded import DEGRADED_MSG, is_exhaustion
from src.rag.orchestrator import AdaptiveRAGPipeline, RAGResponse
from src.rag.semantic_cache import SemanticCache
from src.ui.gradio_helpers import (
    document_choices,
    format_citations_markdown,
    format_effective_laws,
    format_system_info,
    graph_type_counts,
    history_to_turns,
    is_quota_exceeded,
    parse_effective_date,
    rename_document_columns,
)
```

- [ ] **Step 2: Add the `retriever` module-scope singleton**

Change:

```python
vector_store = LegalVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding", device="cuda")
reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=True, device="cuda")
llm = LLMGateway()
metadata_store = MetadataStore()
citation_lookup = CitationLookup(vector_store)
```

to:

```python
vector_store = LegalVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding", device="cuda")
reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=True, device="cuda")
llm = LLMGateway()
metadata_store = MetadataStore()
citation_lookup = CitationLookup(vector_store)
retriever = EnhancedRetriever(vector_store=vector_store, metadata_store=metadata_store,
                               embedder=embedder)
```

(Module scope, next to `citation_lookup` — created once, matching how every other singleton in this file is loaded, not re-created per button click.)

- [ ] **Step 3: Replace the tail of the file**

Replace everything from `demo = gr.ChatInterface(` to the end of the file:

```python
demo = gr.ChatInterface(
    fn=handle_turn,
    type="messages",
    title="Vietnamese Legal Knowledge Bot",
    description=("Adaptive RAG tra cứu văn bản pháp luật Việt Nam (600 văn bản). "
                 "Chạy trên HF ZeroGPU (free tier) -- lần hỏi đầu có thể chậm "
                 "(cold start nạp model)."),
)

if __name__ == "__main__":
    demo.launch()
```

with:

```python
chat_interface = gr.ChatInterface(
    fn=handle_turn,
    type="messages",
    title="Vietnamese Legal Knowledge Bot",
    description=("Adaptive RAG tra cứu văn bản pháp luật Việt Nam (600 văn bản). "
                 "Chạy trên HF ZeroGPU (free tier) -- lần hỏi đầu có thể chậm "
                 "(cold start nạp model)."),
)


def search_documents(legal_type: str, keyword: str):
    """Documents tab search -- pure CPU/Qdrant call, no @spaces.GPU needed."""
    lt = None if legal_type == "Tất cả" else legal_type
    result = vector_store.list_documents(legal_type=lt, keyword=keyword or None,
                                          page=1, page_size=50)
    df = rename_document_columns(pd.DataFrame(result["documents"]))
    return f"Tìm thấy {result['total']} văn bản", df


def system_info_display():
    info = vector_store.get_collection_info()
    return format_system_info(LLMGateway.MODEL_CHAIN[0], info["name"], info["points_count"])


def refresh_document_choices(keyword: str):
    result = vector_store.list_documents(keyword=keyword or None, page=1, page_size=50)
    return gr.update(choices=document_choices(result["documents"]))


def effective_laws_display(at_date_str: str):
    try:
        at_date = parse_effective_date(at_date_str)
    except ValueError:
        return f"Định dạng ngày không hợp lệ: '{at_date_str}' (cần YYYY-MM-DD)"
    laws = metadata_store.get_effective_laws(at_date)
    return format_effective_laws(laws)


def article_history_display(doc_id, article: str):
    if not doc_id or not article:
        return {"error": "Vui lòng chọn văn bản và nhập số điều."}
    return retriever.get_article_history(int(doc_id), article)


def graph_stats_display():
    graph = metadata_store.get_graph()
    counts = graph_type_counts(graph.get("nodes", []), graph.get("edges", []))
    lines = [f"**Tổng số node:** {counts['nodes']} · **Tổng số edge:** {counts['edges']}", "",
             "**Theo loại node:**"]
    lines += [f"- {k}: {v}" for k, v in counts["node_types"].items()]
    lines += ["", "**Theo loại quan hệ (edge):**"]
    lines += [f"- {k}: {v}" for k, v in counts["edge_types"].items()]
    return "\n".join(lines)


with gr.Blocks(title="Vietnamese Legal Knowledge Bot") as demo:
    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            chat_interface.render()

        with gr.Tab("📁 Documents"):
            with gr.Row():
                doc_legal_type = gr.Dropdown(["Tất cả"] + ALL_LEGAL_TYPES, value="Tất cả",
                                              label="Loại văn bản")
                doc_keyword = gr.Textbox(label="Từ khoá", placeholder="Nhập từ khoá tìm kiếm...")
                doc_search_btn = gr.Button("🔍 Tìm kiếm")
            doc_result_label = gr.Markdown()
            doc_result_table = gr.Dataframe()
            doc_search_btn.click(fn=search_documents, inputs=[doc_legal_type, doc_keyword],
                                  outputs=[doc_result_label, doc_result_table])

        with gr.Tab("📊 Thống kê"):
            gr.Markdown(system_info_display())

        with gr.Tab("🕸️ Đồ thị & Hiệu lực"):
            gr.Markdown("### 📅 Luật có hiệu lực tại 1 ngày")
            with gr.Row():
                eff_date_box = gr.Textbox(label="Ngày (YYYY-MM-DD)", placeholder="2026-07-23")
                eff_btn = gr.Button("🔍 Tra cứu hiệu lực")
            eff_table = gr.Dataframe()
            eff_btn.click(fn=effective_laws_display, inputs=[eff_date_box], outputs=[eff_table])

            gr.Markdown("### 📜 Lịch sử sửa đổi 1 điều luật")
            with gr.Row():
                hist_keyword = gr.Textbox(label="Tìm văn bản theo từ khoá")
                hist_search_btn = gr.Button("Tìm văn bản")
            hist_doc_dropdown = gr.Dropdown(label="Chọn văn bản", choices=[])
            hist_article_box = gr.Textbox(label="Số điều", placeholder="Điều 5")
            hist_btn = gr.Button("🔍 Tra cứu lịch sử")
            hist_result = gr.JSON()
            hist_search_btn.click(fn=refresh_document_choices, inputs=[hist_keyword],
                                   outputs=[hist_doc_dropdown])
            hist_btn.click(fn=article_history_display,
                           inputs=[hist_doc_dropdown, hist_article_box], outputs=[hist_result])

            gr.Markdown("### 📊 Thống kê đồ thị trích dẫn chéo")
            gr.Markdown(graph_stats_display())


if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 4: Syntax check**

Run: `.venv/bin/python -m py_compile src/ui/app_gradio.py`
Expected: exit code 0, no output. (This only catches syntax errors — it does NOT import the file, since import triggers the full model/Qdrant load this file's own header comment says can't run in a lightweight local check.)

- [ ] **Step 5: Run the full test suite to confirm no regressions**

Run: `.venv/bin/pytest tests/ -v`
Expected: all tests pass, same count as before this plan plus the 12 new ones from Task 1.

- [ ] **Step 6: Commit**

```bash
git add src/ui/app_gradio.py
git commit -m "feat(ui): Gradio Space gains Documents/Thống kê/Đồ thị tabs alongside Chat"
```

- [ ] **Step 7: Live verification (cannot be done locally — say so if skipped)**

Deploy the change: `python scripts/deploy_space.py` (per the existing M5 deploy workflow — pushes to the live HF Space at `https://huggingface.co/spaces/quockhanh41/vietnamese-legal-bot`).
On the live Space, click through all 4 tabs: confirm 💬 Chat still answers questions (unchanged behavior), 📁 Documents returns results, 📊 Thống kê shows the model/embedding/reranker/collection summary, 🕸️ Đồ thị & Hiệu lực's 3 sections all respond without an unhandled exception.
If this step is not performed in the current session (e.g. ZeroGPU daily quota is exhausted, matching the known 2026-07-23 quota issue noted in the M5 pivot spec), state plainly that live verification is pending — do not report Task 6 as fully done without it.

---

## Self-Review Notes

- **Spec coverage**: Phần 1 (info fixes) → Tasks 2/3/5. Phần 2 (Documents bug) → Task 2. Phần 3 (graph.py) → Task 4, wired in Task 5. Phần 4 (Gradio tabs) → Task 6. All 4 spec sections have a task.
- **Review feedback already folded into this plan** (from `nhan-xet.md`, resolved in the spec before this plan was written): `EnhancedRetriever` at module scope (Task 6 Step 2), Vietnamese column rename without the fabricated `effective_date` column (Task 1 `rename_document_columns` / Task 6 `search_documents`), dropdown/selectbox by value not string-parsing (Task 1 `document_choices`, Task 4 `format_func`), date `try/except` scoped to Gradio only (Task 1 `parse_effective_date` docstring, Task 4 uses `st.date_input` with no try/except).
- **Type consistency check**: `document_choices` returns `(str, int)` tuples — matches `gr.Dropdown(choices=...)`'s expected shape and `article_history_display`'s `doc_id` parameter (received as the `int` from the dropdown's selected tuple value, then explicitly `int(doc_id)` cast to guard against Gradio sometimes passing dropdown values through as strings). `format_effective_laws`/`graph_type_counts` key names (`"Luật"`/`"Hiệu lực từ"`, `"nodes"`/`"edges"`/`"node_types"`/`"edge_types"`) are used identically between their Task 1 test assertions and their Task 6 call sites.
