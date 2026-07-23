# M5 Pivot — Gradio + ZeroGPU Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a genuinely free ($0, no card anywhere), public live demo of the Vietnamese Legal Knowledge Bot as an additive Gradio + ZeroGPU HF Space, without touching the existing (still valid, still tested) FastAPI/Streamlit/Docker stack from M5 Phase 1.

**Architecture:** One new file, `src/ui/app_gradio.py`, calls the existing RAG pipeline in-process (no HTTP hop, no second process) and is decorated with `@spaces.GPU` so the heavy embed/rerank/NLI/generate work runs on free ZeroGPU hardware. `src/api/main.py`, `src/ui/app.py`, `Dockerfile`, `start.sh` are untouched except a 1-line import swap. This replaces Phase 2 (Tasks 16-18) of `docs/superpowers/plans/m5.md`, which assumed a Docker Space — that path turned out to require a paid HF plan with no free exception.

**Tech Stack:** Gradio, HF ZeroGPU (`spaces` package), `huggingface_hub` (HfApi), same RAG stack as the rest of the repo (FlagEmbedding, sentence-transformers, qdrant-client, litellm).

## Global Constraints

- **Spec:** [`docs/superpowers/specs/2026-07-23-m5-gradio-zerogpu-pivot-design.md`](../specs/2026-07-23-m5-gradio-zerogpu-pivot-design.md). Every task implements part of it.
- **$0, no card anywhere** (locked with user 2026-07-23). No HF PRO, no cloud provider requiring a billing account even if usage stays free.
- **`import spaces` is the literal first import** in `src/ui/app_gradio.py` — before `os`/`dotenv`/anything that transitively imports torch.
- **Model device is `"cuda"` at module scope, not `"cpu"`** — verified against HF's `zerogpu.md`: `import spaces` monkey-patches `torch.cuda.*` so `.to("cuda")`/`device="cuda"` works at import time with no physical GPU attached (weights pack to disk, stream to VRAM on first `@spaces.GPU` call). `device="cpu"` would silently defeat ZeroGPU and run everything on the tiny CPU-only main process forever.
- **One `@spaces.GPU` boundary per turn**, including the semantic-cache lookup (its embedding call) and the Groq network generation call — not split into sub-boundaries. Splitting would require restructuring the pipeline internals; not justified.
- **`src/api/main.py`, `src/ui/app.py`, `src/ui/pages/chat.py`, `Dockerfile`, `start.sh`, all Phase-1 M5 hardening: untouched**, except Task 1's import swap in `main.py`. Not deleted — described in the README as the self-hosted Docker deploy path.
- **No local unit test can exercise `@spaces.GPU` behavior** — per HF's own guidance ("iterate ON the Space, not locally"). Local verification for `app_gradio.py` tops out at `python3 -m py_compile`; real verification happens against the live Space.
- **Test conventions**: `.venv/bin/python -m pytest -q` for all pure-logic tests (no GPU, no network, no live models) — matches existing repo convention (currently 484+ passing tests, plus M5 Phase 1's additions).
- **Secrets never printed to chat/terminal** — read via `dotenv_values()`/`source .env` and passed through without echoing values, same pattern used throughout this session.
- **`huggingface_hub` HfApi (Python), not the `hf` CLI's `spaces` subcommand** — the CLI installed on this machine (0.36.2-era `hf` binary) has no `spaces` subcommand; `HfApi().create_repo(..., space_hardware=SpaceHardware.ZERO_A10G)`, `add_space_secret`, `get_space_runtime` are the verified, available methods.

---

## PHASE 1 — Code (TDD, local)

### Task 1: Extract degraded-message logic into `src/rag/degraded.py`

**Files:**
- Create: `src/rag/degraded.py`
- Modify: `src/api/main.py` (remove local `DEGRADED_MSG`/`_is_exhaustion`, import from new module, update 2 call sites)
- Test: `tests/test_degraded.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `degraded.DEGRADED_MSG: str`, `degraded.is_exhaustion(e: Exception) -> bool`. Both `src/api/main.py` and the new `src/ui/app_gradio.py` (Task 3) import from here — single source of truth for the LLM-provider-exhaustion signal and its Vietnamese message.

- [ ] **Step 1: Write the failing test**

Create `tests/test_degraded.py`:
```python
from src.rag.degraded import DEGRADED_MSG, is_exhaustion


def test_is_exhaustion_true_for_gateway_signal():
    assert is_exhaustion(RuntimeError("All models failed. Last error: 429")) is True


def test_is_exhaustion_false_for_other_errors():
    assert is_exhaustion(ValueError("something else")) is False
    assert is_exhaustion(RuntimeError("unrelated runtime error")) is False


def test_degraded_msg_is_vietnamese_and_nonempty():
    assert "quá tải" in DEGRADED_MSG
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_degraded.py -q`
Expected: FAIL (`src.rag.degraded` doesn't exist).

- [ ] **Step 3: Implement**

Create `src/rag/degraded.py`:
```python
"""Shared quota-exhaustion signal + user-facing message for graceful
degradation (M5). Single source of truth so src/api/main.py (FastAPI path)
and src/ui/app_gradio.py (Gradio+ZeroGPU path) show identical behavior:
degraded replies are never cached/persisted (same rule as refusals)."""

DEGRADED_MSG = ("Xin lỗi, hệ thống đang quá tải hạn mức miễn phí. "
                "Vui lòng thử lại sau ít phút.")


def is_exhaustion(e: Exception) -> bool:
    """True for the LLM gateway's 'all providers exhausted' signal
    (src/llm/gateway.py raises RuntimeError("All models failed...") when
    every provider in the fallback chain is rate-limited/out of budget)."""
    return isinstance(e, RuntimeError) and str(e).startswith("All models failed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_degraded.py -q`
Expected: PASS.

- [ ] **Step 5: Wire `main.py` to the shared module**

In `src/api/main.py`, delete the local definitions (currently at `main.py:165-171`):
```python
DEGRADED_MSG = ("Xin lỗi, hệ thống đang quá tải hạn mức miễn phí. "
                "Vui lòng thử lại sau ít phút.")


def _is_exhaustion(e):
    """True for the gateway's 'all providers exhausted' signal (gateway.py)."""
    return isinstance(e, RuntimeError) and str(e).startswith("All models failed")
```
Replace with an import near the other `src.rag.*` imports at the top of the file:
```python
from src.rag.degraded import DEGRADED_MSG, is_exhaustion as _is_exhaustion
```
(Aliasing preserves the existing `_is_exhaustion(e)` call sites at `main.py:307` and `main.py:387` unchanged — no other edit needed in the route handlers.)

- [ ] **Step 6: Run the full existing degraded-path suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/test_api_degraded.py tests/test_degraded.py -q`
Expected: all PASS (no change in behavior, only where the constant/function live).

- [ ] **Step 7: Commit**

```bash
git add src/rag/degraded.py src/api/main.py tests/test_degraded.py
git commit -m "refactor(m5): extract DEGRADED_MSG/is_exhaustion into src/rag/degraded.py

Shared by src/api/main.py and the upcoming Gradio+ZeroGPU app so both
deploy paths show identical quota-exhaustion behavior."
```

---

### Task 2: Pure Gradio helper functions

**Files:**
- Create: `src/ui/gradio_helpers.py`
- Test: `tests/test_gradio_helpers.py` (new)

**Interfaces:**
- Consumes: nothing (pure functions, no model/network/GPU imports — deliberately lightweight so this module is fast to unit test, unlike `app_gradio.py`).
- Produces: `history_to_turns(history: list[dict]) -> list[dict]`, `format_citations_markdown(citations: list[dict]) -> str`, `is_quota_exceeded(e: Exception) -> bool`. All three are imported by `src/ui/app_gradio.py` (Task 3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gradio_helpers.py`:
```python
from src.ui.gradio_helpers import (
    format_citations_markdown,
    history_to_turns,
    is_quota_exceeded,
)


def test_history_to_turns_pairs_user_assistant():
    history = [
        {"role": "user", "content": "Tội trộm cắp bị phạt sao?"},
        {"role": "assistant", "content": "Theo Điều 173..."},
        {"role": "user", "content": "Còn tội cướp giật thì sao?"},
        {"role": "assistant", "content": "Theo Điều 171..."},
    ]
    turns = history_to_turns(history)
    assert turns == [
        {"question": "Tội trộm cắp bị phạt sao?", "answer": "Theo Điều 173..."},
        {"question": "Còn tội cướp giật thì sao?", "answer": "Theo Điều 171..."},
    ]


def test_history_to_turns_drops_trailing_unanswered_question():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2 (no answer yet)"},
    ]
    assert history_to_turns(history) == [{"question": "Q1", "answer": "A1"}]


def test_history_to_turns_empty():
    assert history_to_turns([]) == []


def test_format_citations_markdown_empty():
    assert format_citations_markdown([]) == ""


def test_format_citations_markdown_lists_article_and_title():
    citations = [
        {"block_index": 1, "document_title": "Bộ luật Hình sự 2015", "article": "Điều 173"},
        {"block_index": 2, "document_title": "Bộ luật Hình sự 2015", "article": "Điều 174"},
    ]
    out = format_citations_markdown(citations)
    assert "Căn cứ pháp lý" in out
    assert "1. Điều 173, Bộ luật Hình sự 2015" in out
    assert "2. Điều 174, Bộ luật Hình sự 2015" in out


def test_is_quota_exceeded_true_for_zerogpu_quota_text():
    assert is_quota_exceeded(Exception("ZeroGPU quota exceeded")) is True


def test_is_quota_exceeded_false_for_unrelated_errors():
    assert is_quota_exceeded(ValueError("bad input")) is False
    assert is_quota_exceeded(RuntimeError("All models failed. Last error: 429")) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_gradio_helpers.py -q`
Expected: FAIL (`src.ui.gradio_helpers` doesn't exist).

- [ ] **Step 3: Implement**

Create `src/ui/gradio_helpers.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_gradio_helpers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ui/gradio_helpers.py tests/test_gradio_helpers.py
git commit -m "feat(m5): pure Gradio helper functions (history adapter, citation markdown, quota detection)"
```

---

### Task 3: `src/ui/app_gradio.py` — the Gradio + ZeroGPU app

**Files:**
- Create: `src/ui/app_gradio.py`

**Interfaces:**
- Consumes: `src.rag.degraded.{DEGRADED_MSG, is_exhaustion}` (Task 1), `src.ui.gradio_helpers.{history_to_turns, format_citations_markdown, is_quota_exceeded}` (Task 2), and the existing pipeline building blocks (`LegalVectorStore`, `build_embedder`, `MetadataStore`, `CitationLookup`, `LLMGateway`, `AdaptiveRAGPipeline`, `RAGResponse`, `condense_query`, `SemanticCache`) — all unmodified.
- Produces: a `gr.ChatInterface` instance named `demo`, launched via `demo.launch()` when run directly. This is what HF's Space `app_file` frontmatter (Task 4) points to.

No pytest step here — per the Global Constraints, `@spaces.GPU` behavior cannot be exercised locally (no local GPU, and `spaces` is a no-op off-ZeroGPU per its own docs, so a local "pass" wouldn't prove anything about the real Space). The only local check is a syntax check, which does not execute or resolve any imports.

- [ ] **Step 1: Write the app**

Create `src/ui/app_gradio.py`:
```python
"""HF Space (Gradio + ZeroGPU) front-end -- M5 pivot. Additive: calls the
existing RAG pipeline in-process, no HTTP hop, no separate FastAPI/uvicorn
process. src/api/main.py and src/ui/app.py (Streamlit) are untouched and
remain a self-hosted Docker deploy path -- this file is the live public demo.
See docs/superpowers/specs/2026-07-23-m5-gradio-zerogpu-pivot-design.md."""
import spaces  # MUST be first -- before any torch-touching import

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

QUOTA_MSG = ("Bạn đã dùng hết lượt dùng thử miễn phí hôm nay trên hạ tầng "
             "ZeroGPU. Vui lòng quay lại vào ngày mai!")

# --- Module-scope singletons: loaded ONCE, device="cuda" eagerly. ---
# `import spaces` above monkey-patches torch.cuda so this works with no
# physical GPU attached at import time -- weights get "packed" to disk and
# streamed into real VRAM on the first @spaces.GPU call below. This is HF's
# documented ZeroGPU pattern, not a bug (verified against references/
# zerogpu.md -- see spec's "ZeroGPU boundary" section for the full trace).
vector_store = LegalVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding", device="cuda")
reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=True, device="cuda")
llm = LLMGateway()
metadata_store = MetadataStore()
citation_lookup = CitationLookup(vector_store)


def _embed_query(text: str):
    out = embedder.encode([text], return_dense=True, return_sparse=False)
    return out["dense_vecs"][0].tolist()


semantic_cache = SemanticCache.from_env(_embed_query)

pipeline = AdaptiveRAGPipeline(
    vector_store=vector_store,
    embedder=embedder,
    llm_gateway=llm,
    metadata_store=metadata_store,
    reranker_model="AITeamVN/Vietnamese_Reranker",
    reranker=reranker,
    citation_lookup=citation_lookup,
)


@spaces.GPU(duration=180)
def handle_turn(message, history):
    """One @spaces.GPU allocation per turn -- covers condense, the semantic-
    cache lookup (needs an embedding), and on a miss: retrieve/rerank/
    generate(Groq, network)/NLI verify. Cache lookup is deliberately INSIDE
    this boundary (see spec) so no model-touching call ever runs outside a
    GPU allocation. duration=180 is a placeholder -- Task 9 measures real
    warm latency via gradio_client and tunes this to measured_max * 1.4."""
    turns = history_to_turns(history)
    condensed = condense_query(turns, message, llm)

    cached = semantic_cache.get(condensed, DEFAULT_COLLECTION_NAME) if semantic_cache else None
    if cached is not None:
        yield cached.answer + format_citations_markdown(cached.citations)
        return

    full_answer = ""
    citations = []
    refused = False
    try:
        for token in pipeline.query_stream(condensed):
            if token.startswith("__CITATIONS__"):
                try:
                    citations = json.loads(token[len("__CITATIONS__"):])
                except json.JSONDecodeError:
                    pass
                continue
            if token.startswith("__VERIFICATION__"):
                try:
                    refused = json.loads(token[len("__VERIFICATION__"):]).get("refused", False)
                except json.JSONDecodeError:
                    pass
                continue
            if token.startswith("__STEP__"):
                continue
            full_answer += token
            yield full_answer
    except Exception as e:
        if is_exhaustion(e):
            yield DEGRADED_MSG
            return
        if is_quota_exceeded(e):
            yield QUOTA_MSG
            return
        raise

    yield full_answer + format_citations_markdown(citations)

    if semantic_cache is not None and full_answer and not refused:
        response = RAGResponse(
            answer=full_answer, citations=citations, confidence="unverified",
            disclaimer="", latency_ms=0, pipeline_used="gradio", refused=refused,
        )
        semantic_cache.set(condensed, DEFAULT_COLLECTION_NAME, response)


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

- [ ] **Step 2: Syntax-check locally**

Run: `python3 -m py_compile src/ui/app_gradio.py`
Expected: no output, exit code 0. (This only checks syntax -- it does not execute imports, so it works even though `gradio`/`spaces`/`FlagEmbedding` aren't necessarily importable in every local environment. Do NOT attempt to `import` or run this module locally -- per the Global Constraints, `@spaces.GPU` is only meaningful on the real Space.)

- [ ] **Step 3: Commit**

```bash
git add src/ui/app_gradio.py
git commit -m "feat(m5): Gradio+ZeroGPU app_gradio.py (additive live-demo deploy target)"
```

---

### Task 4: `requirements.txt` (Space root) + `README_GRADIO.md`

**Files:**
- Create: `requirements.txt` (repo root -- this is what HF installs the Gradio Space's Python env from; separate from `pyproject.toml`, which stays for the FastAPI/Docker path)
- Create: `README_GRADIO.md` (repo root -- the Gradio Space's README; `README_HF.md` stays as-is for the Docker self-host path)

**Interfaces:**
- Consumes: nothing.
- Produces: the exact dependency + frontmatter files Task 7's deploy script pushes to the Space.

- [ ] **Step 1: Look up the current stable Gradio version**

Run: `pip index versions gradio 2>&1 | head -3`
Note the first (latest) version printed -- use it for `sdk_version` below instead of guessing (version numbers go stale fast; this command gives the real current one).

- [ ] **Step 2: Create `requirements.txt`**

This list is verified against actual imports in `src/rag/`, `src/pipeline/`, `src/data/`, `src/db/`, `src/llm/`, `src/observability.py` (via `grep`, not guessed) -- these are exactly the modules `app_gradio.py` pulls in transitively:
```
# HF Space (Gradio + ZeroGPU) runtime deps -- M5 pivot.
# Do NOT list gradio/spaces/huggingface_hub: HF preinstalls and manages
# these; pinning them breaks the ZeroGPU runtime. Do NOT list torch: HF
# preinstalls a ZeroGPU-compatible CUDA build on zero-a10g hardware.
FlagEmbedding>=1.2.0
langchain-text-splitters>=0.3.0
litellm>=1.50.0
networkx>=3.0
numpy
qdrant-client>=1.11.0
sentence-transformers>=3.0.0
python-dotenv>=1.0.0
redis>=5.0.0,<6.0.0
langfuse>=3.0.0,<4.0.0
opentelemetry-api>=1.24.0
opentelemetry-sdk>=1.24.0
opentelemetry-exporter-otlp>=1.24.0
```

- [ ] **Step 3: Create `README_GRADIO.md`**

Replace `<GRADIO_VERSION>` with the version found in Step 1:
```markdown
---
title: Vietnamese Legal Knowledge Bot
emoji: 🏛️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: <GRADIO_VERSION>
app_file: src/ui/app_gradio.py
python_version: "3.12"
startup_duration_timeout: 1h
pinned: false
short_description: Adaptive RAG tra cứu văn bản pháp luật Việt Nam (600 văn bản)
---

# Vietnamese Legal Knowledge Bot

Adaptive RAG over 600 Vietnamese legal documents, running on HF's free
ZeroGPU tier. First question after a cold start may take longer while models
load onto GPU. See the repo README for architecture, eval numbers, and the
self-hosted Docker deploy path (full security hardening: CORS whitelist,
per-session rate limit, admin key).
```
(`python_version: "3.12"` -- ZeroGPU officially supports 3.10.13 and 3.12.12; the repo's `pyproject.toml` requires `>=3.11`, so 3.12 satisfies both.)

- [ ] **Step 4: Verify the frontmatter parses as YAML**

Run:
```bash
.venv/bin/python -c "
import yaml
text = open('README_GRADIO.md').read()
front = text.split('---')[1]
data = yaml.safe_load(front)
assert data['sdk'] == 'gradio'
assert data['app_file'] == 'src/ui/app_gradio.py'
print('README_GRADIO.md frontmatter OK:', data)
"
```
Expected: prints the parsed frontmatter dict, no error.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README_GRADIO.md
git commit -m "feat(m5): requirements.txt + README_GRADIO.md for the Gradio+ZeroGPU Space"
```

---

### Task 5: CI liveness job — point at the Gradio Space instead of Streamlit

**Files:**
- Modify: `.github/workflows/smoke.yml`

**Interfaces:**
- Consumes: GitHub secret `PUBLIC_SPACE_URL` (already referenced by the existing liveness job; value changes from the Streamlit Space URL to the Gradio Space URL once Task 7 creates it -- updating the secret's value is a Phase 2 step, not this task).
- Produces: an updated `liveness` job that pings the Gradio Space's root URL instead of Streamlit's `/_stcore/health`.

- [ ] **Step 1: Read the current liveness job**

The existing job (from M5 Phase 1) is:
```yaml
  liveness:
    runs-on: ubuntu-latest
    steps:
      - name: Ping public Space
        run: |
          test -n "${{ secrets.PUBLIC_SPACE_URL }}" || { echo "PUBLIC_SPACE_URL unset"; exit 1; }
          code=$(curl -s -o /dev/null -w "%{http_code}" -L "${{ secrets.PUBLIC_SPACE_URL }}/_stcore/health")
          echo "liveness: $code"; test "$code" = "200"
```
This checked Streamlit's built-in health endpoint, which doesn't exist on a Gradio Space.

- [ ] **Step 2: Replace it with a Gradio-compatible check**

In `.github/workflows/smoke.yml`, replace the `liveness` job body with:
```yaml
  liveness:
    runs-on: ubuntu-latest
    steps:
      - name: Ping public Gradio Space
        run: |
          test -n "${{ secrets.PUBLIC_SPACE_URL }}" || { echo "PUBLIC_SPACE_URL unset"; exit 1; }
          code=$(curl -s -o /dev/null -w "%{http_code}" -L "${{ secrets.PUBLIC_SPACE_URL }}")
          echo "liveness: $code"; test "$code" = "200"
```
(A Gradio Space's root URL returns 200 once the app has booted; there's no dedicated `/health`-style path the way Streamlit has, so this checks the root page directly.)

- [ ] **Step 3: Validate the workflow YAML**

Run: `.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/smoke.yml
git commit -m "ci(m5): liveness job pings the Gradio Space root instead of Streamlit /_stcore/health"
```

---

### Task 6 (Phase 1 gate): Full suite green

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (previous baseline + Task 1's `tests/test_degraded.py` + Task 2's `tests/test_gradio_helpers.py`). Fix any regression before proceeding -- in particular, re-check `tests/test_api_degraded.py` since Task 1 changed where `DEGRADED_MSG`/`is_exhaustion` live.

- [ ] **Step 2: Syntax-check the new Gradio module one more time**

Run: `python3 -m py_compile src/ui/app_gradio.py src/ui/gradio_helpers.py src/rag/degraded.py`
Expected: exit code 0.

---

## PHASE 2 — Deploy + measure (infra / manual verification)

> Mirrors the structure of `docs/superpowers/plans/m5.md`'s own Phase 2: these are checklists against a real external service, not TDD. Record actual outputs -- never claim a number you didn't measure.

### Task 7: `scripts/deploy_space.py` + first deploy

**Files:**
- Create: `scripts/deploy_space.py`

**Interfaces:**
- Produces: an idempotent deploy script (`python scripts/deploy_space.py [<namespace>/<name>]`) that creates the Space with ZeroGPU hardware, pushes the curated file set, and sets secrets from the local `.env` -- re-runnable after future code changes.

- [ ] **Step 1: Write the script**

Create `scripts/deploy_space.py`:
```python
"""M5 pivot: create/update the public Gradio+ZeroGPU HF Space and push the
curated runtime file set (src/, data/metadata/store.db, requirements.txt,
README_GRADIO.md -> README.md). Idempotent -- safe to re-run after code
changes. Secrets are read from the local .env and pushed as Space Secrets;
values are never printed. Usage: python scripts/deploy_space.py [<ns>/<name>]"""
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, SpaceHardware

REPO_ID_DEFAULT = "quockhanh41/vietnamese-legal-bot"
SECRET_KEYS = [
    "QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY", "GROQ_API_KEYS",
    "GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENROUTER_API_KEY", "OPENROUTER_API_KEYS",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "REDIS_URL",
]
ROOT = Path(__file__).resolve().parent.parent


def stage_files(staging_dir: Path) -> None:
    shutil.copytree(ROOT / "src", staging_dir / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (staging_dir / "data" / "metadata").mkdir(parents=True)
    shutil.copy(ROOT / "data/metadata/store.db", staging_dir / "data/metadata/store.db")
    shutil.copy(ROOT / "requirements.txt", staging_dir / "requirements.txt")
    shutil.copy(ROOT / "README_GRADIO.md", staging_dir / "README.md")


def push_secrets(api: HfApi, repo_id: str) -> None:
    values = dotenv_values(ROOT / ".env")
    pushed = []
    for key in SECRET_KEYS:
        val = values.get(key)
        if val:
            api.add_space_secret(repo_id, key, val)
            pushed.append(key)
    print(f"Secrets pushed (values not shown): {pushed}")


def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else REPO_ID_DEFAULT
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio",
                    space_hardware=SpaceHardware.ZERO_A10G, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "space"
        staging.mkdir()
        stage_files(staging)
        api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=str(staging),
                          commit_message="deploy: sync Gradio Space from repo")
    push_secrets(api, repo_id)
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify `store.db` is present and non-stale before deploying**

Run: `.venv/bin/python -c "from src.data.metadata_store import MetadataStore; print(len(MetadataStore().get_all_documents()))"`
Expected: a number >= 600. If not, STOP -- do not deploy stale metadata.

- [ ] **Step 3: Run the deploy script**

Run: `.venv/bin/python scripts/deploy_space.py`
Expected: prints the list of secret KEYS pushed (not values) and the final `https://huggingface.co/spaces/quockhanh41/vietnamese-legal-bot` URL. This call creates the Space (idempotent via `exist_ok=True`) with `zero-a10g` hardware already attached from `create_repo`, so there's no separate manual "set hardware" step.

- [ ] **Step 4: Commit the script**

```bash
git add scripts/deploy_space.py
git commit -m "feat(m5): scripts/deploy_space.py — idempotent Gradio+ZeroGPU Space deploy"
```

---

### Task 8: Verify the Space boots correctly

- [ ] **Step 1: Poll build/runtime stage**

Run:
```bash
.venv/bin/python -c "
from huggingface_hub import HfApi
rt = HfApi().get_space_runtime('quockhanh41/vietnamese-legal-bot')
print('stage:', rt.stage)
print('hardware:', rt.hardware)
"
```
Repeat every ~30s until `stage` is `RUNNING` (building can take a while on first push -- large ML deps). If it lands on `RUNTIME_ERROR` or similar, read the build log in the browser at `https://huggingface.co/spaces/quockhanh41/vietnamese-legal-bot?logs=build` (no logs API in the installed `huggingface_hub`/`hf` CLI versions -- browser is the verified path) before proceeding.

- [ ] **Step 2: Confirm `store.db` and the 600-doc corpus actually shipped**

Once `RUNNING`, open the Space, ask a question that requires the full corpus (e.g. one that was only added after the original 100-doc M1 snapshot) and confirm it answers with citations rather than refusing.

- [ ] **Step 3: Record what you observed**

Note in the conversation/PR (not fabricated): boot time, whether the build log showed model pre-download, any warnings.

---

### Task 9: Measure real `@spaces.GPU` duration, tune, verify quota-exceeded message

- [ ] **Step 1: Call the live Space via `gradio_client` and time it**

Run:
```bash
.venv/bin/python -c "
import time
from gradio_client import Client

c = Client('quockhanh41/vietnamese-legal-bot')
for q in ['Tội trộm cắp tài sản bị xử phạt thế nào?',
         'So sánh mức phạt tội cố ý gây thương tích và tội giết người']:
    t0 = time.time()
    result = c.predict(q, [], api_name='/chat')
    print(f'{time.time()-t0:.1f}s -> {str(result)[:120]}')
"
```
(If the auto-detected API name differs, run `c.view_api()` first and use the printed endpoint name -- don't guess it.) Record the measured seconds for both a MEDIUM and a COMPLEX-shaped question.

- [ ] **Step 2: Set `duration` from the measurement**

In `src/ui/app_gradio.py`, change:
```python
@spaces.GPU(duration=180)
```
to `duration = round(measured_max * 1.4)` from Step 1 (per HF's own sizing methodology in `zerogpu.md` -- not a guess). If MEDIUM and COMPLEX differ a lot, use an input-dependent callable instead:
```python
def _estimate_duration(message, history, *args, **kwargs):
    return <measured_medium * 1.4> if len(message) < 80 else <measured_complex * 1.4>


@spaces.GPU(duration=_estimate_duration)
```
(Exact numbers come from Step 1's real measurement -- fill in here, don't leave the 180s placeholder in the deployed app.)

- [ ] **Step 3: Re-deploy with the tuned duration**

Run: `.venv/bin/python scripts/deploy_space.py`

- [ ] **Step 4: Verify the quota-exceeded path (best-effort, may not be triggerable on demand)**

If/when a `quota exceeded`-style error is observed in the Space's run logs during testing, capture the exact exception text and compare against `src/ui/gradio_helpers.is_quota_exceeded`'s substring match (`"quota" in text and "gpu" in text`). If the real text doesn't match, tighten the function and add a regression test in `tests/test_gradio_helpers.py`, then commit.

- [ ] **Step 5: Commit the duration tuning**

```bash
git add src/ui/app_gradio.py
git commit -m "perf(m5): tune @spaces.GPU duration from measured warm latency"
```

---

### Task 10 (M5 pivot gate): README + roadmap status

**Files:**
- Modify: `README.md` (root)
- Modify: `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md` §7 (M5 status)

- [ ] **Step 1: Add both deploy paths to `README.md`**

Add a "Live demo" section linking `https://huggingface.co/spaces/quockhanh41/vietnamese-legal-bot`, with the measured latency numbers from Task 9 and the known limits (1 of 2 free Gradio+ZeroGPU slots used, visitor daily GPU quota, cold start after idle). Keep/adapt the existing security-hardening note to explicitly frame it as the self-hosted Docker path (`README_HF.md`), not the live one.

- [ ] **Step 2: Update the milestone doc**

In `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md` §7, mark M5 complete and note the deploy-target pivot (Docker Space → Gradio+ZeroGPU) with a one-line pointer to the pivot spec.

- [ ] **Step 3: Confirm the gate**

- URL public sống ✓ (Task 8)
- P95/duration đo thật, không đoán ✓ (Task 9)
- Smoke/liveness CI xanh ✓ (Task 5 + manual `workflow_dispatch` trigger, confirm green)
- Không secret trong repo/logs ✓ (secrets pushed via `add_space_secret`, never printed; existing Phase-1 gitleaks CI still covers the repo)

- [ ] **Step 4: Commit**

```bash
git add README.md docs/Plan_Giai_Doan_Cuoi_2026-07-10.md
git commit -m "docs(m5): mark M5 complete — live demo on Gradio+ZeroGPU, measured latency in README"
```

---

## Self-Review

**Spec coverage:**
- Additive `app_gradio.py`, FastAPI/Docker stack untouched → Task 3, Global Constraints. ✓
- `device="cuda"` at module scope (verified, not the reviewer's incorrect "cpu" claim) → Task 3 Step 1 + inline comment. ✓
- One `@spaces.GPU` boundary including cache lookup → Task 3 Step 1 (`handle_turn`). ✓
- `degraded.py` extraction (and honest "touches main.py" framing) → Task 1. ✓
- Duration sizing via measurement, not guess → Task 9. ✓
- Quota-exceeded UX → Task 2 (`is_quota_exceeded`) + Task 3 (`QUOTA_MSG`) + Task 9 Step 4 (live verification). ✓
- `requirements.txt` trimmed to verified real imports, torch/gradio/spaces excluded → Task 4. ✓
- CI liveness adapted to Gradio → Task 5. ✓
- Deploy mechanics (HfApi, not `hf` CLI `spaces` subcommand which doesn't exist locally) → Task 7. ✓
- Gate mapping → Task 10. ✓

**Placeholder scan:** every code step has concrete, complete code. The two "fill in from measurement" spots (Task 9's duration numbers, Task 4's Gradio version) are explicitly measured-at-execution-time by design (matches the project's existing "measure, don't guess" convention for `RERANK_MAX_K`/P95) -- not vague TODOs.

**Type consistency:** `DEGRADED_MSG`/`is_exhaustion` (Task 1) imported unchanged in Task 3. `history_to_turns`/`format_citations_markdown`/`is_quota_exceeded` (Task 2) imported unchanged in Task 3. `RAGResponse` fields match `src/rag/orchestrator.py`'s dataclass (verified by reading it, not assumed). `SemanticCache.from_env(embed_fn)`, `AdaptiveRAGPipeline(...)` constructor kwargs, `condense_query(history, question, llm)` all matched against the actual current signatures in the codebase (verified via Read, not guessed).
