# Hợp nhất đường stream/non-stream — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gộp hai bản sao logic định tuyến trong `AdaptiveRAGPipeline` thành một, sửa `LLMGateway.generate_stream` báo lỗi bằng exception thay vì yield text, và đặt adaptive làm pipeline mặc định.

**Architecture:** Trích phần định tuyến + truy hồi thành hai generator dùng chung (`_route`, `_retrieve`) trả giá trị qua PEP 380. `query_stream()` tiêu thụ bằng `yield from` (giữ event tiến trình real-time); `query()` tiêu thụ bằng helper `_drain()` (bỏ qua event, chi phí bằng 0). Chữ ký công khai không đổi.

**Tech Stack:** Python 3.11+, pytest, FastAPI, Streamlit, litellm.

**Spec:** [`docs/superpowers/specs/2026-08-18-stream-nonstream-unification-design.md`](../specs/2026-08-18-stream-nonstream-unification-design.md)

## Global Constraints

- **Non-stream là chuẩn.** Nơi nào hai đường lệch, hành vi của `query()` thắng (spec §3.1).
- **Chữ ký công khai giữ nguyên:** `query(question) -> RAGResponse` và `query_stream(question) -> Generator[str, None, None]`. `main.py`, `app_gradio.py`, `chat_turn.py`, `scripts/eval.py` không được phải sửa vì thay đổi signature.
- **`complex_retrieve()` giữ nguyên là public** — `scripts/eval.py --multihop` (harness M3.5) gọi nó.
- **Gate hồi quy:** 533 test hiện có phải xanh sau mỗi task.
- **Tín hiệu exhaustion giữ nguyên chuỗi:** `RuntimeError("All models failed. Last error: …")` — `src/rag/degraded.py:14` khớp bằng `str(e).startswith("All models failed")` và **không được sửa**.
- Chạy test: `.venv/bin/python -m pytest tests/ -q`
- Mọi comment/docstring mới viết bằng tiếng Anh, khớp phong cách file đang sửa.

---

### Task 1: Lưới parity — chốt hãm trước khi đụng code

Hai test này **phải xanh trên `main` trước khi refactor**. Chúng không tố cáo lỗi nào đang có; việc của chúng là bắt lỗi mà thao tác dời ~150 dòng có thể vô tình tạo ra. Nếu chúng đỏ ngay từ đầu thì hiểu biết về code trong spec sai ở đâu đó — dừng lại, báo cáo, đừng sửa test cho xanh.

**Files:**
- Create: `tests/test_stream_nonstream_parity.py`

**Interfaces:**
- Consumes: `AdaptiveRAGPipeline`, `SearchResult` (đang có)
- Produces: các fixture `_make_adaptive()`, `_Conf`, `_Ver`, `_NoMeta`, `_Vec`, `_Reranker` — Task 3 và Task 5 dùng lại

- [ ] **Step 1: Viết file test với fixture dùng chung + hai test parity**

```python
"""Parity between AdaptiveRAGPipeline.query() and .query_stream().

The project measures query() (scripts/eval.py:364, "non-streaming path") but
ships query_stream() (src/api/main.py, src/ui/app_gradio.py). Nothing used to
compare them, so divergences were invisible to every gate the project has.
These tests make "the two paths must agree" a machine-checked claim.

Pure-Python: no Qdrant, no LLM, no models.

Run: .venv/bin/python -m pytest tests/test_stream_nonstream_parity.py -q
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import src.rag.orchestrator as orch
from src.db.qdrant_client import SearchResult
from src.rag.orchestrator import AdaptiveRAGPipeline


class _Vec:
    """Stands in for the numpy vector the embedder returns."""

    def tolist(self):
        return [0.1, 0.2, 0.3]


class _Conf:
    """ConfidenceScorer result double — real floats, so the streaming path can
    JSON-serialize the __VERIFICATION__ payload."""

    label = "high"
    score = 0.9
    refused = False


class _Ver:
    """HallucinationVerifier result double."""

    passed = True
    warnings = []
    entailment_ratio = 0.0
    contradiction_ratio = 0.0
    grounding_ratio = 0.0


class _NoMeta:
    """MetadataStore double with nothing in it.

    get_document -> None keeps get_phase2_status on its fail-open path
    (is_in_effect stays None, so _filter_in_effect keeps every result), and an
    empty graph makes _expand_cross_references a no-op. Both keep the two paths
    comparing retrieval, not metadata lookups.
    """

    def get_document(self, doc_id):
        return None

    def get_timeline(self, doc_id):
        return None

    def get_all_documents(self):
        return {}

    def get_graph(self):
        return {"nodes": [], "edges": []}


class _Reranker:
    """Deterministic cross-encoder double: preserves input order."""

    def compute_score(self, pairs, normalize=True):
        return [1.0 - i * 0.01 for i in range(len(pairs))]


def _sr(i):
    return SearchResult(
        point_id=f"p{i}",
        score=1.0 - i * 0.01,
        chunk_text=f"chunk {i}",
        document_title="Bộ luật hình sự 2015",
        document_number="100/2015/QH13",
        legal_type="Bộ luật",
        article=f"Điều {100 + i}",
        document_id=1,
        parent_chunk_text=f"Điều {100 + i} full text",
    )


def _make_adaptive(n_results=20):
    """AdaptiveRAGPipeline wired to doubles only."""
    results = [_sr(i) for i in range(n_results)]

    vector_store = MagicMock()
    vector_store.search_dense.return_value = list(results)
    vector_store.search_hybrid.return_value = list(results)

    embedder = MagicMock()
    embedder.encode_queries.return_value = [_Vec()]
    embedder.encode.return_value = {"dense_vecs": [_Vec()], "lexical_weights": [{}]}

    llm = MagicMock()
    llm.generate.return_value = "1. truy van mo rong"
    llm.generate_with_model.return_value = ("cau tra loi", "groq/fake-model")
    llm.generate_stream.return_value = iter(["cau ", "tra ", "loi"])

    return AdaptiveRAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        llm_gateway=llm,
        metadata_store=_NoMeta(),
        reranker=_Reranker(),
        citation_lookup=MagicMock(),
    )


@pytest.fixture
def force_route(monkeypatch):
    """Pin the classifier so a test exercises one route on both paths."""

    def _force(complexity):
        monkeypatch.setattr(orch, "parse_citation", lambda q: None)
        monkeypatch.setattr(
            AdaptiveRAGPipeline,
            "_classify_and_scope",
            lambda self, q: (complexity, True, False),
        )

    return _force


@pytest.fixture
def spy_top_k(monkeypatch):
    """Record every (min_k, max_k) _dynamic_top_k is called with."""
    calls = []
    # Accessing a @staticmethod through the class yields the plain function
    # (Python 3.10+), so there is no .__func__ to unwrap here.
    original = AdaptiveRAGPipeline._dynamic_top_k

    def spy(results, min_k=3, max_k=20):
        calls.append({"min_k": min_k, "max_k": max_k})
        return original(results, min_k=min_k, max_k=max_k)

    monkeypatch.setattr(AdaptiveRAGPipeline, "_dynamic_top_k", staticmethod(spy))
    return calls


@pytest.fixture
def capture_post_process(monkeypatch):
    """Record the final result set each path hands to _post_process.

    That set is what becomes context + citations, so equal point_ids means the
    two paths retrieved the same evidence in the same order.
    """
    seen = []

    def fake(self, question, answer, results, context, citations=None,
             rerank_scores=None, generation_model=None):
        seen.append([r.point_id for r in results])
        return _Conf(), _Ver(), answer

    monkeypatch.setattr(AdaptiveRAGPipeline, "_post_process", fake)
    return seen


@pytest.mark.parametrize("route", ["simple", "medium", "complex"])
def test_both_paths_use_same_dynamic_top_k_bounds(route, force_route, spy_top_k):
    """LOCK (green before and after the refactor).

    A divergence here is exactly how RERANK_MAX_K ended up honoured on one path
    and hardcoded on the other.
    """
    force_route(route)

    _make_adaptive().query("cau hoi")
    non_stream = list(spy_top_k)
    spy_top_k.clear()

    list(_make_adaptive().query_stream("cau hoi"))
    stream = list(spy_top_k)

    assert non_stream == stream


@pytest.mark.parametrize("route", ["simple", "medium", "complex"])
def test_both_paths_retrieve_same_points(route, force_route, capture_post_process):
    """LOCK (green before and after the refactor).

    The single most important guard in this suite: it turns "the two paths must
    agree" from a verbal convention into a machine-checked assertion.
    """
    force_route(route)

    _make_adaptive().query("cau hoi")
    list(_make_adaptive().query_stream("cau hoi"))

    assert len(capture_post_process) == 2
    assert capture_post_process[0] == capture_post_process[1]
```

- [ ] **Step 2: Chạy trên `main` — phải XANH**

Run: `.venv/bin/python -m pytest tests/test_stream_nonstream_parity.py -q`
Expected: **6 passed** (3 route × 2 test).

Nếu đỏ: **DỪNG**. Spec §4 kết luận hai đường chỉ lệch ở một điểm không chạm `point_id` cũng không chạm `max_k`. Một test đỏ ở đây nghĩa là kết luận đó sai và spec cần xem lại trước khi refactor. Báo cáo chi tiết chỗ lệch, đừng sửa test cho xanh.

- [ ] **Step 3: Chạy toàn bộ suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 539 passed (533 cũ + 6 mới).

- [ ] **Step 4: Commit**

```bash
git add tests/test_stream_nonstream_parity.py
git commit -m "test: lock stream/non-stream retrieval parity before refactor"
```

---

### Task 2: Gateway báo cạn quota bằng exception

**Files:**
- Modify: `src/llm/gateway.py:325`
- Test: `tests/test_gateway.py`

**Interfaces:**
- Produces: `LLMGateway.generate_stream` raise `RuntimeError("All models failed. Last error: …")` khi mọi model thất bại — cùng dạng tín hiệu `_generate_impl:231`, nên `src.rag.degraded.is_exhaustion` nhận diện được mà không phải sửa.

- [ ] **Step 1: Viết test đỏ**

Thêm vào cuối `tests/test_gateway.py`:

```python
class TestStreamExhaustion:
    """generate_stream used to signal total failure by YIELDING an error string.
    Because nothing raised, src/api/main.py never ran its is_exhaustion branch:
    the error text was streamed as if it were an answer, then written to session
    history and the semantic cache (audit 2026-08-18, #2).
    """

    def test_raises_when_every_model_fails(self, gateway, monkeypatch):
        import litellm

        def always_fail(**kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(litellm, "completion", always_fail)

        with pytest.raises(RuntimeError) as excinfo:
            list(gateway.generate_stream(prompt="hi"))

        assert str(excinfo.value).startswith("All models failed")

    def test_raised_error_is_recognised_as_exhaustion(self, gateway, monkeypatch):
        """The message prefix is load-bearing: src/rag/degraded.py matches on it."""
        import litellm

        from src.rag.degraded import is_exhaustion

        def rate_limited(**kwargs):
            raise RuntimeError("429 rate limit exceeded")

        monkeypatch.setattr(litellm, "completion", rate_limited)

        try:
            list(gateway.generate_stream(prompt="hi"))
        except RuntimeError as e:
            assert is_exhaustion(e) is True
        else:
            pytest.fail("generate_stream did not raise")

    def test_never_yields_an_error_string(self, gateway, monkeypatch):
        """Regression guard: an error must not be able to masquerade as tokens."""
        import litellm

        def always_fail(**kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(litellm, "completion", always_fail)

        collected = []
        with pytest.raises(RuntimeError):
            for token in gateway.generate_stream(prompt="hi"):
                collected.append(token)

        assert collected == []
```

- [ ] **Step 2: Chạy — phải ĐỎ**

Run: `.venv/bin/python -m pytest tests/test_gateway.py::TestStreamExhaustion -q`
Expected: FAIL — hiện `generate_stream` yield `"[Lỗi] Không thể kết nối đến LLM..."` và kết thúc bình thường, không raise. `test_never_yields_an_error_string` fail vì `collected == ["[Lỗi] ..."]`.

- [ ] **Step 3: Sửa gateway**

Trong `src/llm/gateway.py`, đổi dòng cuối `generate_stream` (`:325`):

```python
        yield "[Lỗi] Không thể kết nối đến LLM. Vui lòng thử lại sau."
```

thành:

```python
        raise RuntimeError(f"All models failed. Last error: {last_error}")
```

Để có `last_error`, thêm biến theo dõi trong vòng lặp — sửa thân `generate_stream` từ `:291`:

```python
        # Try each model; within a model, rotate API keys on rate-limit
        last_error = None
        for attempt_model in models_to_try:
            model_kwargs = {}
            if reasoning_effort is not None and self._is_thinking_model(attempt_model):
                model_kwargs["reasoning_effort"] = self._normalize_reasoning_effort(attempt_model, reasoning_effort)

            for api_key in self._keys_for(attempt_model):
                started = False
                try:
                    response = litellm.completion(
                        model=attempt_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        stream=True,
                        **model_kwargs,
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            started = True
                            if used_model is not None:
                                used_model["model"] = attempt_model
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    last_error = e
                    if started:
                        # Mid-stream failure: tokens already reached the client,
                        # so retrying would duplicate output. Task 3 gives this
                        # its own exception type.
                        raise
                    if self._is_rate_limit(e):
                        continue  # rotate to the next key for the same model
                    break  # non-rate-limit failure — try the next model

        # Signal exhaustion the SAME way _generate_impl does: src/rag/degraded.py
        # matches on this message prefix, and both API front-ends already have an
        # is_exhaustion branch waiting for it.
        raise RuntimeError(f"All models failed. Last error: {last_error}")
```

Cập nhật docstring `generate_stream` — bỏ mô tả cũ về việc yield thông báo lỗi, thêm:

```
        Raises:
            RuntimeError: every model in the chain failed ("All models failed…"),
                recognised by src.rag.degraded.is_exhaustion.
```

- [ ] **Step 4: Chạy — phải XANH**

Run: `.venv/bin/python -m pytest tests/test_gateway.py -q`
Expected: PASS.

- [ ] **Step 5: Chạy toàn bộ suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: tất cả xanh. Đặc biệt `tests/test_api_degraded.py::test_stream_degraded_on_exhaustion` phải vẫn xanh — nó vốn mock một `query_stream` **raise**, tức nó đã mô tả đúng hành vi mà tới giờ mới thành thật.

- [ ] **Step 6: Commit**

```bash
git add src/llm/gateway.py tests/test_gateway.py
git commit -m "fix(llm): generate_stream raises on exhaustion instead of yielding an error string

The streamed error text was treated as a normal answer: it bypassed the
is_exhaustion branch in both front-ends, got written to session history, and
was stored in the semantic cache for replay."
```

---

### Task 3: Đứt giữa chừng — exception riêng, không cache, không ghi history

Khác với Task 2: token đã tới client, không rút lại được và không retry được. Cần kiểu exception riêng để caller phân biệt.

**Files:**
- Modify: `src/rag/degraded.py`
- Modify: `src/llm/gateway.py` (nhánh `if started`)
- Modify: `src/api/main.py:386-396`
- Modify: `src/ui/app_gradio.py:129-135`
- Test: `tests/test_api_degraded.py`

**Interfaces:**
- Produces: `src.rag.degraded.StreamInterrupted(RuntimeError)` và `INTERRUPTED_MSG: str`

- [ ] **Step 1: Viết test đỏ**

Thêm vào cuối `tests/test_api_degraded.py`:

```python
def test_stream_interrupted_midway_is_not_cached(monkeypatch):
    """Tokens already reached the client, so the turn cannot be retried — but a
    truncated answer must never be persisted or served to anyone else."""
    from src.rag.degraded import StreamInterrupted

    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    def gen(_condensed):
        yield "phan dau cua cau tra loi"
        raise StreamInterrupted("connection dropped")

    fake = MagicMock(); fake.query_stream = gen
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: fake)
    monkeypatch.setattr(main, "condense_query", lambda h, q, llm: q)

    client = TestClient(main.app)
    r = client.post("/api/chat/stream", json={"question": "q", "session_id": "s1"})

    assert r.status_code == 200
    assert "gián đoạn" in r.text
    cache.set.assert_not_called()
    store.append_turn.assert_not_called()
```

- [ ] **Step 2: Chạy — phải ĐỎ**

Run: `.venv/bin/python -m pytest tests/test_api_degraded.py::test_stream_interrupted_midway_is_not_cached -q`
Expected: FAIL — `ImportError: cannot import name 'StreamInterrupted'`.

- [ ] **Step 3: Thêm kiểu exception + thông báo**

Vào cuối `src/rag/degraded.py`:

```python
INTERRUPTED_MSG = ("\n\n[Lỗi] Kết nối bị gián đoạn giữa chừng. "
                   "Vui lòng thử lại.")


class StreamInterrupted(RuntimeError):
    """Streaming failed AFTER tokens already reached the client.

    Distinct from exhaustion: the turn cannot be retried (retrying would
    duplicate output) and the partial answer must never be cached or written to
    session history. Subclasses RuntimeError but does NOT use the "All models
    failed" prefix, so is_exhaustion() correctly returns False for it.
    """
```

- [ ] **Step 4: Gateway raise kiểu mới**

Trong `src/llm/gateway.py`, thêm import ở đầu file:

```python
from src.rag.degraded import StreamInterrupted
```

và trong `generate_stream` đổi nhánh `if started:` (đã viết ở Task 2 Step 3) thành:

```python
                    if started:
                        # Mid-stream failure: tokens already reached the client,
                        # so retrying would duplicate output. Caller must not
                        # cache or persist the partial answer.
                        raise StreamInterrupted(str(e)) from e
```

- [ ] **Step 5: Xử lý ở `src/api/main.py`**

Thêm `StreamInterrupted` vào import dòng `:36`:

```python
from src.rag.degraded import (DEGRADED_MSG, INTERRUPTED_MSG, StreamInterrupted,
                              is_exhaustion as _is_exhaustion)
```

Trong `event_generator`, thêm nhánh **trước** nhánh `_is_exhaustion` hiện có (`:386`):

```python
            except StreamInterrupted:
                yield {"event": "token",
                       "data": json.dumps({"token": INTERRUPTED_MSG}, ensure_ascii=False)}
                yield {"event": "done", "data": json.dumps({
                    "full_answer": full_answer + INTERRUPTED_MSG, "citations": citations,
                    "confidence": "low", "refused": False, "degraded": True,
                    "latency_ms": (time.time() - start_time) * 1000, "cache_hit": False,
                }, ensure_ascii=False)}
                return  # do NOT persist/cache a truncated answer
```

- [ ] **Step 6: Xử lý ở `src/ui/app_gradio.py`**

Đổi import `:25`:

```python
from src.rag.degraded import (DEGRADED_MSG, INTERRUPTED_MSG, StreamInterrupted,
                              is_exhaustion)
```

Trong `handle_turn`, thêm nhánh **trước** `if is_exhaustion(e):` (`:129`):

```python
    except StreamInterrupted:
        yield full_answer + INTERRUPTED_MSG
        return
```

Lưu ý: nhánh này phải nằm trong cùng khối `try/except` đang có, và `return` sớm để bỏ qua `semantic_cache.set` ở cuối hàm.

- [ ] **Step 7: Chạy — phải XANH**

Run: `.venv/bin/python -m pytest tests/test_api_degraded.py -q`
Expected: PASS.

- [ ] **Step 8: Chạy toàn bộ suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: tất cả xanh.

Kiểm tra vòng import: `gateway.py` giờ import `degraded.py`. `degraded.py` không import gì từ `src` — an toàn. Xác nhận bằng:
Run: `.venv/bin/python -c "import src.llm.gateway; print('ok')"`
Expected: `ok`

- [ ] **Step 9: Commit**

```bash
git add src/rag/degraded.py src/llm/gateway.py src/api/main.py src/ui/app_gradio.py tests/test_api_degraded.py
git commit -m "fix(llm): distinct StreamInterrupted for mid-stream failure

A truncated answer must never be cached or written to session history."
```

---

### Task 4: Trích `_route` — định tuyến dùng chung

**Files:**
- Modify: `src/rag/orchestrator.py` (thêm `_drain`, `_route`; sửa `query()`, `query_stream()`)
- Test: `tests/test_stream_nonstream_parity.py` (Task 1 giữ nguyên, chỉ chạy lại)

**Interfaces:**
- Consumes: `_classify_and_scope`, `parse_citation`, `_get_citation_lookup` (đang có)
- Produces:
  - `_drain(gen)` — module-level, chạy hết generator và trả `StopIteration.value`
  - `AdaptiveRAGPipeline._route(question) -> Generator[str, None, Tuple[str, bool, bool]]`

- [ ] **Step 1: Thêm `_drain` và `_route`**

Thêm sau `RERANK_MAX_K` (`src/rag/orchestrator.py:66`):

```python
def _drain(gen):
    """Run a generator to completion, discarding yields, and return its value.

    Lets the non-streaming path reuse generators that exist to interleave
    `__STEP__` progress events, at zero cost — the events are simply dropped.
    """
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value
```

Thêm method vào `AdaptiveRAGPipeline`, ngay trước `query()`:

```python
    def _route(self, question: str) -> Generator[str, None, Tuple[str, bool, bool]]:
        """Pick the route for `question`, yielding `__STEP__` progress events.

        Returns (complexity, in_scope, is_comparison). A question that parses as
        a citation resolves to real articles by construction, so it takes the
        lookup route WITHOUT an LLM classifier call and is in-scope by
        definition (see AdaptiveRAGPipeline.query for the full rationale).
        """
        parsed_citation = parse_citation(question)
        if parsed_citation is not None:
            citation_lookup = self._get_citation_lookup()
            doc_ids = citation_lookup.resolve_doc(
                parsed_citation.doc_hint,
                parsed_citation.year,
                doc_number=parsed_citation.doc_number,
            )
            aliases = citation_lookup._get_alias_index()
            resolved_titles = [aliases.get(d, str(d)) for d in doc_ids[:3]]
            article_label = f"Điều {parsed_citation.primary.number}"
            if parsed_citation.primary.khoan:
                article_label += f", Khoản {parsed_citation.primary.khoan}"
            resolved_str = ", ".join(resolved_titles) if resolved_titles else "(ambiguous — searched all documents)"
            yield f'__STEP__{json.dumps({"step": "Citation Lookup", "result": f"{article_label} — resolved: {resolved_str}"})}'
            return "lookup", True, False

        complexity, in_scope, is_comparison = self._classify_and_scope(question)
        scope_label = "IN" if in_scope else "OUT"
        yield f'__STEP__{json.dumps({"step": "Classification", "result": f"{complexity.upper()} (scope={scope_label})"})}'
        return complexity, in_scope, is_comparison
```

- [ ] **Step 2: `query()` dùng `_route`**

Thay thân `query()` (`:404-434`) — phần từ `parsed_citation = ...` tới hết — bằng:

```python
    @observe
    def query(self, question: str) -> RAGResponse:
        """Process with adaptive routing."""
        start_time = time.time()

        complexity, in_scope, is_comparison = _drain(self._route(question))

        if complexity == "lookup":
            return self._lookup_pipeline(question, start_time)
        if not in_scope:
            return self._out_of_scope_response(question, start_time)

        if complexity == "simple":
            return self._simple_pipeline(question, start_time, is_comparison)
        elif complexity == "medium":
            return self._medium_pipeline(question, start_time, is_comparison)
        else:
            return self._complex_pipeline(question, start_time, is_comparison)
```

Ghi chú: thứ tự `complexity == "lookup"` trước `not in_scope` giữ đúng hành vi cũ — route trích dẫn không bao giờ đi qua scope gate.

- [ ] **Step 3: `query_stream()` dùng `_route`**

Thay phần đầu `query_stream()` (`:962-1005`, từ `parsed_citation = parse_citation(question)` tới hết khối `else:` chứa scope gate) bằng:

```python
        complexity, in_scope, is_comparison = yield from self._route(question)

        if complexity != "lookup" and not in_scope:
            corpus_summary = self._get_citation_lookup().corpus_summary_compact()
            answer = OUT_OF_SCOPE_ANSWER_TEMPLATE.format(corpus_summary=corpus_summary)
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Out of Scope — refused before retrieval/generation"})}'
            # Yield the refusal text as normal tokens so the UI displays it
            # exactly like any other answer (no retrieval/_post_process ran).
            yield answer
            verification_payload = {
                "confidence_score": 0.0,
                "confidence": "low",
                "refused": True,
                "warnings": [],
                "reason": "out_of_scope",
            }
            yield f'__VERIFICATION__{json.dumps(verification_payload, ensure_ascii=False)}'
            return
```

⚠️ Dựng dict **trước** rồi mới `json.dumps` — không nhúng biểu thức nhiều dòng vào trong `{}` của f-string. Cú pháp đó chỉ hợp lệ từ Python 3.12 (PEP 701), trong khi `pyproject.toml` khai `requires-python = ">=3.11"`. Đây cũng là phong cách file đang dùng ở `:997-1004`.

Xoá biến `citation_lookup` cục bộ ở nhánh lookup cũ; nhánh `if complexity == "lookup":` phía dưới (`:1007`) đổi `citation_lookup` thành `self._get_citation_lookup()`.

- [ ] **Step 4: Chạy lưới parity — phải vẫn XANH**

Run: `.venv/bin/python -m pytest tests/test_stream_nonstream_parity.py -q`
Expected: 6 passed. Nếu đỏ, refactor đã đổi hành vi — revert và xem lại.

- [ ] **Step 5: Chạy toàn bộ suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: tất cả xanh.

- [ ] **Step 6: Commit**

```bash
git add src/rag/orchestrator.py
git commit -m "refactor(rag): extract _route generator shared by query and query_stream"
```

---

### Task 5: Trích `_retrieve` — truy hồi dùng chung

Đây là task lớn nhất. `RERANK_MAX_K` được dùng nhất quán ở mọi điểm gọi sau task này.

**Files:**
- Modify: `src/rag/orchestrator.py`
- Test: `tests/test_stream_nonstream_parity.py`

**Interfaces:**
- Consumes: `_drain`, `_route` (Task 4)
- Produces:
  - `RouteResult` dataclass: `results: List[SearchResult]`, `annotations: Dict[str, Dict]`, `rerank_scores: Optional[List[float]]`
  - `AdaptiveRAGPipeline._retrieve(question, complexity, is_comparison) -> Generator[str, None, RouteResult]`

- [ ] **Step 1: Viết test đỏ cho `RERANK_MAX_K`**

Thêm vào `tests/test_stream_nonstream_parity.py`:

```python
def test_rerank_max_k_honoured_on_both_paths(force_route, spy_top_k, monkeypatch):
    """RERANK_MAX_K exists to cap the reranker pool on HF Space CPU (M2.6 measured
    P95 7.58s @pool 100). It used to be read at exactly one of four call sites —
    and not on the streaming path the UI actually runs (audit 2026-08-18, #3).
    """
    monkeypatch.setattr(orch, "RERANK_MAX_K", 8)
    force_route("medium")

    _make_adaptive().query("cau hoi")
    non_stream_max_k = [c["max_k"] for c in spy_top_k]
    spy_top_k.clear()

    list(_make_adaptive().query_stream("cau hoi"))
    stream_max_k = [c["max_k"] for c in spy_top_k]

    assert non_stream_max_k == [8]
    assert stream_max_k == [8]
```

- [ ] **Step 2: Chạy — phải ĐỎ**

Run: `.venv/bin/python -m pytest tests/test_stream_nonstream_parity.py::test_rerank_max_k_honoured_on_both_paths -q`
Expected: FAIL — `stream_max_k == [15]` (hardcode ở `:1036`), non-stream đã đúng `[8]`.

- [ ] **Step 3: Thêm `RouteResult` và `_retrieve`**

Thêm dataclass sau `RAGResponse` (`src/rag/orchestrator.py:80`):

```python
@dataclass
class RouteResult:
    """Everything a route retrieved, before context building."""
    results: List[SearchResult]
    annotations: Dict[str, Dict]
    rerank_scores: Optional[List[float]] = None
```

Thêm method vào `AdaptiveRAGPipeline`, sau `_route`:

```python
    def _retrieve(
        self, question: str, complexity: str, is_comparison: bool = False,
    ) -> Generator[str, None, RouteResult]:
        """Run one route's retrieval, yielding `__STEP__` progress events.

        The single source of truth for retrieval on BOTH paths: query() drains
        this, query_stream() delegates with `yield from`. Keeping one copy is
        what stops the two paths from drifting apart (audit 2026-08-18, #6).
        """
        rerank_scores = None

        if complexity == "lookup":
            results, route, _ = retrieve_routed(
                question, self.vector_store, self.embedder,
                self._get_citation_lookup(), fill_top_k=10,
            )
            annotations = self._annotate_phase2(results)
            results = self._filter_in_effect(results, annotations)
            yield f'__STEP__{json.dumps({"step": "Routing", "result": f"Citation Lookup ({len(results)} chunks, route={route})"})}'

        elif complexity == "simple":
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Simple Pipeline (Dense Search)"})}'
            query_vector = self.embedder.encode_queries([question])[0]
            results = self.vector_store.search_dense(
                query_vector.tolist(), top_k=20, exclude_superseded=not is_comparison,
            )
            annotations = self._annotate_phase2(results)
            results = self._filter_in_effect(results, annotations, keep_superseded=is_comparison)
            results = results[:self._dynamic_top_k(results, min_k=3, max_k=15)]

        elif complexity == "medium":
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Medium Pipeline (Hybrid Search)"})}'
            output = self.embedder.encode([question], return_dense=True, return_sparse=True)
            dense_vec = output["dense_vecs"][0]
            sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
            results = self.vector_store.search_hybrid(
                dense_vec.tolist(), sparse_vec, top_k=20,
                exclude_superseded=not is_comparison,
            )
            annotations = self._annotate_phase2(results)
            results = self._filter_in_effect(results, annotations, keep_superseded=is_comparison)
            results = results[:self._dynamic_top_k(results, min_k=5, max_k=RERANK_MAX_K)]
            results, rerank_scores = self._rerank_results(question, results)

        else:  # complex
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Complex Pipeline (Rewrite + HyDE + Multi-query)"})}'
            rewritten = self.llm.generate(
                QUERY_REWRITE_PROMPT.format(question=question), temperature=0.1,
                model=MODEL_QUERY_TRANSFORM, reasoning_effort="none",
            ).strip()
            yield f'__STEP__{json.dumps({"step": "Query Rewrite", "result": rewritten})}'

            expanded_raw = self.llm.generate(
                MULTI_QUERY_PROMPT.format(question=rewritten), temperature=0.3,
                model=MODEL_QUERY_TRANSFORM, reasoning_effort="none",
            )
            queries = [q.strip() for q in expanded_raw.split("\n") if q.strip() and q[0].isdigit()]
            queries = [q.split(". ", 1)[-1] if ". " in q else q for q in queries]
            yield f'__STEP__{json.dumps({"step": "Multi-query", "result": ", ".join(queries)})}'

            hyde_text = self.llm.generate(
                HYDE_PROMPT.format(question=question), temperature=0.2,
                model=MODEL_QUERY_TRANSFORM, reasoning_effort="none",
            ).strip()
            yield f'__STEP__{json.dumps({"step": "HyDE (Hypothetical Document)", "result": hyde_text[:100] + "..." if len(hyde_text) > 100 else hyde_text})}'

            all_results = []
            for q in [rewritten] + queries[:2] + [hyde_text]:
                output = self.embedder.encode([q], return_dense=True, return_sparse=True)
                dense_vec = output["dense_vecs"][0]
                sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
                all_results.extend(self.vector_store.search_hybrid(
                    dense_vec.tolist(), sparse_vec, top_k=10,
                    exclude_superseded=not is_comparison,
                ))

            seen = set()
            unique_results = []
            for r in sorted(all_results, key=lambda x: x.score, reverse=True):
                if r.point_id not in seen:
                    seen.add(r.point_id)
                    unique_results.append(r)
            annotations = self._annotate_phase2(unique_results)
            unique_results = self._filter_in_effect(unique_results, annotations, keep_superseded=is_comparison)
            results = unique_results[:self._dynamic_top_k(unique_results, min_k=5, max_k=20)]
            results, rerank_scores = self._rerank_results(question, results)

            # M3.5: append cross-referenced article text (after rerank, capped).
            expanded = self._expand_cross_references(results, is_comparison=is_comparison)
            if expanded:
                results = results + expanded
            # Re-annotate for the final (post-expansion) set.
            annotations = self._annotate_phase2(results)

        return RouteResult(results=results, annotations=annotations, rerank_scores=rerank_scores)
```

- [ ] **Step 4: Thêm `_prepare_generation`**

Method này giết luôn #7 (gọi `_build_context`/`_build_citations` hai lần) và biến "verify trên context chưa nén" từ tác dụng phụ của một dòng thừa thành một quyết định có tên.

Thêm sau `_retrieve`:

```python
    def _prepare_generation(
        self, question: str, route_result: RouteResult, compress: bool,
    ) -> Tuple[List[SearchResult], str, List[Dict], str]:
        """Turn retrieved results into (results, prompt_context, citations,
        verify_context).

        `verify_context` is deliberately the UNCOMPRESSED context, even on the
        complex route where the model is prompted with the compressed one. The
        compressed text is lossy LLM output: verifying against it can flag a
        true claim as unsupported when compression is what dropped the
        supporting sentence. This preserves the non-streaming path's behaviour,
        which is the behaviour every eval number was measured against (spec §4).
        """
        results = self._expand_to_parent(route_result.results)  # Plan §3.4
        context = self._build_context(results, route_result.annotations)
        citations = self._build_citations(results, route_result.annotations)
        verify_context = context
        if compress:
            context = self._compress_context(question, context)
        return results, context, citations, verify_context
```

- [ ] **Step 5: `_post_process` nhận citations dựng sẵn**

Sửa chữ ký `_post_process` (`:540`) thêm tham số `citations`, và thay dòng `citations = self._build_citations(results)` (`:564`) bằng dùng tham số. Điều này sửa #8 (verifier trước đây thấy citation không có cờ amendment):

```python
    def _post_process(self, question: str, answer: str, results: List[SearchResult],
                      context: str, citations: Optional[List[Dict]] = None,
                      rerank_scores: Optional[List[float]] = None,
                      generation_model: Optional[str] = None) -> tuple:
```

và trong thân:

```python
        # Tier 1 + 2: Verifier
        if citations is None:
            citations = self._build_citations(results)
        ver_result = self._verifier.verify(answer, citations, context)
```

Xoá dòng `context_str = context` và thay mọi `context_str` còn lại bằng `context`.

- [ ] **Step 6: Viết lại 4 method route thành một**

Xoá `_simple_pipeline`, `_medium_pipeline`, `_complex_pipeline` và thân `_lookup_pipeline`; thay bằng một method chung, và cho `_lookup_pipeline` gọi nó:

```python
    def _run_route(self, question: str, complexity: str, start_time: float,
                   is_comparison: bool = False) -> RAGResponse:
        """Retrieve → generate → post-process for any non-streaming route."""
        route_result = _drain(self._retrieve(question, complexity, is_comparison))
        results, context, citations, verify_context = self._prepare_generation(
            question, route_result, compress=(complexity == "complex"),
        )

        gen_models = MODEL_COMPLEX if complexity in ("medium", "complex") else MODEL_SIMPLE
        raw, generation_model = self.llm.generate_with_model(
            prompt=GENERATION_PROMPT.format(context=context, question=question),
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            json_mode=False,
            model=gen_models,
        )

        conf_result, ver_result, final_answer = self._post_process(
            question, raw, results, verify_context, citations=citations,
            rerank_scores=route_result.rerank_scores, generation_model=generation_model,
        )

        parsed = self._parse_response(final_answer, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = complexity
        return self._enrich_response(parsed, conf_result, ver_result)

    @observe
    def _lookup_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """LOOKUP: direct metadata lookup for 'Điều N' citations (+ dense fill).

        Deliberately does NOT fall back to vector search when the lookup finds
        nothing (e.g. "Điều 999" doesn't exist) — context stays empty/small so
        the existing refusal path triggers naturally instead of citing a
        plausible-looking wrong article.
        """
        return self._run_route(question, "lookup", start_time)
```

Cập nhật `query()` (Task 4 Step 2) — bỏ ba nhánh riêng:

```python
        if complexity == "lookup":
            return self._lookup_pipeline(question, start_time)
        if not in_scope:
            return self._out_of_scope_response(question, start_time)
        return self._run_route(question, complexity, start_time, is_comparison)
```

- [ ] **Step 7: `complex_retrieve` thành wrapper mỏng**

Giữ public (eval harness M3.5 gọi), bỏ bản sao logic:

```python
    def complex_retrieve(
        self, question: str, is_comparison: bool = False,
    ) -> Tuple[List[SearchResult], Optional[List[float]]]:
        """Complex-route retrieval. Returns (results, rerank_scores) POST-rerank.

        Kept public for the M3.5 eval harness (scripts/eval.py --multihop). NOTE:
        the returned results now include cross-ref expansion, matching what the
        complex route actually feeds the model.
        """
        route_result = _drain(self._retrieve(question, "complex", is_comparison))
        return route_result.results, route_result.rerank_scores
```

⚠️ Đây là **thay đổi hành vi** của `complex_retrieve`: trước đây nó trả kết quả *trước* cross-ref expansion. Kiểm `scripts/eval.py --multihop` sau khi sửa; nếu harness cần bản pre-expansion thì thêm tham số `include_expansion: bool = True` thay vì đổi ngầm.

- [ ] **Step 8: `query_stream` dùng `_retrieve`**

Thay toàn bộ khối phân nhánh (`:1007-1101`) và phần dựng context (`:1103-1114`) bằng:

```python
        route_result = yield from self._retrieve(question, complexity, is_comparison)
        results, context, citations, verify_context = self._prepare_generation(
            question, route_result, compress=(complexity == "complex"),
        )

        yield f'__STEP__{json.dumps({"step": "Search Results", "result": f"Found {len(results)} chunks for context."})}'
        yield f'__STEP__{json.dumps({"step": "Context (Found Chunks)", "result": context})}'
        yield f'__CITATIONS__{json.dumps(citations, ensure_ascii=False)}'
```

và ở phần `_post_process` cuối (`:1141`), truyền `verify_context` + `citations` + `route_result.rerank_scores`:

```python
        conf_result, ver_result, final_answer = self._post_process(
            question, clean_answer, results, verify_context, citations=citations,
            rerank_scores=route_result.rerank_scores,
            generation_model=used_model.get("model"),
        )
```

- [ ] **Step 9: Dọn import chết**

Trong `src/rag/orchestrator.py`:
- `:13` `import statistics, time` → `import time` (`statistics` không dùng ở file này)
- `:315` xoá `import re` cục bộ trong `_parse_response` (đã import ở `:11`)

- [ ] **Step 10: Chạy test parity — tất cả phải XANH**

Run: `.venv/bin/python -m pytest tests/test_stream_nonstream_parity.py -q`
Expected: 7 passed (6 lock + 1 RERANK_MAX_K).

- [ ] **Step 11: Chạy toàn bộ suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: tất cả xanh.

Nếu `tests/test_orchestrator.py` hoặc `tests/test_cross_ref_expansion.py` đỏ vì gọi thẳng `_simple_pipeline`/`_medium_pipeline`/`_complex_pipeline` (các method vừa bị xoá): cập nhật chúng sang `_run_route(question, "<route>", start_time)`. **Không** dựng lại method cũ chỉ để test xanh.

- [ ] **Step 12: Commit**

```bash
git add src/rag/orchestrator.py tests/test_stream_nonstream_parity.py tests/
git commit -m "refactor(rag): single _retrieve generator for both query paths

RERANK_MAX_K is now honoured at every call site, including the streaming path
the UI actually runs. Also removes the duplicate _build_context/_build_citations
pass and gives the verifier the annotated citations."
```

---

### Task 6: Adaptive làm mặc định

**Files:**
- Modify: `src/ui/pages/chat.py:116`
- Modify: `src/api/main.py:43`
- Test: `tests/test_api_wiring.py`

**Interfaces:**
- Produces: `ChatRequest.use_adaptive` mặc định `True` — thay đổi hợp đồng API công khai.

- [ ] **Step 1: Viết test đỏ**

Thêm vào `tests/test_api_wiring.py`:

```python
def test_chat_request_defaults_to_adaptive():
    """The naive pipeline has no scope gate, no confidence gate, no verifier and
    no reranker — none of the M0–M3 safety work. Defaulting to it meant every
    published metric described a pipeline users were not running
    (audit 2026-08-18, #1).
    """
    import src.api.main as main

    assert main.ChatRequest(question="q").use_adaptive is True
```

- [ ] **Step 2: Chạy — phải ĐỎ**

Run: `.venv/bin/python -m pytest tests/test_api_wiring.py::test_chat_request_defaults_to_adaptive -q`
Expected: FAIL — `assert False is True`.

- [ ] **Step 3: Đổi mặc định API**

`src/api/main.py:43`:

```python
    use_adaptive: bool = True
```

Thêm comment ngay trên dòng đó:

```python
    # Adaptive is the default: it is the only pipeline with the scope gate,
    # confidence gate and verifier, and the only one every eval number in
    # docs/ was measured against. Naive stays reachable for A/B comparison.
```

- [ ] **Step 4: Đổi mặc định UI**

`src/ui/pages/chat.py:116`:

```python
        use_adaptive = st.checkbox("Adaptive RAG (v1.0+)", value=True,
                                   help="Bật query routing: SIMPLE/MEDIUM/COMPLEX")
```

- [ ] **Step 5: Chạy toàn bộ suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: tất cả xanh. `test_api_chat_memory.py:86` và `test_answer_format_fixes.py:124` truyền `use_adaptive` tường minh nên không ảnh hưởng.

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py src/ui/pages/chat.py tests/test_api_wiring.py
git commit -m "fix(api,ui): default to the adaptive pipeline

Naive has no scope gate, confidence gate, verifier or reranker — defaulting to
it meant the shipped pipeline was never the measured one."
```

---

### Task 7: Đo lại route complex + cập nhật tài liệu

**Files:**
- Create: `docs/notes/2026-08-18-stream-unification-results.md`
- Modify: `docs/superpowers/specs/2026-08-18-stream-nonstream-unification-design.md` (đánh dấu trạng thái)

- [ ] **Step 1: Chạy Tier-2 trước/sau trên route complex**

Spec §4 xác định thay đổi hành vi duy nhất: route complex giờ verify trên context **chưa nén**. Đo lại phần đó.

```bash
.venv/bin/python scripts/eval.py --set criminal --answers --limit 41
```

Gate (từ B0, spec §9): `refusal_recall ≥ 0.7619`, `false_refusal ≤ 0.0972`.

⚠️ Nhớ ghi chú đã biết: confidence nhiễu ±0.1–0.2 giữa các lần chạy (memory `m3-7-confident-oos-null`) — chạy **ít nhất 2 lần** và lấy trung bình, đừng kết luận từ một lần chạy.

⚠️ Cần Groq budget. Nếu quota đã cạn, **dừng ở đây** và ghi lại là nợ đo, đừng đoán số.

- [ ] **Step 2: Ghi artefact**

Tạo `docs/notes/2026-08-18-stream-unification-results.md` với: số trước/sau, số lần chạy, verdict so với gate, và bất kỳ hồi quy nào quan sát được.

- [ ] **Step 3: Commit**

```bash
git add docs/notes/2026-08-18-stream-unification-results.md docs/superpowers/specs/2026-08-18-stream-nonstream-unification-design.md
git commit -m "docs: Tier-2 re-measurement after stream/non-stream unification"
```

---

## Self-Review

**Spec coverage:**

| Mục spec | Task |
|---|---|
| §5 phương án A: `_route`, `_retrieve`, `_drain` | Task 4, 5 |
| §5 dọn kèm #7 (build_context ×2) | Task 5 Step 4 |
| §5 dọn kèm #8 (annotations vào `_post_process`) | Task 5 Step 5 |
| §5 dọn kèm #3 (bỏ `[:15]`, dùng `RERANK_MAX_K`) | Task 5 Step 3 |
| §5 dọn kèm import thừa | Task 5 Step 9 |
| §6 gateway raise khi cạn chain | Task 2 |
| §6 `StreamInterrupted` + xử lý 2 front-end | Task 3 |
| §7 adaptive mặc định (UI + API) | Task 6 |
| §8 test 1 (max_k parity) | Task 1 |
| §8 test 2 (RERANK_MAX_K cả hai đường) | Task 5 Step 1 |
| §8 test 3 (gateway raise, is_exhaustion bắt được) | Task 2 Step 1 |
| §8 test 4 (không cache/persist khi lỗi) | Task 3 Step 1 |
| §8 test 5 (cùng point_id) | Task 1 |
| §9 đo lại Tier-2 route complex | Task 7 |

Không có mục spec nào thiếu task.

**Rủi ro chưa được spec lường trước, phát hiện khi viết plan:**

1. **`complex_retrieve` đổi ngữ nghĩa** (Task 5 Step 7) — trước trả kết quả pre-expansion, sau là post-expansion. Spec §5 chỉ nói "giữ nguyên như wrapper mỏng" mà không nhận ra điểm này. Đã ghi cảnh báo + phương án dự phòng (`include_expansion` param) ngay trong step.
2. **`gateway.py` giờ import `degraded.py`** (Task 3 Step 4) — chiều import mới. `degraded.py` không import gì từ `src` nên an toàn, đã thêm bước xác nhận (Task 3 Step 8).
3. **`RERANK_MAX_K` đọc lúc import** — test phải dùng `monkeypatch.setattr(orch, "RERANK_MAX_K", 8)`, không phải `setenv`. Đã phản ánh trong Task 5 Step 1.

**Type consistency:** `RouteResult` (`.results`, `.annotations`, `.rerank_scores`) dùng nhất quán ở Task 5 Steps 3, 4, 6, 7, 8. `_prepare_generation` trả tuple 4 phần tử `(results, context, citations, verify_context)` — thứ tự giống nhau ở cả `_run_route` và `query_stream`. `_post_process` nhận `citations` làm tham số thứ 5 (keyword) ở mọi điểm gọi, kể cả trong double của test Task 1.
