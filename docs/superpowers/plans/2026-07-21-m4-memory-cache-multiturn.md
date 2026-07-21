# M4 — Memory · Condensation · Semantic Cache · Multi-turn Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add server-side multi-turn conversation memory, pre-routing query condensation, an embedding-similarity semantic cache, and a `--multiturn` eval — all at the serving layer, leaving the RAG pipeline untouched.

**Architecture:** Session memory + semantic cache each sit behind a small ABC with a Redis backend (when `REDIS_URL` is set) and an in-process fallback. Condensation is a pure function that rewrites a follow-up into a standalone question before routing. A thin `run_chat_turn` helper threads memory→condense→cache→pipeline→memory so the API stays trivial and the logic is unit-testable without Qdrant/models.

**Tech Stack:** Python, FastAPI, Streamlit, `redis` (new dep, lazily imported), pytest, existing `LLMGateway` + `MODEL_QUERY_TRANSFORM`, existing eval harness (`scripts/eval.py`, `src/eval/metrics.py`).

## Global Constraints

- **$0 / free tier** — no paid infra; Redis is optional (Upstash/in-container), code must run with no Redis.
- **Keep both refusal gates intact** — scope gate + confidence gate unchanged; condensation runs BEFORE routing so downstream sees only a standalone query.
- **Fail-open everywhere** — condense error → original question; store error → empty history; cache error → miss. A chat request must NEVER fail because memory/cache/condense is unavailable.
- **Pipeline stays pure** — do not thread `session_id` into `src/rag/orchestrator.py`. It remains a function of one `question: str`.
- **Reuse `MODEL_QUERY_TRANSFORM`** (defined `src/rag/orchestrator.py:53`) for condensation.
- **Cache namespace = `DEFAULT_COLLECTION_NAME`** (imported in `main.py` from `src.data.constants`) — a collection swap auto-invalidates.
- **Constants (module-level):** `MAX_TURNS=10`, `TTL_SECONDS=3600`, `SIM_THRESHOLD=0.95`, `CACHE_MAX=500`, eval pass `recall@5 ≥ 0.5`.
- **Verify every eval `expected_articles` label against the corpus** — labels went stale twice before (memory `vlb-project-state`); reuse pairs from already-verified sets.
- **`make test` (`pytest tests/ -v`) stays green** (~421 tests today). TDD: test first, watch it fail, implement, watch it pass, commit each task.

---

### Task 1: Redis helper + Session store

**Files:**
- Create: `src/rag/redis_util.py`
- Create: `src/rag/session_store.py`
- Create: `tests/test_session_store.py`
- Modify: `pyproject.toml` (add `redis` dependency)

**Interfaces:**
- Produces: `get_redis_client(url: Optional[str]=None) -> redis.Redis | None`
- Produces: `SessionStore` (ABC) with `get_history(session_id: Optional[str]) -> List[Dict]`, `append_turn(session_id, question: str, answer: str) -> None`, `SessionStore.from_env() -> SessionStore`; concrete `MemorySessionStore`, `RedisSessionStore(client)`; module consts `MAX_TURNS=10`, `TTL_SECONDS=3600`.
- Turn dict shape: `{"question": str, "answer": str}`, most-recent-last.

- [ ] **Step 1: Add `redis` to `pyproject.toml`**

Find the `dependencies = [` list and add the line (keep alphabetical-ish placement, match existing quoting):

```toml
    "redis>=5.0.0,<6.0.0",
```

- [ ] **Step 2: Write `src/rag/redis_util.py`**

```python
"""Shared lazy Redis client factory (M4).

`redis` is imported only when a URL is actually present, so the in-memory
fallback path used by tests / eval / local dev needs no redis package or
running server. Any import or connection failure returns None — callers then
fall back to their in-memory backend."""
from __future__ import annotations

import os
from typing import Optional


def get_redis_client(url: Optional[str] = None):
    """Return a live `redis.Redis` for REDIS_URL (or `url`), else None.

    decode_responses=True → values come back as str (we store JSON text).
    Pings once so a set-but-dead URL degrades to None instead of failing later.
    """
    url = url or os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None
```

- [ ] **Step 3: Write the failing test `tests/test_session_store.py`**

```python
from src.rag.session_store import (
    MAX_TURNS,
    MemorySessionStore,
    RedisSessionStore,
    SessionStore,
)


class FakeRedis:
    """Minimal in-memory Redis list emulation (rpush/lrange/ltrim/expire)
    with correct negative-index semantics for our usage."""

    def __init__(self):
        self.store = {}

    @staticmethod
    def _slice(lst, start, stop):
        n = len(lst)
        if start < 0:
            start = max(n + start, 0)
        if stop < 0:
            stop = n + stop
        stop = min(stop, n - 1)
        if start > stop or start >= n:
            return []
        return lst[start:stop + 1]

    def rpush(self, k, v):
        self.store.setdefault(k, []).append(v)

    def lrange(self, k, start, stop):
        return list(self._slice(self.store.get(k, []), start, stop))

    def ltrim(self, k, start, stop):
        self.store[k] = self._slice(self.store.get(k, []), start, stop)

    def expire(self, k, ttl):
        pass


def test_memory_store_appends_and_caps():
    store = MemorySessionStore()
    for i in range(MAX_TURNS + 2):
        store.append_turn("s1", f"q{i}", f"a{i}")
    hist = store.get_history("s1")
    assert len(hist) == MAX_TURNS
    assert hist[0] == {"question": "q2", "answer": "a2"}
    assert hist[-1] == {"question": f"q{MAX_TURNS + 1}", "answer": f"a{MAX_TURNS + 1}"}


def test_memory_store_none_session_is_noop():
    store = MemorySessionStore()
    store.append_turn(None, "q", "a")
    assert store.get_history(None) == []


def test_memory_store_isolates_sessions():
    store = MemorySessionStore()
    store.append_turn("a", "qa", "aa")
    store.append_turn("b", "qb", "ab")
    assert store.get_history("a") == [{"question": "qa", "answer": "aa"}]


def test_redis_store_roundtrip_with_fake_client():
    store = RedisSessionStore(FakeRedis())
    store.append_turn("s1", "q1", "a1")
    store.append_turn("s1", "q2", "a2")
    assert store.get_history("s1") == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]


def test_redis_store_caps_history():
    store = RedisSessionStore(FakeRedis())
    for i in range(MAX_TURNS + 3):
        store.append_turn("s1", f"q{i}", f"a{i}")
    assert len(store.get_history("s1")) == MAX_TURNS


def test_redis_store_errors_degrade_to_empty():
    class Boom:
        def lrange(self, *a):
            raise RuntimeError("down")
    assert RedisSessionStore(Boom()).get_history("s1") == []


def test_from_env_uses_memory_without_redis(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert isinstance(SessionStore.from_env(), MemorySessionStore)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_session_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rag.session_store'`

- [ ] **Step 5: Write `src/rag/session_store.py`**

```python
"""Conversation memory for multi-turn chat (M4.1).

Serving-layer only — the RAG pipeline never sees session_id. One interface,
two backends: Redis when REDIS_URL is set, else a process-local dict. Both cap
at MAX_TURNS and are fail-open (any error → empty history / silent no-op; a
chat request never fails because memory is unavailable)."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Optional

from src.rag.redis_util import get_redis_client

MAX_TURNS = 10
TTL_SECONDS = 3600


class SessionStore(ABC):
    @abstractmethod
    def get_history(self, session_id: Optional[str]) -> List[Dict]:
        """Most-recent-last list of {"question","answer"}; [] if none/error."""

    @abstractmethod
    def append_turn(self, session_id: Optional[str], question: str, answer: str) -> None:
        """Append one turn; no-op if session_id is falsy."""

    @staticmethod
    def from_env() -> "SessionStore":
        client = get_redis_client()
        return RedisSessionStore(client) if client is not None else MemorySessionStore()


class MemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_TURNS))

    def get_history(self, session_id):
        if not session_id:
            return []
        return list(self._data[session_id])

    def append_turn(self, session_id, question, answer):
        if not session_id:
            return
        self._data[session_id].append({"question": question, "answer": answer})


class RedisSessionStore(SessionStore):
    def __init__(self, client) -> None:
        self._r = client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"vlb:session:{session_id}"

    def get_history(self, session_id):
        if not session_id:
            return []
        try:
            raw = self._r.lrange(self._key(session_id), -MAX_TURNS, -1)
            return [json.loads(x) for x in raw]
        except Exception:
            return []

    def append_turn(self, session_id, question, answer):
        if not session_id:
            return
        try:
            key = self._key(session_id)
            self._r.rpush(key, json.dumps({"question": question, "answer": answer}, ensure_ascii=False))
            self._r.ltrim(key, -MAX_TURNS, -1)
            self._r.expire(key, TTL_SECONDS)
        except Exception:
            pass
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_session_store.py -v`
Expected: PASS (7 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/rag/redis_util.py src/rag/session_store.py tests/test_session_store.py
git commit -m "feat(m4): session store with Redis + in-memory fallback"
```

---

### Task 2: Query condensation

**Files:**
- Create: `src/rag/condense.py`
- Create: `tests/test_condense.py`

**Interfaces:**
- Consumes: `MODEL_QUERY_TRANSFORM` from `src.rag.orchestrator`; `observe` from `src.observability`; an `llm` object exposing `generate(prompt, model=, temperature=, max_tokens=, reasoning_effort=) -> str`.
- Produces: `condense_query(history: List[Dict], question: str, llm) -> str`.

- [ ] **Step 1: Write the failing test `tests/test_condense.py`**

```python
from unittest.mock import MagicMock

from src.rag.condense import condense_query


def test_empty_history_returns_original_and_skips_llm():
    llm = MagicMock()
    q = "Tội cướp tài sản bị phạt thế nào?"
    assert condense_query([], q, llm) == q
    llm.generate.assert_not_called()


def test_rewrites_followup_using_history():
    llm = MagicMock()
    llm.generate.return_value = "  Tội cướp giật tài sản bị xử phạt thế nào?  "
    history = [{"question": "Tội cướp tài sản bị phạt thế nào?", "answer": "..."}]
    out = condense_query(history, "vậy còn cướp giật thì sao?", llm)
    assert out == "Tội cướp giật tài sản bị xử phạt thế nào?"
    llm.generate.assert_called_once()


def test_failopen_on_empty_output():
    llm = MagicMock()
    llm.generate.return_value = "   "
    out = condense_query([{"question": "q1", "answer": "a1"}], "vậy còn X?", llm)
    assert out == "vậy còn X?"


def test_failopen_on_exception():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("llm down")
    out = condense_query([{"question": "q1", "answer": "a1"}], "vậy còn X?", llm)
    assert out == "vậy còn X?"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_condense.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rag.condense'`

- [ ] **Step 3: Write `src/rag/condense.py`**

```python
"""Query condensation for multi-turn chat (M4.1).

Rewrites a follow-up into a standalone question BEFORE routing, so every
downstream route stays history-unaware. Pure function — the eval harness calls
it directly. Fail-open: empty history skips the LLM; any error/empty output
returns the original question."""
from __future__ import annotations

from typing import Dict, List

from src.observability import observe
from src.rag.orchestrator import MODEL_QUERY_TRANSFORM

CONDENSE_PROMPT = """Bạn viết lại CÂU HỎI TIẾP THEO thành một câu hỏi ĐỘC LẬP, đầy đủ ngữ cảnh, dựa trên lịch sử hội thoại.

Quy tắc:
- Thay đại từ / tham chiếu ("vậy còn...", "nó", "trường hợp đó", "loại này") bằng chủ thể cụ thể lấy từ lịch sử.
- Nếu câu hỏi ĐÃ độc lập (không phụ thuộc lịch sử), giữ NGUYÊN VĂN.
- Chỉ xuất ra ĐÚNG MỘT câu hỏi tiếng Việt. Không giải thích, không tiền tố, không dấu ngoặc kép.

Lịch sử hội thoại:
{history}

Câu hỏi tiếp theo: {question}

Câu hỏi độc lập:"""


def _format_history(history: List[Dict]) -> str:
    lines = []
    for turn in history:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()
        if q:
            lines.append(f"Người dùng: {q}")
        if a:
            lines.append(f"Trợ lý: {a}")
    return "\n".join(lines)


@observe
def condense_query(history: List[Dict], question: str, llm) -> str:
    if not history:
        return question
    try:
        rewritten = llm.generate(
            CONDENSE_PROMPT.format(history=_format_history(history), question=question),
            model=MODEL_QUERY_TRANSFORM,
            temperature=0,
            max_tokens=120,
            reasoning_effort="none",
        )
    except Exception:
        return question
    rewritten = (rewritten or "").strip()
    return rewritten or question
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_condense.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/condense.py tests/test_condense.py
git commit -m "feat(m4): pre-routing query condensation"
```

---

### Task 3: Semantic cache

**Files:**
- Create: `src/rag/semantic_cache.py`
- Create: `tests/test_semantic_cache.py`

**Interfaces:**
- Consumes: `RAGResponse` from `src.rag.orchestrator`; `get_redis_client` from `src.rag.redis_util`; an `embed_fn: Callable[[str], List[float]]` returning a (normalized) dense vector.
- Produces: `SemanticCache` (ABC) with `get(query: str, collection: str) -> RAGResponse | None`, `set(query, collection, response: RAGResponse) -> None`, `SemanticCache.from_env(embed_fn) -> SemanticCache | None`; concrete `MemorySemanticCache`, `RedisSemanticCache`; consts `SIM_THRESHOLD=0.95`, `CACHE_MAX=500`.

- [ ] **Step 1: Write the failing test `tests/test_semantic_cache.py`**

```python
import src.rag.semantic_cache as sc
from src.rag.orchestrator import RAGResponse
from src.rag.semantic_cache import MemorySemanticCache


def _resp(answer="A"):
    return RAGResponse(
        answer=answer, citations=[], confidence="high", disclaimer="",
        latency_ms=1.0, pipeline_used="simple",
    )


# keyword -> deterministic dense vector (near-orthogonal)
_TABLE = {
    "cuop tai san": [1.0, 0.0, 0.0],
    "cuop giat": [0.0, 1.0, 0.0],
    "thua ke": [0.0, 0.0, 1.0],
    "near_hit": [0.98, 0.20, 0.0],    # cos with cuop tai san ~ 0.980 -> hit
    "near_miss": [0.94, 0.34, 0.0],   # cos ~ 0.940 -> miss
}


def _embed(text):
    return _TABLE[text]


def test_exact_hit_returns_reconstructed_response():
    c = MemorySemanticCache(_embed)
    c.set("cuop tai san", "col1", _resp("hello"))
    hit = c.get("cuop tai san", "col1")
    assert isinstance(hit, RAGResponse)
    assert hit.answer == "hello"


def test_miss_on_dissimilar_query():
    c = MemorySemanticCache(_embed)
    c.set("cuop tai san", "col1", _resp())
    assert c.get("thua ke", "col1") is None


def test_hit_above_threshold():
    c = MemorySemanticCache(_embed)
    c.set("cuop tai san", "col1", _resp("robbery"))
    assert c.get("near_hit", "col1").answer == "robbery"


def test_miss_below_threshold():
    c = MemorySemanticCache(_embed)
    c.set("cuop tai san", "col1", _resp())
    assert c.get("near_miss", "col1") is None


def test_namespaced_by_collection():
    c = MemorySemanticCache(_embed)
    c.set("cuop tai san", "col1", _resp())
    assert c.get("cuop tai san", "col2") is None


def test_fifo_eviction(monkeypatch):
    monkeypatch.setattr(sc, "CACHE_MAX", 2)
    c = MemorySemanticCache(lambda t: [float(t), 0.0, 0.0])
    for i in range(3):
        c.set(str(i), "col", _resp(str(i)))
    entries = c._entries("col")
    assert len(entries) == 2                 # oldest evicted
    assert c.get("0", "col") is None         # "0" was evicted


def test_get_failopen_on_embed_error():
    def boom(_):
        raise RuntimeError("embed down")
    c = MemorySemanticCache(boom)
    assert c.get("anything", "col") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_semantic_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rag.semantic_cache'`

- [ ] **Step 3: Write `src/rag/semantic_cache.py`**

```python
"""Embedding-similarity response cache (M4.2).

Keyed on the (condensed) query embedding, namespaced by collection so a
collection swap auto-invalidates. Redis 7 has no vector search → client-side
cosine scan over a bounded (FIFO, CACHE_MAX) buffer. Serving-layer only and
fail-open (any error → miss). Stores RAGResponse as a plain dict so the Redis
backend can JSON-serialize it."""
from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

from src.rag.orchestrator import RAGResponse
from src.rag.redis_util import get_redis_client

SIM_THRESHOLD = 0.95
CACHE_MAX = 500


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache(ABC):
    def __init__(self, embed_fn: Callable[[str], List[float]]):
        self._embed = embed_fn

    @abstractmethod
    def _entries(self, collection: str) -> List[Tuple[List[float], Dict]]:
        """(embedding, response_dict) pairs for this collection."""

    @abstractmethod
    def _store(self, collection: str, emb: List[float], resp: Dict) -> None:
        """Append one entry, FIFO-evicting to CACHE_MAX."""

    def get(self, query: str, collection: str) -> Optional[RAGResponse]:
        try:
            emb = self._embed(query)
            best, best_sim = None, 0.0
            for cached_emb, resp in self._entries(collection):
                sim = _cosine(emb, cached_emb)
                if sim > best_sim:
                    best, best_sim = resp, sim
            if best is not None and best_sim >= SIM_THRESHOLD:
                return RAGResponse(**best)
        except Exception:
            return None
        return None

    def set(self, query: str, collection: str, response: RAGResponse) -> None:
        try:
            self._store(collection, self._embed(query), asdict(response))
        except Exception:
            pass

    @staticmethod
    def from_env(embed_fn: Callable[[str], List[float]]) -> Optional["SemanticCache"]:
        if os.getenv("SEMANTIC_CACHE_DISABLED"):
            return None
        client = get_redis_client()
        if client is not None:
            return RedisSemanticCache(embed_fn, client)
        return MemorySemanticCache(embed_fn)


class MemorySemanticCache(SemanticCache):
    def __init__(self, embed_fn):
        super().__init__(embed_fn)
        self._buf: Dict[str, List[Tuple[List[float], Dict]]] = {}

    def _entries(self, collection):
        return self._buf.get(collection, [])

    def _store(self, collection, emb, resp):
        buf = self._buf.setdefault(collection, [])
        buf.append((emb, resp))
        while len(buf) > CACHE_MAX:
            del buf[0]


class RedisSemanticCache(SemanticCache):
    def __init__(self, embed_fn, client):
        super().__init__(embed_fn)
        self._r = client

    @staticmethod
    def _key(collection):
        return f"vlb:cache:{collection}"

    def _entries(self, collection):
        raw = self._r.lrange(self._key(collection), 0, -1)
        out = []
        for x in raw:
            item = json.loads(x)
            out.append((item["emb"], item["resp"]))
        return out

    def _store(self, collection, emb, resp):
        key = self._key(collection)
        self._r.rpush(key, json.dumps({"emb": emb, "resp": resp}, ensure_ascii=False))
        self._r.ltrim(key, -CACHE_MAX, -1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_semantic_cache.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/semantic_cache.py tests/test_semantic_cache.py
git commit -m "feat(m4): embedding-similarity semantic cache"
```

---

### Task 4: `run_chat_turn` orchestration helper

**Files:**
- Create: `src/rag/chat_turn.py`
- Create: `tests/test_chat_turn.py`

**Interfaces:**
- Consumes: `condense_query` (Task 2); a `pipeline` with `query(question: str) -> RAGResponse`; a `store` (Task 1); a `cache` (Task 3) or `None`; an `llm`.
- Produces: `run_chat_turn(pipeline, store, cache, llm, collection: str, session_id: Optional[str], question: str) -> ChatTurnResult`; `ChatTurnResult(response, cache_hit: bool, condensed_question: str)`.

- [ ] **Step 1: Write the failing test `tests/test_chat_turn.py`**

```python
from unittest.mock import MagicMock

from src.rag.chat_turn import run_chat_turn
from src.rag.orchestrator import RAGResponse


def _resp(answer="A"):
    return RAGResponse(
        answer=answer, citations=[], confidence="high", disclaimer="",
        latency_ms=1.0, pipeline_used="simple",
    )


def test_cache_miss_runs_pipeline_stores_and_caches():
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None
    pipeline = MagicMock(); pipeline.query.return_value = _resp("hello")

    out = run_chat_turn(pipeline, store, cache, MagicMock(), "col", "s1", "câu hỏi")

    assert out.cache_hit is False
    assert out.response.answer == "hello"
    assert out.condensed_question == "câu hỏi"     # no history → unchanged
    pipeline.query.assert_called_once_with("câu hỏi")
    cache.set.assert_called_once()
    store.append_turn.assert_called_once_with("s1", "câu hỏi", "hello")


def test_cache_hit_skips_pipeline():
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = _resp("cached")
    pipeline = MagicMock()

    out = run_chat_turn(pipeline, store, cache, MagicMock(), "col", "s1", "q")

    assert out.cache_hit is True
    assert out.response.answer == "cached"
    pipeline.query.assert_not_called()
    cache.set.assert_not_called()


def test_condenses_with_history_before_pipeline():
    store = MagicMock()
    store.get_history.return_value = [{"question": "Tội cướp tài sản?", "answer": "..."}]
    llm = MagicMock(); llm.generate.return_value = "Tội cướp giật tài sản bị phạt thế nào?"
    cache = MagicMock(); cache.get.return_value = None
    pipeline = MagicMock(); pipeline.query.return_value = _resp("x")

    out = run_chat_turn(pipeline, store, cache, llm, "col", "s1", "vậy còn cướp giật?")

    assert out.condensed_question == "Tội cướp giật tài sản bị phạt thế nào?"
    pipeline.query.assert_called_once_with("Tội cướp giật tài sản bị phạt thế nào?")


def test_no_cache_object_is_ok():
    store = MagicMock(); store.get_history.return_value = []
    pipeline = MagicMock(); pipeline.query.return_value = _resp("hi")

    out = run_chat_turn(pipeline, store, None, MagicMock(), "col", "s1", "q")

    assert out.cache_hit is False
    assert out.response.answer == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chat_turn.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rag.chat_turn'`

- [ ] **Step 3: Write `src/rag/chat_turn.py`**

```python
"""Single-turn chat orchestration for the non-streaming path (M4).

Threads session memory + condensation + semantic cache around the pure RAG
pipeline. Lives outside the API layer so it is unit-testable without a live
Qdrant/model stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.rag.condense import condense_query
from src.rag.orchestrator import RAGResponse


@dataclass
class ChatTurnResult:
    response: RAGResponse
    cache_hit: bool
    condensed_question: str


def run_chat_turn(pipeline, store, cache, llm, collection: str,
                  session_id: Optional[str], question: str) -> ChatTurnResult:
    history = store.get_history(session_id)
    condensed = condense_query(history, question, llm)

    response = cache.get(condensed, collection) if cache else None
    cache_hit = response is not None
    if response is None:
        response = pipeline.query(condensed)
        if cache:
            cache.set(condensed, collection, response)

    store.append_turn(session_id, question, response.answer)
    return ChatTurnResult(response=response, cache_hit=cache_hit, condensed_question=condensed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat_turn.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/rag/chat_turn.py tests/test_chat_turn.py
git commit -m "feat(m4): run_chat_turn helper (memory+condense+cache around pipeline)"
```

---

### Task 5: Wire the API (`/api/chat`, `/api/chat/stream`)

**Files:**
- Modify: `src/api/main.py`
- Create: `tests/test_api_chat_memory.py`

**Interfaces:**
- Consumes: `SessionStore.from_env`, `SemanticCache.from_env`, `run_chat_turn`, `condense_query`, `RAGResponse`, `DEFAULT_COLLECTION_NAME`.
- Produces: `ChatResponse` gains `cache_hit: bool = False` and `condensed_question: str = ""`; `main._embed_query(text) -> List[float]`; `main._replay_events(resp, start_time) -> List[Dict]`; `app.state.session_store`, `app.state.semantic_cache`.

- [ ] **Step 1: Add imports (top of `src/api/main.py`, after existing `src.rag` imports ~line 29)**

```python
from src.rag.condense import condense_query
from src.rag.session_store import SessionStore
from src.rag.semantic_cache import SemanticCache
from src.rag.chat_turn import run_chat_turn
from src.rag.orchestrator import RAGResponse
```

(Note: `NaiveRAGPipeline, AdaptiveRAGPipeline` are already imported from `src.rag.orchestrator`; add `RAGResponse` to that existing import line instead of duplicating if you prefer.)

- [ ] **Step 2: Add two response fields to `ChatResponse` (after `expanded_results`, ~line 50)**

```python
    cache_hit: bool = False          # M4.2: served from semantic cache?
    condensed_question: str = ""     # M4.1: standalone rewrite of a follow-up
```

- [ ] **Step 3: Initialize stores in `lifespan` (inside the startup block, after `citation_lookup`, ~line 81)**

```python
    app.state.session_store = SessionStore.from_env()
    app.state.semantic_cache = SemanticCache.from_env(_embed_query)
```

- [ ] **Step 4: Add helper functions (near the other `get_*` helpers, after `get_reranker`)**

```python
def _embed_query(text: str):
    """Dense query embedding for the semantic cache. Normalized (dot == cosine),
    matches the collection's distance metric."""
    out = get_embedder().encode([text], return_dense=True, return_sparse=False)
    return out["dense_vecs"][0].tolist()


def _replay_events(resp: RAGResponse, start_time: float):
    """Render a cached RAGResponse as the SSE event sequence the UI expects."""
    events = []
    if resp.citations:
        events.append({"event": "citations",
                       "data": json.dumps(resp.citations, ensure_ascii=False)})
    events.append({"event": "token",
                   "data": json.dumps({"token": resp.answer}, ensure_ascii=False)})
    events.append({"event": "verification",
                   "data": json.dumps({"confidence": resp.confidence,
                                       "confidence_score": resp.confidence_score},
                                      ensure_ascii=False)})
    events.append({"event": "done",
                   "data": json.dumps({
                       "full_answer": resp.answer, "citations": resp.citations,
                       "confidence": resp.confidence, "confidence_score": resp.confidence_score,
                       "latency_ms": (time.time() - start_time) * 1000, "cache_hit": True,
                   }, ensure_ascii=False)})
    return events
```

- [ ] **Step 5: Replace the body of `/api/chat` (`def chat`) with the run_chat_turn flow**

```python
@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Non-streaming chat. Sync `def` so FastAPI runs it in the threadpool.
    Threads session memory + condensation + semantic cache around the pipeline."""
    try:
        pipeline = get_adaptive_pipeline() if request.use_adaptive else get_naive_pipeline()
        result = run_chat_turn(
            pipeline=pipeline,
            store=app.state.session_store,
            cache=app.state.semantic_cache,
            llm=app.state.llm,
            collection=DEFAULT_COLLECTION_NAME,
            session_id=request.session_id,
            question=request.question,
        )
        r = result.response
        return ChatResponse(
            answer=r.answer, citations=r.citations, confidence=r.confidence,
            confidence_score=r.confidence_score, refused=r.refused,
            disclaimer=r.disclaimer, latency_ms=r.latency_ms,
            pipeline_used=r.pipeline_used,
            cache_hit=result.cache_hit, condensed_question=result.condensed_question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 6: Update `/api/chat/stream` — condense + cache-replay fast path + post-stream store**

Replace the `async def chat_stream` body. Compute history/condense/cache lookup off the event loop, replay on hit, and store to memory+cache after a miss completes:

```python
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat (SSE). Cache HIT → replay cached response as SSE (<500ms);
    MISS → stream normally, then persist to memory + cache."""
    try:
        pipeline = get_adaptive_pipeline() if request.use_adaptive else get_naive_pipeline()
        store = app.state.session_store
        cache = app.state.semantic_cache

        history = await run_in_threadpool(store.get_history, request.session_id)
        condensed = await run_in_threadpool(condense_query, history, request.question, app.state.llm)
        cached = None
        if cache is not None:
            cached = await run_in_threadpool(cache.get, condensed, DEFAULT_COLLECTION_NAME)

        async def event_generator():
            start_time = time.time()

            if cached is not None:
                for event in _replay_events(cached, start_time):
                    yield event
                await run_in_threadpool(store.append_turn, request.session_id,
                                        request.question, cached.answer)
                return

            full_answer = ""
            citations = []
            confidence_label = "unverified"
            confidence_score = 0.0
            refused = False

            async for token in iterate_in_threadpool(pipeline.query_stream(condensed)):
                if token.startswith("__STEP__"):
                    try:
                        step_data = json.loads(token[len("__STEP__"):])
                        yield {"event": "step", "data": json.dumps(step_data, ensure_ascii=False)}
                    except json.JSONDecodeError:
                        pass
                    continue
                if token.startswith("__CITATIONS__"):
                    try:
                        citations = json.loads(token[len("__CITATIONS__"):])
                        yield {"event": "citations", "data": json.dumps(citations, ensure_ascii=False)}
                    except json.JSONDecodeError:
                        pass
                    continue
                if token.startswith("__VERIFICATION__"):
                    try:
                        verification = json.loads(token[len("__VERIFICATION__"):])
                        confidence_label = verification.get("confidence", confidence_label)
                        confidence_score = verification.get("confidence_score", confidence_score)
                        refused = verification.get("refused", refused)
                        yield {"event": "verification", "data": json.dumps(verification, ensure_ascii=False)}
                    except json.JSONDecodeError:
                        pass
                    continue

                full_answer += token
                yield {"event": "token", "data": json.dumps({"token": token}, ensure_ascii=False)}

            yield {"event": "done", "data": json.dumps({
                "full_answer": full_answer, "citations": citations,
                "confidence": confidence_label, "confidence_score": confidence_score,
                "latency_ms": (time.time() - start_time) * 1000, "cache_hit": False,
            }, ensure_ascii=False)}

            # Persist AFTER streaming completes (fail-open).
            await run_in_threadpool(store.append_turn, request.session_id,
                                    request.question, full_answer)
            if cache is not None and full_answer and not refused:
                reconstructed = RAGResponse(
                    answer=full_answer, citations=citations, confidence=confidence_label,
                    disclaimer="", latency_ms=(time.time() - start_time) * 1000,
                    pipeline_used="stream", confidence_score=confidence_score, refused=refused,
                )
                await run_in_threadpool(cache.set, condensed, DEFAULT_COLLECTION_NAME, reconstructed)

        return EventSourceResponse(event_generator())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 7: Write the failing test `tests/test_api_chat_memory.py`**

```python
from unittest.mock import MagicMock

import src.api.main as main
from src.rag.orchestrator import RAGResponse


def _resp(answer="A"):
    return RAGResponse(
        answer=answer, citations=[], confidence="high", disclaimer="",
        latency_ms=1.0, pipeline_used="simple",
    )


def test_chat_endpoint_returns_cache_and_condensed_fields(monkeypatch):
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    fake_pipe = MagicMock(); fake_pipe.query.return_value = _resp("hello")
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: fake_pipe)

    out = main.chat(main.ChatRequest(question="q", session_id="s1"))

    assert out.answer == "hello"
    assert out.cache_hit is False
    assert out.condensed_question == "q"
    store.append_turn.assert_called_once_with("s1", "q", "hello")


def test_chat_endpoint_cache_hit_skips_pipeline(monkeypatch):
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = _resp("cached")
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    fake_pipe = MagicMock()
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: fake_pipe)

    out = main.chat(main.ChatRequest(question="q", session_id="s1"))

    assert out.cache_hit is True
    assert out.answer == "cached"
    fake_pipe.query.assert_not_called()


def test_replay_events_shape():
    r = _resp("ans")
    r.citations = [{"n": 1}]
    events = main._replay_events(r, start_time=0.0)
    kinds = [e["event"] for e in events]
    assert kinds == ["citations", "token", "verification", "done"]
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_api_chat_memory.py tests/test_api_wiring.py -v`
Expected: PASS (existing wiring tests still green + 3 new)

- [ ] **Step 9: Commit**

```bash
git add src/api/main.py tests/test_api_chat_memory.py
git commit -m "feat(m4): wire memory, condensation, and semantic cache into chat API"
```

---

### Task 6: Streamlit UI — send `session_id` + surface cache/condense

**Files:**
- Modify: `src/ui/pages/chat.py`

**Interfaces:**
- Consumes: the new `cache_hit` / `condensed_question` response fields; posts `session_id` to `/api/chat` and `/api/chat/stream`.

- [ ] **Step 1: Generate a stable per-session id (near the top of the chat render fn, where `st.session_state.messages` is initialized ~line 128)**

```python
    import uuid
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
```

- [ ] **Step 2: Add `session_id` to BOTH request payloads**

Find the two `requests.post(...)` / stream calls (`f"{API_BASE}/api/chat"` ~line 202 and `f"{API_BASE}/api/chat/stream"` ~line 228). In each JSON body dict, add:

```python
        "session_id": st.session_state.session_id,
```

- [ ] **Step 3: Show a small caption when the answer was cached or the query was condensed**

After the assistant answer is rendered (where the response dict is available), add:

```python
    if data.get("cache_hit"):
        st.caption("⚡ Trả lời từ cache ngữ nghĩa")
    cq = data.get("condensed_question")
    if cq and cq != question:
        st.caption(f"🔗 Hiểu câu hỏi là: *{cq}*")
```

(Adapt `data`/`question` to the actual variable names in scope at the render site — for the streaming path, `cache_hit`/`condensed_question` come from the `done` event payload; wire whichever the existing code already parses.)

- [ ] **Step 4: Manual smoke (no automated UI test — Streamlit has no test harness here)**

Run: `python -c "import ast; ast.parse(open('src/ui/pages/chat.py').read()); print('parse ok')"`
Expected: `parse ok`

Then eyeball: `grep -n "session_id" src/ui/pages/chat.py` shows the uuid init + both payloads.

- [ ] **Step 5: Commit**

```bash
git add src/ui/pages/chat.py
git commit -m "feat(m4): UI sends session_id and surfaces cache/condensed hints"
```

---

### Task 7: Multi-turn eval set + `--multiturn` runner

**Files:**
- Create: `data/eval/eval_set_multiturn_v1.json`
- Modify: `scripts/eval.py`
- Create: `tests/test_multiturn_eval.py`

**Interfaces:**
- Consumes: `condense_query` (Task 2); `retrieve_routed`, `CitationLookup` (already imported in `eval.py`); `recall_at_k`, `_result_to_dict`, `LLMGateway`.
- Produces: `run_multiturn(scenarios, vector_store, embedder, llm) -> Dict`; `eval.py` gains a `--multiturn` flag.
- Eval-set schema: JSON list of scenarios `{"id": str, "legal_sector": str, "turns": [{"question": str, "expected_articles": [{"doc_id": int, "article": str}]}]}`. Turn 0 is standalone; turns ≥1 are follow-ups scored after condensation.

- [ ] **Step 1: Author `data/eval/eval_set_multiturn_v1.json` — ~10 scenarios, 2–3 turns each**

Build one scenario per sector (hinh_su, dan_su, lao_dong, doanh_nghiep, dat_dai, hon_nhan_gia_dinh) plus 4 more across sectors (10 total). **Do NOT invent `expected_articles`** — copy each `{doc_id, article}` pair from an already-verified question in `data/eval/eval_set_*_v*.json` (criminal/sectors/multihop) on the same topic, then author a natural follow-up whose referent resolves to that topic. Example scenario shape (fill real verified pairs):

```json
[
  {
    "id": "mt-hinhsu-01",
    "legal_sector": "hinh_su",
    "turns": [
      {
        "question": "Tội cướp tài sản bị xử phạt như thế nào theo Bộ luật Hình sự 2015?",
        "expected_articles": [{"doc_id": 100803, "article": "Điều 168"}]
      },
      {
        "question": "Vậy còn tội cướp giật tài sản thì sao?",
        "expected_articles": [{"doc_id": 100803, "article": "Điều 171"}]
      }
    ]
  }
]
```

- [ ] **Step 2: Verify every `expected_articles` pair exists in the corpus**

Run (checks each pair resolves via the citation route on the live default collection):

```bash
python - <<'PY'
import json
from src.db.qdrant_client import LegalVectorStore
from src.data.mixed_embedder import build_embedder
from src.rag.citation_lookup import CitationLookup, retrieve_routed

scen = json.load(open("data/eval/eval_set_multiturn_v1.json"))
vs = LegalVectorStore()
emb = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding")
cl = CitationLookup(vs)
missing = []
for s in scen:
    for t in s["turns"]:
        for exp in t["expected_articles"]:
            hits, _r, _p = retrieve_routed(
                f'{exp["article"]} tài liệu {exp["doc_id"]}', vs, emb, cl, fill_top_k=20)
            ok = any(h.document_id == exp["doc_id"] and exp["article"] in (h.article or "") for h in hits)
            if not ok:
                missing.append((s["id"], exp))
print("MISSING:", missing or "none — all labels verified")
PY
```

Expected: `MISSING: none — all labels verified`. If any are missing, fix the label (find the correct doc_id/article from a verified set) before continuing.

- [ ] **Step 3: Write the failing test `tests/test_multiturn_eval.py`**

```python
from unittest.mock import MagicMock

from scripts.eval import run_multiturn


class _Hit:
    def __init__(self, doc_id, article):
        self.document_id = doc_id
        self.article = article
        self.chunk_text = ""
        self.document_title = ""
        self.score = 1.0


def test_run_multiturn_scores_condensed_followups(monkeypatch):
    import scripts.eval as ev

    # Retrieval returns the expected article for the condensed follow-up.
    monkeypatch.setattr(ev, "retrieve_routed",
                        lambda q, vs, emb, cl, fill_top_k=20: ([_Hit(1, "Điều 171")], "hybrid", None))
    # Condensation resolves the pronoun follow-up.
    llm = MagicMock(); llm.generate.return_value = "Tội cướp giật tài sản bị xử phạt thế nào?"

    scenarios = [{
        "id": "mt-01", "legal_sector": "hinh_su",
        "turns": [
            {"question": "Tội cướp tài sản?", "expected_articles": [{"doc_id": 1, "article": "Điều 168"}]},
            {"question": "vậy còn cướp giật?", "expected_articles": [{"doc_id": 1, "article": "Điều 171"}]},
        ],
    }]

    result = run_multiturn(scenarios, MagicMock(), MagicMock(), llm)

    assert result["n_followups"] == 1
    assert result["followup_pass_rate"] == 1.0
    assert result["rows"][0]["turn_index"] == 1
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_multiturn_eval.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_multiturn'`

- [ ] **Step 5: Add `run_multiturn` to `scripts/eval.py`** (place after `run_multihop`, ~line 275)

```python
def run_multiturn(scenarios, vector_store, embedder, llm) -> Dict:
    """M4.4: multi-turn condensation eval. For each scenario, thread history
    turn-by-turn, condense each follow-up (turn index >= 1) with prior turns,
    and score recall@5 of the CONDENSED query vs expected_articles. History
    uses prior questions only (answer="") — deterministic, zero generation.
    A follow-up passes at recall@5 >= 0.5; gate = >=80% of follow-ups pass."""
    from src.eval.metrics import recall_at_k
    from src.rag.condense import condense_query

    citation_lookup = CitationLookup(vector_store)
    rows = []
    for s in scenarios:
        history = []
        for idx, turn in enumerate(s["turns"]):
            question = turn["question"]
            if idx == 0:
                condensed = question
            else:
                condensed = condense_query(history, question, llm)
                hits, _route, _parsed = retrieve_routed(
                    condensed, vector_store, embedder, citation_lookup, fill_top_k=FETCH_K)
                hit_dicts = [_result_to_dict(r) for r in hits]
                recall = recall_at_k(hit_dicts, turn["expected_articles"], 5, lenient=True)
                rows.append({
                    "scenario_id": s["id"],
                    "legal_sector": s.get("legal_sector"),
                    "turn_index": idx,
                    "question": question,
                    "condensed": condensed,
                    "expected_articles": turn["expected_articles"],
                    "recall@5": round(recall, 4),
                    "passed": recall >= 0.5,
                })
            history.append({"question": question, "answer": ""})

    n = len(rows) or 1
    passed = sum(1 for r in rows if r["passed"])
    return {
        "n_scenarios": len(scenarios),
        "n_followups": len(rows),
        "followup_pass_rate": round(passed / n, 4),
        "mean_recall": round(sum(r["recall@5"] for r in rows) / n, 4),
        "rows": rows,
    }
```

- [ ] **Step 6: Add the `--multiturn` CLI flag** (mirror the `--multihop` option ~line 600 and its branch ~line 609)

Add the option decorator alongside `--multihop`:

```python
@click.option("--multiturn", is_flag=True, default=False,
              help="M4.4: multi-turn condensation eval (recall@5 on condensed follow-ups)")
```

Add `multiturn: bool` to the command function signature (next to `multihop: bool`), and add this branch right after the existing `if multihop:` block:

```python
    if multiturn:
        import json as _json
        scenarios = _json.load(open("data/eval/eval_set_multiturn_v1.json"))
        embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding")
        llm = LLMGateway()
        config = {"tier": "multiturn", "collection": vector_store.collection_name}
        mt = run_multiturn(scenarios, vector_store, embedder, llm)
        report = {"config": config, "multiturn": mt}
        _write_report(report)   # use whatever the multihop branch uses to persist/print
        click.echo(_json.dumps({k: v for k, v in mt.items() if k != "rows"}, ensure_ascii=False, indent=2))
        return
```

(Match the exact report-writing / pipeline-construction the `--multihop` branch uses — read lines ~604–635 and follow that pattern for `build_embedder`, `LLMGateway`, and report output. The key deliverable: `run_multiturn` runs and prints `followup_pass_rate`.)

- [ ] **Step 7: Run the unit test to verify it passes**

Run: `pytest tests/test_multiturn_eval.py -v`
Expected: PASS (1 test)

- [ ] **Step 8: Commit**

```bash
git add data/eval/eval_set_multiturn_v1.json scripts/eval.py tests/test_multiturn_eval.py
git commit -m "feat(m4): multi-turn condensation eval set + --multiturn runner"
```

---

### Task 8: Gate run — full suite, eval gate, cache bench, docs

**Files:**
- Create: `docs/notes/m4-cache-bench.md`
- Modify: `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md` (M4 status rows)
- Modify: `.superpowers/sdd/progress.md` (append M4 log) — if the file exists

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: all pass (prior ~421 + the ~22 new). If any fail, fix before proceeding — do not claim done on red.

- [ ] **Step 2: Run the multi-turn eval gate** (needs the live default collection + LLM quota)

Run: `python scripts/eval.py --multiturn`
Expected: prints `followup_pass_rate`. **Gate: ≥ 0.80.** Record the number. If below 0.80, inspect `rows` — is it a condensation miss (bad rewrite) or a label/retrieval miss? Fix condensation prompt or correct the label, re-run.

- [ ] **Step 3: Measure cache-hit latency and write `docs/notes/m4-cache-bench.md`**

Run this micro-bench (embeds once, times a hit vs the embed itself):

```bash
python - <<'PY'
import time
from src.data.mixed_embedder import build_embedder
from src.rag.semantic_cache import MemorySemanticCache
from src.rag.orchestrator import RAGResponse

emb = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding")
embed_fn = lambda t: emb.encode([t], return_dense=True, return_sparse=False)["dense_vecs"][0].tolist()
cache = MemorySemanticCache(embed_fn)
q = "Tội cướp tài sản bị xử phạt như thế nào?"
cache.set(q, "col", RAGResponse(answer="x"*400, citations=[], confidence="high",
          disclaimer="", latency_ms=1.0, pipeline_used="simple"))
# warm
cache.get(q, "col")
ts = []
for _ in range(20):
    t0 = time.time(); cache.get(q, "col"); ts.append((time.time()-t0)*1000)
ts.sort()
print(f"cache-hit latency ms  p50={ts[len(ts)//2]:.1f}  p95={ts[int(len(ts)*0.95)]:.1f}")
PY
```

Write the note with the measured p50/p95 and this framing:

```markdown
# M4 Cache Bench (2026-07-21)

Semantic-cache hit latency (MemorySemanticCache, 1024-d normalized vectors).
Hit path = embed query + cosine scan over cached vectors; skips
retrieval+rerank+generate+verify (seconds on a miss).

| metric | ms |
|---|---|
| p50 hit | <fill> |
| p95 hit | <fill> |

Gate M4 "cache hit < 500ms": <PASS/FAIL>. Dominant cost is the single query
embed; the cosine scan over <=500 vectors is sub-millisecond.
```

- [ ] **Step 4: Update the plan status** (`docs/Plan_Giai_Doan_Cuoi_2026-07-10.md`)

In the M4 table mark 4.1/4.2/4.4 done with the eval number; in the "Trạng thái milestone" table update the M4 row to reflect memory+condensation+cache landed with `followup_pass_rate=<n>` and cache-hit p95=<n>ms. Keep the existing 4.3 entry.

- [ ] **Step 5: Commit**

```bash
git add docs/notes/m4-cache-bench.md docs/Plan_Giai_Doan_Cuoi_2026-07-10.md .superpowers/sdd/progress.md
git commit -m "docs(m4): cache bench + eval gate results, plan status update"
```

---

## Self-Review

**Spec coverage:**
- 4.1 memory → Task 1 (store) + Task 5 (wiring) + Task 6 (UI session_id). ✓
- 4.1 condensation → Task 2 + wired in Task 4/5. ✓
- 4.2 semantic cache → Task 3 + wired Task 4/5; namespace = collection (invalidation). ✓
- 4.4 multi-turn eval → Task 7 (set + runner) + Task 8 (gate run). ✓
- Fail-open everywhere → try/except in store (Task 1), condense (Task 2), cache (Task 3); API degrade paths (Task 5). ✓
- Pipeline untouched → confirmed; only `main.py`, `chat.py`, `eval.py` and new `src/rag/*` files. ✓
- Gates: follow-up ≥80% (Task 8 Step 2), cache <500ms (Task 8 Step 3), `make test` green (Task 8 Step 1). ✓
- `redis` dep added (Task 1 Step 1). ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases". Two intentional adapt-to-local spots are explicit, not vague: Task 6 Step 3 (UI variable names at the render site) and Task 7 Step 6 (match `--multihop`'s report-writing) — both name the exact reference lines to copy. The bench note has `<fill>` placeholders that are *outputs to record*, not unspecified logic.

**Type consistency:** `RAGResponse(answer, citations, confidence, disclaimer, latency_ms, pipeline_used, related_questions=, confidence_score=, refused=)` used identically across Tasks 3/4/5. `run_chat_turn(...) -> ChatTurnResult(response, cache_hit, condensed_question)` matches its consumer in Task 5. `condense_query(history, question, llm)` signature identical in Tasks 2/4/7. `get_history`/`append_turn`/`get`/`set` names match across store/cache and their callers. Cache namespace is `DEFAULT_COLLECTION_NAME` in the app (Task 5) and `vector_store.collection_name` in eval (Task 7) — same value for the default collection, consistent.
