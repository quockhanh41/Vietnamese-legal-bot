import json
import re
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

    out = main.chat(main.ChatRequest(question="q", session_id="s1", use_adaptive=False))

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

    out = main.chat(main.ChatRequest(question="q", session_id="s1", use_adaptive=False))

    assert out.cache_hit is True
    assert out.answer == "cached"
    fake_pipe.query.assert_not_called()


def test_replay_events_shape():
    r = _resp("ans")
    r.citations = [{"n": 1}]
    events = main._replay_events(r, start_time=0.0)
    kinds = [e["event"] for e in events]
    assert kinds == ["citations", "token", "verification", "done"]


def test_replay_events_forwards_refused():
    r = RAGResponse(
        answer="Xin lỗi, tôi không thể trả lời.", citations=[], confidence="low",
        disclaimer="", latency_ms=1.0, pipeline_used="simple", refused=True,
    )
    events = main._replay_events(r, start_time=0.0)
    by_event = {e["event"]: json.loads(e["data"]) for e in events}
    assert by_event["verification"]["refused"] is True
    assert by_event["done"]["refused"] is True


def test_stream_cache_hit_replays_and_skips_pipeline(monkeypatch):
    """Stream cache HIT: replay cached response as SSE, never invoke the
    pipeline, persist the ORIGINAL question to memory."""
    from fastapi.testclient import TestClient

    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = _resp("cached-stream")
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    fake = MagicMock()
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: fake)

    client = TestClient(main.app)
    resp = client.post("/api/chat/stream",
                       json={"question": "q", "use_adaptive": False, "session_id": "s1"})
    assert resp.status_code == 200
    fake.query_stream.assert_not_called()
    m = re.search(r"event: done\r?\ndata: (.*)", resp.text)
    assert m, f"no done event in body: {resp.text!r}"
    done = json.loads(m.group(1))
    assert done["cache_hit"] is True
    assert done["full_answer"] == "cached-stream"
    store.append_turn.assert_called_once_with("s1", "q", "cached-stream")


def test_stream_refused_miss_is_not_cached(monkeypatch):
    """Refusal integrity on the stream path: a refused streamed answer
    (verification marker refused=True) must never be written to the cache."""
    from fastapi.testclient import TestClient

    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None   # miss
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    verification = json.dumps({"confidence": "low", "confidence_score": 0.1, "refused": True})
    fake = MagicMock()
    fake.query_stream = lambda q: iter(["ans ", "text", "__VERIFICATION__" + verification])
    monkeypatch.setattr(main, "get_adaptive_pipeline", lambda: fake)

    client = TestClient(main.app)
    resp = client.post("/api/chat/stream",
                       json={"question": "q", "use_adaptive": True, "session_id": "s1"})
    assert resp.status_code == 200
    cache.set.assert_not_called()
