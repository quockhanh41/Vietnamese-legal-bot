from unittest.mock import MagicMock
import src.api.main as main


async def test_ready_false_before_models_loaded():
    main.app.state.embedder = None
    main.app.state.reranker = None
    out = await main.ready()
    assert out["ready"] is False
    assert out["models_loaded"] == {"embedder": False, "reranker": False}


async def test_ready_true_once_loaded():
    main.app.state.embedder = MagicMock()
    main.app.state.reranker = MagicMock()
    out = await main.ready()
    assert out["ready"] is True


import pytest
from fastapi import HTTPException


def test_chat_degraded_on_exhaustion(monkeypatch):
    main.app.state.session_store = MagicMock()
    main.app.state.semantic_cache = MagicMock()
    main.app.state.llm = MagicMock()
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: MagicMock())

    def boom(**k):
        raise RuntimeError("All models failed. Last error: 429 rate limit")
    monkeypatch.setattr(main, "run_chat_turn", boom)

    out = main.chat(main.ChatRequest(question="q", session_id="s1", use_adaptive=False))
    assert out.degraded is True
    assert out.refused is False
    assert "quá tải" in out.answer


def test_chat_non_exhaustion_still_500(monkeypatch):
    main.app.state.session_store = MagicMock()
    main.app.state.semantic_cache = MagicMock()
    main.app.state.llm = MagicMock()
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: MagicMock())

    def boom(**k):
        raise ValueError("something else")
    monkeypatch.setattr(main, "run_chat_turn", boom)

    with pytest.raises(HTTPException) as e:
        main.chat(main.ChatRequest(question="q", session_id="s1", use_adaptive=False))
    assert e.value.status_code == 500


from fastapi.testclient import TestClient


def test_stream_degraded_on_exhaustion(monkeypatch):
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None
    main.app.state.session_store = store
    main.app.state.semantic_cache = cache
    main.app.state.llm = MagicMock()

    def gen(_condensed):
        raise RuntimeError("All models failed. Last error: 429")
        yield  # unreachable; makes this a generator
    fake = MagicMock(); fake.query_stream = gen
    monkeypatch.setattr(main, "get_naive_pipeline", lambda: fake)
    monkeypatch.setattr(main, "condense_query", lambda h, q, llm: q)

    client = TestClient(main.app)
    r = client.post("/api/chat/stream", json={"question": "q", "session_id": "s1", "use_adaptive": False})
    assert r.status_code == 200
    assert "degraded" in r.text
    assert "quá tải" in r.text
    cache.set.assert_not_called()


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
    r = client.post("/api/chat/stream", json={"question": "q", "session_id": "s1", "use_adaptive": False})

    assert r.status_code == 200
    assert "gián đoạn" in r.text
    cache.set.assert_not_called()
    store.append_turn.assert_not_called()


def test_is_exhaustion_false_for_stream_interrupted():
    """StreamInterrupted subclasses RuntimeError but must NOT be mistaken for
    exhaustion — the two failure modes are handled differently downstream
    (retry-safe degradation vs. never-retry, never-cache truncation)."""
    from src.rag.degraded import StreamInterrupted, is_exhaustion

    assert is_exhaustion(StreamInterrupted("x")) is False
