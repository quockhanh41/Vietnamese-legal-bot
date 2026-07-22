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
