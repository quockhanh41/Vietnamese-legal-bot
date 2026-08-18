"""Wiring tests for the production embedder/reranker/collection stack
(M3 Phase 0 — AITeamVN adoption). No model loading: build_embedder and the
pipeline class are monkeypatched, mirroring tests/test_ingest_embedder_arg.py.
"""
from unittest.mock import MagicMock

import src.api.main as main


def test_get_embedder_uses_aiteamvn_mixed(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        main,
        "build_embedder",
        lambda etype, emodel=None, **k: seen.update(etype=etype, emodel=emodel) or "EMB",
    )
    main.app.state.embedder = None
    out = main.get_embedder()
    assert out == "EMB"
    assert seen["etype"] == "mixed"
    assert seen["emodel"] == "AITeamVN/Vietnamese_Embedding"


def test_adaptive_pipeline_passes_aiteamvn_reranker(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "get_embedder", lambda: MagicMock())
    monkeypatch.setattr(
        main, "AdaptiveRAGPipeline", lambda **k: seen.update(k) or MagicMock()
    )
    main.app.state.vector_store = MagicMock()
    main.app.state.llm = MagicMock()
    main.app.state.metadata_store = MagicMock()
    main.app.state.citation_lookup = MagicMock()
    main.app.state.reranker = MagicMock()
    main.get_adaptive_pipeline()
    assert seen["reranker_model"] == "AITeamVN/Vietnamese_Reranker"


def test_default_collection_is_aiteamvn():
    from src.data.constants import DEFAULT_COLLECTION_NAME

    assert DEFAULT_COLLECTION_NAME == "luat_law_ctx_v4_aiteamvn"


def test_chat_request_defaults_to_adaptive():
    """The naive pipeline has no scope gate, no confidence gate, no verifier and
    no reranker — none of the M0–M3 safety work. Defaulting to it meant every
    published metric described a pipeline users were not running
    (audit 2026-08-18, #1).
    """
    import src.api.main as main

    assert main.ChatRequest(question="q").use_adaptive is True


def test_chat_defaults_route_to_adaptive_pipeline(monkeypatch):
    """The Pydantic default and the dispatch must agree: a request that omits
    use_adaptive has to reach the adaptive pipeline, not just carry True.
    This verifies the default path through src/api/main.py:292 dispatch."""
    from src.rag.orchestrator import RAGResponse

    def _resp(answer="A"):
        return RAGResponse(
            answer=answer, citations=[], confidence="high", disclaimer="",
            latency_ms=1.0, pipeline_used="adaptive",
        )

    # Mock both pipelines to track which one is called.
    adaptive_called = {"called": False}
    naive_called = {"called": False}

    def mock_adaptive():
        adaptive_called["called"] = True
        pipe = MagicMock()
        pipe.query.return_value = _resp("from_adaptive")
        return pipe

    def mock_naive():
        naive_called["called"] = True
        pipe = MagicMock()
        pipe.query.return_value = _resp("from_naive")
        return pipe

    monkeypatch.setattr(main, "get_adaptive_pipeline", mock_adaptive)
    monkeypatch.setattr(main, "get_naive_pipeline", mock_naive)

    # Set up the state as in other tests.
    main.app.state.session_store = MagicMock()
    main.app.state.session_store.get_history.return_value = []
    main.app.state.semantic_cache = MagicMock()
    main.app.state.semantic_cache.get.return_value = None
    main.app.state.llm = MagicMock()

    # Call with use_adaptive OMITTED — should default to True and route to adaptive.
    out = main.chat(main.ChatRequest(question="q", session_id="s1"))

    assert adaptive_called["called"] is True, "adaptive pipeline was not called"
    assert naive_called["called"] is False, "naive pipeline was incorrectly called"
    assert out.answer == "from_adaptive"
