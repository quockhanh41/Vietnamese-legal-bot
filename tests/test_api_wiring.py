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
