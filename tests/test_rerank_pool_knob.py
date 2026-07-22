import importlib


def test_rerank_max_k_default_is_15():
    import src.rag.orchestrator as orch
    importlib.reload(orch)
    assert orch.RERANK_MAX_K == 15


def test_rerank_max_k_reads_env(monkeypatch):
    monkeypatch.setenv("RERANK_MAX_K", "10")
    import src.rag.orchestrator as orch
    importlib.reload(orch)
    assert orch.RERANK_MAX_K == 10
    monkeypatch.delenv("RERANK_MAX_K")
    importlib.reload(orch)  # restore default for other tests


def test_dynamic_top_k_never_exceeds_max_k():
    from src.rag.orchestrator import AdaptiveRAGPipeline
    from src.db.qdrant_client import SearchResult
    results = [SearchResult(point_id=str(i), score=0.9, chunk_text="c",
                            document_title="t", document_number="n", legal_type="l",
                            document_id=1) for i in range(40)]
    k = AdaptiveRAGPipeline._dynamic_top_k(results, min_k=5, max_k=10)
    assert k <= 10
