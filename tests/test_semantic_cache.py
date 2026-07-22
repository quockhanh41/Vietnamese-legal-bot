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
    # Non-degenerate embed to prove FIFO order (avoid zero-vector)
    c = MemorySemanticCache(lambda t: [float(t) + 1.0, 0.0, 0.0])
    for i in range(3):
        c.set(str(i), "col", _resp(str(i)))
    entries = c._entries("col")
    assert len(entries) == 2                 # oldest evicted
    # Verify FIFO order: "0" should be evicted, "1" and "2" should survive
    assert entries[0][1]["answer"] == "1"   # first survivor
    assert entries[1][1]["answer"] == "2"   # second survivor


def test_get_failopen_on_embed_error():
    def boom(_):
        raise RuntimeError("embed down")
    c = MemorySemanticCache(boom)
    assert c.get("anything", "col") is None


def test_set_failopen_on_embed_error():
    """set() should not raise on embed error, silently no-op."""
    def boom(_):
        raise RuntimeError("embed down")
    c = MemorySemanticCache(boom)
    # Should not raise
    c.set("anything", "col", _resp())
    # Verify cache is empty and get still returns None cleanly
    assert c.get("anything", "col") is None


def test_set_failopen_on_store_error():
    """set() should not raise on _store error, silently no-op."""
    c = MemorySemanticCache(_embed)

    # Monkeypatch _store to raise
    original_store = c._store
    def broken_store(*args, **kwargs):
        raise RuntimeError("store down")
    c._store = broken_store

    # Should not raise
    c.set("cuop tai san", "col", _resp())
    # Restore and verify cache is empty
    c._store = original_store
    assert c.get("cuop tai san", "col") is None


def test_get_failopen_on_malformed_entry():
    """get() should return None if RAGResponse reconstruction fails."""
    c = MemorySemanticCache(_embed)
    # Seed buffer directly with a dict that can't reconstruct RAGResponse
    c._buf["col"] = [([1.0, 0.0, 0.0], {"not_a_valid_field": 1})]
    # Embed a query that matches the cached embedding perfectly
    # With exact match on cached entry, should try to reconstruct but fail gracefully
    result = c.get("cuop tai san", "col")
    assert result is None


def test_from_env_disabled():
    """from_env() returns None when SEMANTIC_CACHE_DISABLED is set."""
    import os
    # Temporarily set the env var
    os.environ["SEMANTIC_CACHE_DISABLED"] = "1"
    try:
        result = sc.SemanticCache.from_env(_embed)
        assert result is None
    finally:
        del os.environ["SEMANTIC_CACHE_DISABLED"]


def test_from_env_with_redis(monkeypatch):
    """from_env() returns RedisSemanticCache when get_redis_client returns a truthy client."""
    monkeypatch.setenv("SEMANTIC_CACHE_DISABLED", "")  # Ensure it's not set
    # Mock get_redis_client to return a fake client
    fake_client = object()  # Any truthy object
    monkeypatch.setattr(sc, "get_redis_client", lambda: fake_client)
    result = sc.SemanticCache.from_env(_embed)
    assert isinstance(result, sc.RedisSemanticCache)


def test_from_env_memory_fallback(monkeypatch):
    """from_env() returns MemorySemanticCache when get_redis_client returns None."""
    monkeypatch.setenv("SEMANTIC_CACHE_DISABLED", "")  # Ensure it's not set
    # Mock get_redis_client to return None
    monkeypatch.setattr(sc, "get_redis_client", lambda: None)
    result = sc.SemanticCache.from_env(_embed)
    assert isinstance(result, sc.MemorySemanticCache)
