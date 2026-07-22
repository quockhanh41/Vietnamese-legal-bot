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
