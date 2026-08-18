"""
Unit tests for M3.5 cross-reference graph expansion
(src/rag/orchestrator.py :: AdaptiveRAGPipeline._expand_cross_references).
Pure-Python: no Qdrant, no LLM — get_graph() and scroll_article() are stubbed.

Run: .venv/bin/python -m pytest tests/test_cross_ref_expansion.py -q
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.db.qdrant_client import SearchResult
from src.rag.orchestrator import AdaptiveRAGPipeline


def _sr(point_id, document_id, article, chunk_text="text"):
    return SearchResult(
        point_id=point_id, score=0.9, chunk_text=chunk_text,
        document_title="Luật đất đai 2013", document_number="45/2013/QH13",
        legal_type="Luật", article=article, document_id=document_id,
    )


def _graph(nodes, edges):
    return {"nodes": nodes, "edges": edges}


def _pipeline(graph, scroll_map, alias_ids=None, resolve=None):
    """Build an AdaptiveRAGPipeline with stubbed graph + scroll_article.

    scroll_map: {(doc_id, article_number): [SearchResult, ...]} returned by scroll_article.
    resolve: optional {law_name: [doc_id, ...]} for CitationLookup.resolve_doc.
    """
    vs = MagicMock()
    vs.scroll_article.side_effect = lambda doc_ids, num, cap=1: scroll_map.get((doc_ids[0], num), [])
    p = AdaptiveRAGPipeline(vector_store=vs, embedder=MagicMock(), llm_gateway=MagicMock())
    md = MagicMock()
    md.get_graph.return_value = graph
    md.get_document.return_value = None  # is_in_effect unknown -> kept (fail-open)
    md.get_timeline.return_value = None
    md.get_all_documents.return_value = {}
    p._metadata_store = md
    if resolve is not None:
        cl = MagicMock()
        cl.resolve_doc.side_effect = lambda hint, year=None: resolve.get(hint, list(alias_ids or []))
        cl._get_alias_index.return_value = {i: "" for i in (alias_ids or [])}
        p._citation_lookup = cl
    return p


def test_internal_neighbor_fetched_and_appended():
    graph = _graph(
        nodes=[
            {"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"},
            {"id": "doc_1_điều_50", "doc_id": 1, "article": "Điều 50", "node_type": "article"},
        ],
        edges=[{"source": "doc_1_điều_10", "target": "doc_1_điều_50", "ref_type": "internal"}],
    )
    scroll_map = {(1, "50"): [_sr("p50", 1, "Điều 50", "Điều 50 nội dung tham chiếu")]}
    p = _pipeline(graph, scroll_map)
    seeds = [_sr("p10", 1, "Điều 10")]
    out = p._expand_cross_references(seeds)
    assert [r.article for r in out] == ["Điều 50"]


def test_dedup_neighbor_already_in_results():
    graph = _graph(
        nodes=[
            {"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"},
            {"id": "doc_1_điều_50", "doc_id": 1, "article": "Điều 50", "node_type": "article"},
        ],
        edges=[{"source": "doc_1_điều_10", "target": "doc_1_điều_50", "ref_type": "internal"}],
    )
    scroll_map = {(1, "50"): [_sr("p50", 1, "Điều 50")]}
    p = _pipeline(graph, scroll_map)
    seeds = [_sr("p10", 1, "Điều 10"), _sr("p50b", 1, "Điều 50")]  # 50 already present
    out = p._expand_cross_references(seeds)
    assert out == []


def test_cap_enforced():
    nodes = [{"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"}]
    edges = []
    scroll_map = {}
    for n in range(1, 8):  # 7 neighbors
        art = f"Điều {100 + n}"
        nodes.append({"id": f"doc_1_điều_{100 + n}", "doc_id": 1, "article": art, "node_type": "article"})
        edges.append({"source": "doc_1_điều_10", "target": f"doc_1_điều_{100 + n}", "ref_type": "internal"})
        scroll_map[(1, str(100 + n))] = [_sr(f"p{100 + n}", 1, art)]
    p = _pipeline(_graph(nodes, edges), scroll_map)
    out = p._expand_cross_references([_sr("p10", 1, "Điều 10")])
    assert len(out) == 4  # MAX_EXPANDED


def test_superseded_neighbor_dropped_unless_comparison():
    graph = _graph(
        nodes=[
            {"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"},
            {"id": "doc_2_điều_5", "doc_id": 2, "article": "Điều 5", "node_type": "article"},
        ],
        edges=[{"source": "doc_1_điều_10", "target": "doc_2_điều_5", "ref_type": "external"}],
    )
    scroll_map = {(2, "5"): [_sr("p5", 2, "Điều 5")]}
    p = _pipeline(graph, scroll_map)
    # doc 2 is positively superseded
    doc_meta = MagicMock(); doc_meta.effective_date = None; doc_meta.is_in_effect = False
    doc_meta.ref_count = 0; doc_meta.is_amendment = False
    p._metadata_store.get_document.side_effect = lambda d: doc_meta if d == 2 else None
    assert p._expand_cross_references([_sr("p10", 1, "Điều 10")], is_comparison=False) == []
    assert len(p._expand_cross_references([_sr("p10", 1, "Điều 10")], is_comparison=True)) == 1


def test_external_ref_resolved_via_resolve_doc():
    graph = _graph(
        nodes=[
            {"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"},
            {"id": "ext_luật x_điều_7", "law_name": "luật x", "article": "Điều 7", "node_type": "external_ref"},
        ],
        edges=[{"source": "doc_1_điều_10", "target": "ext_luật x_điều_7", "ref_type": "external"}],
    )
    scroll_map = {(9, "7"): [_sr("p7", 9, "Điều 7")]}
    p = _pipeline(graph, scroll_map, alias_ids=[1, 9], resolve={"luật x": [9]})
    out = p._expand_cross_references([_sr("p10", 1, "Điều 10")])
    assert [r.article for r in out] == ["Điều 7"]


def test_external_ref_ambiguous_skipped():
    graph = _graph(
        nodes=[
            {"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"},
            {"id": "ext_aviation_điều_57", "law_name": "hàng không", "article": "Điều 57", "node_type": "external_ref"},
        ],
        edges=[{"source": "doc_1_điều_10", "target": "ext_aviation_điều_57", "ref_type": "external"}],
    )
    p = _pipeline(graph, {}, alias_ids=[1, 9], resolve={})  # unknown law -> resolve returns all ids
    assert p._expand_cross_references([_sr("p10", 1, "Điều 10")]) == []


def test_no_edges_returns_empty():
    p = _pipeline(_graph([], []), {})
    assert p._expand_cross_references([_sr("p10", 1, "Điều 10")]) == []


def test_exception_returns_empty():
    p = AdaptiveRAGPipeline(vector_store=MagicMock(), embedder=MagicMock(), llm_gateway=MagicMock())
    md = MagicMock(); md.get_graph.side_effect = RuntimeError("boom")
    p._metadata_store = md
    assert p._expand_cross_references([_sr("p10", 1, "Điều 10")]) == []


def test_toggle_off_returns_empty():
    graph = _graph(
        nodes=[{"id": "doc_1_điều_10", "doc_id": 1, "article": "Điều 10", "node_type": "article"}],
        edges=[{"source": "doc_1_điều_10", "target": "doc_1_điều_50", "ref_type": "internal"}],
    )
    p = _pipeline(graph, {})
    p.cross_ref_expansion = False
    assert p._expand_cross_references([_sr("p10", 1, "Điều 10")]) == []


def test_complex_pipeline_appends_expansion_to_citations():
    """The complex route appends expanded blocks so they reach citations."""
    p = AdaptiveRAGPipeline(vector_store=MagicMock(), embedder=MagicMock(), llm_gateway=MagicMock())
    # Stub metadata store so _annotate_phase2 stays offline (no real SQLite).
    md = MagicMock()
    md.get_document.return_value = None
    md.get_timeline.return_value = None
    md.get_all_documents.return_value = {}
    p._metadata_store = md
    seeds = [_sr("p10", 1, "Điều 10", "Điều 10 nội dung")]
    extra = [_sr("p50", 1, "Điều 50", "Điều 50 nội dung")]
    # Drive the real complex retrieval (one seed, so _rerank_results short-circuits
    # without loading a cross-encoder) and stub only the expansion under test.
    p.embedder.encode.return_value = {"dense_vecs": [MagicMock()], "lexical_weights": [{}]}
    p.vector_store.search_hybrid.return_value = list(seeds)
    p._expand_cross_references = MagicMock(return_value=extra)
    p.llm.generate = MagicMock(return_value="Trả lời [1][2]")
    p.llm.generate_with_model = MagicMock(return_value=("Trả lời [1][2]", "mock/model"))
    conf = MagicMock(); conf.label = "high"; conf.score = 0.8; conf.refused = False
    ver = MagicMock(); ver.passed = True; ver.warnings = []
    p._post_process = MagicMock(return_value=(conf, ver, "Trả lời [1][2]"))

    resp = p._run_route("câu hỏi", "complex", start_time=0.0, is_comparison=False)
    cited_articles = [c["article"] for c in resp.citations]
    assert "Điều 50" in cited_articles  # expanded block present
    p._expand_cross_references.assert_called_once()
