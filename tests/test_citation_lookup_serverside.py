"""
CitationLookup server-side path — resolves via get_document_index() /
scroll_article() instead of caching every chunk.
Run: .venv/bin/python -m pytest tests/test_citation_lookup_serverside.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag.citation_lookup import CitationLookup, parse_citation
from src.db.qdrant_client import SearchResult


class _FakePoint:
    """Raw Qdrant-style point (id + payload) for the scroll_docs lenient path."""
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class _FakeStore:
    def __init__(self, docs=None):
        self._index = {
            10: {"document_title": "Bộ luật Hình sự", "document_number": "100/2015/QH13",
                 "legal_type": "Luật", "sector": "hinh_su"},
            20: {"document_title": "Bộ luật Lao động", "document_number": "45/2019/QH14",
                 "legal_type": "Luật", "sector": "lao_dong"},
        }
        self._by = {
            (10, "173"): [SearchResult("p1", 1.0, "Điều 173. Tội trộm cắp...",
                          "Bộ luật Hình sự", "100/2015/QH13", "Luật", article="Điều 173", document_id=10)],
            (10, "40"): [SearchResult("p2", 1.0, "Điều 40. Tử hình...",
                          "Bộ luật Hình sự", "100/2015/QH13", "Luật", article="Điều 40", document_id=10)],
            (20, "40"): [SearchResult("p3", 1.0, "Điều 40. Nghĩa vụ...",
                          "Bộ luật Lao động", "45/2019/QH14", "Luật", article="Điều 40", document_id=20)],
        }
        # {tuple(sorted(doc_ids)): [_FakePoint, ...]} returned by scroll_docs
        self._docs = docs or {}
        # records the doc_ids of every scroll_docs call (the fix skips it when
        # unscoped, so an ambiguous lookup must leave this empty).
        self.scroll_docs_calls = []

    def get_document_index(self):
        return self._index

    def scroll_article(self, doc_ids, article_number, cap=8):
        ids = doc_ids or list(self._index)
        out = []
        for i in ids:
            out.extend(self._by.get((i, article_number), []))
        return out[:cap]

    def scroll_docs(self, doc_ids, limit_per=512):
        self.scroll_docs_calls.append(list(doc_ids))
        return self._docs.get(tuple(sorted(doc_ids)), [])

    def _point_to_result(self, point):
        p = point.payload or {}
        return SearchResult(
            point_id=str(point.id), score=1.0, chunk_text=p.get("chunk_text", ""),
            document_title=p.get("document_title", ""), document_number=p.get("document_number", ""),
            legal_type=p.get("legal_type", ""), article=p.get("article"),
            chapter=p.get("chapter"), url=p.get("url"),
            document_id=p.get("document_id"), parent_chunk_text=p.get("parent_chunk_text"))


def test_lookup_scopes_to_resolved_doc():
    cl = CitationLookup(_FakeStore())
    parsed = parse_citation("Điều 40 Bộ luật Hình sự")
    hits = cl.lookup(parsed)
    assert [h.point_id for h in hits] == ["p2"]  # BLHS art 40, not lao động


def test_lookup_ambiguous_returns_both():
    cl = CitationLookup(_FakeStore())
    parsed = parse_citation("Điều 40 quy định gì?")  # no doc hint
    hits = cl.lookup(parsed)
    assert {h.point_id for h in hits} == {"p2", "p3"}


def test_no_chunk_cache_attribute():
    cl = CitationLookup(_FakeStore())
    cl.lookup(parse_citation("Điều 173 Bộ luật Hình sự"))
    assert getattr(cl, "_chunks_by_doc", None) is None  # cache removed


def test_corpus_summary_uses_doc_index():
    cl = CitationLookup(_FakeStore())
    summary = cl.corpus_summary()
    assert "Bộ luật Hình sự" in summary and "Bộ luật Lao động" in summary


def test_ambiguous_lookup_skips_scroll_docs():
    """Guard for the scale fix: an ambiguous citation (no doc hint) resolves
    to ALL docs → scoped=[] → the lenient fallback (an unfiltered, uncapped
    scroll_docs) must be SKIPPED, or every under-cap ambiguous lookup walks
    the whole collection. Strict (indexed) hits still come back."""
    store = _FakeStore()
    cl = CitationLookup(store)
    hits = cl.lookup(parse_citation("Điều 40 quy định gì?"))  # strict: {p2, p3} < cap
    assert {h.point_id for h in hits} == {"p2", "p3"}
    assert store.scroll_docs_calls == []  # never walked the collection


def test_scoped_lenient_fallback_recovers_stale_article():
    """The lenient fallback still works when SCOPED: a chunk with a stale
    (absent) article_number that strict scroll_article misses, but whose text
    opens with "Điều N.", is recovered via scroll_docs on the scoped doc."""
    stale = _FakePoint("lp1", {
        "chunk_text": "Điều 7. Nghĩa vụ chung của người lao động...",
        "document_id": 20,
        "document_title": "Bộ luật Lao động",
        "document_number": "45/2019/QH14",
        "legal_type": "Luật",
        # no article_number / article payload → strict scroll_article misses it
    })
    store = _FakeStore(docs={(20,): [stale]})
    cl = CitationLookup(store)
    hits = cl.lookup(parse_citation("Điều 7 Bộ luật Lao động"))  # scoped -> [20]
    assert "lp1" in {h.point_id for h in hits}
    assert store.scroll_docs_calls == [[20]]  # scoped call, not the whole corpus


class _RankFakeStore:
    """resolve_doc ranks the law itself before the guiding decree ([20, 30]),
    but scroll_article's single MatchAny filter returns them in Qdrant order
    (here reversed) — lookup() must re-sort strict hits into candidate order."""
    def __init__(self):
        self._index = {
            # third, unrelated doc so resolving to {20, 30} is a strict subset
            # of the corpus (not "all docs" → not treated as ambiguous).
            10: {"document_title": "Bộ luật Hình sự", "document_number": "100/2015/QH13",
                 "legal_type": "Luật", "sector": "hinh_su"},
            20: {"document_title": "Bộ luật Lao động", "document_number": "45/2019/QH14",
                 "legal_type": "Luật", "sector": "lao_dong"},
            30: {"document_title": "Nghị định 145/2020/NĐ-CP hướng dẫn Bộ luật Lao động",
                 "document_number": "145/2020/NĐ-CP", "legal_type": "Nghị định", "sector": "lao_dong"},
        }

    def get_document_index(self):
        return self._index

    def scroll_article(self, doc_ids, article_number, cap=8):
        law = SearchResult("law7", 1.0, "Điều 7...", "Bộ luật Lao động",
                           "45/2019/QH14", "Luật", article="Điều 7", document_id=20)
        decree = SearchResult("decree7", 1.0, "Điều 7...",
                              "Nghị định 145/2020/NĐ-CP hướng dẫn Bộ luật Lao động",
                              "145/2020/NĐ-CP", "Nghị định", article="Điều 7", document_id=30)
        return [decree, law]  # reversed vs candidate order [20, 30]

    def scroll_docs(self, doc_ids, limit_per=512):
        return []


def test_scoped_multidoc_preserves_rank_order():
    cl = CitationLookup(_RankFakeStore())
    parsed = parse_citation("Điều 7 Bộ luật Lao động")
    doc_ids = cl.resolve_doc(parsed.doc_hint, parsed.year, parsed.doc_number)
    assert doc_ids == [20, 30]  # law ranked before referencing decree
    hits = cl.lookup(parsed)
    assert [h.point_id for h in hits] == ["law7", "decree7"]  # candidate order restored
