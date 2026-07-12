"""
Unit tests for src/db/qdrant_client.py — deterministic point IDs
(compute_point_id) and article_number/payload-index wiring in
LegalVectorStore. No network, no real Qdrant: LegalVectorStore.client is
always swapped for an in-memory fake before any method that would otherwise
talk to a server is called.

Run: .venv/bin/python -m pytest tests/test_qdrant_client.py -q
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.db.qdrant_client import LegalVectorStore, compute_point_id


# ── Fakes (no network) ──────────────────────────────────────────────────────

class _FakeCollectionsResponse:
    def __init__(self, names):
        self.collections = [_FakeCollection(n) for n in names]


class _FakeCollection:
    def __init__(self, name):
        self.name = name


class _FakeQdrantClient:
    """Records calls instead of talking to a real Qdrant instance."""

    def __init__(self, existing_collections=None):
        self._existing_collections = list(existing_collections or [])
        self.upserted_points = []
        self.created_collections = []
        self.deleted_collections = []
        self.created_indexes = []  # list of (field_name, field_schema)

    def get_collections(self):
        return _FakeCollectionsResponse(self._existing_collections)

    def delete_collection(self, name):
        self.deleted_collections.append(name)
        if name in self._existing_collections:
            self._existing_collections.remove(name)

    def create_collection(self, collection_name, vectors_config=None,
                          sparse_vectors_config=None, quantization_config=None):
        self.created_collections.append(collection_name)
        self._existing_collections.append(collection_name)
        self.last_vectors_config = vectors_config
        self.last_quantization_config = quantization_config

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.created_indexes.append((field_name, field_schema))

    def upsert(self, collection_name, points, wait=True):
        self.upserted_points.extend(points)


def _make_store(collection_name="test_collection", existing_collections=None):
    """LegalVectorStore backed by a fake client, with zero network.

    Deliberately bypasses LegalVectorStore.__init__: it constructs a real
    QdrantClient(...), whose constructor spawns a background thread that
    probes the server for a version-compatibility check (check_compatibility
    defaults to True) — a genuine network attempt even against localhost.
    __new__ + manual attribute assignment sidesteps that entirely so these
    tests never touch a socket.
    """
    store = LegalVectorStore.__new__(LegalVectorStore)
    store.url = "http://localhost:6333"
    store.api_key = None
    store.collection_name = collection_name
    store.client = _FakeQdrantClient(existing_collections=existing_collections)
    return store


# ── compute_point_id ─────────────────────────────────────────────────────────

def test_same_document_and_chunk_index_yields_same_id():
    meta = {"document_id": 42, "chunk_index": 3}
    id1 = compute_point_id(meta, "chunk text v1")
    id2 = compute_point_id(meta, "chunk text v1")
    assert id1 == id2


def test_same_document_different_chunk_index_yields_different_id():
    id_a = compute_point_id({"document_id": 42, "chunk_index": 0}, "text")
    id_b = compute_point_id({"document_id": 42, "chunk_index": 1}, "text")
    assert id_a != id_b


def test_different_document_same_chunk_index_yields_different_id():
    id_a = compute_point_id({"document_id": 1, "chunk_index": 0}, "text")
    id_b = compute_point_id({"document_id": 2, "chunk_index": 0}, "text")
    assert id_a != id_b


def test_id_is_stable_regardless_of_batch_position():
    """The whole point of keying off metadata instead of an enumerate()
    index: two calls with the SAME metadata but different chunk_text
    (simulating "same chunk, different batch/run") must still agree,
    while an index derived from list position would not."""
    meta = {"document_id": 7, "chunk_index": 5}
    id1 = compute_point_id(meta, "chunk text")
    id2 = compute_point_id(dict(meta), "chunk text")  # fresh dict, same content
    assert id1 == id2


def test_fallback_to_content_hash_when_chunk_index_missing():
    meta_no_index = {"document_id": 42}
    id1 = compute_point_id(meta_no_index, "some chunk body")
    id2 = compute_point_id(meta_no_index, "some chunk body")
    assert id1 == id2  # same content -> same fallback id


def test_fallback_differs_for_different_content():
    meta_no_index = {"document_id": 42}
    id_a = compute_point_id(meta_no_index, "chunk body A")
    id_b = compute_point_id(meta_no_index, "chunk body B")
    assert id_a != id_b


def test_fallback_id_differs_from_chunk_index_id_for_same_document():
    """Sanity: the two derivation paths (chunk_index vs content-hash) must
    not collide for otherwise-identical metadata."""
    id_with_index = compute_point_id({"document_id": 42, "chunk_index": 0}, "text")
    id_fallback = compute_point_id({"document_id": 42}, "text")
    assert id_with_index != id_fallback


def test_point_id_is_a_valid_uuid_string():
    pid = compute_point_id({"document_id": 1, "chunk_index": 0}, "text")
    assert uuid.UUID(pid)  # does not raise


def test_chunk_index_zero_is_not_treated_as_missing():
    """Regression guard: chunk_index=0 is falsy in Python but IS a valid
    index (the first chunk of a document) — must take the chunk_index path,
    not silently fall back to content hashing."""
    id_index_path = compute_point_id({"document_id": 9, "chunk_index": 0}, "text")
    id_fallback_path = compute_point_id({"document_id": 9}, "text")
    assert id_index_path != id_fallback_path


# ── upsert_chunks: point IDs + article_number wiring ────────────────────────

def test_upsert_chunks_uses_compute_point_id():
    store = _make_store()
    metadata_list = [{"document_id": 1, "chunk_index": 0, "article": "Điều 5"}]
    store.upsert_chunks(
        chunk_texts=["noi dung"],
        dense_vectors=[[0.1, 0.2]],
        sparse_vectors=[{0: 0.5}],
        metadata_list=metadata_list,
    )
    [point] = store.client.upserted_points
    assert point.id == compute_point_id(metadata_list[0], "noi dung")


def test_upsert_chunks_rerun_is_idempotent_same_ids():
    """Running upsert_chunks twice with the same input must produce the
    same point IDs both times (the actual idempotency property we care
    about for resumable/re-runnable ingest)."""
    store = _make_store()
    metadata_list = [{"document_id": 1, "chunk_index": 0, "article": "Điều 5"}]
    store.upsert_chunks(["noi dung"], [[0.1]], [{0: 0.5}], metadata_list)
    store.upsert_chunks(["noi dung"], [[0.1]], [{0: 0.5}], metadata_list)
    ids = [p.id for p in store.client.upserted_points]
    assert ids[0] == ids[1]


@pytest.mark.parametrize(
    "article,expected",
    [
        ("Điều 173. Tội trộm cắp tài sản", "173"),
        ("Điều 123a", "123a"),
        ("Điều 51b. Some clause", "51b"),
        (None, None),
        ("", None),
    ],
)
def test_upsert_chunks_sets_article_number(article, expected):
    store = _make_store()
    metadata_list = [{"document_id": 1, "chunk_index": 0, "article": article}]
    store.upsert_chunks(
        chunk_texts=["noi dung"],
        dense_vectors=[[0.1]],
        sparse_vectors=[{0: 0.5}],
        metadata_list=metadata_list,
    )
    [point] = store.client.upserted_points
    assert point.payload["article_number"] == expected
    # Must be a string (or None), never an int — letter suffixes like
    # "123a" would be silently destroyed by int().
    assert isinstance(point.payload["article_number"], str) or point.payload["article_number"] is None


def test_upsert_chunks_preserves_original_article_field():
    """article_number is additive — the original `article` payload field
    (used elsewhere, e.g. src/eval/metrics.py matching) must stay intact."""
    store = _make_store()
    metadata_list = [{"document_id": 1, "chunk_index": 0, "article": "Điều 173. Tội trộm cắp tài sản"}]
    store.upsert_chunks(["noi dung"], [[0.1]], [{0: 0.5}], metadata_list)
    [point] = store.client.upserted_points
    assert point.payload["article"] == "Điều 173. Tội trộm cắp tài sản"
    assert point.payload["article_number"] == "173"


# ── create_collection: payload indexes ──────────────────────────────────────

def test_create_collection_creates_expected_payload_indexes():
    store = _make_store(existing_collections=[])
    store.create_collection()

    assert store.client.created_collections == ["test_collection"]
    index_fields = {name for name, _schema in store.client.created_indexes}
    assert index_fields == {
        "document_id",
        "article_number",
        "legal_type",
        "issuance_date",
        "is_in_effect",
        "sector",  # M1.2: curated-corpus sector label (see qdrant_client.py)
    }


def test_create_collection_is_idempotent_when_already_exists():
    """Calling create_collection() again for an existing collection must
    NOT re-create the collection or re-create payload indexes."""
    store = _make_store(existing_collections=["test_collection"])
    store.create_collection()

    assert store.client.created_collections == []
    assert store.client.created_indexes == []


def test_create_collection_recreate_drops_and_recreates_indexes():
    store = _make_store(existing_collections=["test_collection"])
    store.create_collection(recreate=True)

    assert store.client.deleted_collections == ["test_collection"]
    assert store.client.created_collections == ["test_collection"]
    assert len(store.client.created_indexes) == 6


def test_create_collection_default_is_not_quantized():
    store = _make_store(existing_collections=[])
    store.create_collection()

    assert store.client.last_quantization_config is None
    dense = store.client.last_vectors_config[LegalVectorStore.DENSE_VECTOR_NAME]
    assert dense.on_disk is None


def test_create_collection_quantized_enables_int8_and_on_disk():
    """quantized=True must set int8 scalar quantization (always_ram) and put
    original fp32 dense vectors on disk — the free-tier RAM budget of Plan
    Giai Đoạn Cuối §2.2 depends on both."""
    from qdrant_client.http import models as qmodels

    store = _make_store(existing_collections=[])
    store.create_collection(quantized=True)

    qc = store.client.last_quantization_config
    assert isinstance(qc, qmodels.ScalarQuantization)
    assert qc.scalar.type == qmodels.ScalarType.INT8
    assert qc.scalar.always_ram is True
    dense = store.client.last_vectors_config[LegalVectorStore.DENSE_VECTOR_NAME]
    assert dense.on_disk is True


# ── get_document_index / scroll_article / scroll_docs: server-side helpers ──
# (M2.3 — citation lookup no longer needs a client-side full-collection
# chunk cache; these scroll server-side using the document_id/article_number
# payload indexes created in _create_payload_indexes().)

class _FakePoint:
    def __init__(self, pid, payload): self.id, self.payload = pid, payload


def _seed_points():
    return [
        _FakePoint("p1", {"document_id": 10, "document_title": "Bộ luật Hình sự",
                          "document_number": "100/2015/QH13", "legal_type": "Luật",
                          "sector": "hinh_su", "article": "Điều 173", "article_number": "173",
                          "chunk_text": "Điều 173. Tội trộm cắp..."}),
        _FakePoint("p2", {"document_id": 10, "document_title": "Bộ luật Hình sự",
                          "document_number": "100/2015/QH13", "legal_type": "Luật",
                          "sector": "hinh_su", "article": "Điều 40", "article_number": "40",
                          "chunk_text": "Điều 40. Tử hình..."}),
        _FakePoint("p3", {"document_id": 20, "document_title": "Bộ luật Lao động",
                          "document_number": "45/2019/QH14", "legal_type": "Luật",
                          "sector": "lao_dong", "article": "Điều 40", "article_number": "40",
                          "chunk_text": "Điều 40. Nghĩa vụ..."}),
    ]


def _match_filter(payload, flt):
    """Minimal Qdrant Filter evaluator for must (MatchValue / MatchAny) + must_not."""
    if flt is None: return True
    for c in (flt.must or []):
        val = payload.get(c.key)
        m = c.match
        if hasattr(m, "any"):
            if val not in m.any: return False
        elif val != m.value:
            return False
    for c in (getattr(flt, "must_not", None) or []):
        if payload.get(c.key) == c.match.value: return False
    return True


def _install_scroll(fake, points):
    def scroll(collection_name, scroll_filter=None, limit=256, offset=None,
               with_payload=True, with_vectors=False):
        matched = [p for p in points if _match_filter(p.payload, scroll_filter)]
        # honor projected payload (list of fields) by trimming
        if isinstance(with_payload, list):
            matched = [_FakePoint(p.id, {k: p.payload.get(k) for k in with_payload}) for p in matched]
        return matched, None
    fake.scroll = scroll


def _install_paginated_scroll(fake, points, page_size=2):
    """Fake scroll that returns `page_size` points per call, advancing an
    integer offset cursor and returning offset=None only on the last page —
    so the store's `while True: ... if offset is None: break` loop is exercised
    across multiple pages (real Qdrant behavior)."""
    def scroll(collection_name, scroll_filter=None, limit=256, offset=None,
               with_payload=True, with_vectors=False):
        matched = [p for p in points if _match_filter(p.payload, scroll_filter)]
        if isinstance(with_payload, list):
            matched = [_FakePoint(p.id, {k: p.payload.get(k) for k in with_payload}) for p in matched]
        start = offset or 0
        page = matched[start:start + page_size]
        next_off = start + page_size
        return page, (next_off if next_off < len(matched) else None)
    fake.scroll = scroll


def _store_with_points(points):
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore.__new__(LegalVectorStore)
    s.collection_name = "test"
    fake = _FakeQdrantClient()
    _install_scroll(fake, points)
    s.client = fake
    return s


def test_get_document_index_projects_doc_level():
    s = _store_with_points(_seed_points())
    idx = s.get_document_index()
    assert set(idx.keys()) == {10, 20}
    assert idx[10]["document_title"] == "Bộ luật Hình sự"
    assert idx[10]["sector"] == "hinh_su"
    assert "chunk_text" not in idx[10]  # not retained


def test_scroll_article_filters_by_doc_and_number():
    s = _store_with_points(_seed_points())
    hits = s.scroll_article(doc_ids=[10], article_number="40")
    assert [h.point_id for h in hits] == ["p2"]  # not p3 (doc 20), not p1 (art 173)


def test_scroll_article_all_docs_when_empty():
    s = _store_with_points(_seed_points())
    hits = s.scroll_article(doc_ids=[], article_number="40")
    assert {h.point_id for h in hits} == {"p2", "p3"}


def test_get_document_index_paginates_and_dedups_first_wins():
    pts = [
        _FakePoint("a1", {"document_id": 10, "document_title": "BLHS (first)",
                          "document_number": "100/2015/QH13", "legal_type": "Luật", "sector": "hinh_su"}),
        _FakePoint("a2", {"document_id": 20, "document_title": "BLLĐ",
                          "document_number": "45/2019/QH14", "legal_type": "Luật", "sector": "lao_dong"}),
        _FakePoint("a3", {"document_id": 30, "document_title": "Luật Đất đai",
                          "document_number": "31/2024/QH15", "legal_type": "Luật", "sector": "dat_dai"}),
        _FakePoint("a4", {"document_id": 40, "document_title": "Luật DN",
                          "document_number": "59/2020/QH14", "legal_type": "Luật", "sector": "doanh_nghiep"}),
        # duplicate doc 10 on a later page with a DIFFERENT title — first-seen must win
        _FakePoint("a5", {"document_id": 10, "document_title": "BLHS (later dup)",
                          "document_number": "100/2015/QH13", "legal_type": "Luật", "sector": "hinh_su"}),
    ]
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore.__new__(LegalVectorStore)
    s.collection_name = "test"
    fake = _FakeQdrantClient()
    _install_paginated_scroll(fake, pts, page_size=2)
    s.client = fake
    idx = s.get_document_index()
    assert set(idx.keys()) == {10, 20, 30, 40}          # all pages walked
    assert idx[10]["document_title"] == "BLHS (first)"  # first-seen wins across pages


def test_scroll_docs_walks_all_pages():
    """scroll_docs must follow the offset cursor and return every point across
    pages (not just the first page) — same hot-path loop as get_document_index."""
    pts = _seed_points()  # 3 points -> spans 2 pages at page_size=2
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore.__new__(LegalVectorStore)
    s.collection_name = "test"
    fake = _FakeQdrantClient()
    _install_paginated_scroll(fake, pts, page_size=2)
    s.client = fake
    out = s.scroll_docs(doc_ids=[])
    assert {p.id for p in out} == {"p1", "p2", "p3"}


# ── M2.5: is_in_effect server-side filter + backfill setter ─────────────────

def test_build_filter_exclude_superseded_adds_must_not():
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore.__new__(LegalVectorStore)
    flt = s._build_filter({"legal_type": "Luật"}, exclude_superseded=True)
    assert any(getattr(c, "key", None) == "is_in_effect" for c in (flt.must_not or []))
    # keeps True + absent: only value False is excluded
    mn = [c for c in flt.must_not if c.key == "is_in_effect"][0]
    assert mn.match.value is False


def test_build_filter_none_conditions_but_exclude():
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore.__new__(LegalVectorStore)
    flt = s._build_filter(None, exclude_superseded=True)
    assert flt.must_not and flt.must_not[0].key == "is_in_effect"


def test_set_is_in_effect_calls_set_payload():
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore.__new__(LegalVectorStore)
    s.collection_name = "test"
    class _Rec:
        def __init__(self): self.calls = []
        def set_payload(self, **kw): self.calls.append(kw)
    s.client = _Rec()
    s.set_is_in_effect(10, False)
    call = s.client.calls[0]
    assert call["payload"] == {"is_in_effect": False}
    cond = call["points"].must[0]
    assert cond.key == "document_id" and cond.match.value == 10
