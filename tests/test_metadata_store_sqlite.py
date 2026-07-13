"""
SQLite-backed MetadataStore — behavior + API parity.
Run: .venv/bin/python -m pytest tests/test_metadata_store_sqlite.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import threading
from datetime import date
import pytest
from src.data.metadata_store import MetadataStore, DocumentMeta


@pytest.fixture
def store(tmp_path):
    return MetadataStore(data_dir=str(tmp_path))


def test_uses_sqlite_file_not_json(store, tmp_path):
    store.save_document(1, DocumentMeta(doc_id=1, title="Bộ luật Hình sự"))
    assert (tmp_path / "store.db").exists()


def test_document_round_trip_all_fields(store):
    meta = DocumentMeta(
        doc_id=42, title="Luật Doanh nghiệp",
        ref_count=5, internal_refs=3, external_refs=2,
        articles_referenced=["12", "13a"],
        effective_date="2021-01-01", expiry_date=None, is_in_effect=True,
        replaced_laws=["Luật DN 2014"], version_chain=["2014", "2020"],
        is_amendment=False, target_laws=[], amendment_count=0,
        amended_articles=["7"], graph_node_id="n42",
        graph_in_degree=1, graph_out_degree=4, errors=["x"],
    )
    store.save_document(42, meta)
    got = store.get_document(42)
    assert got.title == "Luật Doanh nghiệp"
    assert got.articles_referenced == ["12", "13a"]
    assert got.is_in_effect is True
    assert got.version_chain == ["2014", "2020"]
    assert got.amended_articles == ["7"]


def test_is_in_effect_none_survives(store):
    store.save_document(7, DocumentMeta(doc_id=7, title="Luật Đầu tư 2025"))
    assert store.get_document(7).is_in_effect is None


def test_get_all_and_missing(store):
    store.save_document(1, DocumentMeta(doc_id=1, title="A"))
    store.save_document(2, DocumentMeta(doc_id=2, title="B"))
    assert set(store.get_all_documents().keys()) == {1, 2}
    assert store.get_document(999) is None


def test_persistence_across_instances(store, tmp_path):
    store.save_document(1, DocumentMeta(doc_id=1, title="A", is_amendment=True,
                                        target_laws=["Bộ luật Lao động"]))
    reopened = MetadataStore(data_dir=str(tmp_path))
    assert reopened.get_document(1).title == "A"
    assert reopened.get_document(1).target_laws == ["Bộ luật Lao động"]


def test_get_documents_by_target(store):
    store.save_document(1, DocumentMeta(doc_id=1, title="NĐ sửa đổi BLLĐ",
                        is_amendment=True, target_laws=["Bộ luật Lao động"]))
    store.save_document(2, DocumentMeta(doc_id=2, title="Luật khác", is_amendment=False))
    hits = store.get_documents_by_target("bộ luật lao động")
    assert [m.doc_id for m in hits] == [1]


def test_amendments_and_articles(store):
    store.save_amendments([
        {"amending_doc_id": 5, "target": "BLHS", "article": "40"},
        {"amending_doc_id": 6, "target": "BLLĐ", "article": "90"},
    ])
    assert store.get_amendments_for_doc(5) == [{"amending_doc_id": 5, "target": "BLHS", "article": "40"}]
    assert len(store.get_amendments()) == 2


def test_timelines_and_effective_laws(store):
    store.save_timelines([
        {"doc_id": 1, "effective_date": "2016-01-01", "expiry_date": None, "is_in_effect": True},
        {"doc_id": 2, "effective_date": "2030-01-01", "expiry_date": None, "is_in_effect": False},
    ])
    assert store.get_timeline(1)["effective_date"] == "2016-01-01"
    eff = store.get_effective_laws(at_date=date(2020, 1, 1))
    assert [t["doc_id"] for t in eff] == [1]


def test_graph_neighbors(store):
    store.save_graph({"nodes": [{"id": "a"}, {"id": "b"}],
                      "edges": [{"source": "a", "target": "b"}]})
    assert store.get_graph()["edges"][0]["target"] == "b"
    assert set(store.get_article_neighbors("a")) == {"b"}


def test_clear(store, tmp_path):
    store.save_document(1, DocumentMeta(doc_id=1, title="A"))
    store.clear()
    assert store.get_all_documents() == {}
    assert store.get_document(1) is None


def test_cross_thread_access_does_not_raise(store):
    # Reproduces the FastAPI scenario: the store (and its sqlite connection)
    # is created on the main thread, but a sync endpoint dispatched to a
    # threadpool worker reads/writes it from a different thread. Without
    # check_same_thread=False + a lock this raises sqlite3.ProgrammingError.
    store.save_document(1, DocumentMeta(doc_id=1, title="A"))
    errors, results = [], []

    def worker():
        try:
            store.save_document(2, DocumentMeta(doc_id=2, title="B"))
            results.append(store.get_document(1).title)
            results.append(store.get_document(2).title)
        except Exception as e:  # noqa
            errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []
    assert results == ["A", "B"]


def test_build_index_shape(store):
    store.save_document(1, DocumentMeta(doc_id=1, title="A", is_amendment=False, is_in_effect=True))
    store.save_document(2, DocumentMeta(doc_id=2, title="B", is_amendment=True, amendment_count=3,
                                        amended_articles=["1", "2"]))
    idx = store.build_index()
    assert idx["total_documents"] == 2
    assert idx["amendment_laws"] == 1
    assert idx["base_laws"] == 1
    assert idx["total_amendments"] == 3
    assert idx["total_articles_amended"] == 2


def test_import_json_dir(tmp_path):
    from src.data.metadata_store import import_json_dir
    (tmp_path / "documents.json").write_text(json.dumps({
        "1": {"doc_id": 1, "title": "Bộ luật Hình sự", "is_in_effect": True,
              "amended_articles": ["40"]},
        "2": {"doc_id": 2, "title": "Luật Đầu tư 2025", "is_in_effect": None},
    }, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "timelines.json").write_text(json.dumps(
        [{"doc_id": 1, "effective_date": "2016-01-01", "is_in_effect": True}],
        ensure_ascii=False), encoding="utf-8")
    n = import_json_dir(str(tmp_path))
    assert n == 2
    store = MetadataStore(data_dir=str(tmp_path))
    assert store.get_document(1).amended_articles == ["40"]
    assert store.get_document(2).is_in_effect is None
    assert store.get_timeline(1)["effective_date"] == "2016-01-01"


def test_init_auto_seeds_from_json(tmp_path):
    (tmp_path / "documents.json").write_text(json.dumps({
        "1": {"doc_id": 1, "title": "Bộ luật Hình sự", "is_in_effect": True,
              "amended_articles": ["40"]},
        "2": {"doc_id": 2, "title": "Luật Đầu tư 2025", "is_in_effect": None},
    }, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "timelines.json").write_text(json.dumps(
        [{"doc_id": 1, "effective_date": "2016-01-01", "is_in_effect": True}],
        ensure_ascii=False), encoding="utf-8")

    # No manual import_json_dir call — construction alone must seed, and must
    # complete without RecursionError.
    store = MetadataStore(data_dir=str(tmp_path))

    assert store.get_document(1).is_in_effect is True
    assert store.get_document(1).amended_articles == ["40"]
    assert store.get_document(2).is_in_effect is None
    assert store.get_timeline(1)["effective_date"] == "2016-01-01"


def test_init_does_not_reseed_when_populated(tmp_path):
    store = MetadataStore(data_dir=str(tmp_path))
    store.save_document(1, DocumentMeta(doc_id=1, title="Original Title"))

    # Different content — if re-seeded on reopen, doc 1 title would change.
    (tmp_path / "documents.json").write_text(json.dumps({
        "1": {"doc_id": 1, "title": "Overwritten Title"},
    }, ensure_ascii=False), encoding="utf-8")

    reopened = MetadataStore(data_dir=str(tmp_path))
    assert reopened.get_document(1).title == "Original Title"
