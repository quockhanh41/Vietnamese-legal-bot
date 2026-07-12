# M2 Scale-Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the retrieval + metadata layers ready to serve a 600-doc corpus efficiently and correctly, without ingesting 600 docs yet.

**Architecture:** Four independent refactors, each preserving observable behavior while moving work server-side or off-RAM: (2.3) citation lookup filters Qdrant server-side instead of caching every chunk; (2.4) metadata store swaps JSON+RAM for SQLite behind an unchanged API; (2.5) `is_in_effect` becomes a Qdrant payload field + server filter, with a client-side backstop kept; (2.6) a measure-only reranker-latency harness.

**Tech Stack:** Python 3.13, `qdrant-client` (payload indexes + `scroll`/`set_payload`/filters), stdlib `sqlite3`, `pytest`, `FlagEmbedding` reranker.

## Global Constraints

- **Preserve refusal behavior** — no change may lower refusal recall; the two refusal gates stay intact.
- **Interface preservation** — 2.3/2.4/2.5 change backends/paths, NOT observable behavior for callers. Existing tests must stay green **without edits** (except where a test asserts a removed private method).
- **No live mutation in dev loop** — write paths (2.5 `set_payload`) are tested on an **ephemeral** Qdrant collection; never mutate `luat_law_ctx_v3` from tests.
- **`article_number` is KEYWORD** — Vietnamese articles carry letter suffixes ("12a"); always string, never `int`.
- **`is_in_effect` semantics** — keep `True` **and** unset/None; drop **only** explicit `False`. Fail-open on unknown.
- **Citation lookup is exact-match** — the `is_in_effect` server filter applies to the hybrid/vector route only, NOT to citation lookup.
- **Test invocation** — `.venv/bin/python -m pytest ...`. Full suite baseline: 298 tests collected.
- **Commits** — branch `m2-scale-prep`, one commit per task, message `feat(M2.x): ...`, footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/data/metadata_store.py` (modify) | SQLite backend behind unchanged `MetadataStore` API | 1, 2 |
| `tests/test_metadata_store_sqlite.py` (create) | SQLite backend behavior + API parity | 1, 2 |
| `src/db/qdrant_client.py` (modify) | `get_document_index()`, `scroll_article()`, `exclude_superseded` search param, `set_is_in_effect()` | 3, 5 |
| `tests/test_qdrant_client.py` (modify) | fake-client tests for the new store helpers | 3, 5 |
| `src/rag/citation_lookup.py` (modify) | drop client-side chunk cache; use server-side helpers | 4 |
| `tests/test_citation_lookup_serverside.py` (create) | lookup/inventory against fake + integration marker | 4 |
| `src/rag/orchestrator.py` (modify) | thread `exclude_superseded=True` into hybrid/vector answer paths | 6 |
| `scripts/backfill_is_in_effect.py` (create) | standalone backfill of `is_in_effect` onto an existing collection | 5 |
| `scripts/ingest.py` (modify) | call backfill after enrichment | 5 |
| `scripts/bench_reranker.py` (create) | reranker P50/P95/P99 latency harness | 7 |

Task order: **1 → 2** (SQLite, no network) then **3 → 4** (citation server-side) then **5 → 6** (is_in_effect) then **7** (bench). Tasks 1/2, 3/4, 5/6 are the coupled pairs; the three groups are mutually independent.

---

### Task 1: SQLite backend for `MetadataStore` (API parity)

Swap the JSON+RAM backend for a single SQLite file, keeping every public method's signature and return type identical. Callers (`enricher.py`, `retriever.py`, `orchestrator.py`) are untouched.

**Files:**
- Modify: `src/data/metadata_store.py` (class `MetadataStore` internals; `DocumentMeta` unchanged)
- Test: `tests/test_metadata_store_sqlite.py` (create)

**Interfaces:**
- Consumes: `DocumentMeta` (unchanged dataclass in same file).
- Produces (all pre-existing signatures, now SQLite-backed):
  - `MetadataStore(data_dir: str = "data/metadata")`
  - `save_document(doc_id: int, meta: DocumentMeta) -> None`
  - `get_document(doc_id: int) -> Optional[DocumentMeta]`
  - `get_all_documents() -> Dict[int, DocumentMeta]`
  - `get_documents_by_target(law_name: str) -> List[DocumentMeta]`
  - `get_articles_amended(doc_id: int) -> List[str]`
  - `save_graph(graph_dict: Dict) -> None` / `get_graph() -> Dict` / `get_article_neighbors(node_id: str, hops: int = 1) -> List[str]`
  - `save_timelines(timelines: List[Dict]) -> None` / `get_timelines() -> List[Dict]` / `get_timeline(doc_id: int) -> Optional[Dict]` / `get_effective_laws(at_date: Optional[date] = None) -> List[Dict]`
  - `save_amendments(amendments: List[Dict]) -> None` / `get_amendments() -> List[Dict]` / `get_amendments_for_doc(doc_id: int) -> List[Dict]`
  - `build_index() -> Dict` / `get_index() -> Dict` / `clear() -> None`
  - New helper for Task 2: `import_json_dir(data_dir: str) -> int` (module-level function).

- [ ] **Step 1: Write failing tests for document round-trip + parity**

Create `tests/test_metadata_store_sqlite.py`:

```python
"""
SQLite-backed MetadataStore — behavior + API parity.
Run: .venv/bin/python -m pytest tests/test_metadata_store_sqlite.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_metadata_store_sqlite.py -q`
Expected: FAIL — `store.db` not created / JSON still used (`AssertionError` on `test_uses_sqlite_file_not_json`).

- [ ] **Step 3: Rewrite `MetadataStore.__init__` + schema**

Replace the JSON file paths / in-memory caches / `_load()` with a SQLite connection and schema. Keep `DocumentMeta` and its `to_dict`/`from_dict` untouched. Add these constants at top of `MetadataStore`:

```python
import sqlite3
import json

_LIST_FIELDS = ("articles_referenced", "replaced_laws", "version_chain",
                "target_laws", "amended_articles", "errors")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id INTEGER PRIMARY KEY, title TEXT, ref_count INTEGER, internal_refs INTEGER,
    external_refs INTEGER, effective_date TEXT, expiry_date TEXT, is_in_effect INTEGER,
    is_amendment INTEGER, amendment_count INTEGER, graph_node_id TEXT,
    graph_in_degree INTEGER, graph_out_degree INTEGER, enriched_at TEXT,
    articles_referenced TEXT, replaced_laws TEXT, version_chain TEXT,
    target_laws TEXT, amended_articles TEXT, errors TEXT);
CREATE INDEX IF NOT EXISTS ix_docs_amend ON documents(is_amendment);
CREATE TABLE IF NOT EXISTS timelines (
    doc_id INTEGER, effective_date TEXT, expiry_date TEXT, is_in_effect INTEGER, blob TEXT);
CREATE INDEX IF NOT EXISTS ix_tl_doc ON timelines(doc_id);
CREATE TABLE IF NOT EXISTS amendments (amending_doc_id INTEGER, target TEXT, blob TEXT);
CREATE INDEX IF NOT EXISTS ix_am_doc ON amendments(amending_doc_id);
CREATE TABLE IF NOT EXISTS graph (id INTEGER PRIMARY KEY CHECK(id=1), blob TEXT);
"""


def __init__(self, data_dir: str = "data/metadata"):
    self.data_dir = Path(data_dir)
    self.data_dir.mkdir(parents=True, exist_ok=True)
    self._db_path = self.data_dir / "store.db"
    self._conn = sqlite3.connect(self._db_path)
    self._conn.row_factory = sqlite3.Row
    self._conn.executescript(_SCHEMA)
    self._conn.commit()
```

`is_in_effect` stores `NULL`/`0`/`1`; a `_to_bool(v)` helper maps `None→None, 0→False, 1→True`. `is_amendment` stores `0/1` (never NULL).

- [ ] **Step 4: Implement document methods**

```python
def _row_to_meta(self, row) -> DocumentMeta:
    d = dict(row)
    for f in _LIST_FIELDS:
        d[f] = json.loads(d[f]) if d[f] else []
    d["is_in_effect"] = None if d["is_in_effect"] is None else bool(d["is_in_effect"])
    d["is_amendment"] = bool(d["is_amendment"])
    return DocumentMeta.from_dict(d)

def save_document(self, doc_id: int, meta: DocumentMeta) -> None:
    meta.enriched_at = datetime.now().isoformat()
    d = meta.to_dict()
    for f in _LIST_FIELDS:
        d[f] = json.dumps(d[f], ensure_ascii=False)
    cols = ["doc_id","title","ref_count","internal_refs","external_refs","effective_date",
            "expiry_date","is_in_effect","is_amendment","amendment_count","graph_node_id",
            "graph_in_degree","graph_out_degree","enriched_at",*_LIST_FIELDS]
    self._conn.execute(
        f"INSERT OR REPLACE INTO documents ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [d.get(c) for c in cols])
    self._conn.commit()

def get_document(self, doc_id: int) -> Optional[DocumentMeta]:
    row = self._conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    return self._row_to_meta(row) if row else None

def get_all_documents(self) -> Dict[int, DocumentMeta]:
    return {r["doc_id"]: self._row_to_meta(r)
            for r in self._conn.execute("SELECT * FROM documents").fetchall()}

def get_documents_by_target(self, law_name: str) -> List[DocumentMeta]:
    norm = law_name.lower()
    out = []
    for r in self._conn.execute("SELECT * FROM documents WHERE is_amendment=1").fetchall():
        meta = self._row_to_meta(r)
        for target in meta.target_laws:
            if norm in target.lower() or target.lower() in norm:
                out.append(meta); break
    return out

def get_articles_amended(self, doc_id: int) -> List[str]:
    meta = self.get_document(doc_id)
    return meta.amended_articles if meta else []
```

- [ ] **Step 5: Implement graph / timeline / amendment / index / clear methods**

```python
def save_graph(self, graph_dict) -> None:
    self._conn.execute("INSERT OR REPLACE INTO graph (id, blob) VALUES (1, ?)",
                       (json.dumps(graph_dict, ensure_ascii=False, default=str),))
    self._conn.commit()

def get_graph(self) -> Dict:
    row = self._conn.execute("SELECT blob FROM graph WHERE id=1").fetchone()
    return json.loads(row["blob"]) if row else {}

def get_article_neighbors(self, node_id, hops: int = 1) -> List[str]:
    edges = self.get_graph().get("edges", [])
    nb = set()
    for e in edges:
        if e.get("source") == node_id: nb.add(e.get("target"))
        if e.get("target") == node_id: nb.add(e.get("source"))
    return list(nb)

def save_timelines(self, timelines: List[Dict]) -> None:
    self._conn.execute("DELETE FROM timelines")
    self._conn.executemany(
        "INSERT INTO timelines (doc_id, effective_date, expiry_date, is_in_effect, blob) VALUES (?,?,?,?,?)",
        [(t.get("doc_id"), t.get("effective_date"), t.get("expiry_date"),
          None if t.get("is_in_effect") is None else int(bool(t.get("is_in_effect"))),
          json.dumps(t, ensure_ascii=False, default=str)) for t in timelines])
    self._conn.commit()

def get_timelines(self) -> List[Dict]:
    return [json.loads(r["blob"]) for r in self._conn.execute("SELECT blob FROM timelines").fetchall()]

def get_timeline(self, doc_id: int) -> Optional[Dict]:
    row = self._conn.execute("SELECT blob FROM timelines WHERE doc_id=?", (doc_id,)).fetchone()
    return json.loads(row["blob"]) if row else None

def get_effective_laws(self, at_date=None) -> List[Dict]:
    if at_date is None:
        at_date = date.today()
    iso = at_date.isoformat()
    rows = self._conn.execute(
        "SELECT blob FROM timelines WHERE effective_date IS NOT NULL AND effective_date<=? "
        "AND (expiry_date IS NULL OR expiry_date>?)", (iso, iso)).fetchall()
    return [json.loads(r["blob"]) for r in rows]

def save_amendments(self, amendments: List[Dict]) -> None:
    self._conn.execute("DELETE FROM amendments")
    self._conn.executemany(
        "INSERT INTO amendments (amending_doc_id, target, blob) VALUES (?,?,?)",
        [(a.get("amending_doc_id"), a.get("target"),
          json.dumps(a, ensure_ascii=False, default=str)) for a in amendments])
    self._conn.commit()

def get_amendments(self) -> List[Dict]:
    return [json.loads(r["blob"]) for r in self._conn.execute("SELECT blob FROM amendments").fetchall()]

def get_amendments_for_doc(self, doc_id: int) -> List[Dict]:
    return [json.loads(r["blob"]) for r in
            self._conn.execute("SELECT blob FROM amendments WHERE amending_doc_id=?", (doc_id,)).fetchall()]

def clear(self) -> None:
    for t in ("documents", "timelines", "amendments", "graph"):
        self._conn.execute(f"DELETE FROM {t}")
    self._conn.commit()
```

`build_index()` / `get_index()`: keep the exact dict shape from the current implementation but compute the aggregates from `get_all_documents()` / `get_graph()` instead of the old in-memory dicts. `get_index()` returns `build_index()` (no separate index.json). Copy the aggregation expressions verbatim from the pre-existing `build_index` body (the `documents.values()` comprehensions) — they operate on the same `DocumentMeta` objects.

- [ ] **Step 6: Run tests to verify pass + full suite green**

Run: `.venv/bin/python -m pytest tests/test_metadata_store_sqlite.py tests/test_metadata_store.py tests/test_retriever_phase2.py -q`
Expected: PASS (new file), PASS (old `DocumentMeta` defaults test untouched), PASS (retriever uses the store).
Run: `.venv/bin/python -m pytest -q`
Expected: same pass count as baseline minus none (298 → 298+new).

- [ ] **Step 7: Commit**

```bash
git add src/data/metadata_store.py tests/test_metadata_store_sqlite.py
git commit -m "feat(M2.4): SQLite-backed MetadataStore behind unchanged API

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: JSON → SQLite migration importer

Preserve the existing 100-doc enrichment: a one-time importer that reads legacy `documents.json`/`graph.json`/`timelines.json`/`amendments.json` into the SQLite store.

**Files:**
- Modify: `src/data/metadata_store.py` (add module-level `import_json_dir`)
- Test: `tests/test_metadata_store_sqlite.py` (add cases)

**Interfaces:**
- Produces: `import_json_dir(data_dir: str) -> int` (returns number of documents imported). Reads `*.json` if present, writes into `MetadataStore(data_dir)`.

- [ ] **Step 1: Write failing test**

Add to `tests/test_metadata_store_sqlite.py`:

```python
import json
from src.data.metadata_store import import_json_dir

def test_import_json_dir(tmp_path):
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metadata_store_sqlite.py::test_import_json_dir -v`
Expected: FAIL — `ImportError: cannot import name 'import_json_dir'`.

- [ ] **Step 3: Implement `import_json_dir`**

```python
def import_json_dir(data_dir: str) -> int:
    """One-time: load legacy *.json enrichment files into the SQLite store."""
    p = Path(data_dir)
    store = MetadataStore(data_dir=data_dir)
    n = 0
    docs_f = p / "documents.json"
    if docs_f.exists():
        raw = json.loads(docs_f.read_text(encoding="utf-8"))
        for k, v in raw.items():
            store.save_document(int(k), DocumentMeta.from_dict(v))
            n += 1
    for name, saver in (("graph.json", store.save_graph),
                        ("timelines.json", store.save_timelines),
                        ("amendments.json", store.save_amendments)):
        f = p / name
        if f.exists():
            saver(json.loads(f.read_text(encoding="utf-8")))
    return n
```

- [ ] **Step 4: Run test to verify pass**

Run: `.venv/bin/python -m pytest tests/test_metadata_store_sqlite.py::test_import_json_dir -v`
Expected: PASS.

- [ ] **Step 5: Migrate the real 100-doc metadata (if JSON present)**

Run: `.venv/bin/python -c "from src.data.metadata_store import import_json_dir; print('imported', import_json_dir('data/metadata'))"`
Expected: prints `imported <N>` (N≈100 if legacy JSON exists; `imported 0` is fine if it was already migrated / absent). `data/metadata/store.db` now exists.

- [ ] **Step 6: Commit**

```bash
git add src/data/metadata_store.py tests/test_metadata_store_sqlite.py
git commit -m "feat(M2.4): JSON->SQLite metadata importer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `LegalVectorStore` server-side helpers for citation lookup

Add the two Qdrant helpers the citation refactor (Task 4) needs: a projected doc-level index scroll (O(docs) memory) and a filtered article scroll (uses the existing `document_id` + `article_number` payload indexes).

**Files:**
- Modify: `src/db/qdrant_client.py`
- Test: `tests/test_qdrant_client.py` (extend the existing fake client)

**Interfaces:**
- Produces:
  - `get_document_index() -> Dict[int, Dict]` — `{document_id: {"document_title","document_number","legal_type","sector"}}`, one projected scroll, no chunk_text retained.
  - `scroll_article(doc_ids: List[int], article_number: str, cap: int = 8) -> List[SearchResult]` — server-side filtered scroll: chunks whose `document_id ∈ doc_ids` AND `article_number == article_number`. Empty `doc_ids` means "all docs" (no `document_id` clause).
  - `scroll_docs(doc_ids: List[int], limit_per: int = 512) -> List[ScoredPoint-like]` — raw points for the lenient fallback (payload incl. `chunk_text`, `article`), filtered by `document_id ∈ doc_ids`.

- [ ] **Step 1: Extend the fake client + write failing tests**

Add to `tests/test_qdrant_client.py`. The existing `_FakeQdrantClient` records calls; extend it to serve `scroll` from an in-memory point list and to assert filters. Add:

```python
from qdrant_client.http import models as qmodels
from src.db.qdrant_client import SearchResult

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
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_qdrant_client.py -k "document_index or scroll_article" -v`
Expected: FAIL — `AttributeError: 'LegalVectorStore' object has no attribute 'get_document_index'`.

- [ ] **Step 3: Implement the helpers**

Add to `LegalVectorStore` in `src/db/qdrant_client.py`:

```python
_DOC_INDEX_FIELDS = ["document_id", "document_title", "document_number", "legal_type", "sector"]

def get_document_index(self) -> Dict[int, Dict]:
    """One projected doc-level scroll (no chunk_text). {doc_id: {title,number,type,sector}}."""
    index: Dict[int, Dict] = {}
    offset = None
    while True:
        points, offset = self.client.scroll(
            collection_name=self.collection_name, limit=256, offset=offset,
            with_payload=self._DOC_INDEX_FIELDS, with_vectors=False)
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("document_id")
            if doc_id is None or doc_id in index:
                continue
            index[doc_id] = {k: payload.get(k) for k in
                             ("document_title", "document_number", "legal_type", "sector")}
        if offset is None:
            break
    return index

def scroll_article(self, doc_ids: List[int], article_number: str, cap: int = 8) -> List[SearchResult]:
    """Server-side filtered scroll: document_id in doc_ids AND article_number == value."""
    must = [FieldCondition(key="article_number", match=MatchValue(value=article_number))]
    if doc_ids:
        must.append(FieldCondition(key="document_id", match=models.MatchAny(any=list(doc_ids))))
    points, _ = self.client.scroll(
        collection_name=self.collection_name, scroll_filter=Filter(must=must),
        limit=cap, with_payload=True, with_vectors=False)
    return [self._point_to_result(p) for p in points]

def scroll_docs(self, doc_ids: List[int], limit_per: int = 512):
    """Raw points for the lenient fallback (keeps chunk_text/article), filtered by document_id."""
    if not doc_ids:
        flt = None
    else:
        flt = Filter(must=[FieldCondition(key="document_id", match=models.MatchAny(any=list(doc_ids)))])
    out, offset = [], None
    while True:
        points, offset = self.client.scroll(
            collection_name=self.collection_name, scroll_filter=flt,
            limit=limit_per, offset=offset, with_payload=True, with_vectors=False)
        out.extend(points)
        if offset is None:
            break
    return out

def _point_to_result(self, point) -> SearchResult:
    payload = point.payload or {}
    return SearchResult(
        point_id=str(point.id), score=1.0, chunk_text=payload.get("chunk_text", ""),
        document_title=payload.get("document_title", ""),
        document_number=payload.get("document_number", ""),
        legal_type=payload.get("legal_type", ""), article=payload.get("article"),
        chapter=payload.get("chapter"), url=payload.get("url"),
        document_id=payload.get("document_id"),
        parent_chunk_text=payload.get("parent_chunk_text"))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_qdrant_client.py -k "document_index or scroll_article" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/db/qdrant_client.py tests/test_qdrant_client.py
git commit -m "feat(M2.3): server-side doc-index + article-scroll helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Refactor `CitationLookup` to server-side filtering

Drop the full-collection client-side chunk cache (`_chunks_by_doc`); resolve the doc inventory via `get_document_index()` and resolve articles via `scroll_article()` + a bounded lenient fallback via `scroll_docs()`. Public API (`lookup`, `resolve_doc`, `corpus_summary`, `corpus_summary_compact`, `_get_alias_index`) unchanged.

**Files:**
- Modify: `src/rag/citation_lookup.py`
- Test: `tests/test_citation_lookup_serverside.py` (create); existing `tests/test_citation_lookup.py` + `tests/test_citation_lookup_multisector.py` must stay green.

**Interfaces:**
- Consumes: `LegalVectorStore.get_document_index()`, `scroll_article()`, `scroll_docs()` (Task 3).
- Produces: same `CitationLookup` public methods. `_get_alias_index()` (used by `orchestrator.py:737`) still returns `Dict[int, str]` (doc_id→title). Internal: replace `_load_collection_cache` with `_load_doc_index`; remove `_chunks_by_doc` / `_get_chunks_by_doc`.

- [ ] **Step 1: Write failing test (server-side lookup, no chunk cache)**

Create `tests/test_citation_lookup_serverside.py`:

```python
"""
CitationLookup server-side path — resolves via get_document_index() /
scroll_article() instead of caching every chunk.
Run: .venv/bin/python -m pytest tests/test_citation_lookup_serverside.py -q
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag.citation_lookup import CitationLookup, parse_citation
from src.db.qdrant_client import SearchResult


class _FakeStore:
    def __init__(self):
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
    def get_document_index(self):
        return self._index
    def scroll_article(self, doc_ids, article_number, cap=8):
        ids = doc_ids or list(self._index)
        out = []
        for i in ids:
            out.extend(self._by.get((i, article_number), []))
        return out[:cap]
    def scroll_docs(self, doc_ids, limit_per=512):
        return []


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_citation_lookup_serverside.py -q`
Expected: FAIL — `lookup` calls `_get_chunks_by_doc()` / `client.scroll` on the fake store (no `.client`), raising `AttributeError`.

- [ ] **Step 3: Replace the cache loader with a doc-index loader**

In `CitationLookup.__init__`, drop `self._chunks_by_doc`. Replace `_load_collection_cache` with:

```python
def _load_doc_index(self) -> None:
    """One projected doc-level scroll → alias / doc_number / sector indexes.
    No chunk_text retained (O(docs) memory)."""
    index = self.vector_store.get_document_index()
    aliases, doc_numbers, sectors = {}, {}, {}
    for doc_id, meta in index.items():
        aliases[doc_id] = meta.get("document_title", "") or ""
        normalized = _normalize_doc_number(meta.get("document_number", "") or "")
        if normalized is not None:
            doc_numbers[doc_id] = normalized
        if meta.get("sector"):
            sectors[doc_id] = meta["sector"]
    self._alias_index, self._doc_number_index, self._sector_index = aliases, doc_numbers, sectors
```

Update `_get_alias_index` / `_get_doc_number_index` to call `_load_doc_index()` instead of `_load_collection_cache()`. Delete `_get_chunks_by_doc`.

- [ ] **Step 4: Rewrite `lookup()` to use server-side scroll**

```python
def lookup(self, parsed: ParsedCitation, cap: int = 8) -> List[SearchResult]:
    target_num = parsed.primary.number
    doc_ids = self.resolve_doc(parsed.doc_hint, parsed.year, parsed.doc_number)
    all_ids = list(self._get_alias_index().keys())
    scoped = [] if set(doc_ids) == set(all_ids) else doc_ids  # [] => server "all docs"

    strict = self.vector_store.scroll_article(scoped, target_num, cap=cap)
    if len(strict) >= cap:
        return strict[:cap]

    # Lenient fallback: chunk whose `article` payload is stale but text opens with "Điều N."
    seen = {r.point_id for r in strict}
    open_pattern = re.compile(r"Điều\s+" + re.escape(target_num) + r"\s*[\.:]")
    lenient: List[SearchResult] = []
    for point in self.vector_store.scroll_docs(scoped):
        payload = point.payload or {}
        text = payload.get("chunk_text", "") or ""
        m = open_pattern.search(text)
        if m and m.start() < _CHUNK_OPEN_WINDOW:
            r = self.vector_store._point_to_result(point)
            if r.point_id not in seen:
                seen.add(r.point_id); lenient.append(r)
        if len(strict) + len(lenient) >= cap:
            break
    return (strict + lenient)[:cap]
```

Remove the now-unused `_point_to_search_result` staticmethod and `artnum` import if no longer referenced (leave `artnum` import if other code in the module uses it — verify with grep before deleting).

- [ ] **Step 5: Run new + existing citation tests**

Run: `.venv/bin/python -m pytest tests/test_citation_lookup_serverside.py tests/test_citation_lookup.py tests/test_citation_lookup_multisector.py -q`
Expected: PASS all. (The multisector tests exercise `resolve_doc` ranking, which is unchanged.)

- [ ] **Step 6: Read-only end-to-end sanity vs live v3**

Run:
```bash
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from src.db.qdrant_client import LegalVectorStore
from src.rag.citation_lookup import CitationLookup, parse_citation
s = LegalVectorStore(collection_name='luat_law_ctx_v3')
cl = CitationLookup(s)
for q in ['Điều 173 Bộ luật Hình sự 2015 quy định về tội gì?',
          'Khoản 2 Điều 40 BLHS quy định gì?',
          'Điều 168 Bộ luật Lao động 2019']:
    hits = cl.lookup(parse_citation(q))
    print(q, '->', [(h.document_title[:30], h.article) for h in hits[:3]])
"
```
Expected: each citation resolves to the right document + article (BLHS Điều 173, BLHS Điều 40, BLLĐ Điều 168). Non-empty, correctly scoped.

- [ ] **Step 7: Commit**

```bash
git add src/rag/citation_lookup.py tests/test_citation_lookup_serverside.py
git commit -m "feat(M2.3): CitationLookup server-side filter, drop chunk cache

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `is_in_effect` payload — write path + server filter capability

Give `LegalVectorStore` (a) an `exclude_superseded` search option that adds `must_not[is_in_effect==False]`, and (b) a `set_is_in_effect(doc_id, value)` backfill using `set_payload` filtered by `document_id`. Then a standalone backfill script + ingest wiring.

**Files:**
- Modify: `src/db/qdrant_client.py`, `scripts/ingest.py`
- Create: `scripts/backfill_is_in_effect.py`
- Test: `tests/test_qdrant_client.py` (extend)

**Interfaces:**
- Produces:
  - `search_dense(query_vector, top_k=20, filter_conditions=None, exclude_superseded=False)`
  - `search_hybrid(dense_vector, sparse_vector, top_k=20, fusion_strategy="rrf", filter_conditions=None, exclude_superseded=False)`
  - `set_is_in_effect(doc_id: int, value: bool) -> None` — `set_payload({"is_in_effect": value}, filter=document_id==doc_id)`.
  - `_build_filter(conditions, exclude_superseded=False) -> Filter` (adds `must_not` clause).

- [ ] **Step 1: Write failing tests**

Add to `tests/test_qdrant_client.py`:

```python
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
```

Note: `_build_filter` currently takes only `conditions`. Change its signature and update the two call sites in `search_dense`/`search_sparse`/`search_hybrid` (they currently call `self._build_filter(filter_conditions) if filter_conditions else None`).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_qdrant_client.py -k "exclude_superseded or set_is_in_effect" -v`
Expected: FAIL — `_build_filter() got an unexpected keyword argument 'exclude_superseded'` / no `set_is_in_effect`.

- [ ] **Step 3: Implement filter + setter**

Replace `_build_filter` and thread the flag:

```python
def _build_filter(self, conditions: Optional[Dict] = None, exclude_superseded: bool = False) -> Optional[Filter]:
    must = []
    for key, value in (conditions or {}).items():
        must.append(FieldCondition(key=key, match=MatchValue(value=value)))
    must_not = []
    if exclude_superseded:
        # keep is_in_effect True AND unset/None; drop only explicit False
        must_not.append(FieldCondition(key="is_in_effect", match=MatchValue(value=False)))
    if not must and not must_not:
        return None
    return Filter(must=must or None, must_not=must_not or None)

def set_is_in_effect(self, doc_id: int, value: bool) -> None:
    self.client.set_payload(
        collection_name=self.collection_name,
        payload={"is_in_effect": value},
        points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]),
        wait=True)
```

In `search_dense`/`search_sparse`/`search_hybrid`, change the filter line to:
`query_filter = self._build_filter(filter_conditions, exclude_superseded=exclude_superseded)`
and add `exclude_superseded: bool = False` to each signature (`search_sparse` may add it for symmetry even if unused by callers).

- [ ] **Step 4: Run tests to verify pass + full suite**

Run: `.venv/bin/python -m pytest tests/test_qdrant_client.py -q`
Expected: PASS. Run `.venv/bin/python -m pytest -q` — no regressions.

- [ ] **Step 5: Ephemeral integration test for the filter semantics**

Create `tests/test_is_in_effect_filter_integration.py` (skips if no Qdrant):

```python
"""
is_in_effect server filter — real Qdrant round-trip on an EPHEMERAL collection.
Skips when QDRANT_URL is unset. Never touches luat_law_ctx_v3.
Run: .venv/bin/python -m pytest tests/test_is_in_effect_filter_integration.py -q
"""
import os, sys, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from dotenv import load_dotenv; load_dotenv()

pytestmark = pytest.mark.skipif(not os.getenv("QDRANT_URL"), reason="no Qdrant")

def test_exclude_superseded_keeps_true_and_absent():
    from qdrant_client.models import PointStruct, SparseVector
    from src.db.qdrant_client import LegalVectorStore
    name = f"eph_{uuid.uuid4().hex[:8]}"
    s = LegalVectorStore(collection_name=name)
    s.create_collection()
    try:
        vec = lambda x: [float(x)] * s.VECTOR_DIM
        pts = [
            PointStruct(id=str(uuid.uuid4()), vector={"dense": vec(1),
                "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={"chunk_text": "keep-true", "document_id": 1, "is_in_effect": True}),
            PointStruct(id=str(uuid.uuid4()), vector={"dense": vec(1),
                "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={"chunk_text": "drop-false", "document_id": 2, "is_in_effect": False}),
            PointStruct(id=str(uuid.uuid4()), vector={"dense": vec(1),
                "sparse": SparseVector(indices=[1], values=[1.0])},
                payload={"chunk_text": "keep-absent", "document_id": 3}),
        ]
        s.client.upsert(collection_name=name, points=pts, wait=True)
        hits = s.search_dense(vec(1), top_k=10, exclude_superseded=True)
        texts = {h.chunk_text for h in hits}
        assert "keep-true" in texts and "keep-absent" in texts
        assert "drop-false" not in texts
        # backfill flips doc 3 to superseded, then it drops too
        s.set_is_in_effect(3, False)
        hits2 = s.search_dense(vec(1), top_k=10, exclude_superseded=True)
        assert "keep-absent" not in {h.chunk_text for h in hits2}
    finally:
        s.client.delete_collection(name)
```

Run: `.venv/bin/python -m pytest tests/test_is_in_effect_filter_integration.py -q`
Expected: PASS (or `skipped` if Qdrant unreachable).

- [ ] **Step 6: Backfill script + ingest wiring**

Create `scripts/backfill_is_in_effect.py`:

```python
"""
Backfill is_in_effect onto chunk payloads of an existing collection, from the
MetadataStore. Idempotent, additive. Only writes docs with a KNOWN status
(True/False); None (unknown) is left unset so the server filter keeps it.

Usage: .venv/bin/python scripts/backfill_is_in_effect.py --collection luat_law_ctx_v3
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()
from src.db.qdrant_client import LegalVectorStore
from src.data.metadata_store import MetadataStore

def main(collection, metadata_dir):
    store = LegalVectorStore(collection_name=collection)
    meta = MetadataStore(data_dir=metadata_dir)
    n = 0
    for doc_id, m in meta.get_all_documents().items():
        if m.is_in_effect is None:
            continue
        store.set_is_in_effect(doc_id, bool(m.is_in_effect))
        n += 1
    print(f"Backfilled is_in_effect on {n} documents in '{collection}'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION"))
    ap.add_argument("--metadata-dir", default="data/metadata")
    a = ap.parse_args()
    main(a.collection, a.metadata_dir)
```

In `scripts/ingest.py`, after enrichment completes and before the final summary, add a call that backfills from the just-written MetadataStore (guarded so ingest without enrichment stays unaffected). Locate the point after enrichment; add:

```python
# M2.5: push known is_in_effect onto chunk payloads for the server-side filter.
if enrich:  # same flag that gated enrichment
    from src.data.metadata_store import MetadataStore
    _meta = MetadataStore(data_dir=metadata_dir)
    _n = 0
    for _doc_id, _m in _meta.get_all_documents().items():
        if _m.is_in_effect is not None:
            store.set_is_in_effect(_doc_id, bool(_m.is_in_effect)); _n += 1
    print(f"  ↳ is_in_effect backfilled on {_n} docs")
```

(If `scripts/ingest.py` has no `enrich`/`metadata_dir` locals, wire the backfill into whichever step runs the `Enricher` — grep `Enricher`/`metadata` in the file; the backfill must run against the SAME collection just ingested.)

- [ ] **Step 7: Commit**

```bash
git add src/db/qdrant_client.py scripts/backfill_is_in_effect.py scripts/ingest.py \
        tests/test_qdrant_client.py tests/test_is_in_effect_filter_integration.py
git commit -m "feat(M2.5): is_in_effect payload write path + server filter capability

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Wire `exclude_superseded=True` into the answer pipelines

Push the server-side exclusion into the hybrid/vector retrieval that feeds answers. Keep `_filter_in_effect` + `_annotate_phase2` as a client-side backstop (correct even before a collection is backfilled) and for amendment badges. Citation lookup is NOT filtered.

**Files:**
- Modify: `src/rag/orchestrator.py`, `src/rag/citation_lookup.py` (`retrieve_routed` hybrid branch)
- Test: `tests/test_orchestrator.py` (add a targeted assertion)

**Interfaces:**
- Consumes: `search_dense(..., exclude_superseded=True)`, `search_hybrid(..., exclude_superseded=True)` (Task 5).
- Produces: no signature changes; only the retrieval calls gain the kwarg.

- [ ] **Step 1: Write failing test — retrieve_routed hybrid branch passes the flag**

Add to `tests/test_orchestrator.py` (or a new `tests/test_exclude_superseded_wiring.py`):

```python
def test_retrieve_routed_hybrid_excludes_superseded():
    """Non-citation query → hybrid search must request exclude_superseded=True."""
    import numpy as np
    from src.rag.citation_lookup import retrieve_routed

    class _Emb:
        def encode(self, qs, return_dense=True, return_sparse=True):
            # dense_vec.tolist() is called downstream → must be a numpy array
            return {"dense_vecs": [np.array([0.1])], "lexical_weights": [{1: 0.5}]}
        def encode_queries(self, qs): return [np.array([0.1])]

    class _Store:
        def __init__(self): self.kw = None
        def search_hybrid(self, dense, sparse, top_k=20, **kw):
            self.kw = kw
            return []

    store = _Store()
    retrieve_routed("Thế nào là tội cướp tài sản?", store, _Emb(), citation_lookup=None, fill_top_k=10)
    assert store.kw.get("exclude_superseded") is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orchestrator.py -k exclude_superseded -v`
(or the new file). Expected: FAIL — `assert None is True` (kwarg not passed).

- [ ] **Step 3: Add the kwarg in `retrieve_routed`**

In `src/rag/citation_lookup.py`, the hybrid branch (end of `retrieve_routed`):

```python
    hits = vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=fill_top_k,
                                      exclude_superseded=True)
```

- [ ] **Step 4: Thread the flag through the orchestrator answer pipelines**

In `src/rag/orchestrator.py`, add `exclude_superseded=True` to each retrieval call that produces answer context inside `_simple_pipeline`, `_medium_pipeline`, `_complex_pipeline`, and their `query_stream` counterparts — i.e. every `self.vector_store.search_dense(...)` / `self.vector_store.search_hybrid(...)` in those methods. Example (`_medium_pipeline`):

```python
        results = self.vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=20,
                                                  exclude_superseded=True)
```

Leave `_annotate_phase2` + `_filter_in_effect` calls exactly as they are (backstop + badges). Do NOT add the flag to any citation-lookup fill search inside `retrieve_routed`'s lookup branch (exact-match route stays unfiltered).

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_orchestrator.py -q && .venv/bin/python -m pytest -q`
Expected: PASS; full suite no regressions (backstop keeps behavior identical on the un-backfilled test doubles).

- [ ] **Step 6: Commit**

```bash
git add src/rag/orchestrator.py src/rag/citation_lookup.py tests/test_orchestrator.py
git commit -m "feat(M2.5): route hybrid/vector retrieval through server-side is_in_effect filter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Reranker latency P95 harness

A measure-only script producing P50/P95/P99 rerank latency across candidate-pool sizes on the current machine. No pipeline behavior changes.

**Files:**
- Create: `scripts/bench_reranker.py`
- Output: `data/eval_reports/reranker_bench_<ts>.{md,json}`

**Interfaces:**
- Consumes: `FlagEmbedding.FlagReranker` (same model/device as `orchestrator._get_reranker`: `BAAI/bge-reranker-v2-m3`, `use_fp16=True`). Reuses `LegalVectorStore` to sample real chunk texts.

- [ ] **Step 1: Write the harness**

Create `scripts/bench_reranker.py`:

```python
"""
Reranker latency harness (M2.6). Measures BGE-Reranker-v2 P50/P95/P99 over
candidate pools of size {10,20,50,100} using REAL chunk texts sampled from a
collection. Measure-only — informs whether to cap candidates at M5.

Usage: .venv/bin/python scripts/bench_reranker.py --collection luat_law_ctx_v3 --repeats 20
"""
import argparse, json, os, sys, time, statistics
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv; load_dotenv()

POOL_SIZES = [10, 20, 50, 100]
QUERIES = [
    "Tội trộm cắp tài sản bị xử phạt như thế nào?",
    "Thời hạn báo trước khi đơn phương chấm dứt hợp đồng lao động?",
    "Điều kiện thành lập doanh nghiệp tư nhân là gì?",
    "Thủ tục cấp giấy chứng nhận quyền sử dụng đất?",
]

def _sample_chunks(collection, n):
    from src.db.qdrant_client import LegalVectorStore
    s = LegalVectorStore(collection_name=collection)
    points, _ = s.client.scroll(collection_name=collection, limit=n,
                                with_payload=["chunk_text"], with_vectors=False)
    return [ (p.payload or {}).get("chunk_text", "")[:512] for p in points if (p.payload or {}).get("chunk_text") ]

def _percentile(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

def main(collection, repeats):
    from FlagEmbedding import FlagReranker
    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    pool = _sample_chunks(collection, max(POOL_SIZES))
    if len(pool) < max(POOL_SIZES):
        print(f"⚠️ only {len(pool)} chunks available; capping pool sizes")
    rows = []
    for size in POOL_SIZES:
        if size > len(pool):
            continue
        passages = pool[:size]
        # warm-up
        reranker.compute_score([[QUERIES[0], passages[0]]], normalize=True)
        lat = []
        for _ in range(repeats):
            q = QUERIES[_ % len(QUERIES)]
            pairs = [[q, p] for p in passages]
            t0 = time.perf_counter()
            reranker.compute_score(pairs, normalize=True)
            lat.append((time.perf_counter() - t0) * 1000)  # ms
        rows.append({"pool": size, "p50": _percentile(lat, 50), "p95": _percentile(lat, 95),
                     "p99": _percentile(lat, 99), "mean": statistics.mean(lat),
                     "pairs_per_s": size / (statistics.mean(lat) / 1000)})
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs("data/eval_reports", exist_ok=True)
    meta = {"device": device, "repeats": repeats, "collection": collection, "rows": rows}
    with open(f"data/eval_reports/reranker_bench_{ts}.json", "w") as f:
        json.dump(meta, f, indent=2)
    md = [f"# Reranker latency ({device}, repeats={repeats}, {collection})", "",
          "| pool | P50 ms | P95 ms | P99 ms | mean ms | pairs/s |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['pool']} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} | "
                  f"{r['mean']:.0f} | {r['pairs_per_s']:.1f} |")
    if rows and rows[-1]["p95"] > 3000:
        md += ["", "> ⚠️ P95 at largest pool exceeds 3s — recommend capping candidates "
               "(e.g. top-10) before rerank at M5 CPU deploy."]
    report = "\n".join(md)
    with open(f"data/eval_reports/reranker_bench_{ts}.md", "w") as f:
        f.write(report + "\n")
    print(report)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="luat_law_ctx_v3")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    main(a.collection, a.repeats)
```

- [ ] **Step 2: Run the harness**

Run: `.venv/bin/python scripts/bench_reranker.py --collection luat_law_ctx_v3 --repeats 20`
Expected: prints a markdown table with P50/P95/P99 for pool sizes present; writes `data/eval_reports/reranker_bench_<ts>.{md,json}`. Sanity: P95 ≥ P50 for every row.

- [ ] **Step 3: Commit (script + report artifact)**

```bash
git add scripts/bench_reranker.py data/eval_reports/reranker_bench_*.md data/eval_reports/reranker_bench_*.json
git commit -m "feat(M2.6): reranker latency P95 harness + baseline report

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Update milestone status

**Files:**
- Modify: `docs/Plan_Giai_Doan_Cuoi_2026-07-10.md` (§7 status table)

- [ ] **Step 1: Update the M2 row**

Change the M2 row to note scale-prep done; ingest/eval deferred. Concretely set the M2 status cell to `🟡 scale-prep xong` with a note: "2.3 citation server-side, 2.4 SQLite store, 2.5 is_in_effect payload filter (+ backfill), 2.6 reranker bench — done <date>, branch m2-scale-prep. Còn 2.1 capacity gate / 2.2 ingest 600 / 2.7 eval+recalibrate — giai đoạn ingest (Kaggle GPU + Qdrant)."

- [ ] **Step 2: Commit**

```bash
git add docs/Plan_Giai_Doan_Cuoi_2026-07-10.md
git commit -m "docs(M2): mark scale-prep done, ingest/eval deferred

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Spec 2.3 (server-side citation) → Tasks 3 + 4 ✓
- Spec 2.4 (SQLite store) → Tasks 1 + 2 ✓
- Spec 2.5 (is_in_effect write + server filter, citation excluded, at_date client-side) → Tasks 5 + 6 ✓ (at_date deliberately untouched — retriever.py unchanged)
- Spec 2.6 (reranker P95 harness) → Task 7 ✓
- Spec §4 cross-cutting (branch, per-task commit, no live mutation, full suite) → Global Constraints + each task ✓
- Spec §4 status update → Task 8 ✓

**Placeholder scan:** All test/impl steps carry complete code. The one soft spot — the exact insertion point of the ingest backfill (Task 5 Step 6) and the exact set of orchestrator retrieval call sites (Task 6 Step 4) — is bounded by an explicit grep instruction and made safe by the client-side backstop (`_filter_in_effect` stays), so partial coverage cannot regress correctness.

**Type consistency:** `get_document_index() -> Dict[int, Dict]` (Task 3) is consumed as `meta.get("document_title"/"document_number"/"sector")` in Task 4 ✓. `scroll_article(doc_ids, article_number, cap)` returns `List[SearchResult]`, consumed by `lookup()` ✓. `_point_to_result` defined in Task 3, used in Task 4 lenient fallback ✓. `exclude_superseded` kwarg defined in Task 5, consumed in Task 6 ✓. `set_is_in_effect(doc_id, value)` defined Task 5, used by backfill script + ingest ✓. `import_json_dir` defined Task 2, used in Step 5 migration ✓.
