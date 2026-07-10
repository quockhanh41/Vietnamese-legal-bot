"""
Unit tests for src/data/ingest_checkpoint.py — SQLite-backed checkpoint/
resume tracking for scripts/ingest.py (M1.2). Pure stdlib sqlite3, no
network, no Qdrant, no models. Every test uses tmp_path so the real
data/ingest_progress/ directory is never touched.

Run: .venv/bin/python -m pytest tests/test_ingest_checkpoint.py -q
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.ingest_checkpoint import IngestCheckpoint


# ── is_done / mark_done_many ────────────────────────────────────────────────

def test_is_done_false_for_unknown_document(tmp_path):
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    assert ckpt.is_done(1) is False
    ckpt.close()


def test_mark_done_many_then_is_done_true(tmp_path):
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(1, 3), (2, 5)])
    assert ckpt.is_done(1) is True
    assert ckpt.is_done(2) is True
    assert ckpt.is_done(3) is False
    ckpt.close()


def test_mark_done_accepts_numpy_integer_types(tmp_path):
    """Regression (Fable verify 2026-07-10): ingest.py streams rows via
    pandas, whose row.to_dict() yields numpy.int64 for the id column —
    raw sqlite3 binding of np.int64 fails with IntegrityError "datatype
    mismatch" on the INTEGER PRIMARY KEY. The int() casts inside
    IngestCheckpoint must absorb this, or a real ingest run crashes at the
    first flush AFTER paying for embedding."""
    np = __import__("numpy")
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(np.int64(46056), np.int64(12))])
    assert ckpt.is_done(np.int64(46056)) is True
    assert ckpt.is_done(46056) is True  # same row, plain-int lookup
    assert ckpt.stats() == {
        "documents_done": 1,
        "chunks_done": 12,
        "path": os.path.join(str(tmp_path), "col.sqlite"),
    }
    ckpt.close()


def test_mark_done_many_empty_list_is_noop(tmp_path):
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([])
    assert ckpt.stats()["documents_done"] == 0
    ckpt.close()


def test_mark_done_many_is_idempotent_on_rerun(tmp_path):
    """Marking the same document done twice (e.g. a re-run without
    --resume, thanks to idempotent point IDs) must not error or double
    count — INSERT OR REPLACE semantics."""
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(1, 3)])
    ckpt.mark_done_many([(1, 3)])
    assert ckpt.is_done(1) is True
    assert ckpt.stats()["documents_done"] == 1
    assert ckpt.stats()["chunks_done"] == 3
    ckpt.close()


def test_mark_done_many_updates_n_chunks_on_rewrite(tmp_path):
    """Re-marking a document with a different n_chunks (e.g. chunker
    config changed between runs) overwrites the stored count rather than
    keeping the stale one."""
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(1, 3)])
    ckpt.mark_done_many([(1, 7)])
    assert ckpt.stats()["chunks_done"] == 7
    ckpt.close()


def test_mark_done_many_is_single_transaction(tmp_path):
    """All (doc_id, n_chunks) pairs passed together must land atomically —
    querying mid-batch isn't observable from a single-threaded test, but we
    can at least confirm all pairs from one call are present together."""
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    docs = [(i, i + 1) for i in range(50)]
    ckpt.mark_done_many(docs)
    assert all(ckpt.is_done(i) for i in range(50))
    assert ckpt.stats()["documents_done"] == 50
    ckpt.close()


# ── stats ────────────────────────────────────────────────────────────────

def test_stats_on_empty_checkpoint(tmp_path):
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    stats = ckpt.stats()
    assert stats["documents_done"] == 0
    assert stats["chunks_done"] == 0
    ckpt.close()


def test_stats_sums_chunks_across_documents(tmp_path):
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(1, 3), (2, 5), (3, 2)])
    stats = ckpt.stats()
    assert stats["documents_done"] == 3
    assert stats["chunks_done"] == 10
    ckpt.close()


# ── file location ────────────────────────────────────────────────────────

def test_checkpoint_file_created_at_expected_path(tmp_path):
    ckpt = IngestCheckpoint("my_collection", base_dir=str(tmp_path))
    expected = tmp_path / "my_collection.sqlite"
    assert expected.exists()
    assert ckpt.path == str(expected)
    ckpt.close()


def test_default_base_dir_is_data_ingest_progress():
    from src.data.ingest_checkpoint import DEFAULT_CHECKPOINT_DIR
    assert DEFAULT_CHECKPOINT_DIR == "data/ingest_progress"


def test_base_dir_created_if_missing(tmp_path):
    nested = tmp_path / "nested" / "dir"
    assert not nested.exists()
    ckpt = IngestCheckpoint("col", base_dir=str(nested))
    assert nested.exists()
    ckpt.close()


def test_different_collections_get_separate_files(tmp_path):
    ckpt_a = IngestCheckpoint("collection_a", base_dir=str(tmp_path))
    ckpt_b = IngestCheckpoint("collection_b", base_dir=str(tmp_path))
    ckpt_a.mark_done_many([(1, 1)])
    assert ckpt_a.is_done(1) is True
    assert ckpt_b.is_done(1) is False  # separate sqlite file, no bleed-over
    ckpt_a.close()
    ckpt_b.close()


# ── recreate ─────────────────────────────────────────────────────────────

def test_recreate_wipes_existing_progress(tmp_path):
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(1, 3), (2, 5)])
    ckpt.close()

    ckpt2 = IngestCheckpoint("col", base_dir=str(tmp_path), recreate=True)
    assert ckpt2.stats()["documents_done"] == 0
    assert ckpt2.is_done(1) is False
    ckpt2.close()


def test_recreate_false_preserves_existing_progress(tmp_path):
    """Sanity counterpart to the recreate test above: without recreate=True,
    re-opening the same collection's checkpoint must NOT lose progress —
    that's the entire point of --resume."""
    ckpt = IngestCheckpoint("col", base_dir=str(tmp_path))
    ckpt.mark_done_many([(1, 3)])
    ckpt.close()

    ckpt2 = IngestCheckpoint("col", base_dir=str(tmp_path), recreate=False)
    assert ckpt2.is_done(1) is True
    ckpt2.close()


def test_recreate_on_nonexistent_file_does_not_error(tmp_path):
    ckpt = IngestCheckpoint("brand_new", base_dir=str(tmp_path), recreate=True)
    assert ckpt.stats()["documents_done"] == 0
    ckpt.close()


# ── context manager ──────────────────────────────────────────────────────

def test_context_manager_closes_connection(tmp_path):
    with IngestCheckpoint("col", base_dir=str(tmp_path)) as ckpt:
        ckpt.mark_done_many([(1, 1)])
        path = ckpt.path
    # Connection closed — further use of the underlying sqlite3 connection
    # would raise; a fresh connection to the same file still sees the data.
    conn = sqlite3.connect(path)
    n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    assert n == 1
