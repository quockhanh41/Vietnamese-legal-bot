"""
SQLite-backed checkpoint/resume tracking for scripts/ingest.py (M1.2).

Why this exists even though upserts are already idempotent: since M0,
Qdrant point IDs are deterministic (uuid5 of "document_id:chunk_index" —
see compute_point_id() in src/db/qdrant_client.py), so re-running ingest
WITHOUT --resume is already safe — it just overwrites the same points
instead of duplicating them. This checkpoint is therefore purely a SPEED /
RESILIENCE optimization: on a flaky connection (Kaggle/Colab session drop,
Qdrant Cloud rate limiting), --resume lets a re-run skip documents that
were already embedded + upserted, instead of paying for BGE-M3 inference
again on unchanged rows.

A document counts as "done" only once ALL of its chunks have been
successfully upserted to Qdrant — see the caller in scripts/ingest.py for
how it batches document (doc_id, n_chunks) pairs and only calls
mark_done_many() right after the store.upsert_chunks() flush that contains
them succeeds (including the final flush after the streaming loop ends).
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, Tuple

# Relative to repo root — matches the existing convention in this project
# (e.g. ingest.py's default --data-path "data/raw/legal_docs_merged.parquet"),
# which assumes scripts are run from the repo root.
DEFAULT_CHECKPOINT_DIR = "data/ingest_progress"


class IngestCheckpoint:
    """Tracks which document_ids have been fully ingested for one collection.

    Backed by a small SQLite database at
    ``{base_dir}/{collection}.sqlite`` (default base_dir:
    ``data/ingest_progress/``). One file per collection, since different
    collections (different chunking strategies / corpora) track unrelated
    progress.
    """

    def __init__(
        self,
        collection: str,
        base_dir: str = DEFAULT_CHECKPOINT_DIR,
        recreate: bool = False,
    ):
        """
        Args:
            collection: Qdrant collection name — determines the sqlite
                filename, so progress is tracked per-collection.
            base_dir: Directory holding checkpoint files (created if
                missing). Overridable in tests (tmp_path) to avoid touching
                the real data/ingest_progress/ directory.
            recreate: If True, wipe any existing progress file for this
                collection first (mirrors --recreate on the Qdrant
                collection itself — an old checkpoint pointing at a
                collection that was just dropped and recreated would cause
                --resume to skip documents that no longer exist in Qdrant).
        """
        self.collection = collection
        self.base_dir = base_dir
        self.path = os.path.join(base_dir, f"{collection}.sqlite")

        os.makedirs(base_dir, exist_ok=True)

        if recreate and os.path.exists(self.path):
            os.remove(self.path)

        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY,
                n_chunks INTEGER NOT NULL,
                completed_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def is_done(self, doc_id: int) -> bool:
        """True if this document_id has already been fully upserted."""
        cur = self._conn.execute(
            "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
            (int(doc_id),),
        )
        return cur.fetchone() is not None

    def mark_done_many(self, docs: Iterable[Tuple[int, int]]) -> None:
        """Mark multiple documents done in a single transaction.

        Args:
            docs: iterable of (document_id, n_chunks) tuples. Idempotent —
                re-marking an already-done document_id (e.g. a re-run
                without --resume) just overwrites its row (INSERT OR
                REPLACE), it does not error or duplicate.
        """
        docs = list(docs)
        if not docs:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:  # commits on success, rolls back on exception
            self._conn.executemany(
                "INSERT OR REPLACE INTO documents "
                "(document_id, n_chunks, completed_at) VALUES (?, ?, ?)",
                # int() casts: callers stream rows via pandas, whose
                # row.to_dict() yields numpy.int64 — sqlite3 refuses to bind
                # those ("datatype mismatch" on the INTEGER PRIMARY KEY).
                [(int(doc_id), int(n_chunks), now) for doc_id, n_chunks in docs],
            )

    def stats(self) -> Dict:
        """Summary of progress recorded so far."""
        cur = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(n_chunks), 0) FROM documents"
        )
        n_docs, n_chunks = cur.fetchone()
        return {
            "documents_done": n_docs,
            "chunks_done": n_chunks,
            "path": self.path,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "IngestCheckpoint":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
