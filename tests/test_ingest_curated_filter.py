"""
Unit tests for the curated-corpus filter mode in scripts/ingest.py
(M1.2) — filter_row(row, curated=...) and the `sector` metadata pass-through
wired in ingest(). Pure Python: no network, no Qdrant, no model loading
(BGEEmbedder is lazy-loaded inside ingest() and is never reached by these
tests — they only exercise filter_row() directly and check the sector
pass-through logic that mirrors what ingest() does).

Run: .venv/bin/python -m pytest tests/test_ingest_curated_filter.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from ingest import filter_row
from src.data.constants import MIN_CONTENT_LENGTH


def _row(**overrides):
    base = {
        "id": 1,
        "legal_type": "Luật",
        "issuance_date": "01/01/2020",
        "content": "x" * (MIN_CONTENT_LENGTH + 10),
        "title": "Some Law",
    }
    base.update(overrides)
    return base


# ── default (non-curated) mode: unchanged behavior ──────────────────────────

def test_default_mode_keeps_luat():
    assert filter_row(_row(legal_type="Luật")) is True


def test_default_mode_drops_nghi_dinh():
    """Regression guard for the exact bug M1.2 describes: without curated
    mode, a Nghị định must still be silently filtered out (legacy
    behavior, unchanged for legal_docs_merged.parquet)."""
    assert filter_row(_row(legal_type="Nghị định")) is False


def test_default_mode_drops_short_content():
    assert filter_row(_row(content="too short")) is False


def test_default_mode_drops_missing_content():
    assert filter_row(_row(content=None)) is False


def test_default_mode_curated_flag_defaults_false():
    """curated must default to False so every existing call site
    (filter_row(row_dict) with no second arg) keeps its old behavior."""
    assert filter_row(_row(legal_type="Nghị định")) is filter_row(
        _row(legal_type="Nghị định"), curated=False
    )


# ── curated mode ─────────────────────────────────────────────────────────

def test_curated_mode_keeps_nghi_dinh():
    """The whole point of M1.2 Requirement 2: corpus_100docs.parquet has 29
    Nghị định that must NOT be dropped in curated mode."""
    assert filter_row(_row(legal_type="Nghị định"), curated=True) is True


def test_curated_mode_keeps_luat_too():
    assert filter_row(_row(legal_type="Luật"), curated=True) is True


def test_curated_mode_still_enforces_min_content_length():
    """Curated mode must not become a rubber stamp — the minimal content
    guard still applies."""
    assert filter_row(_row(content="too short"), curated=True) is False


def test_curated_mode_drops_missing_content():
    assert filter_row(_row(content=None), curated=True) is False


def test_curated_mode_drops_non_string_content():
    assert filter_row(_row(content=12345), curated=True) is False


def test_curated_mode_ignores_legal_type_entirely():
    """Any legal_type value, even one outside ALL_LEGAL_TYPES, passes in
    curated mode as long as content is OK — curated corpora are
    pre-selected by scripts/select_corpus.py, so ingest.py must trust that
    selection rather than re-filtering by type."""
    assert filter_row(_row(legal_type="Something Unlisted"), curated=True) is True


# ── sector metadata pass-through (mirrors the wiring inside ingest()) ──────

def test_sector_column_present_implies_curated_detection():
    """ingest() auto-detects curated mode via `"sector" in dataset.schema.names`.
    This test documents/locks the row-dict-level equivalent: a row that HAS
    a sector key should be treated as curated-corpus data for filtering
    purposes in the caller."""
    row = _row(legal_type="Nghị định", sector="dat_dai")
    curated = "sector" in row
    assert filter_row(row, curated=curated) is True


def test_meta_sector_assignment_mirrors_ingest_wiring():
    """ingest() does `meta["sector"] = row_dict.get("sector")` for every
    chunk when curated. Exercise that exact expression directly (without
    invoking the full ingest() pipeline / embedder)."""
    row_dict = _row(sector="hinh_su")
    meta = {"chunk_text": "noi dung", "document_id": row_dict["id"]}
    curated = "sector" in row_dict
    if curated:
        meta["sector"] = row_dict.get("sector")
    assert meta["sector"] == "hinh_su"


def test_meta_sector_not_added_when_not_curated():
    row_dict = _row()  # no "sector" key
    meta = {"chunk_text": "noi dung", "document_id": row_dict["id"]}
    curated = "sector" in row_dict
    if curated:
        meta["sector"] = row_dict.get("sector")
    assert "sector" not in meta
