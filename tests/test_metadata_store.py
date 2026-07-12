"""DocumentMeta temporal-status defaults — fail-open on unknown.

Measured failure on the 100-doc corpus (2026-07-12): DateExtractor found no
effective_date for Luật Đầu tư 2025 and NĐ 144/2021, the enricher therefore
never set is_in_effect, and the old `False` default marked two CURRENT
documents superseded — the temporal filter (orchestrator: excludes only
`is_in_effect is False`) then hid them from every answer. Unknown must be
None (kept by the filter), not False (positively-known-superseded).

Run: .venv/bin/python -m pytest tests/test_metadata_store.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.metadata_store import DocumentMeta


def test_is_in_effect_defaults_to_unknown_not_false():
    meta = DocumentMeta(doc_id=681550, title="Luật Đầu tư 2025")
    assert meta.is_in_effect is None


def test_unknown_status_survives_serialization_round_trip():
    meta = DocumentMeta(doc_id=425471, title="Nghị định 144/2021/NĐ-CP")
    restored = DocumentMeta.from_dict(meta.to_dict())
    assert restored.is_in_effect is None


def test_explicit_false_is_preserved():
    meta = DocumentMeta(doc_id=215628, title="Luật việc làm 2013", is_in_effect=False)
    restored = DocumentMeta.from_dict(meta.to_dict())
    assert restored.is_in_effect is False
