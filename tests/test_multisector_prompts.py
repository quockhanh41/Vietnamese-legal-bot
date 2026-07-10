"""
Tests for M1.4 — multi-sector prompts + corpus summary shapes.

Covers:
  - CitationLookup.corpus_summary(): sector-grouped when the collection has
    `sector` payloads (curated corpora, luat_law_ctx_v3+), flat "- title"
    bullet list when it doesn't (older collections — pinned separately by
    tests/test_scope_gate.py, unchanged).
  - CitationLookup.corpus_summary_compact(): sector names + counts for
    user-facing refusal text; falls back to the full list without sectors.
  - OUT_OF_SCOPE_ANSWER_TEMPLATE renders with the compact summary.
  - COMPLEXITY_CLASSIFIER_PROMPT / SYSTEM_PROMPT carry the multi-sector
    scope rules and the văn bản hierarchy principle (M1.4 rewrite markers).

Pure-Python: no network, no Qdrant, no models.

Run: .venv/bin/python -m pytest tests/test_multisector_prompts.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.prompts import COMPLEXITY_CLASSIFIER_PROMPT, SYSTEM_PROMPT
from src.rag.citation_lookup import CitationLookup
from src.rag.confidence import OUT_OF_SCOPE_ANSWER_TEMPLATE


FAKE_ALIASES = {
    296661: "Bộ luật hình sự 2015",
    296215: "Bộ luật dân sự 2015",
    333670: "Bộ luật Lao động 2019",
    427301: "Luật Doanh nghiệp 2020",
    523642: "Luật Đất đai 2024",
    238640: "Luật Hôn nhân và gia đình 2014",
    262379: "Nghị định 126/2014/NĐ-CP hướng dẫn thi hành Luật Hôn nhân và gia đình",
}

FAKE_SECTORS = {
    296661: "hinh_su",
    296215: "dan_su",
    333670: "lao_dong",
    427301: "doanh_nghiep",
    523642: "dat_dai",
    238640: "hngd",
    262379: "hngd",
}


def _lookup(sectors=None):
    lookup = CitationLookup(vector_store=None)
    lookup._alias_index = dict(FAKE_ALIASES)
    lookup._sector_index = dict(sectors) if sectors is not None else None
    return lookup


# ── corpus_summary (classifier prompt shape) ────────────────────────────────

def test_corpus_summary_groups_by_sector_when_sectors_known():
    summary = _lookup(FAKE_SECTORS).corpus_summary()
    # Sector headings with counts...
    assert "◆ Hình sự (1 văn bản):" in summary
    assert "◆ Hôn nhân và gia đình (2 văn bản):" in summary
    # ...and every title still listed under them (the scope gate needs the
    # actual titles, not just sector names).
    for title in FAKE_ALIASES.values():
        assert title in summary


def test_corpus_summary_stays_flat_without_sector_info():
    summary = _lookup(sectors=None).corpus_summary()
    assert summary.startswith("- ")
    assert "◆" not in summary
    for title in FAKE_ALIASES.values():
        assert title in summary


def test_corpus_summary_unknown_sector_label_goes_to_khac():
    sectors = dict(FAKE_SECTORS)
    sectors[296661] = "thue"  # label not in _SECTOR_DISPLAY
    summary = _lookup(sectors).corpus_summary()
    assert "◆ Khác (1 văn bản):" in summary
    assert "Bộ luật hình sự 2015" in summary


# ── corpus_summary_compact (user-facing refusal shape) ──────────────────────

def test_compact_summary_shows_counts_not_titles():
    compact = _lookup(FAKE_SECTORS).corpus_summary_compact()
    assert "7 văn bản" in compact
    assert "6 lĩnh vực" in compact
    assert "Hôn nhân và gia đình (2 văn bản)" in compact
    # A 100-doc corpus must not dump titles into a chat message — even at 7
    # docs the compact shape holds: no individual title appears.
    assert "Bộ luật hình sự 2015" not in compact


def test_compact_summary_falls_back_to_full_list_without_sectors():
    lookup = _lookup(sectors=None)
    assert lookup.corpus_summary_compact() == lookup.corpus_summary()


def test_compact_summary_is_cached():
    lookup = _lookup(FAKE_SECTORS)
    first = lookup.corpus_summary_compact()
    lookup._alias_index[999999] = "Should not appear"
    lookup._sector_index[999999] = "hinh_su"
    assert lookup.corpus_summary_compact() == first


def test_out_of_scope_template_renders_with_compact_summary():
    compact = _lookup(FAKE_SECTORS).corpus_summary_compact()
    answer = OUT_OF_SCOPE_ANSWER_TEMPLATE.format(corpus_summary=compact)
    assert "ngoài phạm vi" in answer
    assert "6 lĩnh vực" in answer
    # The old criminal-corpus-only hint must be gone: suggestions now span
    # sectors instead of assuming everything is about criminal liability.
    assert "quyền lợi người lao động" in answer


# ── prompt content pins (M1.4 rewrite) ──────────────────────────────────────

def test_classifier_prompt_has_dynamic_slots():
    assert "{corpus_summary}" in COMPLEXITY_CLASSIFIER_PROMPT
    assert "{question}" in COMPLEXITY_CLASSIFIER_PROMPT


def test_classifier_prompt_no_longer_assumes_criminal_only_corpus():
    """The M0-era rule block opened with 'Cơ sở dữ liệu chủ yếu là LUẬT HÌNH
    SỰ' — false for the 6-sector corpus and it actively taught the gate to
    refuse in-scope civil/labor questions."""
    assert "chủ yếu là LUẬT HÌNH SỰ" not in COMPLEXITY_CLASSIFIER_PROMPT


def test_classifier_prompt_teaches_procedure_in_corpus_is_in_scope():
    """M0's blanket 'thủ tục hành chính -> OUT' rule is wrong now that the
    corpus carries procedural decrees (NĐ 01/2021 đăng ký doanh nghiệp,
    NĐ 101/2024 đăng ký đất đai) — the few-shot must teach the distinction."""
    assert "đăng ký thành lập công ty" in COMPLEXITY_CLASSIFIER_PROMPT
    assert "Vượt đèn đỏ" in COMPLEXITY_CLASSIFIER_PROMPT  # procedure NOT covered -> OUT


def test_classifier_prompt_keeps_criminal_nuance_rule():
    """Still true and still the subtlest rule: criminal liability questions
    about behaviors from uncovered sectors are IN (BLHS has the tội danh)."""
    assert "TRÁCH NHIỆM HÌNH SỰ" in COMPLEXITY_CLASSIFIER_PROMPT


def test_classifier_prompt_warns_about_known_dataset_gaps():
    """Nhà ở / Kinh doanh BĐS were excluded from the corpus (their 2023
    replacements are missing from the dataset) — đất đai questions are IN
    but nhà ở questions must be OUT, and the gate can't know that without
    being told."""
    assert "Luật Nhà ở" in COMPLEXITY_CLASSIFIER_PROMPT
    assert "chung cư" in COMPLEXITY_CLASSIFIER_PROMPT


def test_system_prompt_states_document_hierarchy():
    assert "THỨ BẬC VĂN BẢN" in SYSTEM_PROMPT
    assert "Nghị định" in SYSTEM_PROMPT
