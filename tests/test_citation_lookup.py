"""
Unit tests for src/rag/citation_lookup.py — pure-Python parsing + doc
resolution against a mocked alias index. No network, no Qdrant, no models.

Run: .venv/bin/python -m pytest tests/test_citation_lookup.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.citation_lookup import CitationLookup, parse_citation


# ── parse_citation ──────────────────────────────────────────────────────────

def test_parse_citation_article_with_law_name_and_year():
    parsed = parse_citation("Điều 173 Bộ luật Hình sự 2015 quy định về tội gì?")
    assert parsed is not None
    assert parsed.primary.number == "173"
    assert parsed.year == 2015
    assert parsed.doc_hint is not None
    assert "hình sự" in parsed.doc_hint


def test_parse_citation_khoan_and_abbreviation_hint():
    parsed = parse_citation("Khoản 2 Điều 40 BLHS quy định gì?")
    assert parsed is not None
    assert parsed.primary.number == "40"
    assert parsed.primary.khoan == "2"
    assert parsed.doc_hint == "blhs"


def test_parse_citation_dieu_kien_is_not_a_citation():
    """'Điều kiện' must NOT match — no digit right after 'Điều'."""
    assert parse_citation("Điều kiện để được hưởng án treo là gì?") is None


def test_parse_citation_dieu_tra_is_not_a_citation():
    """'điều tra' (lowercase, mid-sentence) must NOT match either."""
    assert parse_citation("Cơ quan điều tra vụ án này là ai?") is None


def test_parse_citation_nonexistent_article_still_parses():
    """Existence is checked at lookup time, not parse time — 'Điều 999'
    parses fine even though the article doesn't exist in the corpus."""
    parsed = parse_citation("Điều 999 Bộ luật Hình sự quy định gì?")
    assert parsed is not None
    assert parsed.primary.number == "999"


def test_parse_citation_no_citation_at_all_is_none():
    assert parse_citation("Thế nào là tội cướp tài sản?") is None


def test_parse_citation_multi_article_keeps_all_first_is_primary():
    parsed = parse_citation("So sánh Điều 12 và Điều 173 Bộ luật Hình sự")
    assert parsed is not None
    assert [a.number for a in parsed.articles] == ["12", "173"]
    assert parsed.primary.number == "12"


def test_parse_citation_diem_extracted():
    parsed = parse_citation("Điểm a Khoản 1 Điều 51 Bộ luật Hình sự quy định gì?")
    assert parsed is not None
    assert parsed.primary.number == "51"
    assert parsed.primary.khoan == "1"
    assert parsed.primary.diem == "a"


def test_parse_citation_article_with_letter_suffix():
    parsed = parse_citation("Điều 12a Bộ luật Hình sự quy định gì?")
    assert parsed is not None
    assert parsed.primary.number == "12a"


def test_parse_citation_tths_hint_distinct_from_hinh_su():
    parsed = parse_citation("Điều 173 Bộ luật Tố tụng hình sự 2015 quy định về vấn đề gì?")
    assert parsed is not None
    assert "tố tụng" in parsed.doc_hint


# ── resolve_doc (mocked alias index — no Qdrant) ────────────────────────────

def _lookup_with_fake_aliases(aliases):
    """Build a CitationLookup with a pre-populated alias index, bypassing Qdrant."""
    lookup = CitationLookup(vector_store=None)
    lookup._alias_index = dict(aliases)
    return lookup


FAKE_ALIASES = {
    46056: "Bộ Luật Hình sự 1999",
    296661: "Bộ luật hình sự 2015",
    296884: "Bộ luật tố tụng hình sự 2015",
    354053: "Luật sửa đổi Bộ luật Hình sự 2017 (12/2017/QH14)",
    489113: "Luật sửa đổi Bộ luật Tố tụng hình sự 2021 (02/2021/QH15)",
    662377: "Luật sửa đổi Bộ luật Hình sự 2025 (86/2025/QH15)",
}


def test_resolve_doc_no_hint_returns_all_docs():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc(None, None)
    assert set(resolved) == set(FAKE_ALIASES.keys())


def test_resolve_doc_hint_and_year_disambiguates_hinh_su_2015_from_1999():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("bộ luật hình sự", 2015)
    assert resolved == [296661]


def test_resolve_doc_hint_and_year_1999():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("bộ luật hình sự", 1999)
    assert resolved == [46056]


def test_resolve_doc_hint_without_year_is_ambiguous_among_matches():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("bộ luật hình sự", None)
    # Matches every title containing "bộ luật hình sự" as a substring,
    # including the amendment laws that literally contain that phrase —
    # year is what disambiguates further.
    assert 296661 in resolved
    assert 46056 in resolved
    assert 296884 not in resolved  # "tố tụng" doc must not match


def test_resolve_doc_tths_hint_matches_only_tths_docs():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("tths", 2015)
    assert resolved == [296884]


def test_resolve_doc_blhs_abbreviation_expands():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("blhs", 2015)
    assert resolved == [296661]


def test_resolve_doc_no_match_falls_back_to_all_docs():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("luật đất đai", None)
    assert set(resolved) == set(FAKE_ALIASES.keys())


def test_resolve_doc_amendment_hint_matches_amendment_law_only():
    lookup = _lookup_with_fake_aliases(FAKE_ALIASES)
    resolved = lookup.resolve_doc("luật sửa đổi bộ luật hình sự", 2017)
    assert resolved == [354053]
