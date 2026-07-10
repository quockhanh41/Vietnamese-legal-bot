"""
Unit tests for src/rag/citation_lookup.py — M1.3 multisector extensions:
document-number parsing/resolution, the generalized abbreviation table, and
the generalized "bộ luật <tên>" / "luật <tên>" doc-hint pattern, now that the
corpus spans 100 documents across 6 sectors (see scripts/select_corpus.py)
instead of just 6 hình sự documents.

Pure-Python parsing + doc resolution against mocked alias/doc-number indexes
built from *real* corpus_100docs titles and document_number values, plus one
scroll-fake for CitationLookup._load_collection_cache itself. No network, no
Qdrant, no models — same no-I/O guarantee as tests/test_citation_lookup.py.

Run: .venv/bin/python -m pytest tests/test_citation_lookup_multisector.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.citation_lookup import CitationLookup, parse_citation


# ── Fixtures: real titles/numbers pulled from data/raw/corpus_100docs.parquet ──
# (scripts/select_corpus.py CORPUS list — doc_id, sector, exact title/number).

REAL_TITLES = {
    # hình sự
    46056: "Bộ Luật Hình sự 1999",
    296661: "Bộ luật hình sự 2015",
    296884: "Bộ luật tố tụng hình sự 2015",
    354053: "Luật sửa đổi Bộ luật Hình sự 2017",
    387991: "Luật Thi hành án hình sự 2019",
    686325: "Luật Thi hành án hình sự 2025",
    # dân sự
    296215: "Bộ luật dân sự 2015",
    296861: "Bộ luật tố tụng dân sự 2015",
    7022: "Luật Sở hữu trí tuệ 2005",
    458435: "Luật Sở hữu trí tuệ sửa đổi 2022",
    82197: "Luật thi hành án dân sự 2008",
    259728: "Luật Thi hành án dân sự sửa đổi 2014",
    628717: "Luật Thi hành án dân sự 2025",
    # lao động
    142187: "Bộ Luật lao động 2012",
    333670: "Bộ luật Lao động 2019",
    459400: "Nghị định 145/2020/NĐ-CP hướng dẫn Bộ luật Lao động về điều kiện lao động và quan hệ lao động",
    281961: "Luật an toàn, vệ sinh lao động 2015",
    259700: "Luật Bảo hiểm xã hội 2014",
    557190: "Luật Bảo hiểm xã hội 2024",
    82196: "Luật bảo hiểm y tế 2008",
    505750: "Luật Bảo hiểm y tế sửa đổi 2024",
    # doanh nghiệp
    259730: "Luật Doanh nghiệp 2014",
    427301: "Luật Doanh nghiệp 2020",
    659899: "Luật Doanh nghiệp sửa đổi 2025",
    # đất đai
    215836: "Luật đất đai 2013",
    523642: "Luật Đất đai 2024",
    # hôn nhân gia đình
    46450: "Luật Hôn nhân và Gia đình 2000",
    238640: "Luật Hôn nhân và gia đình 2014",
    262379: "Nghị định 126/2014/NĐ-CP hướng dẫn thi hành Luật Hôn nhân và gia đình",
    490095: "Luật Phòng, chống bạo lực gia đình 2022",
}

REAL_DOC_NUMBERS = {
    46056: "15/1999/QH10",
    296661: "100/2015/QH13",
    296884: "101/2015/QH13",
    354053: "12/2017/QH14",
    387991: "41/2019/QH14",
    686325: "127/2025/QH15",
    296215: "91/2015/QH13",
    296861: "92/2015/QH13",
    7022: "50/2005/QH11",
    458435: "07/2022/QH15",
    82197: "26/2008/QH12",
    259728: "64/2014/QH13",
    628717: "106/2025/QH15",
    142187: "10/2012/QH13",
    333670: "45/2019/QH14",
    459400: "145/2020/NĐ-CP",
    281961: "84/2015/QH13",
    259700: "58/2014/QH13",
    557190: "41/2024/QH15",
    82196: "25/2008/QH12",
    505750: "51/2024/QH15",
    259730: "68/2014/QH13",
    427301: "59/2020/QH14",
    659899: "76/2025/QH15",
    215836: "45/2013/QH13",
    523642: "31/2024/QH15",
    46450: "22/2000/QH10",
    238640: "52/2014/QH13",
    262379: "126/2014/NĐ-CP",
    490095: "13/2022/QH15",
}


def _lookup_with_fake_index(aliases=None, doc_numbers=None):
    """Build a CitationLookup with pre-populated indexes, bypassing Qdrant —
    same pattern as tests/test_citation_lookup.py's _lookup_with_fake_aliases,
    extended with the new document_number index."""
    lookup = CitationLookup(vector_store=None)
    lookup._alias_index = dict(aliases if aliases is not None else REAL_TITLES)
    lookup._doc_number_index = dict(doc_numbers if doc_numbers is not None else REAL_DOC_NUMBERS)
    return lookup


# ── parse_citation: document-number extraction ──────────────────────────────

def test_parse_citation_doc_number_qh_format():
    parsed = parse_citation("Điều 10 Luật Doanh nghiệp 59/2020/QH14 quy định gì?")
    assert parsed is not None
    assert parsed.doc_number == "59/2020/QH14"


def test_parse_citation_doc_number_nd_cp_with_diacritic():
    parsed = parse_citation("Điều 5 Nghị định 145/2020/NĐ-CP quy định gì?")
    assert parsed is not None
    assert parsed.doc_number == "145/2020/NĐ-CP"


def test_parse_citation_doc_number_nd_cp_no_diacritic_uppercase():
    parsed = parse_citation("Điều 5 Nghị định 145/2020/ND-CP quy định gì?")
    assert parsed is not None
    assert parsed.doc_number == "145/2020/NĐ-CP"


def test_parse_citation_doc_number_nd_cp_lowercase_no_diacritic():
    parsed = parse_citation("điều 5 nghị định 145/2020/nd-cp quy định gì?")
    assert parsed is not None
    assert parsed.doc_number == "145/2020/NĐ-CP"


def test_parse_citation_doc_number_lowercase_qh():
    parsed = parse_citation("Điều 10 luật doanh nghiệp 59/2020/qh14 quy định gì?")
    assert parsed is not None
    assert parsed.doc_number == "59/2020/QH14"


def test_parse_citation_no_doc_number_is_none():
    parsed = parse_citation("Điều 173 Bộ luật Hình sự 2015 quy định về tội gì?")
    assert parsed is not None
    assert parsed.doc_number is None


# ── resolve_doc: doc_number is an absolute-priority signal ─────────────────

def test_resolve_doc_number_exact_match_wins_over_mismatched_hint_and_year():
    lookup = _lookup_with_fake_index()
    # hint/year point at Bộ luật lao động 2012; doc_number points at Luật
    # Doanh nghiệp 2020 (427301) — doc_number must win outright.
    resolved = lookup.resolve_doc("bộ luật lao động", 2012, doc_number="59/2020/QH14")
    assert resolved == [427301]


def test_resolve_doc_number_case_insensitive_and_no_diacritic_variant_matches():
    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc(None, None, doc_number="145/2020/nd-cp")
    assert resolved == [459400]


def test_resolve_doc_number_no_match_falls_back_to_hint_year():
    lookup = _lookup_with_fake_index()
    # A doc_number outside the corpus falls back to normal hint/year logic
    # rather than resolving to nothing.
    resolved = lookup.resolve_doc("luật doanh nghiệp", 2020, doc_number="999/2099/QH99")
    assert resolved == [427301]


def test_resolve_doc_number_none_uses_hint_year_as_before():
    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc("luật đất đai", 2013, doc_number=None)
    assert resolved == [215836]


# ── new abbreviations: each must expand to something that matches its real
#    corpus title (per-sector coverage, per M1.3 plan §6.2 #8) ─────────────

_ABBREV_CASES = [
    ("blds", 296215),     # Bộ luật dân sự 2015
    ("blttds", 296861),   # Bộ luật tố tụng dân sự 2015
    ("ttds", 296861),
    ("bllđ", 333670),     # Bộ luật Lao động 2019 (with year disambiguation)
    ("blld", 333670),
    ("ldn", 427301),      # Luật Doanh nghiệp 2020
    ("lđđ", 523642),      # Luật Đất đai 2024
    ("ldd", 523642),
    ("hngđ", 238640),     # Luật Hôn nhân và gia đình 2014
    ("hngd", 238640),
    ("bhxh", 557190),     # Luật Bảo hiểm xã hội 2024
    ("bhyt", 505750),     # Luật Bảo hiểm y tế sửa đổi 2024
    ("shtt", 458435),     # Luật Sở hữu trí tuệ sửa đổi 2022
    ("atvslđ", 281961),   # Luật an toàn, vệ sinh lao động 2015
    ("atvsld", 281961),
    ("thads", 628717),    # Luật Thi hành án dân sự 2025
    ("thahs", 686325),    # Luật Thi hành án hình sự 2025
    ("pcblgđ", 490095),   # Luật Phòng, chống bạo lực gia đình 2022
    ("pcblgd", 490095),
]


def test_new_abbreviations_expand_and_match_real_titles():
    """The target law must rank FIRST. Guiding decrees whose title quotes
    the law name (e.g. "Nghị định 126/2014/NĐ-CP hướng dẫn thi hành Luật
    Hôn nhân và gia đình" — which also carries "2014" in its số hiệu, so
    the year filter can't drop it) are allowed to follow, ranked after —
    resolve_doc ranks prefix-match > amendment > referencing instead of
    excluding, because amendment docs carry quoted suffixed articles
    ("Điều 12a") that lookup must still be able to scan (M1.2b)."""
    lookup = _lookup_with_fake_index()
    for abbrev, expected_year_doc in _ABBREV_CASES:
        year = int(REAL_TITLES[expected_year_doc].strip()[-4:])
        resolved = lookup.resolve_doc(abbrev, year)
        assert resolved[0] == expected_year_doc, (
            f"{abbrev!r} (year={year}) -> {resolved}, want [{expected_year_doc}] first"
        )


def test_new_abbreviations_without_year_are_ambiguous_among_real_matches():
    """Without a year, an abbreviation that has multiple corpus versions
    (e.g. BLLĐ 2012 and 2019) must resolve to *all* of them — the two law
    versions ranked ahead of the guiding decree (NĐ 145/2020) whose title
    merely references the code."""
    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc("bllđ", None)
    assert set(resolved) == {142187, 333670, 459400}
    assert set(resolved[:2]) == {142187, 333670}  # laws first
    assert resolved[2] == 459400                  # referencing decree last


# ── full-name doc-hint pattern: "bộ luật X" / "luật X", multisector ────────

def test_parse_and_resolve_bo_luat_lao_dong_with_year():
    parsed = parse_citation("Điều 168 Bộ luật Lao động 2019 quy định gì?")
    assert parsed is not None
    assert parsed.primary.number == "168"
    assert parsed.year == 2019, "must be the year after the hint, not the article number 168"
    assert parsed.doc_hint == "bộ luật lao động"

    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc(parsed.doc_hint, parsed.year)
    assert resolved == [333670]


def test_parse_and_resolve_luat_doanh_nghiep_hint():
    parsed = parse_citation("Điều 10 theo Luật Doanh nghiệp thì công ty phải làm gì?")
    assert parsed is not None
    assert parsed.doc_hint == "luật doanh nghiệp", "must not swallow the trailing 'thì...' clause"

    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc(parsed.doc_hint, parsed.year)
    assert set(resolved) == {259730, 427301, 659899}  # ambiguous without a year


def test_parse_and_resolve_luat_hon_nhan_va_gia_dinh_hint():
    parsed = parse_citation("Điều 27 Luật Hôn nhân và gia đình quy định về ly hôn?")
    assert parsed is not None
    assert parsed.doc_hint == "luật hôn nhân và gia đình"

    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc(parsed.doc_hint, parsed.year)
    # Both law versions rank first; NĐ 126/2014 (title references the law)
    # is a candidate too but must come last — its own "Điều 27" is about a
    # different subject than the law's Điều 27.
    assert set(resolved) == {46450, 238640, 262379}
    assert set(resolved[:2]) == {46450, 238640}
    assert resolved[2] == 262379


def test_parse_and_resolve_nghi_dinh_by_doc_number_ignores_missing_hint():
    parsed = parse_citation("Điều 5 Nghị định 145/2020/NĐ-CP hướng dẫn Bộ luật Lao động")
    assert parsed is not None
    assert parsed.doc_number == "145/2020/NĐ-CP"

    lookup = _lookup_with_fake_index()
    resolved = lookup.resolve_doc(parsed.doc_hint, parsed.year, parsed.doc_number)
    assert resolved == [459400]


# ── year disambiguation across two same-named docs, non-hình-sự sector ─────

def test_year_disambiguates_luat_dat_dai_2013_vs_2024():
    lookup = _lookup_with_fake_index()
    resolved_2013 = lookup.resolve_doc("luật đất đai", 2013)
    assert resolved_2013 == [215836]
    resolved_2024 = lookup.resolve_doc("luật đất đai", 2024)
    assert resolved_2024 == [523642]


def test_parse_luat_dat_dai_2013_question_gets_right_year():
    parsed = parse_citation("Điều 12 Luật đất đai 2013 quy định về gì?")
    assert parsed is not None
    assert parsed.year == 2013


def test_parse_luat_dat_dai_2024_question_gets_right_year():
    parsed = parse_citation("Điều 12 Luật Đất đai 2024 quy định về gì?")
    assert parsed is not None
    assert parsed.year == 2024


# ── no citation at all ──────────────────────────────────────────────────────

def test_parse_citation_no_citation_multisector_sentence_is_none():
    assert parse_citation("Thủ tục thành lập doanh nghiệp gồm những bước nào?") is None


def test_parse_citation_no_citation_hon_nhan_sentence_is_none():
    assert parse_citation("Điều kiện kết hôn theo pháp luật Việt Nam là gì?") is None


# ── _load_collection_cache: doc_number caching from a fake scroll client ───

class _FakePoint:
    def __init__(self, point_id, payload):
        self.id = point_id
        self.payload = payload


class _FakeScrollClient:
    """Minimal stand-in for qdrant_client.QdrantClient — CitationLookup only
    ever calls .scroll() on it. Returns every point on the first call and
    signals end-of-scroll (offset=None) immediately, since these fixtures
    are small enough to fit in one page."""

    def __init__(self, points):
        self._points = points

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        if offset is not None:
            return [], None
        return self._points, None


class _FakeVectorStore:
    def __init__(self, points, collection_name="test_collection"):
        self.client = _FakeScrollClient(points)
        self.collection_name = collection_name


def test_load_collection_cache_populates_doc_number_index_from_payload():
    points = [
        _FakePoint("p1", {
            "document_id": 427301,
            "document_title": "Luật Doanh nghiệp 2020",
            "document_number": "59/2020/QH14",
            "chunk_text": "Điều 1. ...",
        }),
        _FakePoint("p2", {
            "document_id": 459400,
            "document_title": "Nghị định 145/2020/NĐ-CP hướng dẫn Bộ luật Lao động",
            "document_number": "145/2020/nd-cp",  # raw payload spelling variant
            "chunk_text": "Điều 1. ...",
        }),
    ]
    lookup = CitationLookup(vector_store=_FakeVectorStore(points))

    resolved = lookup.resolve_doc(None, None, doc_number="59/2020/QH14")
    assert resolved == [427301]

    # The raw payload spelling ("nd-cp", no diacritic) must have been
    # normalized to canonical "NĐ-CP" at cache-build time too.
    resolved2 = lookup.resolve_doc(None, None, doc_number="145/2020/NĐ-CP")
    assert resolved2 == [459400]

    # corpus_summary() (built from _alias_index) is untouched by the new
    # doc_number cache — still the same bullet-list contract.
    summary = lookup.corpus_summary()
    assert "- Luật Doanh nghiệp 2020" in summary
