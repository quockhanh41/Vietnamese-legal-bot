"""
Tests for VersionResolver.find_replacement (M1.8b fix).

Measured failure on the 100-doc corpus (enrichment 2026-07-10): the original
name-only substring match (year stripped by _normalize_name) marked CURRENT
laws as replaced — 'Luật Doanh nghiệp 2020' matched LDN 2014's clause about
replacing LDN 2005 (an OLDER doc "replacing" a newer one), flagging
LDN 2020 / BLLĐ 2019 / LĐĐ 2024 / BHXH 2024 not-in-effect, which the
retriever's is_in_effect filter would then HIDE from every answer. Only 3/8
temporal pairs resolved correctly.

Pure-Python, no I/O.

Run: .venv/bin/python -m pytest tests/test_version_resolver.py -q
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.timeline.date_extractor import LawTimeline
from src.data.timeline.version_resolver import (
    VersionResolver,
    _extract_doc_numbers,
    _normalize_doc_number,
)


def _tl(doc_id, name, num=None, eff=None, replaced=None):
    return LawTimeline(
        doc_id=doc_id,
        law_name=name,
        doc_number=num,
        effective_date=eff,
        replaced_laws=replaced or [],
    )


# ── helpers ─────────────────────────────────────────────────────────────────

def test_extract_doc_numbers_scopes_to_replaced_target_only():
    """Numbers AFTER 'đã được sửa đổi, bổ sung ... theo' are the replaced
    law's AMENDERS (here: BLLĐ 2019 among them) — only the target before the
    marker counts as replaced."""
    clause = (
        "Luật Bảo hiểm xã hội số 58/2014/QH13 đã được sửa đổi, bổ sung một số "
        "điều theo Luật số 84/2015/QH13, Luật số 35/2018/QH14, Bộ luật số "
        "45/2019/QH14 (sau đây gọi chung là Luật số 58/2014/QH13)"
    )
    assert _extract_doc_numbers(clause) == {"58/2014/QH13"}


def test_extract_doc_numbers_without_marker_takes_all():
    clause = "Luật dân sự số 33/2005/QH11 hết hiệu lực kể từ ngày Luật này có hiệu lực"
    assert _extract_doc_numbers(clause) == {"33/2005/QH11"}


def test_normalize_doc_number_folds_variants():
    assert _normalize_doc_number("01/2021/nd-cp") == "1/2021/NĐ-CP"
    assert _normalize_doc_number("100/2015/QH13") == "100/2015/QH13"


# ── find_replacement ────────────────────────────────────────────────────────

def test_number_match_flags_old_version():
    """BLDS-2005 case: the 2015 clause says 'Luật dân sự số 33/2005/QH11'
    (missing the 'Bộ' prefix, so name matching used to MISS it) — the số
    hiệu must catch it."""
    old = _tl(2463, "Bộ luật Dân sự 2005", num="33/2005/QH11", eff=date(2006, 1, 1))
    new = _tl(296215, "Bộ luật dân sự 2015", num="91/2015/QH13",
              eff=date(2017, 1, 1), replaced=["Luật dân sự số 33/2005/QH11"])
    r = VersionResolver([old, new])
    hit = r.find_replacement(2463)
    assert hit is not None and hit.doc_id == 296215


def test_clause_naming_other_numbers_does_not_flag_newer_version():
    """LDN-2020 case: LDN 2014's clause replaces LDN **2005** by number.
    'Luật Doanh nghiệp 2020' shares the normalized name but must NOT be
    treated as replaced by the OLDER 2014 law."""
    ldn_2014 = _tl(259730, "Luật Doanh nghiệp 2014", num="68/2014/QH13",
                   eff=date(2015, 7, 1),
                   replaced=["Luật doanh nghiệp số 60/2005/QH11"])
    ldn_2020 = _tl(427301, "Luật Doanh nghiệp 2020", num="59/2020/QH14",
                   eff=date(2021, 1, 1))
    r = VersionResolver([ldn_2014, ldn_2020])
    assert r.find_replacement(427301) is None  # 2020 stays in effect
    # ...and 2014 is not flagged either (its replacement clause is about 2005,
    # which isn't in this corpus slice).
    assert r.find_replacement(259730) is None


def test_name_fallback_requires_newer_effective_date():
    """Clause with NO parseable number falls back to name matching, but only
    a strictly NEWER doc may count as the replacement."""
    old = _tl(1, "Luật Thú y 2004", eff=date(2004, 7, 1))
    new = _tl(2, "Luật Thú y 2015", eff=date(2016, 7, 1),
              replaced=["Pháp lệnh Thú y và Luật Thú y trước đây"])
    r = VersionResolver([old, new])
    hit = r.find_replacement(1)
    assert hit is not None and hit.doc_id == 2
    # Direction guard: the OLD doc's clause can never flag the NEW doc.
    old_claiming = _tl(3, "Luật Thú y 2004", eff=date(2004, 7, 1),
                       replaced=["Luật Thú y cũ"])
    newer = _tl(4, "Luật Thú y 2015", eff=date(2016, 7, 1))
    r2 = VersionResolver([old_claiming, newer])
    assert r2.find_replacement(4) is None


def test_name_fallback_without_dates_is_conservative_no_match():
    """Missing effective dates -> can't establish direction -> no flag
    (fail-open: doc stays in effect, mirroring the retriever)."""
    a = _tl(1, "Luật X 2010")  # no effective_date
    b = _tl(2, "Luật X 2020", replaced=["Luật X trước đây"])  # no date either
    r = VersionResolver([a, b])
    assert r.find_replacement(1) is None


def test_amender_numbers_after_marker_do_not_count_as_replaced():
    """BLLĐ-2019 case: BHXH 2024's clause replaces BHXH 2014 and lists the
    laws that AMENDED it ('đã được sửa đổi, bổ sung ... theo Bộ luật số
    45/2019/QH14') — 45/2019 is the CURRENT labor code and must not be
    flagged as replaced."""
    bhxh_2024 = _tl(
        557190, "Luật Bảo hiểm xã hội 2024", num="41/2024/QH15",
        eff=date(2025, 7, 1),
        replaced=[
            "Luật Bảo hiểm xã hội số 58/2014/QH13 đã được sửa đổi, bổ sung "
            "một số điều theo Luật số 84/2015/QH13, Luật số 35/2018/QH14, "
            "Bộ luật số 45/2019/QH14"
        ],
    )
    bhxh_2014 = _tl(259700, "Luật Bảo hiểm xã hội 2014", num="58/2014/QH13",
                    eff=date(2016, 1, 1))
    blld_2019 = _tl(333670, "Bộ luật Lao động 2019", num="45/2019/QH14",
                    eff=date(2021, 1, 1))
    r = VersionResolver([bhxh_2024, bhxh_2014, blld_2019])
    hit = r.find_replacement(259700)
    assert hit is not None and hit.doc_id == 557190  # 2014 replaced ✓
    assert r.find_replacement(333670) is None        # BLLĐ 2019 untouched


def test_amendment_titled_doc_never_replaces():
    """'Luật sửa đổi BLTTHS 2025' MODIFIES BLTTHS 2015 — it must not mark it
    expired even if its extracted clauses mention the code's number."""
    base = _tl(296884, "Bộ luật tố tụng hình sự 2015", num="101/2015/QH13",
               eff=date(2016, 7, 1))
    amend = _tl(662829, "Luật sửa đổi Bộ luật Tố tụng hình sự 2025",
                num="99/2025/QH15", eff=date(2025, 10, 1),
                replaced=["Bộ luật Tố tụng hình sự số 101/2015/QH13"])
    r = VersionResolver([base, amend])
    assert r.find_replacement(296884) is None


# ── hierarchy guard (M1 follow-up, 2026-07-12) ─────────────────────────────
#
# Measured failure on the 100-doc corpus: NĐ 76/2023 (the guidance decree
# for Luật PCBLGĐ 2022) carried the LAW's name in its extracted
# replaced_laws clause. The name fallback's newer-date direction guard
# passed (decree is newer) and the CURRENT law was marked replaced by its
# own implementing decree — hidden from every answer by the temporal
# filter (hg-001/hg-101/hg-301 citation zeros). A lower-rank document
# (Nghị định/Thông tư/Quyết định) can never replace a higher-rank one
# (Luật/Bộ luật) in the legal hierarchy.

def test_guidance_decree_cannot_replace_its_law_name_fallback():
    law = _tl(490095, "Luật Phòng, chống bạo lực gia đình 2022",
              num="13/2022/QH15", eff=date(2023, 7, 1))
    decree = _tl(557470, "Nghị định 76/2023/NĐ-CP hướng dẫn Luật Phòng, "
                 "chống bạo lực gia đình", num="76/2023/NĐ-CP",
                 eff=date(2023, 12, 25),
                 replaced=["Luật Phòng, chống bạo lực gia đình"])
    r = VersionResolver([law, decree])
    assert r.find_replacement(490095) is None


def test_decree_cannot_replace_law_even_with_explicit_number():
    """Extraction garbage can put the law's own number in a decree's
    clause — the hierarchy guard must hold on the number path too."""
    law = _tl(490095, "Luật Phòng, chống bạo lực gia đình 2022",
              num="13/2022/QH15", eff=date(2023, 7, 1))
    decree = _tl(557470, "Nghị định 76/2023/NĐ-CP hướng dẫn",
                 num="76/2023/NĐ-CP", eff=date(2023, 12, 25),
                 replaced=["Luật Phòng, chống bạo lực gia đình số 13/2022/QH15"])
    r = VersionResolver([law, decree])
    assert r.find_replacement(490095) is None


def test_law_still_replaces_law_and_decree_still_replaces_decree():
    old_law = _tl(1, "Luật Việc làm 2013", num="38/2013/QH13",
                  eff=date(2015, 1, 1))
    new_law = _tl(2, "Luật Việc làm 2025", num="74/2025/QH15",
                  eff=date(2026, 1, 1),
                  replaced=["Luật Việc làm số 38/2013/QH13"])
    old_nd = _tl(3, "Nghị định 28/2015/NĐ-CP hướng dẫn Luật Việc làm",
                 num="28/2015/NĐ-CP", eff=date(2015, 5, 1))
    new_nd = _tl(4, "Nghị định 999/2026/NĐ-CP hướng dẫn Luật Việc làm",
                 num="999/2026/NĐ-CP", eff=date(2026, 3, 1),
                 replaced=["Nghị định số 28/2015/NĐ-CP"])
    r = VersionResolver([old_law, new_law, old_nd, new_nd])
    assert r.find_replacement(1) is new_law
    assert r.find_replacement(3) is new_nd


# ── consolidation guard (M2 Phase E, 2026-07-14) ───────────────────────────
#
# Measured failure on the 600-doc corpus: "Văn bản hợp nhất 410/VBHN-BTP ...
# hợp nhất Nghị định hướng dẫn Luật Nuôi con nuôi" carried the LAW's name in
# its extracted replaced_laws clause and, being newer (2017 > 2011), passed
# the name-fallback direction guard — marking the still-in-force Luật Nuôi
# con nuôi 2010 expired. exclude_superseded=True then dropped it from the
# routed pool (hngd recall@5 0.85 -> 0.60). A Văn bản hợp nhất is a
# CONSOLIDATION (merges a law + its amendments into one readable document);
# it never repeals anything, so it can never be a replacement.

def test_consolidation_vbhn_never_replaces_its_law_name_fallback():
    law = _tl(108082, "Luật nuôi con nuôi 2010", num="52/2010/QH12",
              eff=date(2011, 1, 1))
    vbhn = _tl(410, "Văn bản hợp nhất 410/VBHN-BTP năm 2026 hợp nhất Nghị "
               "định hướng dẫn Luật Nuôi con nuôi do Bộ Tư pháp ban hành",
               eff=date(2017, 1, 1),
               replaced=["Luật Nuôi con nuôi"])
    r = VersionResolver([law, vbhn])
    assert r.find_replacement(108082) is None


def test_consolidation_vbhn_never_replaces_even_with_explicit_number():
    """Extraction can put the law's own số hiệu in a VBHN's clause — the
    consolidation guard must hold on the number path too."""
    law = _tl(108082, "Luật nuôi con nuôi 2010", num="52/2010/QH12",
              eff=date(2011, 1, 1))
    vbhn = _tl(410, "Văn bản hợp nhất 410/VBHN-BTP hợp nhất Nghị định "
               "hướng dẫn Luật Nuôi con nuôi", eff=date(2017, 1, 1),
               replaced=["Luật Nuôi con nuôi số 52/2010/QH12"])
    r = VersionResolver([law, vbhn])
    assert r.find_replacement(108082) is None
