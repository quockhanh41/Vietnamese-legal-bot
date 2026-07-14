from scripts.select_corpus_600 import (
    sector_of, estimate_chunks, is_candidate, SPINE_IDS, RELEVANT_TYPES,
    allocate_budget, rank_key,
)

def test_sector_of_maps_token():
    assert sector_of("Doanh nghiệp, Đầu tư", "Nghị định về đăng ký doanh nghiệp") == "doanh_nghiep"
    assert sector_of("Bất động sản", "Nghị định quy định về giá đất") == "dat_dai"

def test_sector_of_hngd_via_title_when_no_token():
    # HNGĐ has no dedicated legal_sectors token → fall back to title hints
    assert sector_of("Văn hóa - Xã hội", "Luật Phòng, chống bạo lực gia đình") == "hngd"

def test_sector_of_returns_none_when_out_of_scope():
    assert sector_of("Giao thông - Vận tải", "Luật Giao thông đường bộ") is None

def test_sector_of_token_beats_hngd_title_hint():
    # a labor decree mentioning "hộ gia đình" must map to its token, not hngd
    assert sector_of("Lao động - Tiền lương",
                     "Nghị định mức lương tối thiểu cho hộ gia đình") == "lao_dong"
    assert sector_of("Bất động sản",
                     "Nghị định giao đất cho hộ gia đình") == "dat_dai"

def test_sector_of_hngd_fallback_only_when_no_token():
    # genuine family-law title with a non-mapping legal_sectors value → hngd fallback
    assert sector_of("Văn hóa - Xã hội", "Luật Nuôi con nuôi") == "hngd"

def test_estimate_chunks_uses_700_char_rule():
    assert estimate_chunks("x" * 6999) == 9    # 6999 // 700 = 9
    assert estimate_chunks("") == 1            # floor of 1

def test_is_candidate_rejects_spine_and_noise_types():
    spine_id = next(iter(SPINE_IDS))
    assert is_candidate({"id": spine_id, "legal_type": "Luật",
                         "legal_sectors": "Doanh nghiệp", "title": "x",
                         "content": "y" * 1000}) is False       # already in spine
    assert is_candidate({"id": -1, "legal_type": "Công văn",
                         "legal_sectors": "Doanh nghiệp", "title": "x",
                         "content": "y" * 1000}) is False       # noise type
    assert is_candidate({"id": -1, "legal_type": "Thông tư",
                         "legal_sectors": "Doanh nghiệp",
                         "title": "Thông tư hướng dẫn Luật Doanh nghiệp",
                         "content": "y" * 5000}) is True


def _cand(cid, chunks, **kw):
    d = {"id": cid, "est_chunks": chunks, "legal_type": "Thông tư",
         "is_consolidated": False, "anchored": False, "year": 2020}
    d.update(kw)
    return d

def test_allocate_budget_keeps_ratio_within_cap():
    ranked = {s: [_cand(f"{s}{i}", 100) for i in range(50)]
              for s in ["hinh_su", "dan_su", "lao_dong", "doanh_nghiep", "dat_dai", "hngd"]}
    spine = {"hinh_su": 2800, "dan_su": 3300, "lao_dong": 1900,
             "doanh_nghiep": 4100, "dat_dai": 2200, "hngd": 1000}
    picks, totals = allocate_budget(ranked, spine, target_added_chunks=20000, ratio_cap=3.0)
    assert max(totals.values()) / min(totals.values()) <= 3.0 + 1e-9
    assert len(picks["hngd"]) >= len(picks["doanh_nghiep"])

def test_allocate_budget_stops_when_candidates_exhausted():
    ranked = {s: [_cand(f"{s}{i}", 100) for i in range(2)] for s in ["hinh_su", "hngd"]}
    picks, totals = allocate_budget(ranked, {"hinh_su": 500, "hngd": 500},
                                    target_added_chunks=99999, ratio_cap=3.0)
    assert len(picks["hinh_su"]) == 2 and len(picks["hngd"]) == 2
    assert max(totals.values()) / min(totals.values()) <= 3.0 + 1e-9

def test_allocate_budget_never_overshoots_on_oversized_candidate():
    # An oversized candidate that can't fit within the cap must be rejected,
    # not force-added onto the min sector.
    ranked = {"a": [_cand("big", 5000)],
              "b": [_cand(f"b{i}", 100) for i in range(3)]}
    picks, totals = allocate_budget(ranked, {"a": 1000, "b": 1000},
                                    target_added_chunks=10000, ratio_cap=3.0)
    assert max(totals.values()) / min(totals.values()) <= 3.0 + 1e-9
    assert all(c["id"] != "big" for c in picks["a"])

def test_rank_key_prioritizes_anchored_then_consolidated():
    anchored = _cand("x", 60, anchored=True)
    consolidated = _cand("y", 60, is_consolidated=True)
    plain = _cand("z", 60)
    ordered = sorted([plain, consolidated, anchored], key=rank_key, reverse=True)
    assert [c["id"] for c in ordered] == ["x", "y", "z"]
