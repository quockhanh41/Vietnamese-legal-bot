"""
Unit tests for src/rag/context_blocks.py — the shared numbered-context-block
parser and per-text premise selector (M0.1 fix, roadmap
docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md "2 follow-up còn lại").

Pure-Python string logic — no model load, no network.

Run: .venv/bin/python -m pytest tests/test_context_blocks.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.context_blocks import (
    extract_block_refs,
    parse_context_blocks,
    select_premise,
)

CONTEXT_3_BLOCKS = (
    "[1] Bộ luật Hình sự 2015 — Điều 17 (Chương IV)\n"
    "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm."
    "\n\n---\n\n"
    "[2] Bộ luật Hình sự 2015 — Điều 20 (Chương IV)\n"
    "Người tổ chức, người thực hành, người xúi giục, người giúp sức đều là đồng phạm."
    "\n\n---\n\n"
    "[3] Bộ luật Hình sự 2015 — Điều 173 (Chương XVI)\n"
    "Tội trộm cắp tài sản bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm."
)


# ── parse_context_blocks ──────────────────────────────────────────────

def test_parse_context_blocks_basic():
    blocks = parse_context_blocks(CONTEXT_3_BLOCKS)
    assert set(blocks.keys()) == {1, 2, 3}
    assert blocks[1].startswith("[1] Bộ luật Hình sự 2015")
    assert "Đồng phạm" in blocks[1]
    assert "trộm cắp" in blocks[3]


def test_parse_context_blocks_empty_context():
    assert parse_context_blocks("") == {}
    assert parse_context_blocks(None) == {}


def test_parse_context_blocks_unnumbered_context_returns_empty():
    """Plain, non-block-formatted context (legacy/test callers) must not be
    mistaken for a numbered block — returns {} so `select_premise` can fall
    back to blind truncation."""
    assert parse_context_blocks("Đây chỉ là một đoạn văn bản bình thường, không đánh số.") == {}


# ── extract_block_refs ────────────────────────────────────────────────

def test_extract_block_refs_single():
    assert extract_block_refs("Một nội dung nào đó (Điều 17 BLHS 2015) [2].") == [2]


def test_extract_block_refs_multiple_dedup_order():
    assert extract_block_refs("Nội dung dựa trên [3][1] và lại nhắc [1] lần nữa.") == [3, 1]


def test_extract_block_refs_none():
    assert extract_block_refs("Không có trích dẫn khối nào ở đây.") == []
    assert extract_block_refs("") == []


# ── select_premise: cites one block ─────────────────────────────────────

def test_select_premise_cites_single_block():
    claim = "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm [1]."
    premise = select_premise(claim, CONTEXT_3_BLOCKS)
    assert premise.startswith("[1] Bộ luật Hình sự 2015")
    assert "Đồng phạm" in premise
    # Must NOT pull in the unrelated block 3 (trộm cắp) content.
    assert "trộm cắp" not in premise


# ── select_premise: cites multiple blocks ───────────────────────────────

def test_select_premise_cites_multiple_blocks_joined_in_order():
    claim = "Đồng phạm gồm nhiều vai trò khác nhau [1][2]."
    premise = select_premise(claim, CONTEXT_3_BLOCKS)
    assert "Đồng phạm là trường hợp có hai người" in premise  # block 1
    assert "Người tổ chức, người thực hành" in premise  # block 2
    assert "trộm cắp" not in premise  # block 3 not cited


def test_select_premise_invalid_ref_ignored_falls_back_to_overlap():
    """A claim citing only an out-of-range block (e.g. [9] on a 3-block
    context) has no *valid* ref — must fall back to lexical overlap, not
    silently include a nonexistent block."""
    claim = "Tội trộm cắp tài sản bị phạt tù theo quy định [9]."
    premise = select_premise(claim, CONTEXT_3_BLOCKS)
    # Best lexical overlap is block 3 (shares "trộm cắp", "tài sản", "phạt tù").
    assert premise.startswith("[3] Bộ luật Hình sự 2015 — Điều 173")


# ── select_premise: no citation → lexical-overlap fallback ─────────────

def test_select_premise_no_citation_falls_back_to_best_overlap_block():
    claim = "Đồng phạm là những người tham gia thực hiện tội phạm với vai trò khác nhau."
    premise = select_premise(claim, CONTEXT_3_BLOCKS)
    # Shares far more words with block 1 (đồng phạm / tội phạm) than block 3.
    assert premise.startswith("[1] Bộ luật Hình sự 2015 — Điều 17")


def test_select_premise_no_citation_picks_most_overlapping_of_several():
    claim = "Người tổ chức và người thực hành đều bị coi là đồng phạm trong vụ án."
    premise = select_premise(claim, CONTEXT_3_BLOCKS)
    assert premise.startswith("[2] Bộ luật Hình sự 2015 — Điều 20")


# ── select_premise: union across a whole answer (self-eval use case) ───

def test_select_premise_union_of_all_refs_in_full_answer():
    answer = (
        "### 📋 Chi tiết\n"
        "- Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm [1].\n"
        "- Người tổ chức, người thực hành cũng là đồng phạm [2].\n"
    )
    premise = select_premise(answer, CONTEXT_3_BLOCKS, cap=3000)
    assert "Đồng phạm là trường hợp có hai người" in premise  # block 1
    assert "Người tổ chức, người thực hành" in premise  # block 2
    assert "trộm cắp" not in premise  # block 3 never cited


# ── select_premise: unnumbered context → old blind-truncation fallback ──

def test_select_premise_unnumbered_context_falls_back_to_blind_truncation():
    """Context that doesn't parse into numbered blocks at all (context isn't
    in `_build_context()` format) must fall back to the pre-M0.1 behavior:
    blind `context[:cap]` — never crash, never silently drop everything."""
    plain_context = "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm." * 50
    claim = "Đồng phạm là gì [1]?"
    premise = select_premise(claim, plain_context, cap=100)
    assert premise == plain_context[:100]


# ── select_premise: cap enforced ────────────────────────────────────────

def test_select_premise_caps_length():
    claim = "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm [1][2][3]."
    premise = select_premise(claim, CONTEXT_3_BLOCKS, cap=50)
    assert len(premise) == 50


def test_select_premise_reuses_preparsed_blocks():
    """Passing a pre-parsed `blocks` dict (as callers do in a per-claim loop)
    must produce the same result as re-parsing from `context` each time."""
    blocks = parse_context_blocks(CONTEXT_3_BLOCKS)
    claim = "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm [1]."
    premise_a = select_premise(claim, CONTEXT_3_BLOCKS)
    premise_b = select_premise(claim, CONTEXT_3_BLOCKS, blocks=blocks)
    assert premise_a == premise_b

