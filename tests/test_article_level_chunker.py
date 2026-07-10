"""
Regression test for src/data/chunking/article_level.py: parent_chunk_text
used to be set to the ENTIRE document (`content.strip()`), which for the
amendment-law documents in this corpus produced a ~267K-char value — later
fed into orchestrator._expand_to_parent (Plan §3.4). It must equal the
chunk's own article text (there's no larger "parent" unit for an
already-article-level chunk), not the whole document.

Run: .venv/bin/python -m pytest tests/test_article_level_chunker.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import src.data.chunker before src.data.chunking.article_level — the latter
# is part of the src.data.chunking package, whose __init__ imports
# hierarchical.py, which imports back from src.data.chunker; importing
# chunker.py standalone first avoids that circular-import trap (same order
# scripts/ingest.py uses).
import src.data.chunker  # noqa: F401
from src.data.chunking.article_level import ArticleLevelChunker


def test_parent_chunk_text_equals_own_article_not_whole_document():
    row = {
        "id": 1,
        "title": "Luật sửa đổi Bộ luật Hình sự 2017",
        "content": (
            "Điều 1. Sửa đổi Điều 173\n"
            "Nội dung sửa đổi điều 173 " + ("x" * 300) + "\n\n"
            "Điều 2. Sửa đổi Điều 174\n"
            "Nội dung sửa đổi điều 174 " + ("y" * 300)
        ),
        "document_number": "12/2017/QH14",
        "legal_type": "Luật",
        "issuing_authority": "Quốc hội",
        "issuance_date": "2017-06-20",
        "url": "",
    }
    chunker = ArticleLevelChunker(min_article_length=50)
    chunks = chunker.chunk_document(row)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.parent_chunk_text == chunk.chunk_text
        # The whole-document bug produced a parent_chunk_text containing
        # BOTH articles; guard that this chunk's parent does not leak the
        # other article's content.
        other_marker = "y" * 300 if chunk.article == "Điều 1. Sửa đổi Điều 173" else "x" * 300
        assert other_marker not in chunk.parent_chunk_text


def test_short_article_is_kept_not_dropped():
    """Regression (M0.4b sibling of HierarchicalChunker Bug 3): a real Điều
    shorter than min_article_length used to be silently dropped from the
    output — "Điều 3. Hiệu lực thi hành" (85 chars) of Luật sửa đổi BLHS 2025
    was missing from the production index entirely. Every real Điều must be
    kept regardless of length."""
    row = {
        "id": 2,
        "title": "Luật sửa đổi Bộ luật Hình sự 2025",
        "content": (
            "Điều 1. Sửa đổi, bổ sung một số điều\n"
            "Nội dung sửa đổi " + ("x" * 200) + "\n\n"
            "Điều 3. Hiệu lực thi hành\n"
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2025.\n"
        ),
        "document_number": "01/2025/QH15",
        "legal_type": "Luật",
        "issuing_authority": "Quốc hội",
        "issuance_date": "2025-06-01",
        "url": "",
    }
    chunks = ArticleLevelChunker(min_article_length=100).chunk_document(row)
    labels = [c.article for c in chunks]
    assert any(a.startswith("Điều 3.") for a in labels), (
        f"Điều 3 (short article) was dropped; labels: {labels}"
    )
    short = next(c for c in chunks if c.article.startswith("Điều 3."))
    assert "hiệu lực thi hành" in short.chunk_text.lower()
    assert short.parent_chunk_text == short.chunk_text


def test_short_preamble_before_first_article_still_dropped():
    """The length filter must still apply to unattributed preamble text —
    only real Điều buffers are exempt."""
    row = {
        "id": 3,
        "title": "Luật test",
        "content": (
            "Căn cứ Hiến pháp;\n\n"  # short preamble, no Điều header → drop
            "Điều 1. Phạm vi điều chỉnh\n"
            "Nội dung " + ("z" * 200) + "\n"
        ),
        "document_number": "",
        "legal_type": "Luật",
        "issuing_authority": "",
        "issuance_date": "",
        "url": "",
    }
    chunks = ArticleLevelChunker(min_article_length=100).chunk_document(row)
    assert all(c.article.startswith("Điều") for c in chunks), (
        f"preamble scrap leaked into chunks: {[c.article for c in chunks]}"
    )


def test_suffixed_quoted_articles_become_labeled_boundaries():
    """M1.2b (2026-07-10): amendment laws insert suffixed articles
    ("Bổ sung Điều 43a và Điều 43b...") whose quoted headers open with a
    curly quote for the FIRST article of a block (“Điều 43a. …) and appear
    bare for subsequent ones (Điều 43b. …). Both must become their own
    chunk with a clean label (no leading quote) — before this fix the
    suffixed/quoted ones stayed glued to the previous chunk with no label,
    invisible to article_number citation lookup, producing chunks up to
    46K chars (past BGE-M3's embed window)."""
    row = {
        "id": 2,
        "title": "Luật Đấu giá tài sản sửa đổi 2024",
        "content": (
            "QUỐC HỘI\n"
            "Điều 1. Sửa đổi, bổ sung một số điều của Luật Đấu giá tài sản\n"
            "28. Bổ sung Điều 43a và Điều 43b vào sau Điều 43 như sau:\n"
            "“Điều 43a. Đấu giá trực tuyến\n"
            "1. Việc đấu giá trực tuyến được thực hiện thông qua Cổng Đấu giá.\n"
            "Điều 43b. Trình tự đấu giá trực tuyến\n"
            "1. Tổ chức hành nghề đấu giá đăng tải Quy chế.”.\n"
            "Điều 2. Hiệu lực thi hành\n"
            "Luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2025.\n"
        ),
    }
    chunks = ArticleLevelChunker(min_article_length=100).chunk_document(row)
    labels = [c.article for c in chunks]

    assert "Điều 43a. Đấu giá trực tuyến" in labels          # quoted-“ header
    assert "Điều 43b. Trình tự đấu giá trực tuyến" in labels  # bare suffixed header
    assert not any(l.startswith("“") for l in labels if l)   # labels quote-free

    # chunk_text stays verbatim — the opening quote is NOT stripped from text
    c43a = next(c for c in chunks if c.article.startswith("Điều 43a"))
    assert c43a.chunk_text.startswith("“Điều 43a.")
    # mid-line mentions ("Bổ sung Điều 43a và...") must NOT create boundaries:
    # the intro line stays in Điều 1's chunk
    c1 = next(c for c in chunks if c.article.startswith("Điều 1"))
    assert "Bổ sung Điều 43a và Điều 43b" in c1.chunk_text


def test_plain_numbered_articles_unaffected_by_suffix_support():
    row = {
        "id": 3,
        "title": "Luật sửa đổi X 2024",
        "content": (
            "Điều 1. Phạm vi điều chỉnh\n"
            "Nội dung một " + ("a" * 120) + "\n"
            "Điều 2. Hiệu lực\n"
            "Nội dung hai " + ("b" * 120) + "\n"
        ),
    }
    chunks = ArticleLevelChunker(min_article_length=100).chunk_document(row)
    assert [c.article for c in chunks] == [
        "Điều 1. Phạm vi điều chỉnh",
        "Điều 2. Hiệu lực",
    ]
