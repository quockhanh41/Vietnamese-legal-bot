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
