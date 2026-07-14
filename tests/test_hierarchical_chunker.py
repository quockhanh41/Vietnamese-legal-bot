"""
Regression tests for src/data/chunking/hierarchical.py: HierarchicalChunker
used to produce chunks whose `article` metadata label did not match the
Điều actually contained in `chunk_text` ("WRONG_LABEL" chunks — 126/1670
points, 7.5%, in the production `luat_law_ctx` collection). Root cause and
fix are documented in
docs/notes/chunker-misalignment-diagnosis-2026-07-10.md — three bugs, all in
`_split_by_structure` / `_control_chunk_sizes`:

  Bug 1: `_control_chunk_sizes` merged consecutive structural chunks
         regardless of which Điều they belonged to, and let the LAST
         chunk's `article` label overwrite the buffer's — so a chunk could
         contain text from Điều 19+20 but be labeled "Điều 21".
  Bug 3: `flush_buffer` dropped any buffer shorter than
         `min_chunk_size // 2` (100 chars) — a short, complete, standalone
         Điều (e.g. a one-line "hiệu lực thi hành" clause) could vanish
         from the index entirely once Bug 1 stopped merging it into a
         neighboring Điều.
  Bug 4: `parent_chunk_text` was snapshotted from `article_buffer` at
         flush time, so a Khoản-1 chunk's parent text only ever contained
         the Điều header + Khoản 1 — missing Khoản 2+ (breaks parent-
         document retrieval, Plan §3.4).

Run: .venv/bin/python -m pytest tests/test_hierarchical_chunker.py -q
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Same import-order workaround as test_article_level_chunker.py: import
# src.data.chunker before src.data.chunking.hierarchical to dodge the
# circular import through src.data.chunking.__init__.
import src.data.chunker  # noqa: F401
from src.data.chunking.hierarchical import HierarchicalChunker


# Mirrors the diagnosis script's header-detection regex (suffix letter
# tolerated, e.g. "Điều 256a.") — used here to assert a chunk's labeled
# article always matches the FIRST Điều header actually present in its text.
HEADER_RE = re.compile(r'(?:^|\n)\s*Điều\s+(\d+[a-zđ]?)\s*[\.:]', re.IGNORECASE)


def _article_num(label):
    """Extract the numeric (+ optional suffix) part of an 'article' label,
    e.g. 'Điều 21. Người bào chữa' -> '21'."""
    if label is None:
        return None
    m = re.match(r'\s*Điều\s+(\d+[a-zđ]?)', label, re.IGNORECASE)
    return m.group(1) if m else None


def _make_row(content, **overrides):
    row = {
        "id": 1,
        "title": "Test Document",
        "content": content,
        "document_number": "1/1999/QH",
        "legal_type": "Luật",
        "issuing_authority": "Quốc hội",
        "issuance_date": "1999-01-01",
        "url": "",
    }
    row.update(overrides)
    return row


def _assert_no_wrong_label(chunks):
    """The core invariant from the diagnosis: for every chunk, if its text
    contains an Điều header, that header's number must match the chunk's
    `article` label. This is exactly the WRONG_LABEL check Fable ran
    against production."""
    for c in chunks:
        expected = _article_num(c.article)
        headers_in_text = HEADER_RE.findall(c.chunk_text)
        if not headers_in_text:
            continue
        first_header = headers_in_text[0]
        assert first_header == expected, (
            f"WRONG_LABEL: chunk labeled article={c.article!r} but text "
            f"starts with header 'Điều {first_header}' — chunk_text[:80]="
            f"{c.chunk_text[:80]!r}"
        )


class TestBug1MergeAcrossArticleBoundary:
    """Consecutive short Điều (BLHS-1999-style: Điều 1, 2, 3 each only a
    few characters) must NOT be merged into a single mislabeled chunk."""

    def _short_articles_doc(self):
        # Each article body is well under min_chunk_size on its own, which
        # is exactly the condition that used to trigger cross-article
        # merging in _control_chunk_sizes.
        return (
            "CHƯƠNG I\nNHỮNG QUY ĐỊNH CHUNG\n\n"
            "Điều 1. Nhiệm vụ của Bộ luật hình sự\n"
            "Ngắn gọn.\n\n"
            "Điều 2. Cơ sở của trách nhiệm hình sự\n"
            "Cũng ngắn.\n\n"
            "Điều 3. Nguyên tắc xử lý\n"
            "Ngắn nữa.\n\n"
            "Điều 4. Trách nhiệm đấu tranh phòng ngừa tội phạm\n"
            + ("Nội dung điều 4 dài hơn nhiều so với các điều trước đó. " * 10)
        )

    def test_each_short_article_keeps_its_own_label(self):
        chunker = HierarchicalChunker(min_chunk_size=200, max_chunk_size=1000)
        row = _make_row(self._short_articles_doc())
        chunks = chunker.chunk_document(row)

        assert len(chunks) >= 4, "expected at least one chunk per Điều"
        _assert_no_wrong_label(chunks)

        # Explicitly: no chunk labeled "Điều 3" may contain "Điều 1" or
        # "Điều 2" text (the exact N-1/N-2 pattern observed in production).
        labels = [c.article for c in chunks]
        assert any(_article_num(l) == "1" for l in labels)
        assert any(_article_num(l) == "2" for l in labels)
        assert any(_article_num(l) == "3" for l in labels)
        for c in chunks:
            if _article_num(c.article) == "3":
                assert "Điều 1." not in c.chunk_text
                assert "Điều 2." not in c.chunk_text

    def test_no_chunk_spans_more_than_one_article(self):
        """Global invariant required by the fix: a chunk_text may contain
        at most one distinct Điều header number."""
        chunker = HierarchicalChunker(min_chunk_size=200, max_chunk_size=1000)
        row = _make_row(self._short_articles_doc())
        chunks = chunker.chunk_document(row)

        for c in chunks:
            headers = {h for h in HEADER_RE.findall(c.chunk_text)}
            assert len(headers) <= 1, (
                f"chunk spans multiple Điều headers {headers}: "
                f"{c.chunk_text[:120]!r}"
            )

    def test_merged_multi_clause_chunk_has_no_single_clause_label(self):
        """When two+ Khoản of the SAME Điều get folded into one buffer
        (both short enough to fit under max_chunk_size), the result
        represents the Điều as a whole — `clause` must be None, not the
        last-seen Khoản's label silently overwriting the first."""
        content = (
            "Điều 10. Điều có nhiều khoản ngắn\n"
            "1. Khoản một ngắn.\n\n"
            "2. Khoản hai ngắn.\n\n"
            "3. Khoản ba ngắn.\n\n"
            "Điều 11. Điều tiếp theo\n"
            "Nội dung điều mười một khá dài để đủ độ dài văn bản. " * 5
        )
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=1000)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        art10_chunks = [c for c in chunks if _article_num(c.article) == "10"]
        assert len(art10_chunks) == 1
        assert art10_chunks[0].clause is None
        assert "1. Khoản một" in art10_chunks[0].chunk_text
        assert "3. Khoản ba" in art10_chunks[0].chunk_text

    def test_single_clause_chunk_keeps_its_clause_label(self):
        """A chunk representing exactly one Khoản should still carry that
        Khoản's label (regression guard: the merge-fix must not blank out
        `clause` unconditionally)."""
        content = (
            "Điều 20. Điều với khoản dài\n"
            "1. " + ("Nội dung khoản một rất dài. " * 40) + "\n\n"
            "2. " + ("Nội dung khoản hai rất dài. " * 40)
        )
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=1000)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        art20_chunks = [c for c in chunks if _article_num(c.article) == "20"]
        assert len(art20_chunks) >= 2
        # Each Khoản is long enough on its own that it should not be merged
        # with its neighbor — both Khoản 1 and Khoản 2 must be represented
        # by a chunk carrying that Khoản's own clause label (the header-only
        # preamble chunk before Khoản 1 legitimately has clause=None).
        clauses = [c.clause for c in art20_chunks if c.clause is not None]
        assert any(cl.startswith("1.") for cl in clauses)
        assert any(cl.startswith("2.") for cl in clauses)


class TestBug3ShortStandaloneArticleNotDropped:
    """A complete but very short Điều (e.g. a one-line 'hiệu lực thi hành'
    clause, well under min_chunk_size // 2 == 100 chars) must still appear
    in the output — dropping it silently removes that Điều from the
    index."""

    def test_short_final_article_survives(self):
        content = (
            "Điều 343. Sửa đổi, bổ sung một số điều\n"
            + ("Nội dung điều 343 khá dài để không bị rơi vào nhánh drop. " * 8)
            + "\n\n"
            "Điều 344. Hiệu lực thi hành\n"
            "Có hiệu lực từ 01/01/2018.\n"  # ~40 chars, well under 100
        )
        chunker = HierarchicalChunker(min_chunk_size=200, max_chunk_size=1000)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        art344 = [c for c in chunks if _article_num(c.article) == "344"]
        assert len(art344) == 1, "Điều 344 (short) must not be dropped"
        assert "Hiệu lực thi hành" in art344[0].chunk_text
        assert "01/01/2018" in art344[0].chunk_text

    def test_article_count_preserved_between_content_and_chunks(self):
        """Every distinct Điều header in the raw content must be
        represented by at least one output chunk's `article` label — the
        general form of Bug 3's invariant."""
        content = "CHƯƠNG I\nQuy định\n\n"
        n_articles = 6
        for i in range(1, n_articles + 1):
            body = "x" if i % 2 == 0 else ("dài " * 30)  # alternate tiny/long
            content += f"Điều {i}. Tiêu đề điều {i}\n{body}\n\n"

        chunker = HierarchicalChunker(min_chunk_size=200, max_chunk_size=1000)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        content_article_nums = {
            m.group(1) for m in re.finditer(r'(?:^|\n)\s*Điều\s+(\d+)\s*\.', content)
        }
        chunk_article_nums = {
            _article_num(c.article) for c in chunks if c.article is not None
        }
        assert content_article_nums == chunk_article_nums, (
            f"missing from chunks: {content_article_nums - chunk_article_nums}"
        )


class TestBug4ParentTextIsFullArticle:
    """parent_chunk_text must be the FULL Điều text (all Khoản), identical
    across every chunk that belongs to that Điều — not a snapshot taken at
    the first Khoản's flush time."""

    def _multi_clause_doc(self):
        return (
            "Điều 50. Điều nhiều khoản dài\n"
            "1. " + ("Nội dung khoản một. " * 40) + "\n\n"
            "2. " + ("Nội dung khoản hai. " * 40) + "\n\n"
            "3. " + ("Nội dung khoản ba. " * 40) + "\n\n"
            "Điều 51. Điều kế tiếp\n"
            "Nội dung điều 51. " * 10
        )

    def test_all_chunks_of_same_article_share_identical_parent_text(self):
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=400)
        row = _make_row(self._multi_clause_doc())
        chunks = chunker.chunk_document(row)

        art50_chunks = [c for c in chunks if _article_num(c.article) == "50"]
        assert len(art50_chunks) >= 2, (
            "expected Điều 50 to be split into multiple Khoản-level chunks "
            "given the small max_chunk_size"
        )
        parents = {c.parent_chunk_text for c in art50_chunks}
        assert len(parents) == 1, (
            f"chunks of the same Điều must share one parent_chunk_text, "
            f"got {len(parents)} distinct values"
        )

    def test_parent_text_contains_every_clause_not_just_the_first(self):
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=400)
        row = _make_row(self._multi_clause_doc())
        chunks = chunker.chunk_document(row)

        art50_chunks = [c for c in chunks if _article_num(c.article) == "50"]
        parent = art50_chunks[0].parent_chunk_text
        assert "khoản một" in parent
        assert "khoản hai" in parent
        assert "khoản ba" in parent

    def test_parent_text_starts_with_article_header(self):
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=400)
        row = _make_row(self._multi_clause_doc())
        chunks = chunker.chunk_document(row)

        art50_chunks = [c for c in chunks if _article_num(c.article) == "50"]
        assert art50_chunks[0].parent_chunk_text.startswith("Điều 50.")

    def test_preamble_chunk_parent_text_is_itself(self):
        """Text before the first Điều has no larger 'parent' Điều unit —
        parent_chunk_text should just be the chunk's own text (unchanged
        behavior, not part of the Bug 4 fix)."""
        content = (
            "Lời nói đầu " + ("dài dòng " * 30) + "\n\n"
            "Điều 1. Điều đầu tiên\n"
            "Nội dung điều 1. " * 10
        )
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=1000)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        preamble = [c for c in chunks if c.article is None]
        for c in preamble:
            assert c.parent_chunk_text == c.chunk_text


class TestOversizeSplitInheritsCorrectArticleLabel:
    """Bug 2 from the diagnosis ('split oversize inherits wrong metadata')
    is described as self-healing once Bug 1 is fixed: a chunk that gets
    split by RecursiveCharacterTextSplitter is now guaranteed to already
    belong to a single Điều, so every sub-chunk's `article` label is
    correct by construction."""

    def test_split_subchunks_all_match_their_own_article_label(self):
        content = (
            "Điều 100. Điều rất dài cần phải chia nhỏ\n"
            + ("Nội dung rất dài lặp đi lặp lại nhiều lần. " * 80)
            + "\n\nĐiều 101. Điều kế tiếp\nNội dung điều 101. " * 5
        )
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=300)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        # Điều 100 must have been split into multiple sub-chunks.
        art100 = [c for c in chunks if _article_num(c.article) == "100"]
        assert len(art100) > 1
        _assert_no_wrong_label(chunks)


class TestArticlePatternSuffixSupport:
    """M2.7 (2026-07-13): ARTICLE_PATTERN now accepts a single letter
    suffix (`\\d+[a-z]?`), fixing the limitation previously locked in here
    (see git history for the prior `TestArticlePatternSuffixLimitation`).
    Driver: the 600-doc corpus gate found WRONG_LABEL>0 on "Văn bản hợp
    nhất" (consolidated) documents, which splice amendment articles in
    place using letter-suffixed numbers (Điều 8a, 11a, 12a...) — previously
    unrecognized as article boundaries, so their content silently merged
    into the preceding Điều's buffer. See
    docs/notes/chunker-thongtu-diagnosis-2026-07-13.md.
    """

    def test_suffix_letter_article_is_recognized_as_its_own_boundary(self):
        content = (
            "Điều 105. Điều gốc\n"
            + ("Nội dung điều 105. " * 15)
            + "\n\nĐiều 105a. Điều bổ sung có hậu tố chữ cái\n"
            + ("Nội dung điều 105a. " * 15)
        )
        chunker = HierarchicalChunker(min_chunk_size=50, max_chunk_size=2000)
        row = _make_row(content)
        chunks = chunker.chunk_document(row)

        # Fixed behavior: "Điều 105a" gets its own label and chunk, and
        # its content no longer bleeds into "Điều 105"'s chunk.
        labels = {_article_num(c.article) for c in chunks}
        assert "105a" in labels
        art105 = [c for c in chunks if _article_num(c.article) == "105"]
        assert not any("Điều 105a." in c.chunk_text for c in art105)
        art105a = [c for c in chunks if _article_num(c.article) == "105a"]
        assert any("Điều 105a." in c.chunk_text for c in art105a)
        _assert_no_wrong_label(chunks)
