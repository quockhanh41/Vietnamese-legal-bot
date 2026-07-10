"""
S5: Article-level chunking — each Điều is one chunk, never split.

Strategy: The most structure-preserving approach. Every Điều becomes exactly
one chunk regardless of length. No merging or splitting. If no Điều structure
is detected, falls back to treating the whole document as one chunk.
"""

import re
from typing import List

from src.data.chunker import LegalChunk, BaseChunker


class ArticleLevelChunker(BaseChunker):
    """
    Article-level chunker — one chunk per Điều.

    Preserves complete semantic units. Every real Điều is kept regardless of
    length; only unattributed preamble text (before the first Điều) shorter
    than min_article_length is dropped.

    Parameters:
        min_article_length: Minimum characters for keeping unattributed
                            (non-Điều) preamble text as a chunk.
    """

    # Header anchored to line start. Two amendment-law quirks (M1.2b,
    # measured on the 100-doc corpus, 2026-07-10):
    #   - suffixed articles: "Bổ sung Điều 43a..." inserts headers like
    #     "Điều 43b." — \d+[a-z]? , otherwise those Điều stay glued to the
    #     previous chunk with no label (invisible to citation lookup);
    #   - the FIRST quoted article of an amendment block opens with a curly
    #     quote (“Điều 43a. …) while subsequent ones are bare — accept an
    #     optional opening quote so both become boundaries consistently.
    ARTICLE_PATTERN = re.compile(
        r'^\s*[“"«]?\s*Điều\s+(\d+[a-z]?)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    # Leading quote/space chars stripped off the header LABEL (metadata
    # only — chunk_text keeps the document's own text verbatim).
    _LABEL_STRIP = '“"« \t'

    CHAPTER_PATTERN = re.compile(
        r'^\s*CHƯƠNG\s+([IVXLCDM]+|\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    def __init__(self, min_article_length: int = 100):
        self.min_article_length = min_article_length

    def chunk_document(self, row: dict) -> List[LegalChunk]:
        content = row.get("content", "")
        if not content or len(content) < self.min_article_length:
            return []

        lines = content.split("\n")
        articles = []  # List of (article_header, chapter, text)
        current_chapter = None
        current_article_header = None
        current_article_text = ""

        def flush_article():
            nonlocal current_article_text
            text = current_article_text.strip()
            # A buffer belonging to a real Điều header must be kept even if
            # short (e.g. "Điều 3. Hiệu lực thi hành", 85 chars) — dropping
            # it silently removes that Điều from the index. Same invariant
            # as HierarchicalChunker's Bug 3 fix, see
            # docs/notes/chunker-misalignment-diagnosis-2026-07-10.md.
            # Only unattributed preamble text is length-filtered.
            if text and (current_article_header is not None
                         or len(text) >= self.min_article_length):
                articles.append({
                    "header": current_article_header or "",
                    "chapter": current_chapter,
                    "text": text,
                })
            current_article_text = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_article_text += "\n"
                continue

            # Track chapter context
            chapter_match = self.CHAPTER_PATTERN.match(stripped)
            if chapter_match:
                current_chapter = stripped
                current_article_text += stripped + "\n"
                continue

            # Article boundary
            article_match = self.ARTICLE_PATTERN.match(stripped)
            if article_match:
                flush_article()
                current_article_header = stripped.lstrip(self._LABEL_STRIP)
                current_article_text = stripped
                continue

            current_article_text += "\n" + stripped if current_article_text else stripped

        flush_article()

        # If no articles detected, treat whole document as one chunk
        if not articles:
            articles.append({
                "header": row.get("title", ""),
                "chapter": None,
                "text": content.strip(),
            })

        # Build LegalChunks
        legal_chunks = []
        for i, art in enumerate(articles):
            chunk = LegalChunk(
                chunk_text=art["text"],
                document_id=row["id"],
                document_title=row["title"],
                document_number=str(row.get("document_number", "")),
                legal_type=row.get("legal_type", ""),
                issuing_authority=row.get("issuing_authority", ""),
                issuance_date=str(row.get("issuance_date", "")),
                url=str(row.get("url", "")),
                chapter=art["chapter"],
                article=art["header"],
                clause=None,
                chunk_index=i,
                # Each chunk already IS one full Điều — there's no larger parent
                # unit to expand to (unlike HierarchicalChunker's Khoản chunks).
                parent_chunk_text=art["text"],
            )
            legal_chunks.append(chunk)

        return legal_chunks
