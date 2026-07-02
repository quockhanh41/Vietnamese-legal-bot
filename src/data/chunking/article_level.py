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

    Preserves complete semantic units. Articles that are very short
    (< min_article_length) are merged with adjacent articles.

    Parameters:
        min_article_length: Minimum characters for a standalone article chunk.
                            Shorter articles get merged with neighbors.
    """

    ARTICLE_PATTERN = re.compile(
        r'^\s*Điều\s+(\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

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
            if text and len(text) >= self.min_article_length:
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
                current_article_header = stripped
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
                parent_chunk_text=content.strip(),
            )
            legal_chunks.append(chunk)

        return legal_chunks
