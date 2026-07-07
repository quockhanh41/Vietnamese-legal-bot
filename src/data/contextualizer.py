"""
Rule-based Contextual Retrieval (Anthropic-style, but from existing metadata —
no LLM calls). See docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md §3.1.

The idea: the text that gets EMBEDDED should carry its legal context ("which
law, which chapter, which article") so retrieval can match it. The stored /
displayed `chunk_text` is left completely unchanged — only a header is
prepended to the string that goes into the embedder.

This addresses the "Trạng thái Tuần 2" finding that `chunk_text` often starts
mid-article (missing the "Điều N." header), which explains why lenient
(text-match) recall was lower than strict (payload-match) recall in early
eval runs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# "Điều N" prefix, optionally followed by a trailing letter (e.g. "Điều 12a"),
# then a period or colon. Used both to detect the article number carried in
# the `article` metadata field and to check whether chunk_text already opens
# with that same article header (anti-duplication).
_ARTICLE_NUM_RE = re.compile(r"[đĐ]i[eề]u\s+(\d+[a-z]?)\s*[\.:]")

# Leading clause number in the `clause` field, e.g. "4. Đối với người..." ->
# "4". In this dataset `clause` holds the full clause *text*, not just a
# number or "Khoản N" label — so we must extract the number ourselves rather
# than embedding the whole (already-duplicated) clause text into the header.
_CLAUSE_NUM_RE = re.compile(r"^\s*(\d+)\s*\.")

# How far into chunk_text we check for an already-present article header
# (mirrors citation_lookup.py's _CHUNK_OPEN_WINDOW).
_CHUNK_OPEN_WINDOW = 100


def _clean(value: Optional[str]) -> Optional[str]:
    """Normalize a metadata field: None/empty/whitespace-only -> None."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _extract_clause_number(clause: Optional[str]) -> Optional[str]:
    """Pull the leading number out of a `clause` field like '4. Đối với...'.

    Returns None if the field is missing or doesn't match the expected
    "N. ..." shape (defensive — don't ever accidentally embed the entire
    clause body twice into the header).
    """
    clause = _clean(clause)
    if clause is None:
        return None
    m = _CLAUSE_NUM_RE.match(clause)
    return m.group(1) if m else None


def _article_number(article: Optional[str]) -> Optional[str]:
    """Extract the article number from a metadata `article` field like
    'Điều 58. Giảm mức hình phạt đã tuyên' -> '58'."""
    article = _clean(article)
    if article is None:
        return None
    m = _ARTICLE_NUM_RE.search(article)
    return m.group(1) if m else None


def _chunk_opens_with_article(chunk_text: str, article_num: str) -> bool:
    """True if chunk_text already begins (within the first
    _CHUNK_OPEN_WINDOW chars) with "Điều <article_num>." / ":"."""
    window = chunk_text[:_CHUNK_OPEN_WINDOW]
    m = _ARTICLE_NUM_RE.search(window)
    return bool(m and m.group(1) == article_num)


def build_context_header(chunk: Dict) -> str:
    """
    Build just the metadata header for a chunk (no chunk body).

    Header format (fields skipped when missing/empty):
        "{document_title} ({legal_type} {document_number}) | {chapter} | {article} | Khoản {n}"

    Anti-duplication: if chunk_text already opens with the same "Điều N."
    header as the `article` field, the article segment is omitted from the
    header so that string isn't embedded twice.

    Returns "" if no metadata fields are available at all.
    """
    chunk_text = chunk.get("chunk_text") or ""

    document_title = _clean(chunk.get("document_title"))
    legal_type = _clean(chunk.get("legal_type"))
    document_number = _clean(chunk.get("document_number"))
    chapter = _clean(chunk.get("chapter"))
    article = _clean(chunk.get("article"))
    clause_num = _extract_clause_number(chunk.get("clause"))

    parts: List[str] = []

    # Doc segment: "Title (Type Number)", degrading gracefully if some
    # sub-fields are missing.
    doc_detail = " ".join(p for p in (legal_type, document_number) if p)
    if document_title and doc_detail:
        parts.append(f"{document_title} ({doc_detail})")
    elif document_title:
        parts.append(document_title)
    elif doc_detail:
        parts.append(doc_detail)

    if chapter:
        parts.append(chapter)

    if article:
        article_num = _article_number(article)
        if article_num and _chunk_opens_with_article(chunk_text, article_num):
            pass  # already duplicated at the top of chunk_text — skip
        else:
            parts.append(article)

    header = " | ".join(parts)

    if clause_num:
        khoan_segment = f"Khoản {clause_num}"
        header = f"{header} | {khoan_segment}" if header else khoan_segment

    return header


def build_contextual_text(chunk: Dict) -> str:
    """
    Build the text that should be EMBEDDED for a chunk: a metadata header
    followed by the (unchanged) chunk body. See build_context_header() for
    the header format and anti-duplication rule.

    If no metadata fields are available at all, returns chunk_text unchanged.
    """
    chunk_text = chunk.get("chunk_text") or ""
    header = build_context_header(chunk)
    if not header:
        return chunk_text
    return f"{header}\n{chunk_text}"


def build_contextual_texts(chunks: List[Dict]) -> List[str]:
    """Convenience batch wrapper around build_contextual_text."""
    return [build_contextual_text(chunk) for chunk in chunks]
