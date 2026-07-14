"""
S3: Hierarchical chunking — regex-based Vietnamese legal document structure.

Strategy: Split by Chương → Mục → Điều → Khoản → Điểm using regex patterns.
Each chunk inherits parent metadata. Size control via merge/split.
"""

import re
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data.chunker import LegalChunk, BaseChunker


class HierarchicalChunker(BaseChunker):
    """
    Regex-based hierarchical chunker for Vietnamese legal documents.

    Structure detection (in order of priority):
    - Level 1: Chapter  (Chương I, CHƯƠNG 1...)
    - Level 2: Section  (Mục 1, MỤC I...)
    - Level 3: Article  (Điều 1, Điều 15...)
    - Level 4: Clause   (1. , 2. , Khoản 1...)
    - Level 5: Point    (a), b), điểm a...)

    Each chunk is an Article or Clause level, inheriting Chapter/Section metadata.
    """

    CHAPTER_PATTERN = re.compile(
        r'^\s*CHƯƠNG\s+([IVXLCDM]+|\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    SECTION_PATTERN = re.compile(
        r'^\s*MỤC\s+([IVXLCDM]+|\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    # Suffix letter tolerated (e.g. "Điều 105a.") — M2.7 (2026-07-13):
    # "Văn bản hợp nhất" (consolidated) documents splice amendment articles
    # in place using letter-suffixed numbers (Điều 8a, 11a, 12a...); without
    # this the header wasn't recognized as a new boundary at all and its
    # content silently merged into the preceding Điều's chunk (WRONG_LABEL
    # on the 600-doc gate). See docs/notes/chunker-thongtu-diagnosis-2026-07-13.md.
    ARTICLE_PATTERN = re.compile(
        r'^\s*Điều\s+(\d+[a-z]?)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    CLAUSE_PATTERN = re.compile(
        r'^\s*(\d+)\s*[\.\)]\s+(.+)',
    )

    POINT_PATTERN = re.compile(
        r'^\s*([a-dđ]\s*[\)\.]\s*.+)',
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, row: dict) -> List[LegalChunk]:
        content = row.get("content", "")
        if not content or len(content) < self.min_chunk_size:
            return []

        structural_chunks = self._split_by_structure(content)
        sized_chunks = self._control_chunk_sizes(structural_chunks)

        legal_chunks = []
        for i, chunk_data in enumerate(sized_chunks):
            chunk = LegalChunk(
                chunk_text=chunk_data["text"],
                document_id=row["id"],
                document_title=row["title"],
                document_number=str(row.get("document_number", "")),
                legal_type=row.get("legal_type", ""),
                issuing_authority=row.get("issuing_authority", ""),
                issuance_date=str(row.get("issuance_date", "")),
                url=str(row.get("url", "")),
                chapter=chunk_data.get("chapter"),
                article=chunk_data.get("article"),
                clause=chunk_data.get("clause"),
                chunk_index=i,
                parent_chunk_text=chunk_data.get("parent_text"),
            )
            legal_chunks.append(chunk)

        return legal_chunks

    def _split_by_structure(self, content: str) -> List[dict]:
        lines = content.split("\n")
        chunks = []
        current_chapter = None
        current_section = None
        current_article = None
        current_clause = None
        buffer = ""
        article_buffer = ""

        # Bug 4 fix: article_buffer is reset every time a new Điều starts, so
        # a parent_text snapshot taken at flush time only ever sees the
        # partial text accumulated *so far* (e.g. header + Khoản 1, missing
        # Khoản 2+). We instead record, for every Điều label, the FULL text
        # accumulated by the time we leave it — then do a second pass after
        # scanning the whole document to stamp that full text onto every
        # chunk belonging to that Điều (see chunk_document's caller below).
        article_full_text: dict = {}

        def close_article():
            """Snapshot the current article's fully-accumulated text before
            current_article/article_buffer get reset for the next Điều (or
            cleared by a Chương/Mục boundary)."""
            if current_article is not None:
                article_full_text[current_article] = article_buffer.strip()

        def flush_buffer():
            nonlocal buffer, article_buffer
            text = buffer.strip()
            if text:
                # Bug 3 fix: any non-empty buffer that belongs to a real
                # Điều must be kept even if short (e.g. a one-line "hiệu lực
                # thi hành" article) — dropping it would silently remove
                # that Điều from the index. Only unattributed scraps that
                # aren't part of any Điều (e.g. stray text before the first
                # Điều) are still filtered by the minimum-length heuristic.
                if current_article is not None or len(text) >= self.min_chunk_size // 2:
                    chunks.append({
                        "text": text,
                        "chapter": current_chapter,
                        "article": current_article,
                        "clause": current_clause,
                        # Placeholder — overwritten below in the second pass
                        # once article_full_text is complete (Bug 4).
                        "parent_text": text,
                    })
            buffer = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if buffer:
                    buffer += "\n"
                if article_buffer:
                    article_buffer += "\n"
                continue

            chapter_match = self.CHAPTER_PATTERN.match(stripped)
            if chapter_match:
                flush_buffer()
                close_article()
                current_chapter = stripped
                current_section = None
                current_article = None
                current_clause = None
                continue

            section_match = self.SECTION_PATTERN.match(stripped)
            if section_match:
                flush_buffer()
                close_article()
                current_section = stripped
                current_article = None
                current_clause = None
                continue

            article_match = self.ARTICLE_PATTERN.match(stripped)
            if article_match:
                flush_buffer()
                close_article()
                current_article = stripped
                current_clause = None
                article_buffer = stripped
                buffer = stripped
                continue

            clause_match = self.CLAUSE_PATTERN.match(stripped)
            if clause_match and current_article:
                flush_buffer()
                current_clause = stripped
                buffer = stripped
                if current_article:
                    article_buffer += "\n" + stripped
                continue

            point_match = self.POINT_PATTERN.match(stripped)
            if point_match and current_article:
                pass

            buffer += "\n" + stripped if buffer else stripped
            if current_article:
                article_buffer += "\n" + stripped if article_buffer else stripped

        flush_buffer()
        close_article()

        if not chunks:
            chunks.append({
                "text": content.strip(),
                "chapter": None,
                "article": None,
                "clause": None,
                "parent_text": content.strip(),
            })
            return chunks

        # Second pass (Bug 4): stamp every chunk belonging to a given Điều
        # with that Điều's full accumulated text, so all chunks of the same
        # Điều share one identical, complete parent_text.
        for chunk in chunks:
            article = chunk.get("article")
            if article is not None and article in article_full_text:
                chunk["parent_text"] = article_full_text[article]

        return chunks

    def _control_chunk_sizes(self, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []

        def start_buffer(chunk: dict) -> dict:
            clause = chunk.get("clause")
            return {
                "text": chunk["text"],
                "chapter": chunk.get("chapter"),
                "article": chunk.get("article"),
                "parent_text": chunk.get("parent_text"),
                # Bug 1 fix: track every distinct clause folded into this
                # buffer so we can tell, at flush time, whether the merged
                # chunk still represents exactly one Khoản (keep its label)
                # or several (the chunk is now Điều-level — no single
                # clause label is correct for it, so use None).
                "_clauses": {clause} if clause is not None else set(),
            }

        def finalize(buf: dict) -> dict:
            clauses = buf.pop("_clauses")
            buf["clause"] = next(iter(clauses)) if len(clauses) == 1 else None
            return buf

        result = []
        buffer = None

        for chunk in chunks:
            text = chunk["text"]
            # Bug 1 fix: only merge consecutive structural chunks that
            # belong to the SAME Điều (None == None covers preamble text
            # that precedes the first Điều). Merging across an Điều
            # boundary is exactly what produced the WRONG_LABEL chunks
            # diagnosed in docs/notes/chunker-misalignment-diagnosis-2026-07-10.md
            # — a chunk's article label must never span text from more
            # than one Điều.
            same_article = buffer is not None and chunk.get("article") == buffer.get("article")
            fits = buffer is not None and len(buffer["text"]) + len(text) <= self.max_chunk_size

            if buffer is not None and same_article and fits:
                buffer["text"] += "\n\n" + text
                buffer["chapter"] = buffer.get("chapter") or chunk.get("chapter")
                buffer["parent_text"] = buffer.get("parent_text") or chunk.get("parent_text")
                clause = chunk.get("clause")
                if clause is not None:
                    buffer["_clauses"].add(clause)
            else:
                if buffer is not None:
                    result.append(finalize(buffer))
                buffer = start_buffer(chunk)

        if buffer is not None:
            result.append(finalize(buffer))

        final_result = []
        for chunk in result:
            if len(chunk["text"]) > self.max_chunk_size:
                sub_texts = self.text_splitter.split_text(chunk["text"])
                # Bug 2 (self-heals once Bug 1 is fixed): every sub_texts
                # piece comes from a chunk that is now guaranteed to belong
                # to a single Điều, so copying "article"/"parent_text" onto
                # each sub-chunk is safe. "clause" can still be imprecise
                # here (a sub-chunk born from splitting a multi-Khoản,
                # clause=None buffer may land entirely inside one Khoản,
                # but we don't re-derive a tighter label) — acceptable per
                # the diagnosis, since the article label (what drove the
                # WRONG_LABEL bug) is always correct.
                for sub in sub_texts:
                    sub_chunk = dict(chunk)
                    sub_chunk["text"] = sub
                    final_result.append(sub_chunk)
            else:
                final_result.append(chunk)

        return final_result
