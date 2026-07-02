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

    ARTICLE_PATTERN = re.compile(
        r'^\s*Điều\s+(\d+)\s*[\.:\)]\s*(.*)',
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

        def flush_buffer():
            nonlocal buffer, article_buffer
            text = buffer.strip()
            if text and len(text) >= self.min_chunk_size // 2:
                chunks.append({
                    "text": text,
                    "chapter": current_chapter,
                    "article": current_article,
                    "clause": current_clause,
                    "parent_text": article_buffer.strip() if current_article else text,
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
                current_chapter = stripped
                current_section = None
                current_article = None
                current_clause = None
                continue

            section_match = self.SECTION_PATTERN.match(stripped)
            if section_match:
                flush_buffer()
                current_section = stripped
                current_article = None
                current_clause = None
                continue

            article_match = self.ARTICLE_PATTERN.match(stripped)
            if article_match:
                flush_buffer()
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

        if not chunks:
            chunks.append({
                "text": content.strip(),
                "chapter": None,
                "article": None,
                "clause": None,
                "parent_text": content.strip(),
            })

        return chunks

    def _control_chunk_sizes(self, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []

        result = []
        buffer = {
            "text": "",
            "chapter": None,
            "article": None,
            "clause": None,
            "parent_text": None,
        }

        for chunk in chunks:
            text = chunk["text"]

            if buffer["text"] and len(buffer["text"]) + len(text) <= self.max_chunk_size:
                buffer["text"] += "\n\n" + text
                buffer["chapter"] = chunk.get("chapter") or buffer.get("chapter")
                buffer["article"] = chunk.get("article") or buffer.get("article")
                buffer["clause"] = chunk.get("clause") or buffer.get("clause")
                buffer["parent_text"] = chunk.get("parent_text") or buffer.get("parent_text")
            else:
                if buffer["text"]:
                    result.append(dict(buffer))
                buffer = {
                    "text": text,
                    "chapter": chunk.get("chapter"),
                    "article": chunk.get("article"),
                    "clause": chunk.get("clause"),
                    "parent_text": chunk.get("parent_text"),
                }

        if buffer["text"]:
            result.append(dict(buffer))

        final_result = []
        for chunk in result:
            if len(chunk["text"]) > self.max_chunk_size:
                sub_texts = self.text_splitter.split_text(chunk["text"])
                for sub in sub_texts:
                    sub_chunk = dict(chunk)
                    sub_chunk["text"] = sub
                    final_result.append(sub_chunk)
            else:
                final_result.append(chunk)

        return final_result
