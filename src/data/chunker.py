"""
Regex-based hierarchical chunking for Vietnamese legal documents.

Strategy (based on actual dataset analysis — all documents are plain text):
1. Split by Vietnamese legal structure: Chương → Điều → Khoản → Điểm
2. Each chunk inherits parent metadata (chapter, article, clause)
3. Size control: min 200 chars, max 1000 chars
4. Merge small consecutive chunks, split oversized at sentence boundary

Dataset reality: NO Markdown headers exist. All documents follow Vietnamese
administrative format (e.g., "CHƯƠNG I...", "Điều 1...", "1. ...", "a) ...").
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class LegalChunk:
    """A single chunk of a legal document with hierarchical metadata."""
    chunk_text: str
    document_id: int
    document_title: str
    document_number: str
    legal_type: str
    issuing_authority: str
    issuance_date: str
    url: str
    chapter: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    chunk_index: int = 0
    parent_chunk_text: Optional[str] = None  # Full article text for context overlap

    def to_dict(self) -> dict:
        return {
            "chunk_text": self.chunk_text,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_number": self.document_number,
            "legal_type": self.legal_type,
            "issuing_authority": self.issuing_authority,
            "issuance_date": str(self.issuance_date),
            "url": self.url,
            "chapter": self.chapter,
            "article": self.article,
            "clause": self.clause,
            "chunk_index": self.chunk_index,
            "parent_chunk_text": self.parent_chunk_text,
        }


class LegalDocumentChunker:
    """
    Regex-based hierarchical chunker for Vietnamese legal documents.

    Structure detection (in order of priority):
    - Level 1: Chapter  (Chương I, Chương II, CHƯƠNG 1...)
    - Level 2: Section  (Mục 1, MỤC I...)
    - Level 3: Article  (Điều 1, Điều 15...)
    - Level 4: Clause   (1. , 2. , Khoản 1...)
    - Level 5: Point    (a), b), điểm a...)

    Each chunk is an Article or Clause level, inheriting Chapter/Section metadata.
    """

    # --- Regex patterns for Vietnamese legal document structure ---

    # "CHƯƠNG I. QUY ĐỊNH CHUNG" or "Chương 1: ..."
    CHAPTER_PATTERN = re.compile(
        r'^\s*CHƯƠNG\s+([IVXLCDM]+|\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    # "Mục 1. ..." or "MỤC I: ..."
    SECTION_PATTERN = re.compile(
        r'^\s*MỤC\s+([IVXLCDM]+|\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    # "Điều 1. Tên điều" or "Điều 15: ..."
    ARTICLE_PATTERN = re.compile(
        r'^\s*Điều\s+(\d+)\s*[\.:\)]\s*(.*)',
        re.IGNORECASE,
    )

    # "1. Nội dung khoản" — numbered clause at start of line
    CLAUSE_PATTERN = re.compile(
        r'^\s*(\d+)\s*[\.\)]\s+(.+)',
    )

    # "a) ...", "b) ...", "điểm a khoản 1"
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

        # Only used for splitting oversized chunks (no langchain header parsing)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, row: dict) -> List[LegalChunk]:
        """
        Chunk a single legal document.

        Args:
            row: Dictionary with keys: id, title, content, document_number,
                 legal_type, issuing_authority, issuance_date, url

        Returns:
            List of LegalChunk objects
        """
        content = row.get("content", "")
        if not content or len(content) < self.min_chunk_size:
            return []

        # Step 1: Split by Vietnamese legal structure (regex-based)
        structural_chunks = self._split_by_structure(content)

        # Step 2: Size control (merge small, split large)
        sized_chunks = self._control_chunk_sizes(structural_chunks)

        # Step 3: Build LegalChunk objects with metadata
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
        """
        Split content by Vietnamese legal document structure using regex.

        Walks line-by-line, detecting Chương → Mục → Điều → Khoản → Điểm.
        Each structural boundary creates a new chunk with inherited metadata.
        """
        lines = content.split("\n")
        chunks = []

        # State tracking
        current_chapter = None
        current_section = None
        current_article = None
        current_clause = None
        buffer = ""

        # Track full article text for parent_chunk_text
        article_buffer = ""

        def flush_buffer():
            """Flush current buffer as a chunk with current metadata."""
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

            # --- Detect structural boundaries ---

            # Chapter: "CHƯƠNG I. QUY ĐỊNH CHUNG"
            chapter_match = self.CHAPTER_PATTERN.match(stripped)
            if chapter_match:
                flush_buffer()
                current_chapter = stripped
                current_section = None
                current_article = None
                current_clause = None
                continue

            # Section: "Mục 1. ..."
            section_match = self.SECTION_PATTERN.match(stripped)
            if section_match:
                flush_buffer()
                current_section = stripped
                current_article = None
                current_clause = None
                continue

            # Article: "Điều 1. Tên điều"
            article_match = self.ARTICLE_PATTERN.match(stripped)
            if article_match:
                flush_buffer()
                current_article = stripped
                current_clause = None
                article_buffer = stripped
                buffer = stripped
                continue

            # Clause: "1. Nội dung"
            clause_match = self.CLAUSE_PATTERN.match(stripped)
            if clause_match and current_article:
                flush_buffer()
                current_clause = stripped
                buffer = stripped
                if current_article:
                    article_buffer += "\n" + stripped
                continue

            # Point: "a) ..."
            point_match = self.POINT_PATTERN.match(stripped)
            if point_match and current_article:
                # Points are small — keep accumulating in current buffer
                pass

            # --- Accumulate into buffer ---
            buffer += "\n" + stripped if buffer else stripped
            if current_article:
                article_buffer += "\n" + stripped if article_buffer else stripped

        # Final flush
        flush_buffer()

        # If no structure detected at all, treat whole document as one chunk
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
        """Merge small chunks, split large chunks."""
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

            # If buffer + this chunk fits, merge
            if buffer["text"] and len(buffer["text"]) + len(text) <= self.max_chunk_size:
                buffer["text"] += "\n\n" + text
                # Keep the most specific metadata (prefer non-None)
                buffer["chapter"] = chunk.get("chapter") or buffer.get("chapter")
                buffer["article"] = chunk.get("article") or buffer.get("article")
                buffer["clause"] = chunk.get("clause") or buffer.get("clause")
                buffer["parent_text"] = chunk.get("parent_text") or buffer.get("parent_text")
            else:
                # Flush buffer
                if buffer["text"]:
                    result.append(dict(buffer))
                buffer = {
                    "text": text,
                    "chapter": chunk.get("chapter"),
                    "article": chunk.get("article"),
                    "clause": chunk.get("clause"),
                    "parent_text": chunk.get("parent_text"),
                }

        # Flush final buffer
        if buffer["text"]:
            result.append(dict(buffer))

        # Split oversized chunks
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
