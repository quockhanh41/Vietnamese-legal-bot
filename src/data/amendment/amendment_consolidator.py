"""
Apply amendment instructions to produce consolidated legal text.

Takes a base law's text and a chronological list of AmendmentInstructions,
then applies each modification to produce the consolidated (amended) version.

Key operations:
- MODIFY_ARTICLE: Replace entire article text
- MODIFY_CLAUSE: Replace a clause within an article
- MODIFY_POINT: Replace a point within a clause
- INSERT_ARTICLE/CLAUSE/POINT: Insert new element after existing one
- REPEAL_ARTICLE/CLAUSE/POINT: Remove an element
- REPLACE_WORD: Inline text substitution

Vietnamese article structure patterns:
    Điều X. Title
    1. Clause text...
    2. Clause text...
       a) Point text...
       b) Point text...

The consolidator works in two modes:
1. Full-text mode: operates on complete document text
2. Article-level mode: operates on individual article text chunks
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import date

from src.data.amendment.amendment_extractor import (
    AmendmentInstruction,
    AmendmentAction,
)

logger = logging.getLogger(__name__)


# ── Structure Parsing Patterns ──────────────────────────────────

# Article boundary: "Điều X. Title" or "Điều X: Title"
_RE_ARTICLE_HEADER = re.compile(
    r'(?:^|\n)\s*Điều\s+(\d+[a-z]?)\s*[\.:]\s*(.+?)(?=\n\s*(?:\d+[\.\)]|$))',
    re.IGNORECASE,
)

# Clause boundary: "\n1. " or "\n1) " (must be at start of line)
_RE_CLAUSE_BOUNDARY = re.compile(
    r'\n\s*(\d+[a-z]?)\s*[\.\)]\s+',
    re.IGNORECASE,
)

# Point boundary: "\na) " or "\n- " or "\n+ "
_RE_POINT_BOUNDARY = re.compile(
    r'\n\s*([a-zđ]+)\s*[\.\)]\s+',
    re.IGNORECASE,
)

# The article number in the header
_RE_ARTICLE_NUM = re.compile(r'Điều\s+(\d+[a-z]?)', re.IGNORECASE)


@dataclass
class ConsolidationResult:
    """Result of applying amendments to a base law."""

    law_name: str
    doc_id: int
    original_text: str
    amended_text: str
    applied_instructions: int = 0
    skipped_instructions: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def is_modified(self) -> bool:
        return self.applied_instructions > 0


class AmendmentConsolidator:
    """
    Apply amendment instructions to base law text.

    Usage:
        consolidator = AmendmentConsolidator()
        result = consolidator.apply_to_article(
            article_text="Điều 123. ...", instructions=[...]
        )
        # or
        result = consolidator.apply_to_full_text(
            full_text="...", instructions=[...]
        )
    """

    def __init__(self):
        pass

    # ── Public API ──────────────────────────────────────────────

    def apply(
        self,
        base_text: str,
        instructions: List[AmendmentInstruction],
        law_name: str = "",
        doc_id: int = 0,
    ) -> ConsolidationResult:
        """
        Apply a list of amendment instructions to base law text.

        Instructions are applied in order (assumed chronological).
        """
        result = ConsolidationResult(
            law_name=law_name,
            doc_id=doc_id,
            original_text=base_text,
            amended_text=base_text,
        )

        current_text = base_text

        for i, instr in enumerate(instructions):
            try:
                new_text = self._apply_one(current_text, instr)
                if new_text != current_text:
                    result.applied_instructions += 1
                    current_text = new_text
                else:
                    result.skipped_instructions += 1
            except Exception as e:
                result.warnings.append(
                    f"Instruction {i} ({instr.action.value} @ "
                    f"Điều {instr.target_article}): {e}"
                )
                result.skipped_instructions += 1

        result.amended_text = current_text
        return result

    def apply_to_article(
        self,
        article_text: str,
        instructions: List[AmendmentInstruction],
    ) -> ConsolidationResult:
        """
        Apply instructions targeting a single article.
        Filters instructions to only those matching this article.
        """
        # Extract article number
        m = _RE_ARTICLE_NUM.search(article_text)
        article_num = m.group(1) if m else None

        if article_num:
            relevant = [
                i for i in instructions
                if i.target_article == article_num
            ]
        else:
            relevant = instructions

        return self.apply(article_text, relevant)

    # ── Single Instruction Application ──────────────────────────

    def _apply_one(
        self, text: str, instr: AmendmentInstruction
    ) -> str:
        """Apply a single instruction to the text."""
        action = instr.action

        if action == AmendmentAction.MODIFY_ARTICLE:
            return self._apply_modify_article(text, instr)

        elif action == AmendmentAction.MODIFY_CLAUSE:
            return self._apply_modify_clause(text, instr)

        elif action == AmendmentAction.MODIFY_POINT:
            return self._apply_modify_point(text, instr)

        elif action == AmendmentAction.MODIFY_MULTI:
            return self._apply_modify_multi(text, instr)

        elif action == AmendmentAction.INSERT_POINT:
            return self._apply_insert_point(text, instr)

        elif action == AmendmentAction.INSERT_ARTICLE:
            return self._apply_insert_article(text, instr)

        elif action == AmendmentAction.INSERT_CLAUSE:
            return self._apply_insert_clause(text, instr)

        elif action == AmendmentAction.REPEAL_ARTICLE:
            return self._apply_repeal_article(text, instr)

        elif action == AmendmentAction.REPEAL_CLAUSE:
            return self._apply_repeal_clause(text, instr)

        elif action == AmendmentAction.REPEAL_POINT:
            return self._apply_repeal_point(text, instr)

        elif action == AmendmentAction.REPLACE_WORD:
            return self._apply_replace_word(text, instr)

        else:
            logger.warning(f"Unknown action: {action}")
            return text

    # ── Modify Operations ───────────────────────────────────────

    def _apply_modify_article(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Replace the full text of an article.

        When the instruction has new_text containing the full article
        ("Điều X. Title\\n1. ..."), find the existing article header
        and replace from there to the next article header.
        """
        if not instr.target_article:
            return text

        article_num = instr.target_article
        new_article_text = instr.new_text or ""

        # Find the article in the text
        pattern = re.compile(
            rf'(?:^|\n)\s*Điều\s+{re.escape(article_num)}\s*[\.:]\s*',
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            logger.debug(f"Article {article_num} not found in text")
            return text

        # Find where this article ends (next article header or end of text)
        start = m.start()
        next_article = re.search(
            r'\n\s*Điều\s+\d+[a-z]?\s*[\.:]',
            text[m.end():],
            re.IGNORECASE,
        )
        if next_article:
            end = m.end() + next_article.start()
        else:
            end = len(text)

        # Assemble the replacement
        # If new_text starts with "Điều X.", strip it to avoid double header
        new_text_stripped = new_article_text
        header_match = re.match(
            rf'\s*Điều\s+{re.escape(article_num)}\s*[\.:]\s*',
            new_article_text, re.IGNORECASE
        )
        if header_match:
            # Keep original header, use only the body of new text
            original_header = text[start:m.end()]
            new_body = new_article_text[header_match.end():]
            replacement = original_header + new_body
        else:
            # Keep original header, append new body
            original_header = text[start:m.end()]
            replacement = original_header + "\n" + new_article_text

        return text[:start] + replacement + text[end:]

    def _apply_modify_clause(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Replace the text of a specific clause within an article.

        Finds "khoản X" within the article and replaces its text.
        """
        if not instr.target_article or not instr.target_clause:
            return text

        new_clause_text = instr.new_text
        if not new_clause_text:
            return text

        article_range = self._find_article_range(text, instr.target_article)
        if not article_range:
            return text

        art_start, art_end = article_range
        article_text = text[art_start:art_end]

        # Find the clause within the article
        clause_pattern = re.compile(
            rf'\n\s*{re.escape(instr.target_clause)}\s*[\.\)]\s*',
            re.IGNORECASE,
        )
        m = clause_pattern.search(article_text)
        if not m:
            logger.debug(
                f"Clause {instr.target_clause} not found in "
                f"Article {instr.target_article}"
            )
            return text

        # Find end of this clause (next clause boundary or article end)
        clause_start_in_article = m.start()
        content_start = m.end()
        rest = article_text[content_start:]

        next_clause = _RE_CLAUSE_BOUNDARY.search(rest)
        if next_clause:
            clause_end_in_article = content_start + next_clause.start()
        else:
            clause_end_in_article = len(article_text)

        # Build replacement
        prefix = article_text[:clause_start_in_article]
        clause_header = m.group(0)
        suffix = article_text[clause_end_in_article:]

        new_article_text = prefix + clause_header + new_clause_text + suffix
        return text[:art_start] + new_article_text + text[art_end:]

    def _apply_modify_point(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Replace the text of a specific point within a clause within an article.
        """
        if not instr.target_article or not instr.target_clause or not instr.target_point:
            return text

        new_point_text = instr.new_text
        if not new_point_text:
            return text

        article_range = self._find_article_range(text, instr.target_article)
        if not article_range:
            return text

        art_start, art_end = article_range
        article_text = text[art_start:art_end]

        # Find the clause first
        clause_range = self._find_clause_range(
            article_text, instr.target_clause
        )
        if not clause_range:
            return text

        cl_start, cl_end = clause_range
        clause_text = article_text[cl_start:cl_end]

        # Find the point
        point_pattern = re.compile(
            rf'\n\s*{re.escape(instr.target_point)}\s*[\.\)]\s*',
            re.IGNORECASE,
        )
        m = point_pattern.search(clause_text)
        if not m:
            logger.debug(
                f"Point {instr.target_point} not found in "
                f"khoản {instr.target_clause} Điều {instr.target_article}"
            )
            return text

        point_start = m.start()
        content_start = m.end()
        rest = clause_text[content_start:]

        next_point = _RE_POINT_BOUNDARY.search(rest)
        if next_point:
            point_end = content_start + next_point.start()
        else:
            point_end = len(clause_text)

        prefix = clause_text[:point_start]
        point_header = m.group(0)
        suffix = clause_text[point_end:]

        new_clause_text = prefix + point_header + new_point_text + suffix
        new_article_text = (
            article_text[:cl_start] + new_clause_text + article_text[cl_end:]
        )

        return text[:art_start] + new_article_text + text[art_end:]

    def _apply_modify_multi(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Apply multi-clause modification by processing sub-instructions.
        """
        result = text
        for sub in instr.sub_instructions:
            result = self._apply_one(result, sub)
        return result

    # ── Insert Operations ───────────────────────────────────────

    def _apply_insert_point(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Insert a new point after an existing point within a clause.

        E.g., "Bổ sung điểm d vào sau điểm c khoản 1 Điều 148"
        """
        if not all([instr.insert_after_point, instr.target_article, instr.target_clause]):
            return text

        new_text = instr.new_text or ""
        new_point_label = instr.target_point or ""

        article_range = self._find_article_range(text, instr.target_article)
        if not article_range:
            return text

        art_start, art_end = article_range
        article_text = text[art_start:art_end]

        # Find the clause
        clause_range = self._find_clause_range(article_text, instr.target_clause)
        if not clause_range:
            return text

        cl_start, cl_end = clause_range
        clause_text = article_text[cl_start:cl_end]

        # Find the insertion point (after_point)
        after_pattern = re.compile(
            rf'\n\s*{re.escape(instr.insert_after_point)}\s*[\.\)]\s*',
            re.IGNORECASE,
        )
        m = after_pattern.search(clause_text)
        if not m:
            logger.debug(
                f"Insertion point {instr.insert_after_point} not found"
            )
            return text

        # Find where the after_point's content ends
        content_start = m.end()
        rest = clause_text[content_start:]

        next_point = _RE_POINT_BOUNDARY.search(rest)
        next_clause = _RE_CLAUSE_BOUNDARY.search(rest)
        end_boundary = None
        if next_point and next_clause:
            end_boundary = min(next_point.start(), next_clause.start())
        elif next_point:
            end_boundary = next_point.start()
        elif next_clause:
            end_boundary = next_clause.start()
        else:
            end_boundary = len(rest)

        insertion_point = m.end() + end_boundary

        # Insert the new point
        insertion = f"\n{new_point_label}) {new_text}"
        new_clause_text = (
            clause_text[:insertion_point] + insertion + clause_text[insertion_point:]
        )
        new_article_text = (
            article_text[:cl_start] + new_clause_text + article_text[cl_end:]
        )

        return text[:art_start] + new_article_text + text[art_end:]

    def _apply_insert_article(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Insert a new article after an existing one.

        E.g., "Bổ sung Điều 123a vào sau Điều 123"
        """
        if not instr.insert_after_article:
            return text

        new_article_text = instr.new_text or ""
        new_article_num = instr.target_article or ""

        # Find the "after" article
        after_pattern = re.compile(
            rf'(?:^|\n)\s*Điều\s+{re.escape(instr.insert_after_article)}\s*[\.:]\s*',
            re.IGNORECASE,
        )
        m = after_pattern.search(text)
        if not m:
            logger.debug(
                f"Insert-after Article {instr.insert_after_article} not found"
            )
            return text

        # Find end of this article
        m_start = m.start()
        rest = text[m.end():]
        next_article = re.search(
            r'\n\s*Điều\s+\d+[a-z]?\s*[\.:]',
            rest,
            re.IGNORECASE,
        )
        if next_article:
            insert_at = m.end() + next_article.start()
        else:
            insert_at = len(text)

        # Build the new article
        if not new_article_text.startswith("Điều"):
            new_article_text = f"\nĐiều {new_article_num}. {new_article_text}"

        return text[:insert_at] + "\n" + new_article_text + "\n" + text[insert_at:]

    def _apply_insert_clause(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Insert a new clause after an existing clause within an article.
        """
        if not instr.insert_after_clause or not instr.target_article:
            return text

        new_text = instr.new_text or ""
        new_clause_label = instr.target_clause or ""

        article_range = self._find_article_range(text, instr.target_article)
        if not article_range:
            return text

        art_start, art_end = article_range
        article_text = text[art_start:art_end]

        # Find the "after" clause
        after_pattern = re.compile(
            rf'\n\s*{re.escape(instr.insert_after_clause)}\s*[\.\)]\s*',
            re.IGNORECASE,
        )
        m = after_pattern.search(article_text)
        if not m:
            logger.debug(f"Insert-after clause {instr.insert_after_clause} not found")
            return text

        # Find end of this clause
        rest = article_text[m.end():]
        next_clause = _RE_CLAUSE_BOUNDARY.search(rest)
        if next_clause:
            insert_at = m.end() + next_clause.start()
        else:
            # End of article
            insert_at = len(article_text)

        insertion = f"\n{new_clause_label}. {new_text}"
        new_article_text = (
            article_text[:insert_at] + insertion + article_text[insert_at:]
        )

        return text[:art_start] + new_article_text + text[art_end:]

    # ── Repeal Operations ───────────────────────────────────────

    def _apply_repeal_article(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Remove an entire article from the text.
        Marks it as "[Bãi bỏ]" rather than deleting, to preserve structure.
        """
        if not instr.target_article:
            return text

        pattern = re.compile(
            rf'(?:^|\n)\s*(Điều\s+{re.escape(instr.target_article)}\s*[\.:])\s*',
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            return text

        # Replace the article header with a repealed marker
        return (
            text[:m.start()]
            + f"\n[Điều {instr.target_article} - Bãi bỏ]\n"
            + text[m.end():]
        )

    def _apply_repeal_clause(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Mark a clause as repealed within an article.
        """
        if not instr.target_article or not instr.target_clause:
            return text

        article_range = self._find_article_range(text, instr.target_article)
        if not article_range:
            return text

        art_start, art_end = article_range
        article_text = text[art_start:art_end]

        pattern = re.compile(
            rf'(\n\s*{re.escape(instr.target_clause)}\s*[\.\)]\s*)',
            re.IGNORECASE,
        )
        m = pattern.search(article_text)
        if not m:
            return text

        new_article_text = (
            article_text[:m.start()]
            + f"\n[khoản {instr.target_clause} - Bãi bỏ]\n"
            + article_text[m.end():]
        )

        return text[:art_start] + new_article_text + text[art_end:]

    def _apply_repeal_point(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Mark a point as repealed within a clause.
        """
        if not all([instr.target_article, instr.target_clause, instr.target_point]):
            return text

        article_range = self._find_article_range(text, instr.target_article)
        if not article_range:
            return text

        art_start, art_end = article_range
        article_text = text[art_start:art_end]

        clause_range = self._find_clause_range(article_text, instr.target_clause)
        if not clause_range:
            return text

        cl_start, cl_end = clause_range
        clause_text = article_text[cl_start:cl_end]

        pattern = re.compile(
            rf'(\n\s*{re.escape(instr.target_point)}\s*[\.\)]\s*)',
            re.IGNORECASE,
        )
        m = pattern.search(clause_text)
        if not m:
            return text

        new_clause_text = (
            clause_text[:m.start()]
            + f"\n[điểm {instr.target_point} - Bãi bỏ]\n"
            + clause_text[m.end():]
        )
        new_article_text = (
            article_text[:cl_start] + new_clause_text + article_text[cl_end:]
        )

        return text[:art_start] + new_article_text + text[art_end:]

    # ── Replace Word Operation ──────────────────────────────────

    def _apply_replace_word(self, text: str, instr: AmendmentInstruction) -> str:
        """
        Replace a word/phrase throughout the text.
        Case-insensitive by default.
        """
        if not instr.old_text or not instr.new_text:
            return text

        # Replace word boundaries for single words, substring for phrases
        old = instr.old_text.strip()
        new = instr.new_text.strip()

        if len(old.split()) == 1 and len(old) < 10:
            # Single word: use word-boundary replacement
            text = re.sub(
                rf'\b{re.escape(old)}\b',
                new,
                text,
                flags=re.IGNORECASE,
            )
        else:
            # Phrase: substring replacement
            text = re.sub(
                re.escape(old),
                new,
                text,
                flags=re.IGNORECASE,
            )

        return text

    # ── Structure Helpers ───────────────────────────────────────

    def _find_article_range(
        self, text: str, article_num: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find the (start, end) of an article in the text.
        End is either the next article header or end of text.
        """
        pattern = re.compile(
            rf'(?:^|\n)\s*Điều\s+{re.escape(article_num)}\s*[\.:]\s*',
            re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            return None

        start = m.start()
        rest = text[m.end():]
        next_article = re.search(
            r'\n\s*Điều\s+\d+[a-z]?\s*[\.:]',
            rest,
            re.IGNORECASE,
        )
        end = m.end() + next_article.start() if next_article else len(text)
        return (start, end)

    def _find_clause_range(
        self, article_text: str, clause_num: str
    ) -> Optional[Tuple[int, int]]:
        """
        Find the (start, end) of a clause within article text.
        """
        pattern = re.compile(
            rf'\n\s*{re.escape(clause_num)}\s*[\.\)]\s*',
            re.IGNORECASE,
        )
        m = pattern.search(article_text)
        if not m:
            return None

        start = m.start()
        rest = article_text[m.end():]
        next_boundary = _RE_CLAUSE_BOUNDARY.search(rest)
        end = m.end() + next_boundary.start() if next_boundary else len(article_text)
        return (start, end)

    def extract_article_text(self, full_text: str, article_num: str) -> Optional[str]:
        """
        Extract the text of a single article from full document text.
        Useful for working with individual articles from Qdrant chunks.
        """
        r = self._find_article_range(full_text, article_num)
        if r:
            return full_text[r[0]:r[1]]
        return None

    def list_articles(self, full_text: str) -> List[Tuple[str, str]]:
        """
        List all articles in a document. Returns [(article_num, header_line), ...].
        """
        articles = []
        for m in _RE_ARTICLE_HEADER.finditer(full_text):
            num = m.group(1)
            title = m.group(2).strip()[:100] if m.group(2) else ""
            articles.append((num, title))
        return articles
