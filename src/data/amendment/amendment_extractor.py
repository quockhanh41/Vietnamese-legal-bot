"""
Extract amendment instructions from Vietnamese amendment laws ("Luật sửa đổi, bổ sung").

Vietnamese amendment laws follow a characteristic structure:
  ┌─────────────────────────────────────────────────────────┐
  │ Preamble: "Căn cứ Hiến pháp... Quốc hội ban hành       │
  │           Luật sửa đổi, bổ sung một số điều của [TARGET] │
  │                                                         │
  │ Điều 1. Sửa đổi, bổ sung một số điều của [TARGET]       │
  │                                                         │
  │   1. Sửa đổi, bổ sung Điều X như sau:                   │
  │      "Điều X. ..."  (full article replacement)          │
  │                                                         │
  │   2. Sửa đổi, bổ sung khoản Y Điều Z như sau:           │
  │      "new clause text"                                  │
  │                                                         │
  │   3. Bổ sung điểm X vào sau điểm Y khoản Z Điều W:      │
  │      "new point text"                                   │
  │                                                         │
  │   4. Bãi bỏ Điều V                                      │
  │                                                         │
  │ Điều 2. Hiệu lực thi hành                               │
  │   "Luật này có hiệu lực từ ngày ..."                    │
  └─────────────────────────────────────────────────────────┘

Key design decisions:
- One amendment law can amend MULTIPLE target laws (e.g., 2021 amendment
  modifies both BLTTHS and Luật Tổ chức CQĐTHS)
- Numbering resets between target-law sections
- Amendment actions include MODIFY (sửa đổi), INSERT (bổ sung),
  REPEAL (bãi bỏ), and REPLACE_WORD (thay thế)
- Multi-clause modifications have sub-items a), b), c), etc.
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ── Enums & Data Classes ────────────────────────────────────────


class AmendmentAction(str, Enum):
    """Type of amendment action."""
    MODIFY_ARTICLE = "modify_article"           # Full article rewrite
    MODIFY_CLAUSE = "modify_clause"             # Single clause rewrite
    MODIFY_POINT = "modify_point"               # Single point rewrite
    MODIFY_MULTI = "modify_multi"               # Multi-clause modification (sub-items)
    MODIFY_MULTI_ARTICLE = "modify_multi_article"  # Multiple articles at once
    INSERT_POINT = "insert_point"               # Add new point after existing one
    INSERT_ARTICLE = "insert_article"           # Add new article after existing one
    INSERT_CLAUSE = "insert_clause"             # Add new clause after existing one
    REPEAL_ARTICLE = "repeal_article"           # Remove entire article
    REPEAL_CLAUSE = "repeal_clause"             # Remove clause
    REPEAL_POINT = "repeal_point"               # Remove point
    REPLACE_WORD = "replace_word"               # Inline word/phrase substitution
    BATCH_REPLACE = "batch_replace"             # Batch word replacements with sub-items
    BATCH_INSERT = "batch_insert"               # Batch word/phrase insertions with sub-items
    UNKNOWN = "unknown"


@dataclass
class AmendmentInstruction:
    """A single amendment instruction extracted from an amendment law."""

    # Source
    amending_doc_id: int
    amending_law_name: str

    # Target
    target_law_name: str                   # Which law is being amended
    action: AmendmentAction = AmendmentAction.UNKNOWN

    # What is being changed
    target_article: Optional[str] = None    # "123"
    target_clause: Optional[str] = None     # "1"
    target_point: Optional[str] = None      # "a"

    # For multi-clause: list of clauses being amended
    target_clauses: List[str] = field(default_factory=list)

    # For INSERT actions: where to insert
    insert_after_article: Optional[str] = None
    insert_after_clause: Optional[str] = None
    insert_after_point: Optional[str] = None

    # New content
    new_text: Optional[str] = None          # Replacement/insertion text
    old_text: Optional[str] = None          # For REPLACE_WORD: text being replaced

    # Structure
    instruction_number: str = ""           # "1", "2a", etc.
    sub_instructions: List[AmendmentInstruction] = field(default_factory=list)

    # Metadata
    raw_instruction: str = ""              # Original text of the instruction
    confidence: float = 1.0

    def __repr__(self) -> str:
        loc = []
        if self.target_article:
            loc.append(f"Điều {self.target_article}")
        if self.target_clause:
            loc.append(f"khoản {self.target_clause}")
        if self.target_point:
            loc.append(f"điểm {self.target_point}")

        location = " ".join(loc) if loc else "N/A"
        return (
            f"<Amendment {self.action.value} @ {location} "
            f"in '{self.target_law_name[:40]}'>"
        )


# ── Regex Patterns ──────────────────────────────────────────────

# ── Amendment Detection ──

# Detect amendment law from title: "Sửa đổi, bổ sung..."
_RE_IS_AMENDMENT = re.compile(
    r'(?:sửa\s+đổi|bổ\s+sung|sửa\s+đổi,\s*bổ\s+sung)',
    re.IGNORECASE,
)

# Extract target law from title: "Luật sửa đổi Bộ luật Hình sự"
_RE_TARGET_FROM_TITLE = re.compile(
    r'(?:Luật|Nghị\s+quyết)\s+(?:sửa\s+đổi,\s*bổ\s+sung\s+'
    r'(?:một\s+số\s+điều\s+của\s+)?|sửa\s+đổi\s+)'
    r'(.+?)(?:\s+số\s+[\d/]+[A-Za-z]*)?\s*$',
    re.IGNORECASE,
)

# ── Section Detection ──

# Find sections that target specific laws
_RE_TARGET_SECTION = re.compile(
    r'(?:^|\n)\s*(?:Điều\s+1[\.:]\s*)?'
    r'Sửa\s+đổi,\s*bổ\s+sung\s+(?:một\s+số\s+điều\s+của\s+)?'
    r'(.+?)\s*(?:số\s+[\d/]+[A-Za-z]*)?'
    r'(?=\n\s*\d+[\.\)]\s*|\n)',
    re.IGNORECASE,
)

# Find "Hiệu lực thi hành" boundary (end of amendments)
_RE_EFFECTIVENESS_BOUNDARY = re.compile(
    r'(?:^|\n)\s*(?:Điều\s+\d+[\.:]\s*)?(?:Hiệu\s+lực\s+thi\s+hành|Điều\s+khoản\s+thi\s+hành)',
    re.IGNORECASE,
)

# ── Instruction Detection ──

# Top-level numbered instruction: "1. Sửa đổi..." or "1) Sửa đổi..."
_RE_INSTRUCTION_BOUNDARY = re.compile(
    r'(?:^|\n)\s*(\d+[a-z]?)\s*[\.\)]\s*'
    r'(Sửa\s+đổi|Bổ\s+sung|Bãi\s+bỏ|Thay\s+thế|Bổ\s+xung)',
    re.IGNORECASE,
)

# Sub-item: "a) Sửa đổi..." (only within a parent instruction)
_RE_SUBITEM = re.compile(
    r'\n\s*([a-zđ]\)|(?:[a-zđ])\s*[\.\)])\s*'
    r'(Sửa\s+đổi|Bổ\s+sung|Bãi\s+bỏ|Thay\s+thế|Bổ\s+xung)',
    re.IGNORECASE,
)

# ── Action Classification Patterns ──

# "Sửa đổi, bổ sung Điều X như sau:"
_RE_MODIFY_ARTICLE = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+(?:tên\s+điều\s+và\s+)?(?:khoản\s+\d+\s+)?Điều\s+(\d+[a-z]?)\s+'
    r'(?:của\s+.+?\s+)?như\s+sau\s*:?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# "Sửa đổi, bổ sung tên điều và khoản X Điều Y như sau:"
_RE_MODIFY_RENAME_AND_CLAUSE = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+tên\s+điều\s+và\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung tên điều, các khoản 1, 2 và 3 Điều X như sau:"
_RE_MODIFY_RENAME_MULTI = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+tên\s+điều,\s*các\s+khoản\s+'
    r'([\d\s,;]+(?:và\s+\d+[a-z]?)?)\s+'
    r'Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung tên điều, khoản X và khoản Y Điều Z như sau:"
_RE_MODIFY_RENAME_TWO_CLAUSES = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+tên\s+điều,\s*khoản\s+(\d+[a-z]?)\s+và\s+khoản\s+(\d+[a-z]?)\s+'
    r'Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bổ sung một số từ, cụm từ, dấu... vào các điều, khoản, điểm" (batch insert words)
_RE_BATCH_INSERT = re.compile(
    r'Bổ\s+sung\s+(?:một\s+số\s+)?(?:từ|dấu|(?:các\s+)?từ,\s*cụm\s+từ)',
    re.IGNORECASE,
)

# "Bổ sung điểm X vào khoản Y Điều Z như sau:" (no explicit after-point)
_RE_INSERT_POINT_INTO_CLAUSE = re.compile(
    r'Bổ\s+sung\s+điểm\s+([a-zđ]+)\s+vào\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bổ sung khoản X vào Điều Y như sau:" (no explicit after-clause)
_RE_INSERT_CLAUSE_INTO_ARTICLE = re.compile(
    r'Bổ\s+sung\s+khoản\s+(\d+[a-z]?)\s+vào\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bổ sung Điều X như sau:" (insert new article, no explicit "after")
_RE_INSERT_ARTICLE_NEW = re.compile(
    r'Bổ\s+sung\s+Điều\s+(\d+[a-z]+)\s+như\s+sau\s*:?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# "Sửa đổi, bổ sung khoản X Điều Y như sau:"
_RE_MODIFY_CLAUSE = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung điểm X khoản Y Điều Z như sau:"
_RE_MODIFY_POINT = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+điểm\s+([a-z])\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung các khoản 1, 2 và 3 Điều X như sau:"
_RE_MODIFY_MULTI = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+các\s+khoản\s+'
    r'([\d\s,;]+(?:và\s+\d+[a-z]?)?)\s+'
    r'Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung khoản X và khoản Y Điều Z như sau:" (without "các")
_RE_MODIFY_MULTI_NO_CAC = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+khoản\s+(\d+[a-z]?)\s+và\s+khoản\s+(\d+[a-z]?)\s+'
    r'Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung Điều X và Điều Y như sau:" (two articles)
_RE_MODIFY_TWO_ARTICLES = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+Điều\s+(\d+[a-z]?)\s+và\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Sửa đổi, bổ sung các điều X, Y, Z như sau:" (multiple articles)
_RE_MODIFY_MULTI_ARTICLE = re.compile(
    r'Sửa\s+đổi,\s*bổ\s+sung\s+các\s+điều\s+'
    r'([\d\s,;]+(?:và\s+\d+[a-z]?)?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bổ sung điểm X vào sau điểm Y khoản Z Điều W như sau:"
_RE_INSERT_POINT = re.compile(
    r'Bổ\s+sung\s+điểm\s+([a-zđ]+)\s+vào\s+sau\s+điểm\s+([a-zđ]+)\s+'
    r'khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bổ sung Điều X vào sau Điều Y như sau:"
_RE_INSERT_ARTICLE = re.compile(
    r'Bổ\s+sung\s+Điều\s+(\d+[a-z]?)\s+vào\s+sau\s+Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bổ sung khoản X vào sau khoản Y Điều Z như sau:"
_RE_INSERT_CLAUSE = re.compile(
    r'Bổ\s+sung\s+khoản\s+(\d+[a-z]?)\s+vào\s+sau\s+khoản\s+(\d+[a-z]?)\s+'
    r'Điều\s+(\d+[a-z]?)\s+như\s+sau\s*:?',
    re.IGNORECASE,
)

# "Bãi bỏ Điều X" / "Bãi bỏ khoản X Điều Y" / "Bãi bỏ điểm X khoản Y Điều Z"
_RE_REPEAL_ARTICLE = re.compile(
    r'Bãi\s+bỏ\s+Điều\s+(\d+[a-z]?)\s*(?:của\s+.*?)?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

_RE_REPEAL_CLAUSE = re.compile(
    r'Bãi\s+bỏ\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)',
    re.IGNORECASE,
)

_RE_REPEAL_POINT = re.compile(
    r'Bãi\s+bỏ\s+điểm\s+([a-zđ]+)\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)',
    re.IGNORECASE,
)

# "Thay thế cụm từ/từ/dấu "X" bằng "Y""
_RE_REPLACE_WORD = re.compile(
    r'Thay\s+thế\s+(?:cụm\s+từ|từ|dấu)\s+[“\"“](.+?)[”\"”]\s+bằng\s+[“\"“](.+?)[”\"”]',
    re.IGNORECASE,
)

# "Thay thế từ, cụm từ, dấu phẩy tại các điều, khoản, điểm sau đây..."
_RE_BATCH_REPLACE = re.compile(
    r'Thay\s+thế\s+(?:từ|dấu|(?:các\s+)?từ,\s*cụm\s+từ)',
    re.IGNORECASE,
)

# ── Target Law in Instruction ──

# Detect when an instruction targets a DIFFERENT law than the current section
# e.g., "Sửa đổi, bổ sung Điều 44 của Luật Tổ chức Cơ quan điều tra hình sự"
_RE_TARGET_IN_INSTRUCTION = re.compile(
    r'(?:của|các)\s+((?:Luật|Bộ\s+luật|Nghị\s+định|Thông\s+tư|Pháp\s+lệnh)\s+'
    r'.+?)(?:\s+số\s+[\d/]+[A-Za-z]*)?\s*(?:như\s+sau|$|\.)',
    re.IGNORECASE,
)

# ── Text Extraction ──

# Extract quoted new text after an instruction (smart quotes "..." or "...")
_RE_QUOTED_TEXT = re.compile(
    r'[“\"“](.+?)[”\"”]',
    re.DOTALL,
)

# Pattern to find the start of the next top-level instruction
_RE_NEXT_INSTRUCTION = re.compile(
    r'\n\s*\d+[\.\)]\s*(?:Sửa|Bổ|Bãi|Thay)',
    re.IGNORECASE,
)

# ── Article number list parsing ──

_RE_ARTICLE_NUMBERS = re.compile(r'(\d+[a-z]?)')


# ── Main Extractor Class ────────────────────────────────────────


class AmendmentExtractor:
    """
    Extract amendment instructions from Vietnamese amendment laws.

    Usage:
        extractor = AmendmentExtractor()
        instructions = extractor.extract(content=text, doc_id=123,
                                         title="Luật sửa đổi Bộ luật Hình sự 2017")
        print(extractor.stats(instructions))
    """

    # Instruction-level patterns in priority order (most specific first)
    INSTRUCTION_PATTERNS: List[Tuple[str, 're.Pattern']] = [
        ("modify_multi", _RE_MODIFY_RENAME_TWO_CLAUSES),     # "tên điều, khoản X và khoản Y"
        ("modify_multi", _RE_MODIFY_RENAME_MULTI),           # "tên điều, các khoản X,Y,Z"
        ("modify_clause", _RE_MODIFY_RENAME_AND_CLAUSE),     # "tên điều và khoản X"
        ("modify_clause", _RE_MODIFY_CLAUSE),
        ("modify_point", _RE_MODIFY_POINT),
        ("modify_multi", _RE_MODIFY_MULTI),
        ("modify_multi", _RE_MODIFY_MULTI_NO_CAC),           # "khoản X và khoản Y"
        ("modify_multi_article", _RE_MODIFY_MULTI_ARTICLE),  # "các điều X, Y, Z"
        ("modify_multi_article", _RE_MODIFY_TWO_ARTICLES),   # "Điều X và Điều Y"
        ("insert_article", _RE_INSERT_ARTICLE_NEW),          # "Bổ sung Điều X" (new)
        ("insert_point", _RE_INSERT_POINT_INTO_CLAUSE),      # "Bổ sung điểm X vào khoản Y"
        ("insert_point", _RE_INSERT_POINT),
        ("insert_clause", _RE_INSERT_CLAUSE_INTO_ARTICLE),   # "Bổ sung khoản X vào Điều Y"
        ("insert_article", _RE_INSERT_ARTICLE),
        ("insert_clause", _RE_INSERT_CLAUSE),
        ("repeal_article", _RE_REPEAL_ARTICLE),
        ("repeal_clause", _RE_REPEAL_CLAUSE),
        ("repeal_point", _RE_REPEAL_POINT),
        ("replace_word", _RE_REPLACE_WORD),
        ("batch_replace", _RE_BATCH_REPLACE),
        ("batch_insert", _RE_BATCH_INSERT),
        ("modify_article", _RE_MODIFY_ARTICLE),              # catch-all last
    ]

    def __init__(self):
        pass

    # ── Public API ──────────────────────────────────────────────

    @staticmethod
    def is_amendment_law(title: str) -> bool:
        """Check if a document is an amendment law based on its title."""
        if not title:
            return False
        return bool(_RE_IS_AMENDMENT.search(title))

    def extract_target_laws(self, title: str, content: str = "") -> List[str]:
        """
        Extract the names of laws targeted by this amendment.

        Returns a list because some amendment laws modify multiple laws.
        Result is deduplicated (case-insensitive).
        """
        targets = []
        seen_normalized = set()

        def _add(name: str):
            n = self._clean_law_name(name)
            if not n:
                return
            key = n.lower()
            if key not in seen_normalized:
                seen_normalized.add(key)
                targets.append(n)

        # Strategy 1: Extract from title
        m = _RE_TARGET_FROM_TITLE.search(title)
        if m:
            _add(m.group(1))

        # Strategy 2: Scan content for explicit target mentions
        if content:
            for m in _RE_TARGET_SECTION.finditer(content):
                _add(m.group(1))

        # Strategy 3: Heuristic - check title for "của [Law Name]"
        if not targets:
            m_alt = re.search(
                r'(?:của|các)\s+((?:Bộ\s+)?[Ll]uật\s+.+?)(?:\s+số\s+[\d/]+|$)',
                title, re.IGNORECASE
            )
            if m_alt:
                _add(m_alt.group(1))

        return targets

    def extract(
        self,
        content: str,
        doc_id: int,
        title: str = "",
    ) -> List[AmendmentInstruction]:
        """
        Extract all amendment instructions from an amendment law.

        Args:
            content: Full text of the amendment law
            doc_id: Document ID of the amendment law
            title: Document title

        Returns:
            List of AmendmentInstruction objects
        """
        if not content:
            return []

        # 1. Identify target laws
        target_laws = self.extract_target_laws(title, content)
        default_target = target_laws[0] if target_laws else ""

        # 2. Strip preamble and split into sections
        sections = self._split_sections(content, default_target)

        # 3. Extract instructions from each section
        all_instructions = []
        for section_target, section_text in sections:
            instructions = self._extract_section_instructions(
                section_text, doc_id, title, section_target
            )
            all_instructions.extend(instructions)

        return all_instructions

    # ── Section Splitting ───────────────────────────────────────

    def _split_sections(
        self, content: str, default_target: str
    ) -> List[Tuple[str, str]]:
        """
        Split amendment content into sections, each for a different target law.

        Returns list of (target_law_name, section_text).
        """
        # Strip preamble (everything before first "Điều 1" or first numbered instruction)
        body_start = self._find_body_start(content)

        # Cut off at effectiveness article
        body_end = self._find_body_end(content)

        body = content[body_start:body_end]

        # Find explicit target-law section boundaries
        section_starts = list(_RE_TARGET_SECTION.finditer(body))

        if not section_starts:
            # No explicit sections — entire body targets the default
            return [(default_target, body)]

        sections = []
        for i, m in enumerate(section_starts):
            target_name = self._clean_law_name(m.group(1))
            if not target_name:
                target_name = default_target

            start = m.start()
            end = section_starts[i + 1].start() if i + 1 < len(section_starts) else len(body)
            sections.append((target_name, body[start:end]))

        return sections

    def _find_body_start(self, content: str) -> int:
        """Find where the amendment body begins (after preamble)."""
        # Look for "Điều 1" or first numbered item
        m_dieu1 = re.search(
            r'(?:^|\n)\s*Điều\s+1[\.:]\s*(?:Sửa\s+đổi|Bổ\s+sung)',
            content, re.IGNORECASE
        )
        if m_dieu1:
            return m_dieu1.start()

        # Fallback: First numbered amendment instruction
        m_first = re.search(
            r'(?:^|\n)\s*\d+[\.\)]\s*(?:Sửa|Bổ|Bãi)',
            content, re.IGNORECASE
        )
        if m_first:
            return m_first.start()

        # Last resort: "Quốc hội ban hành..." line
        m_ban_hanh = re.search(
            r'Quốc\s+hội\s+ban\s+hành\s+Luật\s+sửa\s+đổi',
            content, re.IGNORECASE
        )
        if m_ban_hanh:
            # Find end of that line
            end_of_line = content.find('\n', m_ban_hanh.end())
            if end_of_line > 0:
                return end_of_line + 1

        return 0

    def _find_body_end(self, content: str) -> int:
        """Find where the amendment body ends (effectiveness article)."""
        m = _RE_EFFECTIVENESS_BOUNDARY.search(content)
        if m:
            return m.start()
        return len(content)

    # ── Section Instruction Extraction ──────────────────────────

    def _extract_section_instructions(
        self,
        section_text: str,
        doc_id: int,
        law_name: str,
        target_law: str,
    ) -> List[AmendmentInstruction]:
        """Extract instructions from a single target-law section."""
        # Split into individual instructions
        chunks = self._split_into_instructions(section_text)
        instructions = []

        for instr_num, chunk in chunks:
            instr = self._parse_instruction(
                chunk, doc_id, law_name, target_law, instr_num
            )
            if instr:
                instructions.append(instr)

        return instructions

    def _split_into_instructions(self, text: str) -> List[Tuple[str, str]]:
        """
        Split section text into individual numbered instructions.

        Returns list of (instruction_number, instruction_text).
        """
        chunks = []
        matches = list(_RE_INSTRUCTION_BOUNDARY.finditer(text))

        for i, m in enumerate(matches):
            num = m.group(1)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunks.append((num, text[start:end]))

        return chunks

    # ── Instruction Parsing ─────────────────────────────────────

    def _parse_instruction(
        self,
        text: str,
        doc_id: int,
        law_name: str,
        target_law: str,
        instr_num: str,
    ) -> Optional[AmendmentInstruction]:
        """Parse a single instruction chunk into an AmendmentInstruction."""
        # Check for target-law change within this instruction
        detected_target = self._detect_target_in_instruction(text)
        if detected_target:
            target_law = detected_target

        # Classify the action — returns (action, match, matched_pattern)
        action, match, matched_re = self._classify_action(text)

        instr = AmendmentInstruction(
            amending_doc_id=doc_id,
            amending_law_name=law_name,
            target_law_name=target_law,
            action=action,
            instruction_number=instr_num,
            raw_instruction=text[:500],
        )

        # ── Dispatch based on which pattern matched ──

        if action == AmendmentAction.MODIFY_ARTICLE:
            instr.target_article = match.group(1) if match else None
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.MODIFY_CLAUSE:
            instr.target_clause = match.group(1) if match else None
            instr.target_article = match.group(2) if match else None
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.MODIFY_POINT:
            instr.target_point = match.group(1) if match else None
            instr.target_clause = match.group(2) if match else None
            instr.target_article = match.group(3) if match else None
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.MODIFY_MULTI:
            if matched_re in (_RE_MODIFY_MULTI_NO_CAC, _RE_MODIFY_RENAME_TWO_CLAUSES):
                # "khoản X và khoản Y Điều Z" — groups: (clause1, clause2, article)
                instr.target_clauses = [match.group(1), match.group(2)]
                instr.target_article = match.group(3)
            elif matched_re in (_RE_MODIFY_RENAME_MULTI, _RE_MODIFY_MULTI):
                # "các khoản X, Y, Z Điều W" — groups: (clauses_str, article)
                clauses_str = match.group(1) if match else ""
                instr.target_article = match.group(2) if match else None
                instr.target_clauses = self._parse_number_list(clauses_str)
            # Parse sub-instructions
            instr.sub_instructions = self._parse_sub_instructions(
                text, doc_id, law_name, target_law, instr
            )

        elif action == AmendmentAction.MODIFY_MULTI_ARTICLE:
            if matched_re is _RE_MODIFY_TWO_ARTICLES:
                # "Điều X và Điều Y" — groups: (article1, article2)
                instr.target_clauses = [
                    match.group(1), match.group(2)
                ]
            else:
                # "các điều X, Y, Z" — groups: (articles_str,)
                articles_str = match.group(1) if match else ""
                instr.target_clauses = self._parse_number_list(articles_str)
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.INSERT_POINT:
            if matched_re is _RE_INSERT_POINT_INTO_CLAUSE:
                # "Bổ sung điểm X vào khoản Y Điều Z" — groups: (point, clause, article)
                instr.target_point = match.group(1)
                instr.target_clause = match.group(2)
                instr.target_article = match.group(3)
            else:
                # "Bổ sung điểm X vào sau điểm Y khoản Z Điều W" — groups: (point, after_point, clause, article)
                instr.target_point = match.group(1) if match else None
                instr.insert_after_point = match.group(2) if match else None
                instr.target_clause = match.group(3) if match else None
                instr.target_article = match.group(4) if match else None
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.INSERT_ARTICLE:
            instr.target_article = match.group(1) if match else None
            # Pattern may or may not have group(2) ("vào sau Điều Y")
            if matched_re is _RE_INSERT_ARTICLE_NEW:
                instr.insert_after_article = None   # New article, no explicit "after"
            else:
                try:
                    instr.insert_after_article = match.group(2) if match and match.lastindex >= 2 else None
                except (IndexError, AttributeError):
                    instr.insert_after_article = None
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.INSERT_CLAUSE:
            if matched_re is _RE_INSERT_CLAUSE_INTO_ARTICLE:
                # "Bổ sung khoản X vào Điều Y" — groups: (clause, article)
                instr.target_clause = match.group(1)
                instr.target_article = match.group(2)
            else:
                # "Bổ sung khoản X vào sau khoản Y Điều Z" — groups: (clause, after_clause, article)
                instr.target_clause = match.group(1) if match else None
                instr.insert_after_clause = match.group(2) if match else None
                instr.target_article = match.group(3) if match else None
            instr.new_text = self._extract_quoted_text(text)

        elif action == AmendmentAction.REPEAL_ARTICLE:
            instr.target_article = match.group(1) if match else None

        elif action == AmendmentAction.REPEAL_CLAUSE:
            instr.target_clause = match.group(1) if match else None
            instr.target_article = match.group(2) if match else None

        elif action == AmendmentAction.REPEAL_POINT:
            instr.target_point = match.group(1) if match else None
            instr.target_clause = match.group(2) if match else None
            instr.target_article = match.group(3) if match else None

        elif action == AmendmentAction.REPLACE_WORD:
            instr.old_text = match.group(1) if match else None
            instr.new_text = match.group(2) if match else None

        elif action in (AmendmentAction.BATCH_REPLACE, AmendmentAction.BATCH_INSERT):
            # Has sub-items a), b), c) etc. with individual replacements/insertions
            instr.sub_instructions = self._parse_sub_instructions(
                text, doc_id, law_name, target_law, instr
            )

        # Fallback: try to extract article/clause from text heuristically
        if action == AmendmentAction.UNKNOWN:
            instr = self._heuristic_parse(text, instr)

        return instr

    def _classify_action(self, text: str) -> Tuple[AmendmentAction, Optional[re.Match], Optional['re.Pattern']]:
        """
        Classify the amendment action type from instruction text.
        Returns (action, regex_match, matched_pattern).
        Priority: most specific patterns first.
        """
        # Try each pattern
        for action_name, pattern in self.INSTRUCTION_PATTERNS:
            m = pattern.search(text)
            if m:
                return (AmendmentAction(action_name), m, pattern)

        # Heuristic fallback: scan the text for keywords
        text_lower = text.lower()
        if 'bãi bỏ' in text_lower:
            return (AmendmentAction.REPEAL_ARTICLE, None, None)
        if 'bổ sung' in text_lower and 'vào sau' in text_lower:
            return (AmendmentAction.INSERT_ARTICLE, None, None)
        if 'bổ sung' in text_lower:
            return (AmendmentAction.MODIFY_ARTICLE, None, None)
        if 'sửa đổi' in text_lower and 'điều' in text_lower:
            return (AmendmentAction.MODIFY_ARTICLE, None, None)

        return (AmendmentAction.UNKNOWN, None, None)

    def _detect_target_in_instruction(self, text: str) -> Optional[str]:
        """
        Check if an instruction targets a DIFFERENT law than the current section.
        E.g., "Sửa đổi, bổ sung Điều 44 của Luật Tổ chức Cơ quan điều tra hình sự"
        """
        # Look for "của [Law Name]" in the first line of the instruction
        first_line = text.split('\n')[0] if text else ""
        m = _RE_TARGET_IN_INSTRUCTION.search(first_line)
        if m:
            name = self._clean_law_name(m.group(1))
            if name:
                logger.debug(f"Detected target law change: {name}")
                return name
        return None

    def _parse_sub_instructions(
        self,
        text: str,
        doc_id: int,
        law_name: str,
        target_law: str,
        parent: AmendmentInstruction,
    ) -> List[AmendmentInstruction]:
        """Parse sub-instructions (a), b), c)) within a MODIFY_MULTI instruction."""
        subs = []
        sub_matches = list(_RE_SUBITEM.finditer(text))

        for i, m in enumerate(sub_matches):
            sub_num = m.group(1).rstrip(')').rstrip('.').strip()
            start = m.start()
            end = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(text)
            sub_text = text[start:end]

            # Classify this sub-instruction
            sub_action, sub_match, _ = self._classify_action(sub_text)

            sub = AmendmentInstruction(
                amending_doc_id=doc_id,
                amending_law_name=law_name,
                target_law_name=target_law,
                action=sub_action,
                instruction_number=f"{parent.instruction_number}{sub_num}",
                raw_instruction=sub_text[:300],
            )

            if sub_action == AmendmentAction.MODIFY_CLAUSE:
                sub.target_clause = sub_match.group(1) if sub_match else None
                sub.target_article = sub_match.group(2) if sub_match else parent.target_article
                sub.new_text = self._extract_quoted_text(sub_text)

            elif sub_action == AmendmentAction.MODIFY_POINT:
                sub.target_point = sub_match.group(1) if sub_match else None
                sub.target_clause = sub_match.group(2) if sub_match else None
                sub.target_article = sub_match.group(3) if sub_match else parent.target_article
                sub.new_text = self._extract_quoted_text(sub_text)

            elif sub_action == AmendmentAction.REPLACE_WORD:
                sub.old_text = sub_match.group(1) if sub_match else None
                sub.new_text = sub_match.group(2) if sub_match else None

            elif sub_action == AmendmentAction.MODIFY_ARTICLE:
                sub.target_article = sub_match.group(1) if sub_match else parent.target_article
                sub.new_text = self._extract_quoted_text(sub_text)

            else:
                # Heuristic fallback for sub-instructions
                sub = self._heuristic_parse(sub_text, sub)
                sub.target_article = sub.target_article or parent.target_article

            subs.append(sub)

        return subs

    # ── Text Extraction ─────────────────────────────────────────

    def _extract_quoted_text(self, text: str) -> Optional[str]:
        """
        Extract text between Vietnamese-style quotes.

        Vietnamese legal documents use either "" or "" (Unicode smart quotes).
        For multi-paragraph quoted text, takes the longest quoted span.
        """
        matches = list(_RE_QUOTED_TEXT.finditer(text))
        if not matches:
            # Try finding text after "như sau:\n" until next numbered item
            m_boundary = re.search(r'như\s+sau\s*:\s*\n', text, re.IGNORECASE)
            if m_boundary:
                rest = text[m_boundary.end():]
                # Go until next instruction boundary or end
                m_next = _RE_NEXT_INSTRUCTION.search(rest)
                if m_next:
                    rest = rest[:m_next.start()]
                # Strip quotes if present
                rest = rest.strip().strip('"“”‘’"').strip()
                if rest:
                    return rest
            return None

        # Return the longest match (usually the full article text)
        longest = max(matches, key=lambda m: len(m.group(1)))
        return longest.group(1).strip()

    # ── Heuristic Parsing Fallback ──────────────────────────────

    def _heuristic_parse(
        self, text: str, instr: AmendmentInstruction
    ) -> AmendmentInstruction:
        """Fallback parser: extract article/clause numbers from instruction text."""
        # Look for "Điều X" pattern
        m_article = re.search(r'Điều\s+(\d+[a-z]?)', text, re.IGNORECASE)
        if m_article and not instr.target_article:
            instr.target_article = m_article.group(1)

        # Look for "khoản X" pattern
        m_clause = re.search(r'khoản\s+(\d+[a-z]?)', text, re.IGNORECASE)
        if m_clause and not instr.target_clause:
            instr.target_clause = m_clause.group(1)

        # Look for "điểm X" pattern
        m_point = re.search(r'điểm\s+([a-zđ]+)', text, re.IGNORECASE)
        if m_point and not instr.target_point:
            instr.target_point = m_point.group(1)

        # Try to extract quoted text regardless
        quoted = self._extract_quoted_text(text)
        if quoted and not instr.new_text:
            instr.new_text = quoted

        instr.confidence = 0.3  # Low confidence for heuristic parses
        return instr

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _clean_law_name(name: str) -> str:
        """Clean up a law name extracted from text."""
        name = name.strip().strip('.,;:')
        # Remove trailing "số XX/YYYY/QHXX" if present
        name = re.sub(r'\s+số\s+[\d/]+[A-Za-z]*\s*$', '', name)
        # Remove trailing year
        name = re.sub(r'\s+(?:năm\s+)?\d{4}\s*$', '', name)
        # Remove quotes
        name = name.strip('"“”"\'')
        # Remove leading "của"
        name = re.sub(r'^của\s+', '', name, flags=re.IGNORECASE).strip()
        return name if len(name) > 3 else ""

    @staticmethod
    def _parse_number_list(clauses_str: str) -> List[str]:
        """Parse "1, 2, 3 và 4" into ['1', '2', '3', '4']."""
        return _RE_ARTICLE_NUMBERS.findall(clauses_str)

    # ── Statistics ──────────────────────────────────────────────

    def stats(self, instructions: List[AmendmentInstruction]) -> Dict:
        """Summary statistics for extracted instructions."""
        by_action = {}
        articles_affected = set()
        target_laws = set()

        def _count(instrs):
            count = 0
            for i in instrs:
                count += 1
                by_action.setdefault(i.action.value, 0)
                by_action[i.action.value] += 1
                if i.target_article:
                    articles_affected.add(i.target_article)
                if i.target_law_name:
                    target_laws.add(i.target_law_name)
                count += _count(i.sub_instructions)
            return count

        total = _count(instructions)

        return {
            "total_instructions": total,
            "by_action": dict(sorted(by_action.items())),
            "articles_affected": len(articles_affected),
            "target_laws": sorted(target_laws),
        }
