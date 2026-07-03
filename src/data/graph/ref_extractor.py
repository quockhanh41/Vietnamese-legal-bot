"""
Reference extraction for Vietnamese legal documents.

Extracts cross-references from legal text using regex patterns
optimized for Vietnamese legal language conventions.

Patterns detected (in priority order):
1. "Điều X của Luật Y"         → external reference to another law
2. "Điều X của Bộ luật Y"      → external reference to a code
3. "Điều X của Luật/Bộ luật này" → internal reference within same law
4. "khoản X Điều Y"             → clause-level reference
5. "theo quy định tại Điều X"   → generic reference
6. "Điều X và Điều Y"           → multi-article reference

Output: List[Reference] with normalized entity identifiers.
"""

from __future__ import annotations

import re
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum


class RefType(Enum):
    """Type of legal reference."""
    INTERNAL = "internal"          # Same law, e.g. "Điều 5 của Luật này"
    EXTERNAL = "external"          # Different law, e.g. "Điều 27 Luật Hôn nhân..."
    CLAUSE = "clause"              # Clause-level, e.g. "khoản 2 Điều 3"
    MULTI = "multi"                # Multi-article, e.g. "Điều 5 và Điều 6"
    CHAPTER = "chapter"            # Chapter-level, e.g. "tại Chương III"


@dataclass
class Reference:
    """A single cross-reference extracted from legal text."""
    source_doc_id: int
    source_article: Optional[str] = None  # "Điều 117" or None if unknown
    source_clause: Optional[str] = None   # "khoản 1" or None

    ref_type: RefType = RefType.INTERNAL

    # Target identifiers (normalized)
    target_law_name: Optional[str] = None   # "Luật Hôn nhân và gia đình"
    target_article: Optional[str] = None    # "Điều 27"
    target_clause: Optional[str] = None     # "khoản 1"
    target_chapter: Optional[str] = None    # "Chương III"

    # Raw text for debugging
    raw_match: str = ""

    # Metadata
    confidence: float = 1.0  # 1.0 for regex, < 1.0 for fuzzy/LLM


# ── Regex Patterns ──────────────────────────────────────────────

# Article number: "Điều 117", "Điều 123a", "điều 5"
_ARTICLE_NUM = r'(?:Điều|điều)\s+(\d+[a-z]?)'
_CLAUSE_NUM = r'(?:khoản|khoan|khoản)\s+(\d+[a-z]?)'
_CHAPTER_NUM = r'(?:Chương|chương)\s+([IVXLCDM]+)'

# Pattern 1: "Điều X của Luật Y" — external reference
# Captures the law name up to common delimiters
_RE_EXTERNAL_LAW = re.compile(
    rf'{_ARTICLE_NUM}\s+(?:của\s+)?(?:Luật|Bộ\s+luật)\s+'
    rf'(?!này)([^,.();]+?)'
    rf'(?=\s*(?:,|\.|\)|\s+năm|\s+số|$))',
    re.IGNORECASE
)

# Pattern 2: "Điều X của Bộ luật Y" (same as above but explicit code)
_RE_EXTERNAL_CODE = re.compile(
    rf'{_ARTICLE_NUM}\s+(?:của\s+)?Bộ\s+luật\s+'
    rf'(?!này)([^,.();]+?)'
    rf'(?=\s*(?:,|\.|\)|\s+năm|\s+số|$))',
    re.IGNORECASE
)

# Pattern 3: "Điều X của Luật/Bộ luật này" — internal reference
_RE_INTERNAL = re.compile(
    rf'{_ARTICLE_NUM}\s+(?:của\s+)?(?:Luật|Bộ\s+luật)\s+này',
    re.IGNORECASE
)

# Pattern 4: "khoản X Điều Y" — clause+article (may appear in any context)
_RE_CLAUSE_DIEU = re.compile(
    rf'{_CLAUSE_NUM}\s+{_ARTICLE_NUM}',
    re.IGNORECASE
)

# Pattern 5: "theo quy định tại (khoản X)? Điều Y"
_RE_THEO_QUY_DINH = re.compile(
    rf'theo\s+quy\s+định\s+(?:tại\s+)?(?:{_CLAUSE_NUM}\s+)?{_ARTICLE_NUM}',
    re.IGNORECASE
)

# Pattern 6: "tại (khoản X)? Điều Y của Luật Z"
_RE_TAI_DIEU_LUAT = re.compile(
    rf'tại\s+(?:{_CLAUSE_NUM}\s+)?{_ARTICLE_NUM}\s+(?:của\s+)?(?:Luật|Bộ\s+luật)\s+'
    rf'(?!này)([^,.();]+?)'
    rf'(?=\s*(?:,|\.|\)|\s+năm|\s+số|$))',
    re.IGNORECASE
)

# Pattern 7: "Điều X và Điều Y" — multi-article
_RE_MULTI_DIEU = re.compile(
    rf'{_ARTICLE_NUM}\s+và\s+{_ARTICLE_NUM}',
    re.IGNORECASE
)

# Pattern 8: "tại Chương X" — chapter reference
_RE_CHAPTER = re.compile(
    rf'(?:tại|theo|quy\s+định\s+tại)\s+{_CHAPTER_NUM}',
    re.IGNORECASE
)

# Pattern 9: Article-only mention (loose, low confidence)
_RE_ARTICLE_LOOSE = re.compile(
    rf'(?<!\w){_ARTICLE_NUM}(?!\w)',
    re.IGNORECASE
)


class ReferenceExtractor:
    """
    Extract cross-references from Vietnamese legal text.

    Usage:
        extractor = ReferenceExtractor()
        refs = extractor.extract(
            text=doc_content,
            doc_id=123,
            law_name="Luật Doanh nghiệp 2020",
            current_article="Điều 5"
        )
    """

    # Common noise patterns to skip
    _SKIP_PATTERNS = [
        re.compile(r'Điều\s+\d+\s+Hiến\s+pháp', re.IGNORECASE),
    ]

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def extract(
        self,
        text: str,
        doc_id: int,
        law_name: str = "",
        current_article: Optional[str] = None,
    ) -> List[Reference]:
        """
        Extract all references from a legal text segment.

        Args:
            text: The legal text to scan
            doc_id: Document ID from the dataset
            law_name: Full name of the current law (e.g. "Luật Doanh nghiệp")
            current_article: Which article this text belongs to

        Returns:
            List of Reference objects
        """
        refs: List[Reference] = []
        text_clean = text.replace('\n', ' ')

        # 1. External references: "Điều X của Luật Y"
        for m in _RE_EXTERNAL_LAW.finditer(text_clean):
            article_num = m.group(1)
            law_name_ref = self._clean_law_name(m.group(2))
            if self._is_noise(law_name_ref):
                continue
            refs.append(Reference(
                source_doc_id=doc_id,
                source_article=current_article,
                ref_type=RefType.EXTERNAL,
                target_law_name=law_name_ref,
                target_article=f"Điều {article_num}",
                raw_match=m.group(0)[:200],
                confidence=0.95,
            ))

        # 2. External code references: "Điều X của Bộ luật Y"
        for m in _RE_EXTERNAL_CODE.finditer(text_clean):
            article_num = m.group(1)
            code_name = self._clean_law_name(m.group(2))
            if self._is_noise(code_name):
                continue
            refs.append(Reference(
                source_doc_id=doc_id,
                source_article=current_article,
                ref_type=RefType.EXTERNAL,
                target_law_name=f"Bộ luật {code_name}",
                target_article=f"Điều {article_num}",
                raw_match=m.group(0)[:200],
                confidence=0.95,
            ))

        # 3. Internal references: "Điều X của Luật/Bộ luật này"
        for m in _RE_INTERNAL.finditer(text_clean):
            article_num = m.group(1)
            refs.append(Reference(
                source_doc_id=doc_id,
                source_article=current_article,
                ref_type=RefType.INTERNAL,
                target_law_name=law_name,
                target_article=f"Điều {article_num}",
                raw_match=m.group(0)[:200],
                confidence=0.95,
            ))

        # 4. Clause+Article: "khoản X Điều Y"
        for m in _RE_CLAUSE_DIEU.finditer(text_clean):
            clause_num = m.group(1)
            article_num = m.group(2)
            # Determine if this is internal or needs context
            refs.append(Reference(
                source_doc_id=doc_id,
                source_article=current_article,
                source_clause=f"khoản {clause_num}",
                ref_type=RefType.CLAUSE,
                target_article=f"Điều {article_num}",
                target_clause=f"khoản {clause_num}",
                raw_match=m.group(0)[:200],
                confidence=0.85,
            ))

        # 5. "tại Điều X của Luật Y" — external
        for m in _RE_TAI_DIEU_LUAT.finditer(text_clean):
            groups = m.groups()
            clause_num = groups[0]  # May be None
            article_num = groups[1]
            law_name_ref = self._clean_law_name(groups[2])
            if self._is_noise(law_name_ref):
                continue
            refs.append(Reference(
                source_doc_id=doc_id,
                source_article=current_article,
                source_clause=f"khoản {clause_num}" if clause_num else None,
                ref_type=RefType.EXTERNAL,
                target_law_name=law_name_ref,
                target_article=f"Điều {article_num}",
                target_clause=f"khoản {clause_num}" if clause_num else None,
                raw_match=m.group(0)[:200],
                confidence=0.90,
            ))

        # 6. Multi-article: "Điều X và Điều Y"
        for m in _RE_MULTI_DIEU.finditer(text_clean):
            a1, a2 = m.group(1), m.group(2)
            for art_num in [a1, a2]:
                refs.append(Reference(
                    source_doc_id=doc_id,
                    source_article=current_article,
                    ref_type=RefType.MULTI,
                    target_article=f"Điều {art_num}",
                    raw_match=m.group(0)[:200],
                    confidence=0.80,
                ))

        # 7. Chapter references
        for m in _RE_CHAPTER.finditer(text_clean):
            chapter_num = m.group(1)
            refs.append(Reference(
                source_doc_id=doc_id,
                source_article=current_article,
                ref_type=RefType.CHAPTER,
                target_chapter=f"Chương {chapter_num}",
                raw_match=m.group(0)[:200],
                confidence=0.75,
            ))

        # Deduplicate by (target_law_name, target_article, target_clause)
        refs = self._deduplicate(refs)

        # Filter by confidence
        refs = [r for r in refs if r.confidence >= self.min_confidence]

        return refs

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _clean_law_name(name: str) -> str:
        """Normalize a law name: trim, remove trailing noise."""
        name = name.strip().rstrip(',').rstrip('.').rstrip(';')
        # Remove leading/trailing whitespace artifacts
        name = re.sub(r'\s+', ' ', name)
        # Truncate at obvious noise boundaries
        name = re.split(r'\s+(?:số|năm|theo)', name, flags=re.IGNORECASE)[0]
        return name.strip()

    @staticmethod
    def _is_noise(law_name: str) -> bool:
        """Filter out false-positive law names."""
        if len(law_name) < 3:
            return True
        noise_words = {'này', 'số', 'năm', 'của', 'và', 'theo'}
        if law_name.lower().strip() in noise_words:
            return True
        return False

    @staticmethod
    def _deduplicate(refs: List[Reference]) -> List[Reference]:
        """Remove duplicate references, keeping highest confidence."""
        seen: Dict[Tuple, Reference] = {}
        for ref in refs:
            key = (ref.ref_type.value, ref.target_law_name,
                   ref.target_article, ref.target_clause, ref.target_chapter)
            if key not in seen or ref.confidence > seen[key].confidence:
                seen[key] = ref
        return list(seen.values())

    # ── Statistics ───────────────────────────────────────────────

    def stats(self, refs: List[Reference]) -> Dict:
        """Return summary statistics for a set of references."""
        by_type = {}
        for ref in refs:
            t = ref.ref_type.value
            by_type[t] = by_type.get(t, 0) + 1

        external_laws = set(
            r.target_law_name for r in refs
            if r.ref_type == RefType.EXTERNAL and r.target_law_name
        )

        return {
            "total": len(refs),
            "by_type": by_type,
            "unique_external_laws": len(external_laws),
            "external_laws": sorted(external_laws)[:20],
        }
