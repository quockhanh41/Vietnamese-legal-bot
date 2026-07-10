"""
Resolve which version of a law is valid at a given point in time.

Given a set of LawTimeline objects, answers questions like:
- "What laws are currently in effect?"
- "What was the law in 2018?"
- "What laws were replaced by this one?"
- "Show me the version chain for Điều X"

This is the query-time component of Temporal Versioning.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import date, datetime

from src.data.timeline.date_extractor import LawTimeline


# Số hiệu văn bản inside a replacement clause, e.g. "Luật Doanh nghiệp số
# 68/2014/QH13 ... hết hiệu lực". Covers Quốc hội laws and Chính phủ decrees;
# tolerates missing diacritics in NĐ-CP (see also
# src/rag/citation_lookup._DOC_NUMBER_RE, kept separate on purpose: this one
# scans legal prose, that one parses user questions).
_DOC_NUM_IN_CLAUSE_RE = re.compile(
    r"\b(\d{1,3}[a-zA-Z]?/\d{4}/(?:QH\d+|N[ĐD]-?CP))", re.IGNORECASE
)


def _normalize_doc_number(raw: str) -> str:
    """Uppercase, strip spaces, fold 'ND-CP' -> 'NĐ-CP', drop leading zeros —
    so numbers compare equal across payload/prose spelling variants."""
    s = re.sub(r"\s+", "", raw or "").upper()
    s = s.replace("NDCP", "NĐ-CP").replace("ND-CP", "NĐ-CP").replace("NĐCP", "NĐ-CP")
    return s.lstrip("0") or s


# Marks the boundary after which numbers in a replacement clause refer to
# laws that AMENDED the replaced law, not to laws being replaced: "Luật BHXH
# số 58/2014/QH13 ĐÃ ĐƯỢC SỬA ĐỔI, BỔ SUNG một số điều theo Luật số
# 84/2015/QH13, Bộ luật số 45/2019/QH14..." — only 58/2014 is replaced;
# 45/2019 is BLLĐ 2019, alive and well (M1.8b: matching it marked the
# current labor code expired).
_AMENDED_BY_MARKER_RE = re.compile(
    r"(?:đã\s+)?được\s+sửa\s+đổi(?:\s*,\s*bổ\s+sung)?", re.IGNORECASE
)

# Amendment-law titles ("Luật sửa đổi ...", "... sửa đổi 2025", "Nghị định
# ... sửa đổi, bổ sung ..."): amendments MODIFY a law, they never replace it
# wholesale — their clauses must not mark anything expired. Same detection
# ingest uses to route amendment docs to the article-level chunker.
_AMENDMENT_TITLE_RE = re.compile(r"(?:sửa\s+đổi|bổ\s+sung)", re.IGNORECASE)


def _extract_doc_numbers(clause: str) -> Set[str]:
    """Normalized số hiệu of the laws a replaced_laws clause actually
    replaces — numbers after the 'đã được sửa đổi...' marker are the
    replaced law's AMENDERS and are excluded."""
    target_part = _AMENDED_BY_MARKER_RE.split(clause or "", maxsplit=1)[0]
    return {_normalize_doc_number(m) for m in _DOC_NUM_IN_CLAUSE_RE.findall(target_part)}


@dataclass
class VersionResult:
    """Result of a temporal query."""
    law_name: str
    doc_id: int
    effective_date: Optional[date]
    expiry_date: Optional[date]
    status: str  # "current", "future", "expired", "unknown"
    replaced_by: Optional[str] = None
    replaces: List[str] = None


class VersionResolver:
    """
    Time-aware query engine for legal documents.

    Usage:
        resolver = VersionResolver(timelines)
        current_laws = resolver.at_time()  # All currently effective
        laws_2018 = resolver.at_time(datetime(2018, 6, 1))
        chain = resolver.version_chain("Luật Doanh nghiệp")
    """

    def __init__(self, timelines: List[LawTimeline]):
        self.timelines = timelines
        self._by_name: Dict[str, List[LawTimeline]] = {}
        self._by_id: Dict[int, LawTimeline] = {}
        self._build_index()

    def _build_index(self):
        """Build lookup indices."""
        for tl in self.timelines:
            self._by_id[tl.doc_id] = tl
            # Index by name fragments for fuzzy matching
            name_key = self._normalize_name(tl.law_name)
            if name_key not in self._by_name:
                self._by_name[name_key] = []
            self._by_name[name_key].append(tl)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize law name for lookup: lowercase, remove year/number."""
        import re
        name = name.lower().strip()
        # Remove trailing year: "Luật Doanh nghiệp 2020" → "Luật Doanh nghiệp"
        name = re.sub(r'\s+(?:năm\s+)?\d{4}\s*$', '', name)
        # Remove "số XX/YYYY/QHXX"
        name = re.sub(r'\s+số\s+[\d/]+[a-z]*\d*\s*$', '', name, flags=re.IGNORECASE)
        return name.strip()

    def at_time(self, when: Optional[date] = None,
                law_name: Optional[str] = None) -> List[VersionResult]:
        """
        Find all laws (or a specific law) valid at a given time.

        Args:
            when: Point in time. None = now.
            law_name: Optional filter by law name (fuzzy match).

        Returns:
            List of VersionResult with status per law.
        """
        if when is None:
            when = date.today()

        candidates = self.timelines
        if law_name:
            norm = self._normalize_name(law_name)
            candidates = []
            for key, tls in self._by_name.items():
                if norm in key or key in norm:
                    candidates.extend(tls)

        results = []
        for tl in candidates:
            status = self._classify_status(tl, when)

            results.append(VersionResult(
                law_name=tl.law_name,
                doc_id=tl.doc_id,
                effective_date=tl.effective_date,
                expiry_date=tl.expiry_date,
                status=status,
                replaced_by=None,
                replaces=tl.replaced_laws if tl.replaced_laws else None,
            ))

        # Sort: current first, then future, then expired, then unknown
        sort_order = {"current": 0, "future": 1, "expired": 2, "unknown": 3}
        results.sort(key=lambda r: (sort_order.get(r.status, 99),
                                     r.effective_date or date.min))

        return results

    def _classify_status(self, tl: LawTimeline, when: date) -> str:
        """Classify whether a law is current, future, expired, or unknown."""
        if tl.effective_date is None:
            return "unknown"

        if tl.effective_date > when:
            return "future"  # Not yet in effect at 'when'

        if tl.expiry_date and tl.expiry_date <= when:
            return "expired"  # Already expired at 'when'

        return "current"

    def version_chain(self, law_name: str) -> List[LawTimeline]:
        """
        Get the version chain for a law: all versions sorted by date.

        E.g., "Luật Doanh nghiệp" → [1999, 2005, 2014, 2020]
        """
        norm = self._normalize_name(law_name)
        matches = []

        for key, tls in self._by_name.items():
            if norm in key or key in norm:
                matches.extend(tls)

        # Also check replaced_laws for links
        for tl in self.timelines:
            for replaced in (tl.replaced_laws or []):
                if norm in self._normalize_name(replaced):
                    if tl not in matches:
                        matches.append(tl)

        # Sort by effective date
        matches.sort(key=lambda t: t.effective_date or date.min)
        return matches

    def find_replacement(self, doc_id: int) -> Optional[LawTimeline]:
        """
        Find which law replaced this one (if any).

        Scans other timelines' replaced_laws clauses. Two-tier matching
        (M1.8b — the original name-only substring match misfired at 100-doc
        scale, marking CURRENT laws expired: _normalize_name strips the year,
        so 'Luật Doanh nghiệp 2020' matched LDN **2014**'s clause about
        replacing LDN 2005 and got flagged as replaced by an OLDER law):

        1. Số hiệu match — if this doc's number ("59/2020/QH14") appears in
           the clause, that's authoritative. If the clause names OTHER
           specific numbers only, it is NOT about us: skip it.
        2. Name fallback (clause has no parseable number) — only accepted
           when the replacing doc is strictly NEWER (direction guard); both
           effective dates must be known, else conservatively no match
           (fail-open: docs stay in effect, mirroring the retriever's
           fail-open on unknown status).
        """
        tl = self._by_id.get(doc_id)
        if not tl:
            return None

        norm = self._normalize_name(tl.law_name)
        own_num = _normalize_doc_number(tl.doc_number) if tl.doc_number else None

        for other in self.timelines:
            if other.doc_id == doc_id:
                continue
            if _AMENDMENT_TITLE_RE.search(other.law_name or ""):
                # Amendments modify, they don't replace — e.g. "Luật sửa đổi
                # BLTTHS 2025" must not mark BLTTHS 2015 expired.
                continue
            for replaced in (other.replaced_laws or []):
                clause_nums = _extract_doc_numbers(replaced)
                if clause_nums:
                    if own_num and own_num in clause_nums:
                        return other
                    continue  # clause names specific other numbers — not us
                if norm in self._normalize_name(replaced):
                    if (
                        tl.effective_date
                        and other.effective_date
                        and other.effective_date > tl.effective_date
                    ):
                        return other
        return None

    def effective_at(self, doc_id: int, when: Optional[date] = None) -> bool:
        """Check if a specific law is effective at a given time."""
        if when is None:
            when = date.today()

        tl = self._by_id.get(doc_id)
        if not tl or not tl.effective_date:
            return False

        if tl.effective_date > when:
            return False
        if tl.expiry_date and tl.expiry_date <= when:
            return False
        return True

    def article_effective_date(self, doc_id: int,
                                article: str) -> Optional[date]:
        """
        Get the effective date for a specific article.
        Falls back to the law's general effective date.
        """
        tl = self._by_id.get(doc_id)
        if not tl:
            return None

        # Check article-specific dates first
        if article in tl.article_dates:
            return tl.article_dates[article]

        # Fallback to general effective date
        return tl.effective_date

    @property
    def current_laws(self) -> List[LawTimeline]:
        """All laws currently in effect."""
        return [tl for tl in self.timelines if tl.is_in_effect]

    @property
    def future_laws(self) -> List[LawTimeline]:
        """Laws that will take effect in the future."""
        today = date.today()
        return [tl for tl in self.timelines
                if tl.effective_date and tl.effective_date > today]

    @property
    def stats(self) -> Dict:
        """Resolver statistics."""
        today = date.today()
        return {
            "total_laws": len(self.timelines),
            "current": sum(1 for t in self.timelines if t.is_in_effect),
            "future": sum(1 for t in self.timelines
                         if t.effective_date and t.effective_date > today),
            "expired": sum(1 for t in self.timelines if t.expiry_date),
            "unknown": sum(1 for t in self.timelines if t.effective_date is None),
            "with_replacements": sum(1 for t in self.timelines if t.replaced_laws),
        }
