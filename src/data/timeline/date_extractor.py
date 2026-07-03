"""
Extract effective and expiry dates from Vietnamese legal documents.

Vietnamese laws typically contain a final article "Điều X. Hiệu lực thi hành"
that specifies when the law takes effect. Common patterns:

1. "Luật/Bộ luật này có hiệu lực thi hành từ ngày DD tháng MM năm YYYY"
2. "Luật này có hiệu lực từ ngày DD/MM/YYYY"
3. "có hiệu lực thi hành sau XX ngày, kể từ ngày công bố"
4. "Luật X hết hiệu lực kể từ ngày Luật này có hiệu lực thi hành"
5. Multiple effective dates for different articles
6. "từ ngày Luật này được thông qua" (event-based, needs resolution)

Output: LawTimeline with effective_date, expiry_date, and per-article dates.
"""

from __future__ import annotations

import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import date


# ── Regex Patterns ──────────────────────────────────────────────

# Find the "Hiệu lực thi hành" article
_RE_HIEU_LUC_ARTICLE = re.compile(
    r'(?:^|\n)\s*Điều\s+(\d+[a-z]?)\.\s*Hiệu\s+lực\s+thi\s+hành\s*\n'
    r'(.*?)'
    r'(?=\n\s*(?:Điều\s+\d+\.\s*[A-Z]|'
    r'Chương\s+[IVXLCDM]+|Mục\s+\d+|$))',
    re.IGNORECASE | re.DOTALL
)

# DD/MM/YYYY or DD-MM-YYYY
_RE_DATE_DMY = re.compile(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})')

# "ngày DD tháng MM năm YYYY"
_RE_DATE_VN = re.compile(
    r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
    re.IGNORECASE
)

# "có hiệu lực thi hành từ ngày ..."
_RE_EFFECTIVE_FROM = re.compile(
    r'(?:có\s+)?hiệu\s+lực\s+thi\s+hành\s+từ\s+ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
    re.IGNORECASE
)

# "sau XX ngày kể từ ngày công bố/ban hành"
_RE_AFTER_DAYS = re.compile(
    r'sau\s+(\d+)\s*ngày\s*(?:kể\s+từ\s+ngày\s+(?:công\s+bố|ban\s+hành|ký))?',
    re.IGNORECASE
)

# "hết hiệu lực kể từ ngày..."
_RE_EXPIRY = re.compile(
    r'(?:hết|chấm\s+dứt)\s+hiệu\s+lực\s+(?:kể\s+từ|từ)\s+ngày\s+'
    r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
    re.IGNORECASE
)

# "Luật X hết hiệu lực kể từ ngày Luật này có hiệu lực" — reference-based expiry
_RE_EXPIRY_REF = re.compile(
    r'(?:Luật|Bộ\s+luật|Nghị\s+định)\s+.*?\s+hết\s+hiệu\s+lực\s+'
    r'(?:kể\s+từ\s+ngày\s+)?(?:Luật|Bộ\s+luật)\s+này\s+có\s+hiệu\s+lực',
    re.IGNORECASE
)

# Amendment law: "Điều 2. Hiệu lực thi hành" (amendment laws are short)
_RE_AMENDMENT_HIEU_LUC = re.compile(
    r'(?:^|\n)\s*Điều\s+2\.\s*Hiệu\s+lực\s+thi\s+hành\s*\n'
    r'(.*?)'
    r'(?=\n\s*(?:Điều\s+3\.|$))',
    re.IGNORECASE | re.DOTALL
)


@dataclass
class LawTimeline:
    """Effective/expiry dates for a single legal document."""

    doc_id: int
    law_name: str

    issuance_date: Optional[str] = None       # Ngày ban hành (from metadata)
    effective_date: Optional[date] = None      # Ngày có hiệu lực chính
    expiry_date: Optional[date] = None         # Ngày hết hiệu lực (if any)

    # Multiple effective dates (article-specific)
    article_dates: Dict[str, date] = field(default_factory=dict)
    # e.g., {"Điều 48": date(2026, 3, 1), "Điều 56": date(2026, 2, 28)}

    # Source of truth
    source_text: str = ""                      # The article text used
    is_relative: bool = False                  # True if date is relative ("sau X ngày")
    relative_days: Optional[int] = None        # Number of days after publication
    replaced_laws: List[str] = field(default_factory=list)  # Laws this one replaces

    @property
    def is_in_effect(self) -> bool:
        """Check if this law is currently in effect."""
        if not self.effective_date:
            return False
        today = date.today()
        if self.effective_date > today:
            return False  # Not yet in effect
        if self.expiry_date and self.expiry_date <= today:
            return False  # Already expired
        return True

    @property
    def effective_period(self) -> Tuple[Optional[date], Optional[date]]:
        """Return (start_date, end_date) for this law's validity period."""
        return (self.effective_date, self.expiry_date)


class DateExtractor:
    """
    Extract effective and expiry dates from Vietnamese legal documents.

    Usage:
        extractor = DateExtractor()
        timeline = extractor.extract(doc_id=123, content=text,
                                     issuance_date="20/11/2012",
                                     law_name="Luật Doanh nghiệp 2020")
    """

    def extract(
        self,
        doc_id: int,
        content: str,
        issuance_date: str,
        law_name: str = "",
    ) -> LawTimeline:
        """
        Extract timeline information from a legal document.

        Args:
            doc_id: Document ID
            content: Full document text
            issuance_date: DD/MM/YYYY date string from metadata
            law_name: Document title

        Returns:
            LawTimeline with extracted dates
        """
        timeline = LawTimeline(
            doc_id=doc_id,
            law_name=law_name,
            issuance_date=issuance_date,
        )

        if not content or not isinstance(content, str):
            return timeline

        # 1. Find the "Hiệu lực thi hành" article
        article_text = self._find_effectiveness_article(content)

        if not article_text:
            # Fallback: search entire text for dates
            article_text = content

        timeline.source_text = article_text[:500]

        # 2. Extract the primary effective date
        timeline.effective_date = self._extract_primary_date(
            article_text, issuance_date
        )

        # 3. Extract article-specific dates
        timeline.article_dates = self._extract_article_dates(article_text)

        # 4. Extract expiry information
        expiry_date, replaced = self._extract_expiry(article_text)
        timeline.expiry_date = expiry_date
        timeline.replaced_laws = replaced

        # 5. Handle relative dates
        if not timeline.effective_date:
            relative = self._extract_relative_date(article_text, issuance_date)
            if relative:
                timeline.effective_date = relative
                timeline.is_relative = True

        # 6. Fallback: search last 3000 chars for any effective date
        if not timeline.effective_date:
            fallback_date = self._extract_primary_date(content[-3000:], issuance_date)
            if fallback_date:
                timeline.effective_date = fallback_date
                timeline.source_text = content[-3000:][:500]

        return timeline

    # ── Private Methods ──────────────────────────────────────────

    def _find_effectiveness_article(self, content: str) -> Optional[str]:
        """Find the text of the effectiveness article.

        Strategy: Search from the END of the document, since the
        'Hiệu lực thi hành' article is always the last or near-last article.
        """
        # Strategy 1: Find LAST occurrence of "Hiệu lực thi hành" heading
        last_idx = content.rfind('Hiệu lực thi hành')
        if last_idx > 0 and last_idx > len(content) * 0.7:
            # We're in the last 30% of the doc — likely the real article
            # Take text from this point to end, or to next major boundary
            snippet = content[last_idx:last_idx + 1000]
            return snippet

        # Strategy 2: Try regex pattern (searches from beginning)
        m = _RE_HIEU_LUC_ARTICLE.search(content)
        if m:
            return m.group(2)

        # Strategy 3: Amendment law pattern "Điều 2. Hiệu lực thi hành"
        m = _RE_AMENDMENT_HIEU_LUC.search(content)
        if m:
            return m.group(1)

        # Strategy 4: Any "Hiệu lực thi hành" (not at the very beginning)
        idx = content.lower().find('hiệu lực thi hành')
        if idx > 500:  # Not in preamble
            return content[idx:idx + 800]

        return None

    def _extract_primary_date(self, text: str, issuance: str) -> Optional[date]:
        """Extract the main effective date from article text."""
        # Priority 1: "có hiệu lực thi hành từ ngày DD/MM/YYYY"
        m = _RE_EFFECTIVE_FROM.search(text)
        if m:
            return self._make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        # Priority 2: "ngày DD tháng MM năm YYYY"
        m = _RE_DATE_VN.search(text)
        if m:
            return self._make_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        # Priority 3: Any DD/MM/YYYY in the article
        m = _RE_DATE_DMY.search(text)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Validate: must be a reasonable date
            if 1990 <= y <= 2030 and 1 <= d <= 31 and 1 <= mo <= 12:
                return self._make_date(d, mo, y)

        return None

    def _extract_article_dates(self, text: str) -> Dict[str, date]:
        """Extract per-article effective dates."""
        result = {}

        # Pattern: "Điều XX ... có hiệu lực thi hành từ ngày DD tháng MM năm YYYY"
        article_date_re = re.compile(
            r'(?:Điều|điều)\s+(\d+[a-z]?).*?'
            r'(?:có\s+)?hiệu\s+lực\s+thi\s+hành\s+từ\s+ngày\s+'
            r'(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
            re.IGNORECASE
        )

        for m in article_date_re.finditer(text):
            art_num = m.group(1)
            d, mo, y = int(m.group(2)), int(m.group(3)), int(m.group(4))
            result[f"Điều {art_num}"] = self._make_date(d, mo, y)

        # Also try DD/MM/YYYY format
        article_date_re2 = re.compile(
            r'(?:Điều|điều)\s+(\d+[a-z]?).*?'
            r'(?:có\s+)?hiệu\s+lực\s+thi\s+hành\s+từ\s+ngày\s+'
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            re.IGNORECASE
        )

        for m in article_date_re2.finditer(text):
            art_num = m.group(1)
            if f"Điều {art_num}" not in result:
                d, mo, y = int(m.group(2)), int(m.group(3)), int(m.group(4))
                result[f"Điều {art_num}"] = self._make_date(d, mo, y)

        return result

    def _extract_expiry(self, text: str) -> Tuple[Optional[date], List[str]]:
        """Extract expiry date and list of replaced laws."""
        expiry = None
        replaced = []

        # Pattern 1: Explicit date "hết hiệu lực kể từ ngày DD/MM/YYYY"
        m = _RE_EXPIRY.search(text)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            expiry = self._make_date(d, mo, y)

        # Pattern 2: "Luật X hết hiệu lực kể từ ngày Luật này có hiệu lực"
        # Find law names that are being replaced
        replaced_re = re.compile(
            r'(?:Luật|Bộ\s+luật)\s+([^.]+?)\s+(?:số\s+[\d/]+/[A-Za-z]+)?'
            r'\s*hết\s+hiệu\s+lực',
            re.IGNORECASE
        )
        for m in replaced_re.finditer(text):
            name = m.group(1).strip()
            if name and len(name) > 5:
                replaced.append(f"Luật {name}" if not name.startswith('Luật') else name)

        return expiry, replaced

    def _extract_relative_date(self, text: str, issuance: str) -> Optional[date]:
        """Handle relative dates like 'sau 45 ngày kể từ ngày công bố'."""
        m = _RE_AFTER_DAYS.search(text)
        if not m:
            return None

        days = int(m.group(1))
        issuance_date = self._parse_issuance_date(issuance)

        if issuance_date:
            from datetime import timedelta
            return issuance_date + timedelta(days=days)

        return None

    @staticmethod
    def _parse_issuance_date(issuance: str) -> Optional[date]:
        """Parse DD/MM/YYYY issuance date string."""
        if not issuance:
            return None
        m = _RE_DATE_DMY.match(issuance.strip())
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(y, mo, d)
            except ValueError:
                return None
        return None

    @staticmethod
    def _make_date(day: int, month: int, year: int) -> Optional[date]:
        """Safely create a date object."""
        try:
            return date(year, month, day)
        except ValueError:
            return None

    # ── Statistics ───────────────────────────────────────────────

    def stats(self, timelines: List[LawTimeline]) -> Dict:
        """Summary statistics for a set of timelines."""
        with_dates = sum(1 for t in timelines if t.effective_date is not None)
        in_effect = sum(1 for t in timelines if t.is_in_effect)
        expired = sum(1 for t in timelines if t.expiry_date is not None)
        with_article_dates = sum(1 for t in timelines if t.article_dates)
        with_replaced = sum(1 for t in timelines if t.replaced_laws)

        years = {}
        for t in timelines:
            if t.effective_date:
                y = t.effective_date.year
                years[y] = years.get(y, 0) + 1

        return {
            "total": len(timelines),
            "with_effective_date": with_dates,
            "currently_in_effect": in_effect,
            "expired": expired,
            "with_article_dates": with_article_dates,
            "with_replaced_laws": with_replaced,
            "by_effective_year": dict(sorted(years.items())),
        }
