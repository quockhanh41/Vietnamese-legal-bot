"""
Citation-lookup route: parse "Điều N" style Vietnamese legal citations out of
a question and resolve them via direct Qdrant metadata lookup instead of pure
vector similarity search.

Why: vector search does badly on citation-style queries — see
docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md, "Trạng thái Tuần 2":
"Điều 173 Bộ luật Hình sự 2015 quy định về tội gì?" measured recall@5=0.125,
MRR=0.03 (both dense AND hybrid — BGE-M3 sparse doesn't rescue it because
amendment-law chunks quote many "Điều N" verbatim while amending them,
polluting the results). Citation-style queries are also the most common
query shape in legal search, so this is worth a dedicated route.

TODO(scale): client-side scroll-and-filter (see CitationLookup.lookup) is
fine at ~1670 chunks / 6 documents. At corpus scale this needs an indexed
`article_number` payload field written at ingest time (a Qdrant payload
index), instead of scanning every chunk of each candidate document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.db.qdrant_client import LegalVectorStore, SearchResult
from src.eval.metrics import artnum


# ── Parsing ────────────────────────────────────────────────────────────────

# "Điều" followed directly by digits (optionally one trailing letter, e.g.
# "Điều 12a"). Requiring a digit right after "Điều" is what keeps "Điều
# kiện", "điều khoản", "điều tra" etc. from matching — none of them have a
# digit there.
_ARTICLE_RE = re.compile(r"[đĐ]i[eề]u\s+(\d+[a-z]?)\b", re.IGNORECASE)
_KHOAN_RE = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)
_DIEM_RE = re.compile(r"điểm\s+([a-zđ])\b", re.IGNORECASE)
# Any 19xx/20xx 4-digit number. Deliberately not anchored to a fixed list of
# known years any more (M1.3: corpus grew from 6 hình sự docs to 100 docs
# across 6 sectors spanning 1999-2025) — see parse_citation() for how this is
# kept from grabbing an article/khoản number instead of an actual year.
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Document-number citation, e.g. "59/2020/QH14" (luật) or "145/2020/NĐ-CP"
# (nghị định). Accepts the no-diacritic/lowercase "ND-CP"/"nd-cp" spelling
# and an optional missing hyphen ("NDCP") — case handled by IGNORECASE, but
# "Đ" vs "D" is a *different letter* in Vietnamese, not a case variant, so
# IGNORECASE alone does not fold one into the other; the character class
# N[DĐ]-?CP covers it explicitly. No whitespace is tolerated around the
# "/" separators — document numbers are always written tight in practice.
_DOC_NUMBER_RE = re.compile(r"\b(\d+)/(\d{4})/(QH\d+|N[DĐ]-?CP)\b", re.IGNORECASE)


def _normalize_doc_number(raw: str) -> Optional[str]:
    """
    Normalize a document-number citation to canonical form for exact-match
    doc resolution: "<num>/<year>/QHxx" or "<num>/<year>/NĐ-CP", uppercase,
    full diacritics. Returns None if `raw` doesn't contain one.

    Used both on the free-text question (parse_citation) and on each doc's
    payload `document_number` (CitationLookup._load_collection_cache), so
    the two sides compare equal regardless of source-text case/diacritic
    spelling.
    """
    m = _DOC_NUMBER_RE.search(raw or "")
    if not m:
        return None
    num, year, suffix = m.group(1), m.group(2), m.group(3)
    # Official numbers zero-pad ("01/2021/NĐ-CP") but users often type
    # "1/2021/NĐ-CP" — strip leading zeros on BOTH sides so they compare
    # equal ("0" alone can't occur: numbering starts at 1).
    num = num.lstrip("0") or num
    suffix_upper = suffix.upper()
    canonical_suffix = suffix_upper if suffix_upper.startswith("QH") else "NĐ-CP"
    return f"{num}/{year}/{canonical_suffix}"


# Abbreviation -> canonical (lowercase) expansion. Matched against the hint
# with spaces stripped, so "BLHS", "BL HS", "TTHS", "BLTTHS", "BL TTHS" all
# normalize consistently. Each expansion must be a literal substring of the
# real (lowercased) document title it targets — that's what resolve_doc()
# matches against.
_ABBREV_EXPANSIONS: Dict[str, str] = {
    "blhs": "bộ luật hình sự",
    "bltths": "bộ luật tố tụng hình sự",
    "tths": "bộ luật tố tụng hình sự",
    "blds": "bộ luật dân sự",
    "blttds": "bộ luật tố tụng dân sự",
    "ttds": "bộ luật tố tụng dân sự",
    "bllđ": "bộ luật lao động",
    "blld": "bộ luật lao động",
    "ldn": "luật doanh nghiệp",
    "lđđ": "luật đất đai",
    "ldd": "luật đất đai",
    "hngđ": "luật hôn nhân và gia đình",
    "hngd": "luật hôn nhân và gia đình",
    "bhxh": "luật bảo hiểm xã hội",
    "bhyt": "luật bảo hiểm y tế",
    "shtt": "luật sở hữu trí tuệ",
    "atvslđ": "luật an toàn, vệ sinh lao động",
    "atvsld": "luật an toàn, vệ sinh lao động",
    "thads": "luật thi hành án dân sự",
    "thahs": "luật thi hành án hình sự",
    "pcblgđ": "luật phòng, chống bạo lực gia đình",
    "pcblgd": "luật phòng, chống bạo lực gia đình",
}

# Generic "<luật name>" capture, used both for the bare "bộ luật X"/"luật X"
# pattern and (with a "luật sửa đổi ..." prefix) for the amendment-phrase
# pattern below. Captures up to 10 whitespace/comma-separated word tokens
# after "luật"/"bộ luật" — long enough for the longest real title ("Luật Bảo
# vệ quyền lợi người tiêu dùng", "Luật an toàn, vệ sinh lao động", commas and
# all) — and stops at the first digit (a year) or at a small set of Vietnamese
# connector words that show up right after a doc name in a sentence but never
# inside a title itself (e.g. "theo Luật Doanh nghiệp THÌ ..." must not eat
# "thì..."). This is what keeps it from swallowing the rest of the question.
_DOC_HINT_STOPWORDS = (
    "thì", "là", "được", "theo", "quy", "định", "này", "đó", "khi",
    "nếu", "mà", "để", "như", "sẽ", "phải", "không", "vậy",
)
_WORD_TOKEN = r"[^\s\d.,;:!?\"'()\n]+"
_DOC_NAME_GROUP = (
    r"(?:(?!(?:" + "|".join(_DOC_HINT_STOPWORDS) + r")\b)" + _WORD_TOKEN + r"[ \t,]*){1,10}"
)

# Doc-name hint patterns, most specific first (see parse_citation()):
#   1-2. amendment-law phrasing, hình sự only (kept verbatim — tests pin
#        these exact captures; the general amendment pattern below would
#        reproduce them anyway, but there's no reason to risk it).
#   3.   amendment-law phrasing, any sector ("luật sửa đổi ... <tên>").
#   4.   full name, "bộ luật <tên>" / "luật <tên>" (one regex covers both —
#        the "bộ\s*luật" alternative is tried first so "Bộ luật Tố tụng
#        hình sự" captures the "bộ luật" prefix rather than being split).
#   5.   abbreviation, generated from _ABBREV_EXPANSIONS keys instead of
#        hardcoded so every new sector alias automatically gets a pattern.
_DOC_HINT_PATTERNS: List[re.Pattern] = [
    re.compile(r"luật\s*sửa\s*đổi[^.?!\n]{0,40}?bộ\s*luật\s*tố\s*tụng\s*hình\s*sự", re.IGNORECASE),
    re.compile(r"luật\s*sửa\s*đổi[^.?!\n]{0,40}?bộ\s*luật\s*hình\s*sự", re.IGNORECASE),
    re.compile(r"luật\s*sửa\s*đổi[^.?!\n]{0,40}?(?:bộ\s*luật|luật)\s+" + _DOC_NAME_GROUP, re.IGNORECASE),
    re.compile(r"(?:bộ\s*luật|luật)\s+" + _DOC_NAME_GROUP, re.IGNORECASE),
    re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in sorted(_ABBREV_EXPANSIONS, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    ),
]

# How far into a chunk "Điều N." must appear for the chunk to count as
# *opening* with that article (vs. just mentioning it mid-chunk, which is
# the amendment-law pollution problem this whole module exists to avoid).
_CHUNK_OPEN_WINDOW = 100

# Display names for the curated-corpus `sector` payload labels
# (scripts/select_corpus.py assigns them at selection time; ingest stamps
# them onto every chunk). Used by corpus_summary()/corpus_summary_compact().
_SECTOR_DISPLAY: Dict[str, str] = {
    "hinh_su": "Hình sự",
    "dan_su": "Dân sự - Tố tụng dân sự",
    "lao_dong": "Lao động - Bảo hiểm",
    "doanh_nghiep": "Doanh nghiệp - Thương mại - Đầu tư",
    "dat_dai": "Đất đai",
    "hngd": "Hôn nhân và gia đình",
}


@dataclass
class ArticleRef:
    """A single 'Điều N' mention, with any nearby Khoản/Điểm."""
    number: str
    khoan: Optional[str] = None
    diem: Optional[str] = None


@dataclass
class ParsedCitation:
    """Result of parsing a citation-style question. See parse_citation()."""
    articles: List[ArticleRef] = field(default_factory=list)
    doc_hint: Optional[str] = None   # normalized (lowercased) free-text hint
    year: Optional[int] = None
    doc_number: Optional[str] = None  # normalized "num/year/QHxx" or "num/year/NĐ-CP", see _normalize_doc_number

    @property
    def primary(self) -> ArticleRef:
        """The first (primary) article mentioned."""
        return self.articles[0]


def parse_citation(question: str) -> Optional[ParsedCitation]:
    """
    Parse "Điều N" style citations out of a Vietnamese legal question.

    Pure function — no I/O. Returns None if no article number is found,
    e.g. "Điều kiện để được hưởng án treo là gì?" or "điều tra vụ án" are
    NOT citations — the regex requires a digit right after "Điều", which
    neither has.
    """
    matches = list(_ARTICLE_RE.finditer(question))
    if not matches:
        return None

    articles = []
    for m in matches:
        number = m.group(1).lower()
        window_start = max(0, m.start() - 40)
        window_end = min(len(question), m.end() + 40)
        window = question[window_start:window_end]
        khoan_m = _KHOAN_RE.search(window)
        diem_m = _DIEM_RE.search(window)
        articles.append(ArticleRef(
            number=number,
            khoan=khoan_m.group(1) if khoan_m else None,
            diem=diem_m.group(1) if diem_m else None,
        ))

    doc_number_m = _DOC_NUMBER_RE.search(question)
    doc_number = _normalize_doc_number(doc_number_m.group(0)) if doc_number_m else None

    doc_hint = None
    hint_end = 0
    for pattern in _DOC_HINT_PATTERNS:
        hm = pattern.search(question)
        if hm:
            doc_hint = hm.group(0).lower().strip()
            hint_end = hm.end()
            break

    # Year = first 19xx/20xx number *after* the doc-hint, if there is a
    # hint (so e.g. "Điều 168 Bộ luật Lao động 2019" doesn't pick up a
    # stray year-shaped number sitting before the hint). No hint -> first
    # year anywhere in the question, same as before.
    year_m = _YEAR_RE.search(question, hint_end) if doc_hint else _YEAR_RE.search(question)
    year = int(year_m.group(1)) if year_m else None

    return ParsedCitation(articles=articles, doc_hint=doc_hint, year=year, doc_number=doc_number)


# ── Metadata lookup ──────────────────────────────────────────────────────

class CitationLookup:
    """
    Resolves a ParsedCitation to concrete chunks via direct Qdrant metadata
    lookup (scroll + filter), instead of vector similarity search.

    Builds a document alias index (document_id -> document_title) lazily
    from the collection itself on first use — no hardcoded doc_ids, so the
    corpus can grow without code changes.
    """

    def __init__(self, vector_store: LegalVectorStore):
        self.vector_store = vector_store
        self._alias_index: Optional[Dict[int, str]] = None
        self._doc_number_index: Optional[Dict[int, str]] = None
        self._sector_index: Optional[Dict[int, str]] = None
        self._chunks_by_doc: Optional[Dict[int, List]] = None
        self._corpus_summary: Optional[str] = None
        self._corpus_summary_compact: Optional[str] = None

    # ── Alias index + chunk cache ───────────────────────────────────
    #
    # Both are built from a single full collection scroll (see
    # _load_collection_cache). We'd prefer a server-side `scroll_filter` on
    # document_id per lookup (the original design) instead of caching every
    # chunk client-side, but Qdrant Cloud requires a payload index to filter
    # scroll by a field and none exists for document_id here — creating one
    # is an ingest-time change out of scope for this route. At 1670 chunks a
    # single full scroll + in-memory grouping is cheap, cached once, and
    # avoids provisioning an index just for this.
    #
    # TODO(scale): create a payload index on document_id (and, per the
    # module docstring, article_number) at ingest time once the corpus
    # grows past a few thousand chunks — then this can go back to
    # server-side scroll_filter per doc_id instead of a full-collection
    # client-side cache.

    def _get_alias_index(self) -> Dict[int, str]:
        if self._alias_index is None:
            self._load_collection_cache()
        return self._alias_index

    def _get_doc_number_index(self) -> Dict[int, str]:
        if self._doc_number_index is None:
            self._load_collection_cache()
        return self._doc_number_index

    def _get_chunks_by_doc(self) -> Dict[int, List]:
        if self._chunks_by_doc is None:
            self._load_collection_cache()
        return self._chunks_by_doc

    def _load_collection_cache(self) -> None:
        """One full scroll over the collection; caches (document_id ->
        document_title), (document_id -> normalized document_number), and
        (document_id -> [chunks]) on the instance."""
        aliases: Dict[int, str] = {}
        doc_numbers: Dict[int, str] = {}
        sectors: Dict[int, str] = {}
        chunks_by_doc: Dict[int, List] = {}
        client = self.vector_store.client
        collection = self.vector_store.collection_name
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                payload = p.payload or {}
                doc_id = payload.get("document_id")
                if doc_id is None:
                    continue
                if doc_id not in aliases:
                    aliases[doc_id] = payload.get("document_title", "")
                if doc_id not in doc_numbers:
                    normalized = _normalize_doc_number(payload.get("document_number", ""))
                    if normalized is not None:
                        doc_numbers[doc_id] = normalized
                if doc_id not in sectors and payload.get("sector"):
                    sectors[doc_id] = payload["sector"]
                chunks_by_doc.setdefault(doc_id, []).append(p)
            if offset is None:
                break
        self._alias_index = aliases
        self._doc_number_index = doc_numbers
        self._sector_index = sectors
        self._chunks_by_doc = chunks_by_doc

    def corpus_summary(self) -> str:
        """
        Document inventory for the SCOPE-gate prompt (src/llm/prompts.py
        COMPLEXITY_CLASSIFIER_PROMPT) — never hardcodes doc titles, so the
        corpus can grow without a prompt edit. Built once from the alias
        index (same source as citation-lookup doc resolution) and cached.

        Two shapes (M1.4):
        - collection has `sector` payloads (curated corpora, luat_law_ctx_v3+):
          titles grouped under sector headings — the scope gate reads "what
          fields does this corpus cover" at a glance instead of scanning a
          flat 100-line list;
        - no sector info (older collections, e.g. luat_law_ctx_v2): the
          original flat "- title" bullet list, unchanged.
        """
        if self._corpus_summary is None:
            aliases = self._get_alias_index()
            sectors = self._sector_index or {}
            titles = sorted({t for t in aliases.values() if t})
            if not titles:
                self._corpus_summary = "(không có văn bản nào)"
            elif not sectors:
                self._corpus_summary = "\n".join(f"- {t}" for t in titles)
            else:
                by_sector: Dict[str, List[str]] = {}
                for doc_id, title in aliases.items():
                    if not title:
                        continue
                    label = _SECTOR_DISPLAY.get(sectors.get(doc_id), "Khác")
                    by_sector.setdefault(label, []).append(title)
                lines = []
                # Stable order: theo _SECTOR_DISPLAY, nhóm lạ xếp cuối.
                known_order = list(_SECTOR_DISPLAY.values()) + ["Khác"]
                for label in known_order:
                    group = by_sector.pop(label, None)
                    if not group:
                        continue
                    lines.append(f"◆ {label} ({len(group)} văn bản):")
                    lines.extend(f"  - {t}" for t in sorted(group))
                for label in sorted(by_sector):  # bất kỳ nhãn nào ngoài known_order
                    group = by_sector[label]
                    lines.append(f"◆ {label} ({len(group)} văn bản):")
                    lines.extend(f"  - {t}" for t in sorted(group))
                self._corpus_summary = "\n".join(lines)
        return self._corpus_summary

    def corpus_summary_compact(self) -> str:
        """
        One-paragraph inventory for USER-FACING text (the out-of-scope
        refusal answer, src/rag/confidence.py OUT_OF_SCOPE_ANSWER_TEMPLATE).
        At 100 docs the full title list is unreadable in a chat message —
        show sector names + counts instead. Falls back to the full bullet
        list when the collection has no sector info (small corpora, where
        listing titles IS the more helpful message).
        """
        if self._corpus_summary_compact is None:
            aliases = self._get_alias_index()
            sectors = self._sector_index or {}
            if not sectors:
                self._corpus_summary_compact = self.corpus_summary()
            else:
                counts: Dict[str, int] = {}
                for doc_id, title in aliases.items():
                    if not title:
                        continue
                    label = _SECTOR_DISPLAY.get(sectors.get(doc_id), "Khác")
                    counts[label] = counts.get(label, 0) + 1
                known_order = list(_SECTOR_DISPLAY.values()) + ["Khác"]
                parts = [
                    f"{label} ({counts[label]} văn bản)"
                    for label in known_order
                    if counts.get(label)
                ]
                total = sum(counts.values())
                self._corpus_summary_compact = (
                    f"**{total} văn bản luật và nghị định** thuộc {len(parts)} lĩnh vực: "
                    + ", ".join(parts) + "."
                )
        return self._corpus_summary_compact

    # ── Doc resolution ─────────────────────────────────────────────

    def resolve_doc(
        self,
        hint: Optional[str],
        year: Optional[int],
        doc_number: Optional[str] = None,
    ) -> List[int]:
        """
        Resolve a free-text document hint (+ optional year) to a list of
        candidate document_ids, most-likely first.

        `doc_number` (from ParsedCitation.doc_number, e.g. "59/2020/QH14")
        is an absolute priority signal: an exact (case-insensitive,
        Đ/D-normalized — see _normalize_doc_number) match against a doc's
        document_number wins outright, ignoring hint/year entirely, because
        a document number is unambiguous where a free-text name never is.
        No match on doc_number falls through to the hint/year logic below.

        No hint, or hint matches nothing -> every known doc_id (ambiguous:
        caller searches every document).
        """
        # Re-normalize doc_number here too (not just trust the caller already
        # did, e.g. in parse_citation) so resolve_doc's own case-insensitive /
        # Đ-D-normalized matching guarantee holds no matter what a caller
        # passes in ("59/2020/qh14" resolves the same as "59/2020/QH14").
        doc_number_norm = _normalize_doc_number(doc_number) if doc_number else None
        if doc_number_norm:
            for doc_id, num in self._get_doc_number_index().items():
                if num == doc_number_norm:
                    return [doc_id]
            # doc_number parsed but didn't match any known document (e.g.
            # a typo, or a doc outside the corpus) -> fall through to
            # hint/year like doc_number was never given.

        aliases = self._get_alias_index()
        all_ids = list(aliases.keys())
        if not hint:
            return all_ids

        hint_norm = hint.strip().lower()
        expanded = _ABBREV_EXPANSIONS.get(hint_norm.replace(" ", ""), hint_norm)

        # Multi-sector corpus hazard (M1.3): guiding-decree titles CONTAIN
        # the law name they implement ("Nghị định 126/2014/NĐ-CP hướng dẫn
        # thi hành Luật Hôn nhân và gia đình"), so a bare substring test
        # can't tell the law itself from documents merely referencing it —
        # and both may define their own "Điều 27" with unrelated content.
        # Rank instead of exclude (excluding would lose amendment laws,
        # whose chunks carry the quoted suffixed articles like "Điều 12a"
        # recovered in M1.2b):
        #   1. title STARTS with the expansion  -> the law itself
        #   2. contains it + "sửa đổi" in title -> amendment of that law
        #   3. merely contains it               -> decree/other reference
        prefix, amendment, referencing = [], [], []
        for doc_id, title in aliases.items():
            t = title.lower()
            if expanded not in t:
                continue
            if t.startswith(expanded):
                prefix.append(doc_id)
            elif "sửa đổi" in t:
                amendment.append(doc_id)
            else:
                referencing.append(doc_id)
        matched = prefix + amendment + referencing

        if year is not None and matched:
            year_str = str(year)
            with_year = [doc_id for doc_id in matched if year_str in aliases[doc_id]]
            if with_year:
                return with_year
            # Hint matched documents, but none carry this year in the title —
            # fall through and return the hint-only matches below.

        if matched:
            return matched

        return all_ids

    # ── Chunk lookup ───────────────────────────────────────────────

    def lookup(self, parsed: ParsedCitation, cap: int = 8) -> List[SearchResult]:
        """
        Direct metadata lookup for the primary article in `parsed`.

        For each candidate document (from resolve_doc), scroll its chunks
        and keep the ones that are actually about the target article:
        either the `article` payload field matches, or (chunker metadata is
        occasionally stale) the chunk text itself opens with "Điều N." /
        "Điều N:" within the first `_CHUNK_OPEN_WINDOW` characters.

        Returns [] if nothing matches — the caller must NOT fall back to
        vector search on empty lookup (e.g. "Điều 999" doesn't exist; vector
        search would still return *something* plausible-looking and wrong).
        """
        target_num = parsed.primary.number
        doc_ids = self.resolve_doc(parsed.doc_hint, parsed.year, parsed.doc_number)
        open_pattern = re.compile(r"Điều\s+" + re.escape(target_num) + r"\s*[\.:]")

        strict_hits: List[Tuple[int, object]] = []
        lenient_hits: List[Tuple[int, object]] = []

        chunks_by_doc = self._get_chunks_by_doc()
        for doc_id in doc_ids:
            if len(strict_hits) >= cap:
                break
            for point in chunks_by_doc.get(doc_id, []):
                payload = point.payload or {}
                art_num = artnum(payload.get("article"))
                if art_num is not None and art_num == target_num:
                    strict_hits.append((doc_id, point))
                    continue
                chunk_text = payload.get("chunk_text", "") or ""
                m = open_pattern.search(chunk_text)
                if m and m.start() < _CHUNK_OPEN_WINDOW:
                    lenient_hits.append((doc_id, point))

        ordered = strict_hits + lenient_hits
        return [self._point_to_search_result(doc_id, point) for doc_id, point in ordered[:cap]]

    @staticmethod
    def _point_to_search_result(doc_id: int, point) -> SearchResult:
        payload = point.payload or {}
        return SearchResult(
            point_id=str(point.id),
            score=1.0,  # lookup is exact metadata match, not a similarity score
            chunk_text=payload.get("chunk_text", ""),
            document_title=payload.get("document_title", ""),
            document_number=payload.get("document_number", ""),
            legal_type=payload.get("legal_type", ""),
            article=payload.get("article"),
            chapter=payload.get("chapter"),
            url=payload.get("url"),
            document_id=payload.get("document_id", doc_id),
            parent_chunk_text=payload.get("parent_chunk_text"),
        )


# ── Shared retrieval helper (used by both the pipeline and scripts/eval.py) ─

def retrieve_routed(
    question: str,
    vector_store: LegalVectorStore,
    embedder,
    citation_lookup: "CitationLookup",
    fill_top_k: int = 10,
) -> Tuple[List[SearchResult], str, Optional[ParsedCitation]]:
    """
    Route a question to citation lookup or hybrid vector search.

    - If the question parses as a citation: run CitationLookup.lookup(). If
      it finds chunks, supplement with dense search results (deduped by
      point_id) up to `fill_top_k` total, so the LLM still gets surrounding
      context. If it finds NOTHING, return an empty list — no vector-search
      fallback (see CitationLookup.lookup docstring for why).
    - Otherwise: hybrid search (the existing baseline retrieval path).

    Returns (results, route, parsed) where route is "lookup" or "hybrid".
    Shared by AdaptiveRAGPipeline (src/rag/orchestrator.py) and the "routed"
    Tier 1 eval mode (scripts/eval.py) so both measure/use identical logic.
    """
    parsed = parse_citation(question)

    if parsed is not None:
        lookup_hits = citation_lookup.lookup(parsed)
        if lookup_hits:
            seen = {r.point_id for r in lookup_hits}
            if len(lookup_hits) < fill_top_k:
                query_vector = embedder.encode_queries([question])[0]
                fill = vector_store.search_dense(query_vector.tolist(), top_k=fill_top_k * 2)
                for r in fill:
                    if len(lookup_hits) >= fill_top_k:
                        break
                    if r.point_id not in seen:
                        seen.add(r.point_id)
                        lookup_hits.append(r)
            return lookup_hits[:fill_top_k], "lookup", parsed
        # Parsed as a citation but nothing found (e.g. "Điều 999" — the
        # article doesn't exist). Deliberately NOT falling back to vector
        # search here — see CitationLookup.lookup docstring.
        return [], "lookup", parsed

    output = embedder.encode([question], return_dense=True, return_sparse=True)
    dense_vec = output["dense_vecs"][0]
    sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
    hits = vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=fill_top_k)
    return hits, "hybrid", parsed
