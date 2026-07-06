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
_YEAR_RE = re.compile(r"\b(1999|2015|2017|2021|2025)\b")

# Doc-name hint patterns, most specific first: amendment-law phrasing and
# "tố tụng" variants must be tried before the plain "Hình sự" patterns they
# textually contain (as a substring once "tố tụng"/"sửa đổi" is stripped
# away), otherwise the more specific hint would never be reached.
_DOC_HINT_PATTERNS: List[re.Pattern] = [
    re.compile(r"luật\s*sửa\s*đổi[^.?!\n]{0,40}?bộ\s*luật\s*tố\s*tụng\s*hình\s*sự", re.IGNORECASE),
    re.compile(r"luật\s*sửa\s*đổi[^.?!\n]{0,40}?bộ\s*luật\s*hình\s*sự", re.IGNORECASE),
    re.compile(r"bộ\s*luật\s*tố\s*tụng\s*hình\s*sự", re.IGNORECASE),
    re.compile(r"\bbltths\b", re.IGNORECASE),
    re.compile(r"\btths\b", re.IGNORECASE),
    re.compile(r"bộ\s*luật\s*hình\s*sự", re.IGNORECASE),
    re.compile(r"\bblhs\b", re.IGNORECASE),
]

# Abbreviation -> canonical (lowercase) expansion. Matched against the hint
# with spaces stripped, so "BLHS", "BL HS", "TTHS", "BLTTHS", "BL TTHS" all
# normalize consistently.
_ABBREV_EXPANSIONS: Dict[str, str] = {
    "blhs": "bộ luật hình sự",
    "bltths": "bộ luật tố tụng hình sự",
    "tths": "bộ luật tố tụng hình sự",
}

# How far into a chunk "Điều N." must appear for the chunk to count as
# *opening* with that article (vs. just mentioning it mid-chunk, which is
# the amendment-law pollution problem this whole module exists to avoid).
_CHUNK_OPEN_WINDOW = 100


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

    doc_hint = None
    for pattern in _DOC_HINT_PATTERNS:
        hm = pattern.search(question)
        if hm:
            doc_hint = hm.group(0).lower().strip()
            break

    year_m = _YEAR_RE.search(question)
    year = int(year_m.group(1)) if year_m else None

    return ParsedCitation(articles=articles, doc_hint=doc_hint, year=year)


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
        self._chunks_by_doc: Optional[Dict[int, List]] = None

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

    def _get_chunks_by_doc(self) -> Dict[int, List]:
        if self._chunks_by_doc is None:
            self._load_collection_cache()
        return self._chunks_by_doc

    def _load_collection_cache(self) -> None:
        """One full scroll over the collection; caches (document_id ->
        document_title) and (document_id -> [chunks]) on the instance."""
        aliases: Dict[int, str] = {}
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
                chunks_by_doc.setdefault(doc_id, []).append(p)
            if offset is None:
                break
        self._alias_index = aliases
        self._chunks_by_doc = chunks_by_doc

    # ── Doc resolution ─────────────────────────────────────────────

    def resolve_doc(self, hint: Optional[str], year: Optional[int]) -> List[int]:
        """
        Resolve a free-text document hint (+ optional year) to a list of
        candidate document_ids, most-likely first.

        No hint, or hint matches nothing -> every known doc_id (ambiguous:
        caller searches every document).
        """
        aliases = self._get_alias_index()
        all_ids = list(aliases.keys())
        if not hint:
            return all_ids

        hint_norm = hint.strip().lower()
        expanded = _ABBREV_EXPANSIONS.get(hint_norm.replace(" ", ""), hint_norm)

        matched = [doc_id for doc_id, title in aliases.items() if expanded in title.lower()]

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
        doc_ids = self.resolve_doc(parsed.doc_hint, parsed.year)
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
