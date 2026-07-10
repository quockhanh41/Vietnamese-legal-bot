"""
Main RAG orchestrator — ties together query processing, retrieval, and generation.

MVP version: Naive RAG (dense search → LLM generate)
v1.0: Adaptive RAG (query classifier + routing)
"""

import json
import os
import re
import sys
import statistics, time
from typing import List, Dict, Optional, Generator, Tuple

from dataclasses import dataclass, field, replace
# Ensure project root is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.db.qdrant_client import LegalVectorStore, SearchResult
from src.data.embedder import BGEEmbedder
from src.data.metadata_store import MetadataStore
from src.pipeline.retriever import get_phase2_status
from src.llm.gateway import LLMGateway
from src.llm.prompts import (
    SYSTEM_PROMPT,
    GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
    COMPLEXITY_CLASSIFIER_PROMPT,
)

from src.rag.confidence import ConfidenceScorer, REFUSAL_ANSWER, OUT_OF_SCOPE_ANSWER_TEMPLATE
from src.rag.verifier import HallucinationVerifier
from src.rag.citation_lookup import parse_citation, CitationLookup, retrieve_routed

@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    citations: List[Dict]
    confidence: str
    disclaimer: str
    latency_ms: float
    pipeline_used: str  # "simple", "medium", "complex"
    related_questions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0   # NEW: 0.0-1.0
    refused: bool = False            # NEW: did confidence gate refuse?


class NaiveRAGPipeline:
    """
    MVP Naive RAG: dense search → LLM generate.
    Simple but functional baseline.
    """

    def __init__(
        self,
        vector_store: LegalVectorStore,
        embedder: BGEEmbedder,
        llm_gateway: LLMGateway,
        reranker: Optional[object] = None,
        top_k: int = 5,
        metadata_store: Optional[MetadataStore] = None,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm_gateway
        self.top_k = top_k
        # Phase 2 (temporal + amendment) metadata — lazy default so callers that
        # don't care about Phase 2 (e.g. unit tests) don't pay a JSON-load cost.
        # Callers with an existing MetadataStore (api/main.py) should pass it in
        # to avoid loading the same JSON files twice.
        self._metadata_store = metadata_store

    @property
    def metadata_store(self) -> MetadataStore:
        if self._metadata_store is None:
            self._metadata_store = MetadataStore()
        return self._metadata_store

    # ── Methods inherited by AdaptiveRAGPipeline ──
    def query(self, question: str) -> RAGResponse:
        """Process a single question through the naive RAG pipeline."""
        start_time = time.time()

        # Step 1: Embed the question
        query_vector = self.embedder.encode_queries([question])[0]

        # Step 2: Dense search
        results = self.vector_store.search_dense(
            query_vector.tolist(),
            top_k=self.top_k,
        )

        # Phase 2 (Plan §3.6): default temporal filter + amendment annotations.
        # Small-to-big (Plan §3.4): expand Khoản hits to their parent Điều.
        annotations = self._annotate_phase2(results)
        results = self._filter_in_effect(results, annotations)
        results = self._expand_to_parent(results)

        # Step 3: Build context from retrieved chunks
        context = self._build_context(results, annotations)

        # Step 4: Generate answer
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw_response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            json_mode=False,
        )

        # Step 5: Parse response
        citations = self._build_citations(results, annotations)
        parsed = self._parse_response(raw_response, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "naive"

        return parsed

    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Process a question with streaming response.

        Yields:
            First: citation metadata as JSON string (event: citations)
            Then: tokens from LLM generation
            Finally: end marker (event: done)
        """
        # Embed question
        query_vector = self.embedder.encode_queries([question])[0]

        # Search
        results = self.vector_store.search_dense(
            query_vector.tolist(),
            top_k=self.top_k,
        )

        # Phase 2 (Plan §3.6) + small-to-big (Plan §3.4) — see query() above.
        annotations = self._annotate_phase2(results)
        results = self._filter_in_effect(results, annotations)
        results = self._expand_to_parent(results)

        # Build context
        context = self._build_context(results, annotations)

        # Build citations from search results
        citations = self._build_citations(results, annotations)

        # Yield citations metadata first (as a special marker)
        yield f'__CITATIONS__{json.dumps(citations, ensure_ascii=False)}'

        # Stream generation
        prompt = GENERATION_PROMPT.format(context=context, question=question)

        for token in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        ):
            yield token

    _PARENT_TEXT_MAX_LEN = 3000  # guard against bad/oversized parent_chunk_text data

    def _expand_to_parent(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Small-to-big retrieval (Plan §3.4): retrieval matches at the precise
        Khoản level, but the LLM needs the full Điều for context. Swaps each
        result's chunk_text for its parent_chunk_text (already computed at
        chunk time by HierarchicalChunker) and collapses results that share
        the same parent article, so a multi-Khoản hit on one Điều doesn't
        repeat that Điều as several context blocks.

        Order is preserved — the first (highest-scored/reranked) occurrence
        of an article wins the slot; later duplicates are dropped entirely
        rather than merged, keeping the returned list 1:1 with the `[n]`
        blocks `_build_context` will produce from it.
        """
        seen: set = set()
        expanded: List[SearchResult] = []
        for r in results:
            key = (r.document_id, r.article) if r.article else None
            if key is not None and key in seen:
                continue
            parent_text = r.parent_chunk_text or r.chunk_text
            if not parent_text or len(parent_text) > self._PARENT_TEXT_MAX_LEN:
                parent_text = r.chunk_text  # guard: never trust oversized/bad parent data
            if key is not None:
                seen.add(key)
            expanded.append(replace(r, chunk_text=parent_text))
        return expanded

    def _annotate_phase2(self, results: List[SearchResult]) -> Dict[str, Dict]:
        """Look up Phase 2 (temporal + amendment) status per result (Plan §3.6)."""
        return {
            r.point_id: get_phase2_status(r.document_id, r.article, self.metadata_store)
            for r in results
        }

    @staticmethod
    def _filter_in_effect(
        results: List[SearchResult], annotations: Dict[str, Dict]
    ) -> List[SearchResult]:
        """
        Default temporal filter (Plan §3.6): drop chunks whose source law is
        known to no longer be in effect. Fail-open — a chunk with no Phase 2
        metadata, or is_in_effect=None (unknown), is KEPT. Only an explicit
        is_in_effect=False (a law we positively know was superseded) excludes.
        """
        return [
            r for r in results
            if annotations.get(r.point_id, {}).get("is_in_effect") is not False
        ]

    def _build_context(self, results: List[SearchResult], annotations: Optional[Dict[str, Dict]] = None) -> str:
        """Format search results into a context string for the LLM."""
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[{i}] {result.document_title}"
            if result.article:
                header += f" — {result.article}"
            if result.chapter:
                header += f" ({result.chapter})"

            block = header
            status = (annotations or {}).get(result.point_id)
            if status and status.get("is_amended"):
                laws = ", ".join(status["amending_laws"][:2])
                block += f"\n⚠️ Đã được sửa đổi bởi: {laws}"

            # Cap raised from 500: blocks now usually hold the full parent
            # Điều text (Plan §3.4), which runs longer than 500 chars.
            block += f"\n{result.chunk_text[:2000]}"
            parts.append(block)

        return "\n\n---\n\n".join(parts)

    def _build_citations(self, results: List[SearchResult], annotations: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        """Build citation list from search results.

        `results` MUST be the exact same list (same order, same slice) passed
        to `_build_context()` for this route — `block_index` is assigned with
        the identical 1-based enumeration `_build_context` uses for its `[n]`
        headers, so a `[n]` the model cites in the answer maps 1:1 to the
        citation at that `block_index` here. Returns ALL of `results` (not
        just the top few) so every `[n]` the model could possibly cite
        resolves to a readable card.
        """
        citations = []
        for i, r in enumerate(results, 1):
            status = (annotations or {}).get(r.point_id, {})
            citations.append({
                "block_index": i,
                "document_title": r.document_title,
                "article": r.article,
                "chunk_text": r.chunk_text[:200] if r.chunk_text else "",
                "legal_type": r.legal_type,
                "url": r.url,
                "score": round(r.score, 4),
                "is_amended": status.get("is_amended", False),
                "amending_laws": status.get("amending_laws", []),
            })
        return citations

    def _parse_response(self, raw: str, citations: List[Dict]) -> RAGResponse:
        """Parse markdown response into RAGResponse."""
        import re
        clean_raw = re.sub(r'<think>[\s\S]*?</think>', '', raw, flags=re.IGNORECASE).strip()
        
        return RAGResponse(
            answer=clean_raw,
            citations=citations,
            confidence="high" if len(citations) >= 2 else "medium",
            disclaimer="⚠️ Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.",
            latency_ms=0,
            pipeline_used="naive",
            related_questions=[]
        )


class AdaptiveRAGPipeline(NaiveRAGPipeline):
    """
    v1.0 Adaptive RAG: query classifier + dynamic routing.

    SIMPLE → dense search → generate
    MEDIUM → hybrid search → rerank → generate
    COMPLEX → full pipeline (rewrite → multi-query → HyDE → hybrid → rerank → compress → generate)

    Key improvements:
    - Dynamic Top-K: adaptive number of results based on score distribution
    - Re-ranking: BGE-Reranker v2 cross-encoder for MEDIUM+COMPLEX
    - Context Compression: LLM extracts relevant sentences (COMPLEX only)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reranker = None  # Lazy-loaded FlagReranker
        self._confidence_scorer = None  # Lazy-loaded
        self._verifier = None  # Lazy-loaded
        self._citation_lookup = None  # Lazy-loaded CitationLookup

    def _get_citation_lookup(self) -> CitationLookup:
        """Lazy-init CitationLookup (builds a document alias index from Qdrant on first use)."""
        if self._citation_lookup is None:
            self._citation_lookup = CitationLookup(self.vector_store)
        return self._citation_lookup

    def _get_reranker(self):
        """Lazy-load BGE-Reranker v2."""
        if self._reranker is None:
            try:
                from FlagEmbedding import FlagReranker
                self._reranker = FlagReranker(
                    'BAAI/bge-reranker-v2-m3',
                    use_fp16=True,
                    device='cpu',
                )
                print("✅ BGE-Reranker v2 loaded")
            except Exception as e:
                print(f"⚠️ Reranker not available (skip): {e}")
                self._reranker = False  # Sentinel — don't retry
        return self._reranker if self._reranker else None

    @staticmethod
    def _dynamic_top_k(results: List[SearchResult], min_k: int = 3, max_k: int = 20) -> int:
        """
        Calculate adaptive Top-K based on score distribution (Plan §4.5).
        High scores with small variance → confident → fewer results.
        Low/flat scores → ambiguous → more results.
        """
        if not results:
            return min_k
        scores = [r.score for r in results if r.score is not None]
        if not scores:
            return min_k
        # Normalize to 0-1 regardless of original scale (cosine / RRF)
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return (min_k + max_k) // 2
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        n_high = sum(1 for s in normalized if s > 0.7)
        if n_high >= 8:
            return min(10, max_k)
        elif n_high >= 4:
            return min(12, max_k)
        else:
            return max_k

    def query(self, question: str) -> RAGResponse:
        """Process with adaptive routing."""
        start_time = time.time()

        # Step 0: Citation-lookup route — parsed BEFORE the (LLM) complexity
        # classifier, so citation-style queries ("Điều 173 BLHS 2015 quy
        # định về tội gì?") skip the classifier call entirely and go
        # straight to a direct metadata lookup instead of vector search
        # (vector search measured recall@5=0.125 on this query shape — see
        # docs/Danh_Gia_va_Roadmap_Production_2026-07-06.md).
        parsed_citation = parse_citation(question)
        if parsed_citation is not None:
            return self._lookup_pipeline(question, start_time)

        # Step 1: Classify query complexity + scope gate (merged into a
        # single LLM call — see COMPLEXITY_CLASSIFIER_PROMPT). A question
        # that resolves to real articles via the citation route above is
        # in-scope by construction and never reaches this gate.
        complexity, in_scope = self._classify_and_scope(question)

        if not in_scope:
            return self._out_of_scope_response(question, start_time)

        # Step 2-6: Route to appropriate pipeline
        if complexity == "simple":
            return self._simple_pipeline(question, start_time)
        elif complexity == "medium":
            return self._medium_pipeline(question, start_time)
        else:
            return self._complex_pipeline(question, start_time)

    def _out_of_scope_response(self, question: str, start_time: float) -> RAGResponse:
        """Refuse immediately, WITHOUT retrieval/generation — the SCOPE gate
        already determined this corpus cannot answer the question at all."""
        corpus_summary = self._get_citation_lookup().corpus_summary()
        answer = OUT_OF_SCOPE_ANSWER_TEMPLATE.format(corpus_summary=corpus_summary)
        return RAGResponse(
            answer=answer,
            citations=[],
            confidence="low",
            disclaimer="⚠️ Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.",
            latency_ms=(time.time() - start_time) * 1000,
            pipeline_used="out-of-scope",
            related_questions=[],
            confidence_score=0.0,
            refused=True,
        )

    def _lookup_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """LOOKUP: direct metadata lookup for 'Điều N' citations (+ dense fill) → generate.

        Deliberately does NOT fall back to vector search when the lookup finds
        nothing (e.g. "Điều 999" doesn't exist) — context stays empty/small so
        the existing refusal path (confidence gate + system prompt instruction)
        triggers naturally instead of citing a plausible-looking wrong article.
        """
        citation_lookup = self._get_citation_lookup()
        results, _route, _parsed = retrieve_routed(
            question, self.vector_store, self.embedder, citation_lookup, fill_top_k=10,
        )

        annotations = self._annotate_phase2(results)
        results = self._filter_in_effect(results, annotations)
        results = self._expand_to_parent(results)

        context = self._build_context(results, annotations)
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results, annotations)
        context = self._build_context(results, annotations)

        # Post-process: verify + confidence (same as every other route)
        conf_result, ver_result, final_answer = self._post_process(question, raw, results, context)

        parsed = self._parse_response(final_answer, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "lookup"
        parsed = self._enrich_response(parsed, conf_result, ver_result)
        return parsed

    def _classify_and_scope(self, question: str) -> Tuple[str, bool]:
        """
        Combined SCOPE + complexity classification — ONE LLM call (zero added
        latency/cost vs. the old complexity-only classifier).

        Returns (complexity, in_scope):
            complexity: "simple" | "medium" | "complex"
            in_scope: whether the question can be answered from this corpus
                (see COMPLEXITY_CLASSIFIER_PROMPT for the SCOPE rules).

        Parsing is deliberately lenient: regex-search for SCOPE:/CLASS:
        rather than exact-match, since the (thinking) model may wrap labels
        in extra text even after <think> stripping. An unparseable response
        NEVER blocks the pipeline — it defaults to in_scope=True (fail open)
        + complexity="medium", so a classifier hiccup degrades to "normal
        pipeline, maybe wrong route" instead of wrongly refusing a real
        question outright.
        """
        corpus_summary = self._get_citation_lookup().corpus_summary()
        prompt = COMPLEXITY_CLASSIFIER_PROMPT.format(corpus_summary=corpus_summary, question=question)
        result = self.llm.generate(
            prompt=prompt, temperature=0, max_tokens=150, reasoning_effort="none",
        )
        result = result.strip().upper()

        scope_m = re.search(r'SCOPE\s*:\s*(IN|OUT)\b', result, re.IGNORECASE)
        class_m = re.search(r'\b(SIMPLE|MEDIUM|COMPLEX)\b', result, re.IGNORECASE)

        in_scope = scope_m.group(1).upper() == "IN" if scope_m else True  # fail open
        complexity = class_m.group(1).lower() if class_m else "medium"  # safe default

        return complexity, in_scope

    def _post_process(self, question: str, answer: str, results: List[SearchResult],
                      context: str, rerank_scores: Optional[List[float]] = None) -> tuple:
        """
        Post-generation: verifier (Tier 1 + 2) + confidence scoring + refusal gate.

        Args:
            rerank_scores: Cross-encoder scores from `_rerank_results`, if reranking
                ran for this query (None otherwise — distinct from "ran but empty").

        Returns:
            (conf_result, ver_result, final_answer)
        """
        # Lazy init
        if self._confidence_scorer is None:
            self._confidence_scorer = ConfidenceScorer(self.llm)
        if self._verifier is None:
            self._verifier = HallucinationVerifier()

        # Tier 1 + 2: Verifier
        citations = self._build_citations(results)
        context_str = context
        ver_result = self._verifier.verify(answer, citations, context_str)

        # Semantic verification (NLI) is "skipped" when no claims were labeled —
        # pass None so confidence scoring can drop this signal instead of treating
        # a 0.0 ratio as "answer entails nothing".
        entailment_ratio = ver_result.entailment_ratio if ver_result.entailment_labels else None

        # Same convention for grounding (Tier-1.5, M0.2): `grounded_claims` is
        # only empty when there were no groundable ("Chi tiết" bullet) claims
        # to check at all (see `_verify_grounding`'s early return) — pass None
        # so confidence scoring drops the signal instead of treating "no
        # claims to ground" as "0% grounded".
        grounding_ratio = ver_result.grounding_ratio if ver_result.grounded_claims else None

        result = self._confidence_scorer.score(
            question=question,
            answer=answer,
            context=context_str,
            search_results=results,
            rerank_scores=rerank_scores,
            entailment_ratio=entailment_ratio,
            grounding_ratio=grounding_ratio,
        )

        # Refusal gate
        final_answer = answer
        if result.refused:
            final_answer = REFUSAL_ANSWER

        return result, ver_result, final_answer

    def _enrich_response(self, parsed: RAGResponse, conf_result, ver_result) -> RAGResponse:
        """Set confidence and verification fields on RAGResponse."""
        parsed.confidence = conf_result.label
        parsed.confidence_score = conf_result.score
        parsed.refused = conf_result.refused
        # Store verification data in disclaimer for now (API model updated separately)
        if not ver_result.passed:
            parsed.disclaimer += "\n\n" + "\n".join(ver_result.warnings[:3])
        return parsed

    def _simple_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """SIMPLE: dense search → generate (2 LLM calls including classify)."""
        query_vector = self.embedder.encode_queries([question])[0]
        # Use a broader initial pool, then dynamically select
        results = self.vector_store.search_dense(query_vector.tolist(), top_k=20)

        # Phase 2 (Plan §3.6): default temporal filter, computed on the broad
        # pool before dynamic-top-k picks the final candidate count.
        annotations = self._annotate_phase2(results)
        results = self._filter_in_effect(results, annotations)

        top_k = self._dynamic_top_k(results, min_k=3, max_k=15)
        results = results[:top_k]
        results = self._expand_to_parent(results)  # Plan §3.4
        context = self._build_context(results, annotations)

        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results, annotations)
        context = self._build_context(results, annotations)

        # Post-process: verify + confidence
        conf_result, ver_result, final_answer = self._post_process(question, raw, results, context)

        parsed = self._parse_response(final_answer, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "simple"
        parsed = self._enrich_response(parsed, conf_result, ver_result)
        return parsed

    def _medium_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """MEDIUM: hybrid search → dynamic top-k → rerank → generate."""
        output = self.embedder.encode([question], return_dense=True, return_sparse=True)
        dense_vec = output["dense_vecs"][0]
        sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}

        # Broader initial pool
        results = self.vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=20)

        # Phase 2 (Plan §3.6): default temporal filter, before dynamic-top-k/rerank.
        annotations = self._annotate_phase2(results)
        results = self._filter_in_effect(results, annotations)

        top_k = self._dynamic_top_k(results, min_k=5, max_k=15)
        results = results[:top_k]

        # Re-rank (if available)
        results, rerank_scores = self._rerank_results(question, results)

        results = results[:15]
        results = self._expand_to_parent(results)  # Plan §3.4
        context = self._build_context(results, annotations)
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results, annotations)
        context = self._build_context(results, annotations)

        # Post-process: verify + confidence
        conf_result, ver_result, final_answer = self._post_process(
            question, raw, results, context, rerank_scores=rerank_scores,
        )

        parsed = self._parse_response(final_answer, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "medium"
        parsed = self._enrich_response(parsed, conf_result, ver_result)
        return parsed

    def _complex_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """COMPLEX: full pipeline (rewrite → multi-query → HyDE → hybrid → rerank → compress → generate)."""
        # 1. Query rewrite
        rewritten = self.llm.generate(QUERY_REWRITE_PROMPT.format(question=question), temperature=0.1).strip()

        # 2. Multi-query expansion
        expanded_raw = self.llm.generate(MULTI_QUERY_PROMPT.format(question=rewritten), temperature=0.3)
        queries = [q.strip() for q in expanded_raw.split("\n") if q.strip() and q[0].isdigit()]
        queries = [q.split(". ", 1)[-1] if ". " in q else q for q in queries]

        # 3. HyDE — generate hypothetical document
        hyde_text = self.llm.generate(HYDE_PROMPT.format(question=question), temperature=0.2).strip()

        # 4. Hybrid search with multiple queries + HyDE (broader pool each)
        all_queries = [rewritten] + queries[:2] + [hyde_text]
        all_results = []
        for q in all_queries:
            output = self.embedder.encode([q], return_dense=True, return_sparse=True)
            dense_vec = output["dense_vecs"][0]
            sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
            results = self.vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=10)
            all_results.extend(results)

        # 5. Deduplicate and apply dynamic top-k
        seen = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.point_id not in seen:
                seen.add(r.point_id)
                unique_results.append(r)
        # Phase 2 (Plan §3.6): default temporal filter, before dynamic-top-k/rerank.
        annotations = self._annotate_phase2(unique_results)
        unique_results = self._filter_in_effect(unique_results, annotations)

        top_k = self._dynamic_top_k(unique_results, min_k=5, max_k=20)
        results = unique_results[:top_k]

        # 6. Re-rank (if available)
        results, rerank_scores = self._rerank_results(question, results)

        # 7. Context Compression — extract only relevant sentences
        results = self._expand_to_parent(results)  # Plan §3.4
        context = self._build_context(results, annotations)
        context = self._compress_context(question, context)

        # 8. Generate
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results, annotations)
        context = self._build_context(results, annotations)

        # Post-process: verify + confidence
        conf_result, ver_result, final_answer = self._post_process(
            question, raw, results, context, rerank_scores=rerank_scores,
        )

        parsed = self._parse_response(final_answer, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "complex"
        parsed = self._enrich_response(parsed, conf_result, ver_result)
        return parsed

    # ── Re-ranking ───────────────────────────────────────────

    def _rerank_results(
        self, question: str, results: List[SearchResult]
    ) -> Tuple[List[SearchResult], Optional[List[float]]]:
        """
        Re-rank results using BGE-Reranker v2 (if available). Falls back to score sort.

        Returns:
            (results, rerank_scores) — rerank_scores is None when reranking did not
            run (too few results / reranker unavailable / error), so callers can
            tell "no signal" apart from "signal is empty".
        """
        if len(results) <= 1:
            return results, None
        reranker = self._get_reranker()
        if reranker is None:
            # Fallback: already sorted by Qdrant score — keep as-is
            return results, None
        try:
            passages = [r.chunk_text[:512] for r in results]
            pairs = [[question, p] for p in passages]
            scores = reranker.compute_score(pairs, normalize=True)
            # Re-sort by reranker score
            indexed = list(enumerate(results))
            indexed.sort(key=lambda x: scores[x[0]], reverse=True)
            sorted_results = [results[i] for i, _ in indexed]
            sorted_scores = [scores[i] for i, _ in indexed]
            return sorted_results, sorted_scores
        except Exception as e:
            print(f"⚠️ Reranking failed (skip): {e}")
            return results, None

    # ── Context Compression ──────────────────────────────────

    def _compress_context(self, question: str, context: str) -> str:
        """Compress context by extracting only sentences relevant to the query (COMPLEX only)."""
        if not context or len(context) < 500:
            return context
        COMPRESS_PROMPT = """Dưới đây là các tài liệu pháp luật có liên quan đến câu hỏi.
Trích xuất CHỈ những câu có liên quan trực tiếp đến câu hỏi.
Bỏ qua các câu không liên quan. Giữ nguyên nguyên văn, CHỈ lược bỏ.

Câu hỏi: {question}

Tài liệu:
{context}

Các câu liên quan (giữ nguyên văn):"""
        try:
            compressed = self.llm.generate(
                COMPRESS_PROMPT.format(question=question, context=context), temperature=0, max_tokens=1500,
            ).strip()
            return compressed if len(compressed) > 50 else context
        except Exception:
            return context  # Fallback to original context on any error

    # ── Streaming ────────────────────────────────────────────


    def query_stream(self, question: str) -> Generator[str, None, None]:
        """Process a question with streaming response and intermediate steps."""
        start_time = time.time()

        rerank_scores = None

        # Step 0: Citation-lookup route — parsed BEFORE the (LLM) complexity
        # classifier; citation-style queries ("Điều 173 BLHS 2015 quy định
        # về tội gì?") skip the classifier call entirely (see
        # AdaptiveRAGPipeline.query for the full rationale).
        parsed_citation = parse_citation(question)
        if parsed_citation is not None:
            complexity = "lookup"
            citation_lookup = self._get_citation_lookup()
            doc_ids = citation_lookup.resolve_doc(
                parsed_citation.doc_hint,
                parsed_citation.year,
                doc_number=parsed_citation.doc_number,
            )
            aliases = citation_lookup._get_alias_index()
            resolved_titles = [aliases.get(d, str(d)) for d in doc_ids[:3]]
            article_label = f"Điều {parsed_citation.primary.number}"
            if parsed_citation.primary.khoan:
                article_label += f", Khoản {parsed_citation.primary.khoan}"
            resolved_str = ", ".join(resolved_titles) if resolved_titles else "(ambiguous — searched all documents)"
            yield f'__STEP__{json.dumps({"step": "Citation Lookup", "result": f"{article_label} — resolved: {resolved_str}"})}'
        else:
            # Step 1: Classify query complexity + scope gate (merged into a
            # single LLM call — see AdaptiveRAGPipeline._classify_and_scope).
            complexity, in_scope = self._classify_and_scope(question)
            scope_label = "IN" if in_scope else "OUT"
            yield f'__STEP__{json.dumps({"step": "Classification", "result": f"{complexity.upper()} (scope={scope_label})"})}'

            if not in_scope:
                corpus_summary = self._get_citation_lookup().corpus_summary()
                answer = OUT_OF_SCOPE_ANSWER_TEMPLATE.format(corpus_summary=corpus_summary)
                yield f'__STEP__{json.dumps({"step": "Routing", "result": "Out of Scope — refused before retrieval/generation"})}'
                # Yield the refusal text as normal tokens so the UI displays
                # it exactly like any other answer (no retrieval/_post_process
                # ran, so there is nothing else to stream).
                yield answer
                verification_payload = {
                    "confidence_score": 0.0,
                    "confidence": "low",
                    "refused": True,
                    "warnings": [],
                    "reason": "out_of_scope",
                }
                yield f'__VERIFICATION__{json.dumps(verification_payload, ensure_ascii=False)}'
                return

        if complexity == "lookup":
            results, route, _ = retrieve_routed(
                question, self.vector_store, self.embedder, citation_lookup, fill_top_k=10,
            )
            annotations = self._annotate_phase2(results)  # Plan §3.6
            results = self._filter_in_effect(results, annotations)
            yield f'__STEP__{json.dumps({"step": "Routing", "result": f"Citation Lookup ({len(results)} chunks, route={route})"})}'

        elif complexity == "simple":
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Simple Pipeline (Dense Search)"})}'
            query_vector = self.embedder.encode_queries([question])[0]
            results = self.vector_store.search_dense(query_vector.tolist(), top_k=20)
            annotations = self._annotate_phase2(results)  # Plan §3.6
            results = self._filter_in_effect(results, annotations)
            # Dynamic top-k
            top_k = self._dynamic_top_k(results, min_k=3, max_k=15)
            results = results[:top_k]

        elif complexity == "medium":
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Medium Pipeline (Hybrid Search)"})}'
            output = self.embedder.encode([question], return_dense=True, return_sparse=True)
            dense_vec = output["dense_vecs"][0]
            sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
            results = self.vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=20)
            annotations = self._annotate_phase2(results)  # Plan §3.6
            results = self._filter_in_effect(results, annotations)
            # Dynamic top-k + rerank
            top_k = self._dynamic_top_k(results, min_k=5, max_k=15)
            results = results[:top_k]
            results, rerank_scores = self._rerank_results(question, results)

        else: # complex
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Complex Pipeline (Rewrite + HyDE + Multi-query)"})}'
            
            # 1. Query rewrite
            rewritten = self.llm.generate(
                QUERY_REWRITE_PROMPT.format(question=question), temperature=0.1
            ).strip()
            yield f'__STEP__{json.dumps({"step": "Query Rewrite", "result": rewritten})}'

            # 2. Multi-query expansion
            expanded_raw = self.llm.generate(
                MULTI_QUERY_PROMPT.format(question=rewritten), temperature=0.3
            )
            queries = [q.strip() for q in expanded_raw.split("\n") if q.strip() and q[0].isdigit()]
            queries = [q.split(". ", 1)[-1] if ". " in q else q for q in queries]
            yield f'__STEP__{json.dumps({"step": "Multi-query", "result": ", ".join(queries)})}'

            # 3. HyDE
            hyde_text = self.llm.generate(
                HYDE_PROMPT.format(question=question), temperature=0.2
            ).strip()
            yield f'__STEP__{json.dumps({"step": "HyDE (Hypothetical Document)", "result": hyde_text[:100] + "..." if len(hyde_text) > 100 else hyde_text})}'

            # 4. Search
            all_queries = [rewritten] + queries[:2] + [hyde_text]
            all_results = []
            for q in all_queries:
                output = self.embedder.encode([q], return_dense=True, return_sparse=True)
                dense_vec = output["dense_vecs"][0]
                sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
                all_results.extend(self.vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=10))

            # Deduplicate + dynamic top-k
            seen = set()
            unique_results = []
            for r in sorted(all_results, key=lambda x: x.score, reverse=True):
                if r.point_id not in seen:
                    seen.add(r.point_id)
                    unique_results.append(r)
            annotations = self._annotate_phase2(unique_results)  # Plan §3.6
            unique_results = self._filter_in_effect(unique_results, annotations)
            top_k = self._dynamic_top_k(unique_results, min_k=5, max_k=20)
            results = unique_results[:top_k]

            # Re-rank (if available)
            results, rerank_scores = self._rerank_results(question, results)

        # Build context & citations
        results = self._expand_to_parent(results)  # Plan §3.4
        context = self._build_context(results, annotations)
        citations = self._build_citations(results, annotations)

        # Compress context for COMPLEX
        if complexity == "complex":
            context = self._compress_context(question, context)

        yield f'__STEP__{json.dumps({"step": "Search Results", "result": f"Found {len(results)} chunks for context."})}'
        yield f'__STEP__{json.dumps({"step": "Context (Found Chunks)", "result": context})}'
        yield f'__CITATIONS__{json.dumps(citations, ensure_ascii=False)}'

        # Stream generation
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        yield f'__STEP__{json.dumps({"step": "LLM Prompt", "result": prompt})}'

        full_answer = ""
        for token in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        ):
            full_answer += token
            yield token

        # Post-process (verify + confidence) on the full answer, same as the
        # non-streaming path — run after generation so it doesn't add latency
        # to the first token, but still reaches the UI before the turn is "done".
        # Strip <think> blocks first: the raw stream includes Qwen3 reasoning
        # (the UI hides it for display), and verifying reasoning text as claims
        # wrecks the entailment/overlap signals.
        clean_answer = re.sub(r'<think>[\s\S]*?(?:</think>|$)', '', full_answer, flags=re.IGNORECASE).strip()
        conf_result, ver_result, final_answer = self._post_process(
            question, clean_answer, results, context, rerank_scores=rerank_scores,
        )
        verification_payload = {
            "confidence_score": conf_result.score,
            "confidence": conf_result.label,
            "refused": conf_result.refused,
            "warnings": ver_result.warnings,
            "entailment_ratio": ver_result.entailment_ratio,
            "contradiction_ratio": ver_result.contradiction_ratio,
            "grounding_ratio": ver_result.grounding_ratio,
        }
        if conf_result.refused:
            verification_payload["refusal_answer"] = final_answer
        yield f'__VERIFICATION__{json.dumps(verification_payload, ensure_ascii=False)}'