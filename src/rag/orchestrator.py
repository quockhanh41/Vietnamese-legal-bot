"""
Main RAG orchestrator — ties together query processing, retrieval, and generation.

MVP version: Naive RAG (dense search → LLM generate)
v1.0: Adaptive RAG (query classifier + routing)
"""

import json
import os
import sys
import statistics, time
from typing import List, Dict, Optional, Generator

from dataclasses import dataclass, field
# Ensure project root is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.db.qdrant_client import LegalVectorStore, SearchResult
from src.data.embedder import BGEEmbedder
from src.llm.gateway import LLMGateway
from src.llm.prompts import (
    SYSTEM_PROMPT,
    GENERATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
    COMPLEXITY_CLASSIFIER_PROMPT,
)

from src.rag.confidence import ConfidenceScorer, REFUSAL_ANSWER
from src.rag.verifier import HallucinationVerifier

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
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm_gateway
        self.top_k = top_k

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

        # Step 3: Build context from retrieved chunks
        context = self._build_context(results)

        # Step 4: Generate answer
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw_response = self.llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
            json_mode=False,
        )

        # Step 5: Parse response
        citations = self._build_citations(results)
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

        # Build context
        context = self._build_context(results)

        # Build citations from search results
        citations = self._build_citations(results)

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

    def _build_context(self, results: List[SearchResult]) -> str:
        """Format search results into a context string for the LLM."""
        parts = []
        for i, result in enumerate(results, 1):
            header = f"[{i}] {result.document_title}"
            if result.article:
                header += f" — {result.article}"
            if result.chapter:
                header += f" ({result.chapter})"

            parts.append(f"{header}\n{result.chunk_text[:500]}")

        return "\n\n---\n\n".join(parts)

    def _build_citations(self, results: List[SearchResult]) -> List[Dict]:
        """Build citation list from search results."""
        citations = []
        for r in results[:3]:  # Top 3 for citations
            citations.append({
                "document_title": r.document_title,
                "article": r.article,
                "chunk_text": r.chunk_text[:200] if r.chunk_text else "",
                "legal_type": r.legal_type,
                "url": r.url,
                "score": round(r.score, 4),
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

        # Step 0: Classify query complexity
        complexity = self._classify(question)

        # Step 1-5: Route to appropriate pipeline
        if complexity == "simple":
            return self._simple_pipeline(question, start_time)
        elif complexity == "medium":
            return self._medium_pipeline(question, start_time)
        else:
            return self._complex_pipeline(question, start_time)

    def _classify(self, question: str) -> str:
        """Classify query complexity: simple/medium/complex."""
        prompt = COMPLEXITY_CLASSIFIER_PROMPT.format(question=question)
        result = self.llm.generate(prompt=prompt, temperature=0, max_tokens=10)
        result = result.strip().upper()
        if result in ("SIMPLE", "MEDIUM", "COMPLEX"):
            return result.lower()
        return "medium"  # Safe default

    def _post_process(self, question: str, answer: str, results: List[SearchResult],
                      context: str) -> tuple:
        """
        Post-generation: verifier (Tier 1 + 2) + confidence scoring + refusal gate.

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

        # Get reranker scores if available (from _rerank_results output)
        # We don't store them separately, so pass None and let confidence handle it
        rerank_scores = None
        result = self._confidence_scorer.score(
            question=question,
            answer=answer,
            context=context_str,
            search_results=results,
            rerank_scores=rerank_scores,
            entailment_ratio=ver_result.entailment_ratio,
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
        top_k = self._dynamic_top_k(results, min_k=3, max_k=15)
        results = results[:top_k]
        context = self._build_context(results)

        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results)
        context = self._build_context(results)

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
        top_k = self._dynamic_top_k(results, min_k=5, max_k=15)
        results = results[:top_k]

        # Re-rank (if available)
        results = self._rerank_results(question, results)

        context = self._build_context(results[:15])
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results[:15])
        context = self._build_context(results[:15])

        # Post-process: verify + confidence
        conf_result, ver_result, final_answer = self._post_process(question, raw, results[:15], context)

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
        top_k = self._dynamic_top_k(unique_results, min_k=5, max_k=20)
        results = unique_results[:top_k]

        # 6. Re-rank (if available)
        results = self._rerank_results(question, results)

        # 7. Context Compression — extract only relevant sentences
        context = self._build_context(results)
        context = self._compress_context(question, context)

        # 8. Generate
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results)
        context = self._build_context(results)

        # Post-process: verify + confidence
        conf_result, ver_result, final_answer = self._post_process(question, raw, results, context)

        parsed = self._parse_response(final_answer, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "complex"
        parsed = self._enrich_response(parsed, conf_result, ver_result)
        return parsed

    # ── Re-ranking ───────────────────────────────────────────

    def _rerank_results(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results using BGE-Reranker v2 (if available). Falls back to score sort."""
        if len(results) <= 1:
            return results
        reranker = self._get_reranker()
        if reranker is None:
            # Fallback: already sorted by Qdrant score — keep as-is
            return results
        try:
            passages = [r.chunk_text[:512] for r in results]
            pairs = [[question, p] for p in passages]
            scores = reranker.compute_score(pairs, normalize=True)
            # Re-sort by reranker score
            indexed = list(enumerate(results))
            indexed.sort(key=lambda x: scores[x[0]], reverse=True)
            return [results[i] for i, _ in indexed]
        except Exception as e:
            print(f"⚠️ Reranking failed (skip): {e}")
            return results

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

        # Step 0: Classify query complexity
        complexity = self._classify(question)
        yield f'__STEP__{json.dumps({"step": "Classification", "result": complexity.upper()})}'

        if complexity == "simple":
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Simple Pipeline (Dense Search)"})}'
            query_vector = self.embedder.encode_queries([question])[0]
            results = self.vector_store.search_dense(query_vector.tolist(), top_k=20)
            # Dynamic top-k
            top_k = self._dynamic_top_k(results, min_k=3, max_k=15)
            results = results[:top_k]
            
        elif complexity == "medium":
            yield f'__STEP__{json.dumps({"step": "Routing", "result": "Medium Pipeline (Hybrid Search)"})}'
            output = self.embedder.encode([question], return_dense=True, return_sparse=True)
            dense_vec = output["dense_vecs"][0]
            sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}
            results = self.vector_store.search_hybrid(dense_vec.tolist(), sparse_vec, top_k=20)
            # Dynamic top-k + rerank
            top_k = self._dynamic_top_k(results, min_k=5, max_k=15)
            results = results[:top_k]
            results = self._rerank_results(question, results)
            
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
            top_k = self._dynamic_top_k(unique_results, min_k=5, max_k=20)
            results = unique_results[:top_k]

            # Re-rank (if available)
            results = self._rerank_results(question, results)

        # Build context & citations
        context = self._build_context(results)
        citations = self._build_citations(results)

        # Compress context for COMPLEX
        if complexity == "complex":
            context = self._compress_context(question, context)

        yield f'__STEP__{json.dumps({"step": "Search Results", "result": f"Found {len(results)} chunks for context."})}'
        yield f'__STEP__{json.dumps({"step": "Context (Found Chunks)", "result": context})}'
        yield f'__CITATIONS__{json.dumps(citations, ensure_ascii=False)}'

        # Stream generation
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        yield f'__STEP__{json.dumps({"step": "LLM Prompt", "result": prompt})}'
        
        for token in self.llm.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.1,
        ):
            yield token