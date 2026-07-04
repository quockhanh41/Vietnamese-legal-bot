"""
Main RAG orchestrator — ties together query processing, retrieval, and generation.

MVP version: Naive RAG (dense search → LLM generate)
v1.0: Adaptive RAG (query classifier + routing)
"""

import json
import os
import sys
import time
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass

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


from dataclasses import dataclass, field

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
        top_k: int = 5,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = llm_gateway
        self.top_k = top_k

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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        result = self.llm.generate(
            prompt=prompt,
            temperature=0,
            max_tokens=10,
        )
        result = result.strip().upper()
        if result in ("SIMPLE", "MEDIUM", "COMPLEX"):
            return result.lower()
        return "medium"  # Safe default

    def _simple_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """SIMPLE: dense search → generate (2 LLM calls including classify)."""
        query_vector = self.embedder.encode_queries([question])[0]
        results = self.vector_store.search_dense(query_vector.tolist(), top_k=5)
        context = self._build_context(results)

        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results)
        parsed = self._parse_response(raw, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "simple"
        return parsed

    def _medium_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """MEDIUM: hybrid search → rerank → generate."""
        output = self.embedder.encode([question], return_dense=True, return_sparse=True)
        dense_vec = output["dense_vecs"][0]
        sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}

        results = self.vector_store.search_hybrid(
            dense_vec.tolist(), sparse_vec, top_k=10
        )
        context = self._build_context(results[:5])  # Top-5 after fusion

        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(results[:5])
        parsed = self._parse_response(raw, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "medium"
        return parsed

    def _complex_pipeline(self, question: str, start_time: float) -> RAGResponse:
        """COMPLEX: full pipeline."""
        # 1. Query rewrite
        rewritten = self.llm.generate(
            QUERY_REWRITE_PROMPT.format(question=question),
            temperature=0.1,
        ).strip()

        # 2. Multi-query expansion
        expanded_raw = self.llm.generate(
            MULTI_QUERY_PROMPT.format(question=rewritten),
            temperature=0.3,
        )
        queries = [q.strip() for q in expanded_raw.split("\n") if q.strip() and q[0].isdigit()]
        queries = [q.split(". ", 1)[-1] if ". " in q else q for q in queries]

        # 3. HyDE — generate hypothetical document
        hyde_text = self.llm.generate(
            HYDE_PROMPT.format(question=question),
            temperature=0.2,
        ).strip()

        # 4. Hybrid search with multiple queries + HyDE
        all_queries = [rewritten] + queries[:2] + [hyde_text]

        all_results = []
        for q in all_queries:
            output = self.embedder.encode([q], return_dense=True, return_sparse=True)
            dense_vec = output["dense_vecs"][0]
            sparse_vec = output["lexical_weights"][0] if output["lexical_weights"] else {}

            results = self.vector_store.search_hybrid(
                dense_vec.tolist(), sparse_vec, top_k=10
            )
            all_results.extend(results)

        # Deduplicate and take top
        seen = set()
        unique_results = []
        for r in sorted(all_results, key=lambda x: x.score, reverse=True):
            if r.point_id not in seen:
                seen.add(r.point_id)
                unique_results.append(r)

        context = self._build_context(unique_results[:5])

        # 5. Generate
        prompt = GENERATION_PROMPT.format(context=context, question=question)
        raw = self.llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.1, json_mode=False)

        citations = self._build_citations(unique_results[:5])
        parsed = self._parse_response(raw, citations)
        parsed.latency_ms = (time.time() - start_time) * 1000
        parsed.pipeline_used = "complex"
        return parsed
