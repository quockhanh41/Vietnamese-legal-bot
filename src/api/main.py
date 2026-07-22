"""
FastAPI backend for Vietnamese Legal Knowledge Bot.
Phase 2: Integrated with cross-referencing, temporal versioning, and amendments.
"""

import json
import time
from typing import Optional
from contextlib import asynccontextmanager
from datetime import date as DateType

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from src.db.qdrant_client import LegalVectorStore
from src.data.mixed_embedder import build_embedder
from src.data.constants import DEFAULT_COLLECTION_NAME
from src.data.metadata_store import MetadataStore
from src.llm.gateway import LLMGateway
from src.rag.orchestrator import NaiveRAGPipeline, AdaptiveRAGPipeline, RAGResponse
from src.pipeline.retriever import EnhancedRetriever
from src.rag.citation_lookup import CitationLookup
from src.rag.condense import condense_query
from src.rag.session_store import SessionStore
from src.rag.semantic_cache import SemanticCache
from src.rag.chat_turn import run_chat_turn


# --- Models ---
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_adaptive: bool = False
    expand_context: bool = False  # Phase 2: cross-reference expansion
    expansion_hops: int = 1


class ChatResponse(BaseModel):
    answer: str
    citations: list = []
    confidence: str = "medium"
    confidence_score: float = 0.0   # Phase B: exact score 0.0-1.0
    refused: bool = False            # Phase B: confidence gate refused?
    disclaimer: str = ""
    latency_ms: float = 0
    pipeline_used: str = "naive"
    expanded_results: int = 0  # Phase 2: context expansion count
    cache_hit: bool = False          # M4.2: served from semantic cache?
    condensed_question: str = ""     # M4.1: standalone rewrite of a follow-up


class EnrichRequest(BaseModel):
    collection_name: Optional[str] = None


class ArticleHistoryRequest(BaseModel):
    doc_id: int
    article: str


class DocumentFilter(BaseModel):
    legal_type: Optional[str] = None
    keyword: Optional[str] = None
    page: int = 1
    page_size: int = 20


# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.vector_store = LegalVectorStore(
        collection_name=DEFAULT_COLLECTION_NAME
    )
    app.state.embedder = None  # Lazy load (mixed embedder is heavy)
    app.state.llm = LLMGateway()
    app.state.metadata_store = MetadataStore()
    app.state.enhanced_retriever = None  # Lazy load
    app.state.reranker = None  # Lazy load
    app.state.citation_lookup = CitationLookup(app.state.vector_store)
    app.state.session_store = SessionStore.from_env()
    app.state.semantic_cache = SemanticCache.from_env(_embed_query)
    # Warm up the corpus cache and reranker model in the background so the server
    # can start accepting connections immediately — both otherwise load on the
    # first request (the reranker model load alone is ~60s on the complex route).
    import asyncio
    asyncio.create_task(run_in_threadpool(app.state.citation_lookup.corpus_summary_compact))
    asyncio.create_task(run_in_threadpool(get_reranker))
    yield
    # Shutdown
    pass


app = FastAPI(
    title="Vietnamese Legal Knowledge Bot",
    description="Adaptive RAG chatbot for Vietnamese legal documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedder():
    """Lazy-load the production embedder — AITeamVN mixed hybrid (AITeamVN dense
    + BGE-M3 sparse). Heavy; instantiated on first use. Must match the embedder
    that built DEFAULT_COLLECTION_NAME (M3 Phase 0)."""
    if app.state.embedder is None:
        app.state.embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding")
    return app.state.embedder


def get_enhanced_retriever() -> EnhancedRetriever:
    """Lazy-load enhanced retriever with Phase 2 capabilities."""
    if app.state.enhanced_retriever is None:
        app.state.enhanced_retriever = EnhancedRetriever(
            vector_store=app.state.vector_store,
            metadata_store=app.state.metadata_store,
            embedder=get_embedder(),
        )
    return app.state.enhanced_retriever


def get_reranker():
    """Lazy-load the reranker model globally."""
    if getattr(app.state, "reranker", None) is None:
        try:
            from FlagEmbedding import FlagReranker
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            app.state.reranker = FlagReranker(
                "AITeamVN/Vietnamese_Reranker",
                use_fp16=True,
                device=device,
            )
        except Exception as e:
            print(f"Reranker not available (skip): {e}")
            app.state.reranker = False  # Sentinel
    return app.state.reranker if app.state.reranker else None


def _embed_query(text: str):
    """Dense query embedding for the semantic cache. Normalized (dot == cosine),
    matches the collection's distance metric."""
    out = get_embedder().encode([text], return_dense=True, return_sparse=False)
    return out["dense_vecs"][0].tolist()


def _replay_events(resp: RAGResponse, start_time: float):
    """Render a cached RAGResponse as the SSE event sequence the UI expects."""
    events = []
    if resp.citations:
        events.append({"event": "citations",
                       "data": json.dumps(resp.citations, ensure_ascii=False)})
    events.append({"event": "token",
                   "data": json.dumps({"token": resp.answer}, ensure_ascii=False)})
    events.append({"event": "verification",
                   "data": json.dumps({"confidence": resp.confidence,
                                       "confidence_score": resp.confidence_score,
                                       "refused": resp.refused},
                                      ensure_ascii=False)})
    events.append({"event": "done",
                   "data": json.dumps({
                       "full_answer": resp.answer, "citations": resp.citations,
                       "confidence": resp.confidence, "confidence_score": resp.confidence_score,
                       "refused": resp.refused,
                       "latency_ms": (time.time() - start_time) * 1000, "cache_hit": True,
                   }, ensure_ascii=False)})
    return events


def get_naive_pipeline() -> NaiveRAGPipeline:
    return NaiveRAGPipeline(
        vector_store=app.state.vector_store,
        embedder=get_embedder(),
        llm_gateway=app.state.llm,
        metadata_store=app.state.metadata_store,
        citation_lookup=app.state.citation_lookup,
    )


def get_adaptive_pipeline() -> AdaptiveRAGPipeline:
    return AdaptiveRAGPipeline(
        vector_store=app.state.vector_store,
        embedder=get_embedder(),
        llm_gateway=app.state.llm,
        metadata_store=app.state.metadata_store,
        reranker_model="AITeamVN/Vietnamese_Reranker",
        reranker=get_reranker(),
        citation_lookup=app.state.citation_lookup,
    )


# --- Routes ---
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Non-streaming chat. Sync `def` so FastAPI runs it in the threadpool.
    Threads session memory + condensation + semantic cache around the pipeline."""
    try:
        pipeline = get_adaptive_pipeline() if request.use_adaptive else get_naive_pipeline()
        result = run_chat_turn(
            pipeline=pipeline,
            store=app.state.session_store,
            cache=app.state.semantic_cache,
            llm=app.state.llm,
            collection=DEFAULT_COLLECTION_NAME,
            session_id=request.session_id,
            question=request.question,
        )
        r = result.response
        return ChatResponse(
            answer=r.answer, citations=r.citations, confidence=r.confidence,
            confidence_score=r.confidence_score, refused=r.refused,
            disclaimer=r.disclaimer, latency_ms=r.latency_ms,
            pipeline_used=r.pipeline_used,
            cache_hit=result.cache_hit, condensed_question=result.condensed_question,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat (SSE). Cache HIT → replay cached response as SSE (<500ms);
    MISS → stream normally, then persist to memory + cache."""
    try:
        pipeline = get_adaptive_pipeline() if request.use_adaptive else get_naive_pipeline()
        store = app.state.session_store
        cache = app.state.semantic_cache

        history = await run_in_threadpool(store.get_history, request.session_id)
        condensed = await run_in_threadpool(condense_query, history, request.question, app.state.llm)
        cached = None
        if cache is not None:
            cached = await run_in_threadpool(cache.get, condensed, DEFAULT_COLLECTION_NAME)

        async def event_generator():
            start_time = time.time()

            if cached is not None:
                for event in _replay_events(cached, start_time):
                    yield event
                await run_in_threadpool(store.append_turn, request.session_id,
                                        request.question, cached.answer)
                return

            full_answer = ""
            citations = []
            # Falls back to these when the pipeline never emits __VERIFICATION__
            # (e.g. NaiveRAGPipeline, which has no confidence/verification step
            # — hence "unverified", not a fabricated level).
            confidence_label = "unverified"
            confidence_score = 0.0
            refused = False

            # pipeline.query_stream is a sync generator (blocking model/network
            # calls) — run its iteration in the threadpool so it doesn't block
            # the event loop while other requests are being served.
            async for token in iterate_in_threadpool(pipeline.query_stream(condensed)):
                # Intercept step marker
                if token.startswith("__STEP__"):
                    try:
                        step_data = json.loads(token[len("__STEP__"):])
                        yield {"event": "step", "data": json.dumps(step_data, ensure_ascii=False)}
                    except json.JSONDecodeError:
                        pass
                    continue

                # Intercept citations marker
                if token.startswith("__CITATIONS__"):
                    try:
                        citations = json.loads(token[len("__CITATIONS__"):])
                        yield {"event": "citations", "data": json.dumps(citations, ensure_ascii=False)}
                    except json.JSONDecodeError:
                        pass
                    continue

                # Intercept verification marker (confidence + hallucination check,
                # computed after the full answer has streamed)
                if token.startswith("__VERIFICATION__"):
                    try:
                        verification = json.loads(token[len("__VERIFICATION__"):])
                        confidence_label = verification.get("confidence", confidence_label)
                        confidence_score = verification.get("confidence_score", confidence_score)
                        refused = verification.get("refused", refused)
                        yield {"event": "verification", "data": json.dumps(verification, ensure_ascii=False)}
                    except json.JSONDecodeError:
                        pass
                    continue

                full_answer += token
                yield {"event": "token", "data": json.dumps({"token": token}, ensure_ascii=False)}

            # Final event with metadata
            yield {"event": "done", "data": json.dumps({
                "full_answer": full_answer, "citations": citations,
                "confidence": confidence_label, "confidence_score": confidence_score,
                "latency_ms": (time.time() - start_time) * 1000, "cache_hit": False,
            }, ensure_ascii=False)}

            # Persist AFTER streaming completes (fail-open).
            await run_in_threadpool(store.append_turn, request.session_id,
                                    request.question, full_answer)
            if cache is not None and full_answer and not refused:
                reconstructed = RAGResponse(
                    answer=full_answer, citations=citations, confidence=confidence_label,
                    disclaimer="", latency_ms=(time.time() - start_time) * 1000,
                    pipeline_used="stream", confidence_score=confidence_score, refused=refused,
                )
                await run_in_threadpool(cache.set, condensed, DEFAULT_COLLECTION_NAME, reconstructed)

        return EventSourceResponse(event_generator())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents(
    legal_type: Optional[str] = None,
    keyword: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """List/search documents (for Document Browser)."""
    try:
        store = app.state.vector_store
        result = store.list_documents(
            legal_type=legal_type,
            keyword=keyword,
            page=page,
            page_size=page_size,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def stats():
    """Get system statistics (Phase 2 enhanced)."""
    try:
        store = app.state.vector_store
        info = store.get_collection_info()

        # Add Phase 2 stats if enrichment has been run
        phase2 = {}
        try:
            index = app.state.metadata_store.get_index()
            phase2 = {
                "documents_enriched": index.get("total_documents", 0),
                "currently_in_effect": index.get("currently_in_effect", 0),
                "amendment_laws": index.get("amendment_laws", 0),
                "graph_nodes": index.get("total_graph_nodes", 0),
                "graph_edges": index.get("total_graph_edges", 0),
                "total_amendments": index.get("total_amendments", 0),
                "last_enriched": index.get("last_enriched"),
            }
        except Exception:
            pass  # Enrichment not yet run

        return {
            "indexed_documents": info.get("points_count", 0),
            "collection_name": info.get("name", ""),
            "phase2": phase2,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# Phase 2 Endpoints
# ═══════════════════════════════════════════════════════════════

@app.post("/api/pipeline/enrich")
async def run_enrichment(request: EnrichRequest = None):
    """
    Run the Phase 2 enrichment pipeline on all documents.

    Extracts cross-references, timelines, and amendment data,
    storing results in the metadata store.
    """
    try:
        from src.pipeline.enricher import EnrichmentPipeline

        pipeline = EnrichmentPipeline(
            vector_store=app.state.vector_store,
            metadata_store=app.state.metadata_store,
        )

        result = pipeline.run()

        return {
            "status": "completed" if not result.errors else "completed_with_errors",
            "docs_processed": result.docs_processed,
            "total_docs": result.total_docs,
            "base_laws": result.base_laws,
            "amendment_laws": result.amendment_laws,
            "graph_nodes": result.graph_nodes,
            "graph_edges": result.graph_edges,
            "with_effective_date": result.with_effective_date,
            "currently_in_effect": result.currently_in_effect,
            "total_amendments": result.total_amendments,
            "articles_amended": result.articles_amended,
            "elapsed_seconds": result.elapsed_seconds,
            "errors": result.errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pipeline/status")
async def enrichment_status():
    """Get Phase 2 enrichment status and index."""
    try:
        index = app.state.metadata_store.get_index()
        return {
            "enriched": index.get("total_documents", 0) > 0,
            "index": index,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/article/history")
async def article_history(
    doc_id: int = Query(..., description="Document ID"),
    article: str = Query(..., description="Article number, e.g. 'Điều 123'"),
):
    """Get amendment history for a specific article."""
    try:
        retriever = get_enhanced_retriever()
        history = retriever.get_article_history(doc_id, article)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/stats")
async def graph_stats():
    """Get knowledge graph statistics."""
    try:
        graph_data = app.state.metadata_store.get_graph()
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        # Count by type
        node_types = {}
        for n in nodes:
            nt = n.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types = {}
        for e in edges:
            et = e.get("ref_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "node_types": node_types,
            "edge_types": edge_types,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/neighbors")
async def graph_neighbors(
    doc_id: int = Query(..., description="Document ID"),
    article: str = Query(..., description="Article number"),
    hops: int = Query(1, description="Number of hops (1-2)"),
):
    """Get neighboring articles in the cross-reference graph."""
    try:
        node_id = f"doc_{doc_id}_{article.lower().replace(' ', '_')}"
        neighbors = app.state.metadata_store.get_article_neighbors(node_id, hops)
        return {
            "node_id": node_id,
            "hops": hops,
            "neighbors": neighbors[:20],
            "count": len(neighbors),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/timeline/effective")
async def effective_laws(
    at_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (default: today)"),
):
    """Get laws effective at a given date."""
    try:
        from datetime import date as dt

        target_date = dt.fromisoformat(at_date) if at_date else dt.today()

        effective = app.state.metadata_store.get_effective_laws(target_date)

        return {
            "at_date": target_date.isoformat(),
            "count": len(effective),
            "laws": [
                {
                    "doc_id": tl.get("doc_id"),
                    "law_name": tl.get("law_name"),
                    "effective_date": tl.get("effective_date"),
                }
                for tl in effective[:50]
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/enhanced")
async def enhanced_search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    expand_context: bool = Query(False, description="Enable cross-reference expansion"),
    expansion_hops: int = Query(1, description="Number of expansion hops"),
):
    """Enhanced search with Phase 2 context expansion and metadata."""
    try:
        retriever = get_enhanced_retriever()
        results = retriever.retrieve(
            query=query,
            top_k=top_k,
            expand_context=expand_context,
            expansion_hops=expansion_hops,
        )
        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "expand_context": expand_context,
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
