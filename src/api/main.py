"""
FastAPI backend for Vietnamese Legal Knowledge Bot.
"""

import json
import time
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.db.qdrant_client import LegalVectorStore
from src.data.embedder import BGEEmbedder
from src.llm.gateway import LLMGateway
from src.rag.orchestrator import NaiveRAGPipeline, AdaptiveRAGPipeline


# --- Models ---
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_adaptive: bool = False  # MVP: False, v1.0+: True


class ChatResponse(BaseModel):
    answer: str
    citations: list = []
    confidence: str = "medium"
    disclaimer: str = ""
    latency_ms: float = 0
    pipeline_used: str = "naive"


class DocumentFilter(BaseModel):
    legal_type: Optional[str] = None
    keyword: Optional[str] = None
    page: int = 1
    page_size: int = 20


# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.vector_store = LegalVectorStore()
    app.state.embedder = None  # Lazy load (BGE-M3 is heavy)
    app.state.llm = LLMGateway()
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
    """Lazy-load BGE-M3 embedder."""
    if app.state.embedder is None:
        app.state.embedder = BGEEmbedder()
    return app.state.embedder


def get_naive_pipeline() -> NaiveRAGPipeline:
    return NaiveRAGPipeline(
        vector_store=app.state.vector_store,
        embedder=get_embedder(),
        llm_gateway=app.state.llm,
    )


def get_adaptive_pipeline() -> AdaptiveRAGPipeline:
    return AdaptiveRAGPipeline(
        vector_store=app.state.vector_store,
        embedder=get_embedder(),
        llm_gateway=app.state.llm,
    )


# --- Routes ---
@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    try:
        if request.use_adaptive:
            pipeline = get_adaptive_pipeline()
        else:
            pipeline = get_naive_pipeline()

        response = pipeline.query(request.question)
        return ChatResponse(
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            disclaimer=response.disclaimer,
            latency_ms=response.latency_ms,
            pipeline_used=response.pipeline_used,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint (SSE)."""
    try:
        if request.use_adaptive:
            pipeline = get_adaptive_pipeline()
        else:
            pipeline = get_naive_pipeline()

        async def event_generator():
            start_time = time.time()
            full_answer = ""
            citations = []
            citations_sent = False

            for token in pipeline.query_stream(request.question):
                # Intercept citations marker
                if token.startswith("__CITATIONS__"):
                    try:
                        citations = json.loads(token[len("__CITATIONS__"):])
                        yield {
                            "event": "citations",
                            "data": json.dumps(citations, ensure_ascii=False),
                        }
                        citations_sent = True
                    except json.JSONDecodeError:
                        pass
                    continue

                full_answer += token
                yield {
                    "event": "token",
                    "data": json.dumps({"token": token}, ensure_ascii=False),
                }

            # Final event with metadata
            yield {
                "event": "done",
                "data": json.dumps({
                    "full_answer": full_answer,
                    "citations": citations,
                    "confidence": "high" if len(citations) >= 2 else "medium",
                    "latency_ms": (time.time() - start_time) * 1000,
                }, ensure_ascii=False),
            }

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
    """Get system statistics."""
    try:
        store = app.state.vector_store
        info = store.get_collection_info()
        return {
            "indexed_documents": info.get("points_count", 0),
            "collection_name": info.get("name", ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
