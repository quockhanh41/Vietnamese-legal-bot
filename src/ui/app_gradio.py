"""HF Space (Gradio + ZeroGPU) front-end -- M5 pivot. Additive: calls the
existing RAG pipeline in-process, no HTTP hop, no separate FastAPI/uvicorn
process. src/api/main.py and src/ui/app.py (Streamlit) are untouched and
remain a self-hosted Docker deploy path -- this file is the live public demo.
See docs/superpowers/specs/2026-07-23-m5-gradio-zerogpu-pivot-design.md."""
import spaces  # MUST be first -- before any torch-touching import

import json

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from FlagEmbedding import FlagReranker

from src.data.constants import DEFAULT_COLLECTION_NAME
from src.data.metadata_store import MetadataStore
from src.data.mixed_embedder import build_embedder
from src.db.qdrant_client import LegalVectorStore
from src.llm.gateway import LLMGateway
from src.rag.citation_lookup import CitationLookup
from src.rag.condense import condense_query
from src.rag.degraded import DEGRADED_MSG, is_exhaustion
from src.rag.orchestrator import AdaptiveRAGPipeline, RAGResponse
from src.rag.semantic_cache import SemanticCache
from src.ui.gradio_helpers import (
    format_citations_markdown,
    history_to_turns,
    is_quota_exceeded,
)

QUOTA_MSG = ("Bạn đã dùng hết lượt dùng thử miễn phí hôm nay trên hạ tầng "
             "ZeroGPU. Vui lòng quay lại vào ngày mai!")

# --- Module-scope light singletons ---
vector_store = LegalVectorStore(collection_name=DEFAULT_COLLECTION_NAME)
llm = LLMGateway()
metadata_store = MetadataStore()
citation_lookup = CitationLookup(vector_store)

# Heavy models lazily initialized inside @spaces.GPU context to guarantee
# clean CUDA initialization under ZeroGPU runtime.
_embedder = None
_reranker = None
_semantic_cache = None
_pipeline = None


def get_pipeline():
    global _embedder, _reranker, _semantic_cache, _pipeline
    if _pipeline is None:
        _embedder = build_embedder("mixed", "AITeamVN/Vietnamese_Embedding", device="cuda")
        _reranker = FlagReranker("AITeamVN/Vietnamese_Reranker", use_fp16=True, device="cuda")

        def _embed_query(text: str):
            out = _embedder.encode([text], return_dense=True, return_sparse=False)
            return out["dense_vecs"][0].tolist()

        _semantic_cache = SemanticCache.from_env(_embed_query)
        _pipeline = AdaptiveRAGPipeline(
            vector_store=vector_store,
            embedder=_embedder,
            llm_gateway=llm,
            metadata_store=metadata_store,
            reranker_model="AITeamVN/Vietnamese_Reranker",
            reranker=_reranker,
            citation_lookup=citation_lookup,
        )
    return _pipeline, _semantic_cache


@spaces.GPU(duration=90)
def handle_turn(message, history):
    """One @spaces.GPU allocation per turn -- covers condense, the semantic-
    cache lookup (needs an embedding), and on a miss: retrieve/rerank/
    generate(Groq, network)/NLI verify."""
    pipeline_obj, cache_obj = get_pipeline()

    turns = history_to_turns(history)
    condensed = condense_query(turns, message, llm)

    cached = cache_obj.get(condensed, DEFAULT_COLLECTION_NAME) if cache_obj else None
    if cached is not None:
        yield cached.answer + format_citations_markdown(cached.citations)
        return

    full_answer = ""
    citations = []
    refused = False
    try:
        for token in pipeline_obj.query_stream(condensed):
            if token.startswith("__CITATIONS__"):
                try:
                    citations = json.loads(token[len("__CITATIONS__"):])
                except json.JSONDecodeError:
                    pass
                continue
            if token.startswith("__VERIFICATION__"):
                try:
                    refused = json.loads(token[len("__VERIFICATION__"):]).get("refused", False)
                except json.JSONDecodeError:
                    pass
                continue
            if token.startswith("__STEP__"):
                continue
            full_answer += token
            yield full_answer
    except Exception as e:
        if is_exhaustion(e):
            yield DEGRADED_MSG
            return
        if is_quota_exceeded(e):
            yield QUOTA_MSG
            return
        raise

    yield full_answer + format_citations_markdown(citations)

    if cache_obj is not None and full_answer and not refused:
        response = RAGResponse(
            answer=full_answer, citations=citations, confidence="unverified",
            disclaimer="", latency_ms=0, pipeline_used="gradio", refused=refused,
        )
        cache_obj.set(condensed, DEFAULT_COLLECTION_NAME, response)


demo = gr.ChatInterface(
    fn=handle_turn,
    type="messages",
    title="Vietnamese Legal Knowledge Bot",
    description=("Adaptive RAG tra cứu văn bản pháp luật Việt Nam (600 văn bản). "
                 "Chạy trên HF ZeroGPU (free tier) -- lần hỏi đầu có thể chậm "
                 "(cold start nạp model)."),
)

if __name__ == "__main__":
    demo.launch()
