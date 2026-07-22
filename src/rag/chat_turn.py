"""Single-turn chat orchestration for the non-streaming path (M4).

Threads session memory + condensation + semantic cache around the pure RAG
pipeline. Lives outside the API layer so it is unit-testable without a live
Qdrant/model stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.rag.condense import condense_query
from src.rag.orchestrator import RAGResponse


@dataclass
class ChatTurnResult:
    response: RAGResponse
    cache_hit: bool
    condensed_question: str


def run_chat_turn(pipeline, store, cache, llm, collection: str,
                  session_id: Optional[str], question: str) -> ChatTurnResult:
    history = store.get_history(session_id)
    condensed = condense_query(history, question, llm)

    response = cache.get(condensed, collection) if cache else None
    cache_hit = response is not None
    if response is None:
        response = pipeline.query(condensed)
        if cache:
            cache.set(condensed, collection, response)

    store.append_turn(session_id, question, response.answer)
    return ChatTurnResult(response=response, cache_hit=cache_hit, condensed_question=condensed)
