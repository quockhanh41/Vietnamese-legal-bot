from unittest.mock import MagicMock

from src.rag.chat_turn import run_chat_turn
from src.rag.orchestrator import RAGResponse


def _resp(answer="A"):
    return RAGResponse(
        answer=answer, citations=[], confidence="high", disclaimer="",
        latency_ms=1.0, pipeline_used="simple",
    )


def test_cache_miss_runs_pipeline_stores_and_caches():
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = None
    pipeline = MagicMock(); pipeline.query.return_value = _resp("hello")

    out = run_chat_turn(pipeline, store, cache, MagicMock(), "col", "s1", "câu hỏi")

    assert out.cache_hit is False
    assert out.response.answer == "hello"
    assert out.condensed_question == "câu hỏi"     # no history → unchanged
    pipeline.query.assert_called_once_with("câu hỏi")
    cache.set.assert_called_once()
    store.append_turn.assert_called_once_with("s1", "câu hỏi", "hello")


def test_cache_hit_skips_pipeline():
    store = MagicMock(); store.get_history.return_value = []
    cache = MagicMock(); cache.get.return_value = _resp("cached")
    pipeline = MagicMock()

    out = run_chat_turn(pipeline, store, cache, MagicMock(), "col", "s1", "q")

    assert out.cache_hit is True
    assert out.response.answer == "cached"
    pipeline.query.assert_not_called()
    cache.set.assert_not_called()
    store.append_turn.assert_called_once_with("s1", "q", "cached")


def test_condenses_with_history_before_pipeline():
    store = MagicMock()
    store.get_history.return_value = [{"question": "Tội cướp tài sản?", "answer": "..."}]
    llm = MagicMock(); llm.generate.return_value = "Tội cướp giật tài sản bị phạt thế nào?"
    cache = MagicMock(); cache.get.return_value = None
    pipeline = MagicMock(); pipeline.query.return_value = _resp("x")

    out = run_chat_turn(pipeline, store, cache, llm, "col", "s1", "vậy còn cướp giật?")

    assert out.condensed_question == "Tội cướp giật tài sản bị phạt thế nào?"
    pipeline.query.assert_called_once_with("Tội cướp giật tài sản bị phạt thế nào?")
    store.append_turn.assert_called_once_with("s1", "vậy còn cướp giật?", "x")


def test_no_cache_object_is_ok():
    store = MagicMock(); store.get_history.return_value = []
    pipeline = MagicMock(); pipeline.query.return_value = _resp("hi")

    out = run_chat_turn(pipeline, store, None, MagicMock(), "col", "s1", "q")

    assert out.cache_hit is False
    assert out.response.answer == "hi"
