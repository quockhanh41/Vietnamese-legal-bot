from unittest.mock import MagicMock

from src.rag.condense import condense_query


def test_empty_history_returns_original_and_skips_llm():
    llm = MagicMock()
    q = "Tội cướp tài sản bị phạt thế nào?"
    assert condense_query([], q, llm) == q
    llm.generate.assert_not_called()


def test_rewrites_followup_using_history():
    llm = MagicMock()
    llm.generate.return_value = "  Tội cướp giật tài sản bị xử phạt thế nào?  "
    history = [{"question": "Tội cướp tài sản bị phạt thế nào?", "answer": "..."}]
    out = condense_query(history, "vậy còn cướp giật thì sao?", llm)
    assert out == "Tội cướp giật tài sản bị xử phạt thế nào?"
    llm.generate.assert_called_once()


def test_failopen_on_empty_output():
    llm = MagicMock()
    llm.generate.return_value = "   "
    out = condense_query([{"question": "q1", "answer": "a1"}], "vậy còn X?", llm)
    assert out == "vậy còn X?"


def test_failopen_on_exception():
    llm = MagicMock()
    llm.generate.side_effect = RuntimeError("llm down")
    out = condense_query([{"question": "q1", "answer": "a1"}], "vậy còn X?", llm)
    assert out == "vậy còn X?"
