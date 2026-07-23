from src.ui.gradio_helpers import (
    format_citations_markdown,
    history_to_turns,
    is_quota_exceeded,
)


def test_history_to_turns_pairs_user_assistant():
    history = [
        {"role": "user", "content": "Tội trộm cắp bị phạt sao?"},
        {"role": "assistant", "content": "Theo Điều 173..."},
        {"role": "user", "content": "Còn tội cướp giật thì sao?"},
        {"role": "assistant", "content": "Theo Điều 171..."},
    ]
    turns = history_to_turns(history)
    assert turns == [
        {"question": "Tội trộm cắp bị phạt sao?", "answer": "Theo Điều 173..."},
        {"question": "Còn tội cướp giật thì sao?", "answer": "Theo Điều 171..."},
    ]


def test_history_to_turns_drops_trailing_unanswered_question():
    history = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2 (no answer yet)"},
    ]
    assert history_to_turns(history) == [{"question": "Q1", "answer": "A1"}]


def test_history_to_turns_empty():
    assert history_to_turns([]) == []


def test_format_citations_markdown_empty():
    assert format_citations_markdown([]) == ""


def test_format_citations_markdown_lists_article_and_title():
    citations = [
        {"block_index": 1, "document_title": "Bộ luật Hình sự 2015", "article": "Điều 173"},
        {"block_index": 2, "document_title": "Bộ luật Hình sự 2015", "article": "Điều 174"},
    ]
    out = format_citations_markdown(citations)
    assert "Căn cứ pháp lý" in out
    assert "1. Điều 173, Bộ luật Hình sự 2015" in out
    assert "2. Điều 174, Bộ luật Hình sự 2015" in out


def test_is_quota_exceeded_true_for_zerogpu_quota_text():
    assert is_quota_exceeded(Exception("ZeroGPU quota exceeded")) is True


def test_is_quota_exceeded_false_for_unrelated_errors():
    assert is_quota_exceeded(ValueError("bad input")) is False
    assert is_quota_exceeded(RuntimeError("All models failed. Last error: 429")) is False
