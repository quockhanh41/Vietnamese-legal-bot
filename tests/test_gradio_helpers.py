from datetime import date

import pandas as pd
import pytest

from src.ui.gradio_helpers import (
    document_choices,
    format_citations_markdown,
    format_effective_laws,
    format_system_info,
    graph_type_counts,
    history_to_turns,
    is_quota_exceeded,
    parse_effective_date,
    rename_document_columns,
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


def test_document_choices_returns_label_value_tuples():
    documents = [
        {"document_id": 101, "title": "Luật Đất đai 2024"},
        {"document_id": 205, "title": "Bộ luật Hình sự 2015"},
    ]
    assert document_choices(documents) == [
        ("Luật Đất đai 2024", 101),
        ("Bộ luật Hình sự 2015", 205),
    ]


def test_document_choices_empty():
    assert document_choices([]) == []


def test_rename_document_columns():
    df = pd.DataFrame([{"title": "Luật X", "legal_type": "Luật", "issuance_date": "2024-01-01"}])
    renamed = rename_document_columns(df)
    assert list(renamed.columns) == ["Tên văn bản", "Loại văn bản", "Ngày ban hành"]


def test_format_system_info_includes_all_fields():
    info = format_system_info("groq/llama-3.3-70b-versatile", "luat_law_ctx_v4_aiteamvn", 60958)
    assert "groq/llama-3.3-70b-versatile" in info
    assert "AITeamVN/Vietnamese_Embedding" in info
    assert "AITeamVN/Vietnamese_Reranker" in info
    assert "luat_law_ctx_v4_aiteamvn" in info
    assert "60,958" in info


def test_parse_effective_date_empty_string_returns_none():
    assert parse_effective_date("") is None


def test_parse_effective_date_valid_string():
    assert parse_effective_date("2026-07-23") == date(2026, 7, 23)


def test_parse_effective_date_malformed_raises_value_error():
    with pytest.raises(ValueError):
        parse_effective_date("not-a-date")


def test_format_effective_laws_builds_dataframe():
    laws = [{"law_name": "Luật Đất đai 2024", "effective_date": "2025-01-01"}]
    df = format_effective_laws(laws)
    assert list(df.columns) == ["Luật", "Hiệu lực từ"]
    assert df.iloc[0]["Luật"] == "Luật Đất đai 2024"


def test_format_effective_laws_empty():
    df = format_effective_laws([])
    assert len(df) == 0


def test_graph_type_counts_tallies_by_type():
    nodes = [{"node_type": "article"}, {"node_type": "article"}, {"node_type": "law"}]
    edges = [{"ref_type": "amends"}, {"ref_type": "refers_to"}]
    counts = graph_type_counts(nodes, edges)
    assert counts["nodes"] == 3
    assert counts["edges"] == 2
    assert counts["node_types"] == {"article": 2, "law": 1}
    assert counts["edge_types"] == {"amends": 1, "refers_to": 1}


def test_graph_type_counts_empty():
    counts = graph_type_counts([], [])
    assert counts == {"nodes": 0, "edges": 0, "node_types": {}, "edge_types": {}}
