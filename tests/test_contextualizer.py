"""
Unit tests for src/data/contextualizer.py — pure-Python, no I/O, no models.

Run: .venv/bin/python -m pytest tests/test_contextualizer.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.contextualizer import build_contextual_text, build_contextual_texts


def test_full_metadata_builds_expected_header():
    chunk = {
        "chunk_text": "2. Người bị kết án phạt tiền đã tích cực chấp hành...",
        "document_title": "Bộ Luật Hình sự 1999",
        "document_number": "15/1999/QH10",
        "legal_type": "Luật",
        "chapter": "Chương V",
        "article": "Điều 58. Giảm mức hình phạt đã tuyên",
        "clause": "2. Người bị kết án phạt tiền đã tích cực chấp hành...",
    }
    result = build_contextual_text(chunk)
    expected_header = (
        "Bộ Luật Hình sự 1999 (Luật 15/1999/QH10) | Chương V | "
        "Điều 58. Giảm mức hình phạt đã tuyên | Khoản 2"
    )
    assert result == expected_header + "\n" + chunk["chunk_text"]


def test_missing_chapter_and_clause_are_skipped():
    chunk = {
        "chunk_text": "Người bị kết án cải tạo không giam giữ...",
        "document_title": "Bộ Luật Hình sự 1999",
        "document_number": "15/1999/QH10",
        "legal_type": "Luật",
        "chapter": None,
        "article": "Điều 58. Giảm mức hình phạt đã tuyên",
        "clause": None,
    }
    result = build_contextual_text(chunk)
    expected_header = (
        "Bộ Luật Hình sự 1999 (Luật 15/1999/QH10) | Điều 58. Giảm mức hình phạt đã tuyên"
    )
    assert result == expected_header + "\n" + chunk["chunk_text"]
    assert "Chương" not in result.split("\n")[0]
    assert "Khoản" not in result.split("\n")[0]


def test_anti_duplication_when_chunk_already_opens_with_article_header():
    chunk_text = (
        "Điều 173. Tội trộm cắp tài sản\n\n"
        "1. Người nào trộm cắp tài sản của người khác trị giá từ 2.000.000 đồng..."
    )
    chunk = {
        "chunk_text": chunk_text,
        "document_title": "Bộ luật hình sự 2015",
        "document_number": "100/2015/QH13",
        "legal_type": "Luật",
        "chapter": None,
        "article": "Điều 173. Tội trộm cắp tài sản",
        "clause": None,
    }
    result = build_contextual_text(chunk)
    header_line = result.split("\n", 1)[0]
    # Article segment must NOT appear in the header — it's already at the
    # top of chunk_text, so embedding it in the header too would duplicate it.
    assert "Điều 173" not in header_line
    assert header_line == "Bộ luật hình sự 2015 (Luật 100/2015/QH13)"
    # Body (with its own article header) is untouched.
    assert result == header_line + "\n" + chunk_text


def test_no_metadata_returns_chunk_text_unchanged():
    chunk = {"chunk_text": "Một đoạn văn bản không có metadata gì cả."}
    assert build_contextual_text(chunk) == chunk["chunk_text"]


def test_no_metadata_with_explicit_none_fields_returns_chunk_text_unchanged():
    chunk = {
        "chunk_text": "Một đoạn văn bản không có metadata gì cả.",
        "document_title": None,
        "document_number": None,
        "legal_type": None,
        "chapter": None,
        "article": None,
        "clause": None,
    }
    assert build_contextual_text(chunk) == chunk["chunk_text"]


def test_build_contextual_texts_batch_matches_single_calls():
    chunks = [
        {
            "chunk_text": "Body one",
            "document_title": "Luật A",
            "legal_type": "Luật",
            "document_number": "1/2020/QH14",
            "article": "Điều 1. Phạm vi điều chỉnh",
        },
        {"chunk_text": "Body two, no metadata"},
    ]
    results = build_contextual_texts(chunks)
    assert results == [build_contextual_text(c) for c in chunks]
    assert results[1] == "Body two, no metadata"
