"""
Unit tests for scripts/download_zalo.py — loaders + count validation.
Pure: tiny in-memory fixtures, no network, no real dataset.

Run: .venv/bin/python -m pytest tests/test_zalo_data.py -q
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.download_zalo import (
    load_zalo_corpus,
    load_zalo_questions,
    validate_zalo,
    ZaloValidationError,
    corpus_rows_to_laws,
    build_questions,
)

CORPUS = [
    {"law_id": "01/2020/qh14", "articles": [
        {"article_id": "1", "title": "Điều 1", "text": "Nội dung điều 1."},
        {"article_id": "2", "title": "Điều 2", "text": "Nội dung điều 2."},
    ]},
    {"law_id": "02/2021/nd-cp", "articles": [
        {"article_id": "5", "title": "Điều 5", "text": "Nội dung điều 5."},
    ]},
]

QUESTIONS_ITEMS_WRAPPED = {"items": [
    {"question_id": "q1", "question": "Câu hỏi 1?",
     "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "2"}]},
]}

QUESTIONS_FLAT = [
    {"question_id": "q1", "question": "Câu hỏi 1?",
     "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "2"}]},
]


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(p)


def test_load_corpus(tmp_path):
    corpus = load_zalo_corpus(_write(tmp_path, "c.json", CORPUS))
    assert len(corpus) == 2
    assert corpus[0]["law_id"] == "01/2020/qh14"
    assert corpus[0]["articles"][1]["article_id"] == "2"


def test_load_questions_items_wrapped(tmp_path):
    qs = load_zalo_questions(_write(tmp_path, "q.json", QUESTIONS_ITEMS_WRAPPED))
    assert len(qs) == 1
    assert qs[0]["relevant_articles"] == [{"law_id": "01/2020/qh14", "article_id": "2"}]


def test_load_questions_flat_list(tmp_path):
    qs = load_zalo_questions(_write(tmp_path, "q.json", QUESTIONS_FLAT))
    assert len(qs) == 1


def test_validate_counts_ok():
    stats = validate_zalo(CORPUS, QUESTIONS_FLAT,
                          expected_laws=2, expected_articles=3, expected_questions=1)
    assert stats == {"laws": 2, "articles": 3, "questions": 1}


def test_validate_counts_mismatch_raises():
    with pytest.raises(ZaloValidationError) as e:
        validate_zalo(CORPUS, QUESTIONS_FLAT,
                      expected_laws=99, expected_articles=3, expected_questions=1)
    assert "99" in str(e.value) and "2" in str(e.value)  # expected vs actual in message


def test_validate_rejects_question_with_unknown_article():
    bad_q = [{"question_id": "qX", "question": "?",
              "relevant_articles": [{"law_id": "01/2020/qh14", "article_id": "404"}]}]
    with pytest.raises(ZaloValidationError):
        validate_zalo(CORPUS, bad_q,
                      expected_laws=2, expected_articles=3, expected_questions=1)


def test_corpus_rows_to_laws_groups_and_orders():
    rows = [
        {"id": "01/2020/qh14#1", "title": "Điều 1", "text": "Nội dung điều 1."},
        {"id": "02/2021/nd-cp#5", "title": "Điều 5", "text": "Nội dung điều 5."},
        {"id": "01/2020/qh14#2", "title": "Điều 2", "text": "Nội dung điều 2."},
    ]
    laws = corpus_rows_to_laws(rows)
    assert [law["law_id"] for law in laws] == ["01/2020/qh14", "02/2021/nd-cp"]
    assert [a["article_id"] for a in laws[0]["articles"]] == ["1", "2"]
    assert laws[0]["articles"][0] == {
        "article_id": "1", "title": "Điều 1", "text": "Nội dung điều 1."}
    assert laws[1]["articles"] == [
        {"article_id": "5", "title": "Điều 5", "text": "Nội dung điều 5."}]


def test_build_questions_multi_qrel_and_zero_qrel():
    query_rows = [
        {"query_id": "q1", "question": "Câu hỏi 1?"},
        {"query_id": "q2", "question": "Câu hỏi 2?"},
    ]
    qrel_rows = [
        {"corpus_id": "01/2020/qh14#1", "query_id": "q1"},
        {"corpus_id": "02/2021/nd-cp#5", "query_id": "q1"},
    ]
    questions = build_questions(query_rows, qrel_rows)
    assert [q["question_id"] for q in questions] == ["q1", "q2"]
    assert questions[0]["question"] == "Câu hỏi 1?"
    assert questions[0]["relevant_articles"] == [
        {"law_id": "01/2020/qh14", "article_id": "1"},
        {"law_id": "02/2021/nd-cp", "article_id": "5"},
    ]
    assert questions[1]["relevant_articles"] == []
