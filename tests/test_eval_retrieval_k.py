import scripts.eval as ev
from src.eval.metrics import recall_at_k


def test_retrieval_k_includes_1_and_3():
    assert ev.RETRIEVAL_K == (1, 3, 5, 10)
    assert ev.FETCH_K == 10  # still fetch top-10; @1/@3 are slices, no extra search


def test_recall_at_1_and_3_are_slices():
    # top-3 has the match at rank 3 -> recall@1 miss, recall@3 hit
    results = [
        {"document_id": 9, "article": "Điều 9", "chunk_text": ""},
        {"document_id": 8, "article": "Điều 8", "chunk_text": ""},
        {"document_id": 1, "article": "Điều 1", "chunk_text": ""},
    ]
    expected = [{"doc_id": 1, "article": "Điều 1"}]
    assert recall_at_k(results, expected, 1) == 0.0
    assert recall_at_k(results, expected, 3) == 1.0
