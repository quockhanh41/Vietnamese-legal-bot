from unittest.mock import MagicMock

from scripts.eval import run_multiturn


class _Hit:
    def __init__(self, doc_id, article):
        self.document_id = doc_id
        self.article = article
        self.chunk_text = ""
        self.document_title = ""
        self.score = 1.0


def test_run_multiturn_scores_condensed_followups(monkeypatch):
    import scripts.eval as ev

    # Retrieval returns the expected article for the condensed follow-up.
    monkeypatch.setattr(ev, "retrieve_routed",
                        lambda q, vs, emb, cl, fill_top_k=20: ([_Hit(1, "Điều 171")], "hybrid", None))
    # Condensation resolves the pronoun follow-up.
    llm = MagicMock(); llm.generate.return_value = "Tội cướp giật tài sản bị xử phạt thế nào?"

    scenarios = [{
        "id": "mt-01", "legal_sector": "hinh_su",
        "turns": [
            {"question": "Tội cướp tài sản?", "expected_articles": [{"doc_id": 1, "article": "Điều 168"}]},
            {"question": "vậy còn cướp giật?", "expected_articles": [{"doc_id": 1, "article": "Điều 171"}]},
        ],
    }]

    result = run_multiturn(scenarios, MagicMock(), MagicMock(), llm)

    assert result["n_followups"] == 1
    assert result["followup_pass_rate"] == 1.0
    assert result["rows"][0]["turn_index"] == 1
