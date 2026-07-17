"""Schema validation for the M3.5 multi-hop eval cohort
(data/eval/eval_set_multihop_v1.json). Labels are hand-verified against the
live luat_law_ctx_v4_aiteamvn collection; this test guards the schema/shape so
a malformed edit can't silently break the measurement harness.

Run: .venv/bin/python -m pytest tests/test_multihop_eval_set.py -q
"""

import json
import os
import re

PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval", "eval_set_multihop_v1.json")


def _load():
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def test_multihop_set_schema():
    data = _load()
    assert isinstance(data, list), "cohort must be a JSON list (run_multihop iterates it)"
    assert 10 <= len(data) <= 12, f"expected 10-12 questions, got {len(data)}"

    ids = set()
    for q in data:
        for field in ("id", "question", "type", "difficulty", "legal_sector",
                      "expected_behavior", "ground_truth", "expected_articles"):
            assert field in q, f"{q.get('id')} missing {field}"
        assert q["type"] == "multi-hop", f"{q['id']} type must be multi-hop"
        assert q["expected_behavior"] == "answer", f"{q['id']} must be an answer question"
        assert q["id"] not in ids, f"duplicate id {q['id']}"
        ids.add(q["id"])

        # multi-hop MUST label >= 2 articles (primary + cross-referenced)
        arts = q["expected_articles"]
        assert len(arts) >= 2, f"{q['id']} needs >=2 expected_articles"
        for a in arts:
            # matcher (src/eval/metrics.py strict_match) reads `doc_id` + `article`
            assert "doc_id" in a and isinstance(a["doc_id"], int), f"{q['id']} expected_articles need int doc_id"
            assert "article" in a and re.match(r"Điều\s+\d+", a["article"]), \
                f"{q['id']} article must look like 'Điều N', got {a.get('article')!r}"

    sectors = {q["legal_sector"] for q in data}
    assert len(sectors) >= 4, f"want >=4 sectors, got {sectors}"
