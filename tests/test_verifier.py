"""
Unit tests for src/rag/verifier.py claim splitting — pure string logic, no
NLI model load, no network.

Regression test for the 2026-07-07 false-refusal diagnosis
(docs/notes/crim-false-refusal-diagnosis-2026-07-07.md): every answer this
pipeline generates ends with a "### 💡 Câu hỏi liên quan" section (mandated
by SYSTEM_PROMPT) containing 3 suggested follow-up QUESTIONS. Before the fix,
_split_claims() included that header line and the follow-up questions as
"claims" to be NLI-verified against the retrieved context — but questions are
interrogative, not factual assertions, so they can never sensibly be
"entailed" by a premise. This systematically deflated entailment_ratio (a
confidence signal) for every single question, not just the 4 false refusals.

Run: .venv/bin/python -m pytest tests/test_verifier.py -q
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.verifier import HallucinationVerifier

ANSWER_WITH_RELATED_QUESTIONS = """### 💬 Trả lời ngắn
**Đồng phạm** là những người tham gia thực hiện tội phạm với vai trò khác nhau, được quy định tại **Điều 17 Bộ luật Hình sự 2015**.

### 📋 Chi tiết
- **Người tổ chức**: Chủ mưu, cầm đầu, chỉ huy việc thực hiện tội phạm (**Điều 17 BLHS 2015**).
- **Người thực hành**: Trực tiếp thực hiện hành vi phạm tội (**Điều 17 BLHS 2015**).

### 💡 Câu hỏi liên quan
- Trách nhiệm hình sự của đồng phạm được quy định như thế nào?
- Có phân biệt mức hình phạt giữa các loại đồng phạm không?
- Trường hợp người dưới 16 tuổi tham gia đồng phạm có bị xử lý không?"""


def test_split_claims_excludes_related_questions_section():
    verifier = HallucinationVerifier()
    claims = verifier._split_claims(ANSWER_WITH_RELATED_QUESTIONS)

    # None of the follow-up questions (nor the section header) should survive.
    joined = " ".join(claims)
    assert "Câu hỏi liên quan" not in joined
    assert "Trách nhiệm hình sự của đồng phạm" not in joined
    assert "Có phân biệt mức hình phạt" not in joined
    assert "Trường hợp người dưới 16 tuổi" not in joined

    # The real, substantive claims from "Trả lời ngắn" / "Chi tiết" must
    # still be present — the fix must not eat legitimate content.
    assert any("Đồng phạm" in c and "Điều 17" in c for c in claims)
    assert any("Người tổ chức" in c for c in claims)
    assert any("Người thực hành" in c for c in claims)


def test_split_claims_no_related_questions_section_unaffected():
    """Answers without the trailer section behave exactly as before."""
    verifier = HallucinationVerifier()
    answer = """### 💬 Trả lời ngắn
Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm.

### 📋 Chi tiết
- Người tổ chức là người chủ mưu, cầm đầu việc thực hiện tội phạm."""
    claims = verifier._split_claims(answer)
    assert len(claims) == 2
    assert "Đồng phạm là trường hợp" in claims[0]
    assert "Người tổ chức" in claims[1]


def test_split_claims_phrase_mid_sentence_does_not_truncate():
    """"câu hỏi liên quan đến X" occurs in natural Vietnamese mid-sentence —
    only a standalone header line may cut the claim list, never a real claim
    that happens to contain the phrase."""
    verifier = HallucinationVerifier()
    answer = """### 📋 Chi tiết
- Đối với các câu hỏi liên quan đến thủ tục tố tụng, Bộ luật Tố tụng Hình sự 2015 là văn bản điều chỉnh.
- Người tổ chức là người chủ mưu, cầm đầu việc thực hiện tội phạm (Điều 17 BLHS 2015).

### 💡 Câu hỏi liên quan
- Trách nhiệm hình sự của đồng phạm được quy định như thế nào?"""
    claims = verifier._split_claims(answer)
    assert len(claims) == 2
    assert any("thủ tục tố tụng" in c for c in claims)
    assert any("Người tổ chức" in c for c in claims)
    assert not any("Trách nhiệm hình sự của đồng phạm" in c for c in claims)


def test_split_claims_empty_text():
    verifier = HallucinationVerifier()
    assert verifier._split_claims("") == []
    assert verifier._split_claims(None) == []
