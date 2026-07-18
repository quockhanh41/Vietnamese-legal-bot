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


# ── Tier 1.5: Grounding (`[n]` block citations) ──────────────────────────
#
# Regression coverage for the "quote-then-answer" grounding feature (roadmap
# §3.2): SYSTEM_PROMPT now asks the model to cite the numbered context block
# `[n]` at the end of each "Chi tiết" bullet. `_verify_grounding` checks that
# (a) every claim's `[n]` actually falls within the retrieved block range,
# and (b) an "Điều X" mentioned alongside `[n]` is actually present in that
# block's text. This is an observability signal only — it must never affect
# `result.passed` (which stays governed solely by contradiction).

CONTEXT_2_BLOCKS = (
    "[1] Bộ luật Hình sự 2015 — Điều 17 (Chương IV)\n"
    "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm."
    "\n\n---\n\n"
    "[2] Bộ luật Hình sự 2015 — Điều 20 (Chương IV)\n"
    "Người tổ chức, người thực hành, người xúi giục, người giúp sức đều là đồng phạm."
)


def test_grounding_well_grounded_answer():
    verifier = HallucinationVerifier()
    answer = """### 📋 Chi tiết
- Đồng phạm là trường hợp có từ hai người trở lên cùng cố ý thực hiện tội phạm (Điều 17 BLHS 2015) [1].
- Người tổ chức, người thực hành đều được coi là đồng phạm (Điều 20 BLHS 2015) [2]."""

    result = _run_grounding(verifier, answer, CONTEXT_2_BLOCKS)

    assert result.grounding_ratio == 1.0
    assert result.invalid_block_refs == []
    assert result.ungrounded_article_citations == []
    assert all(result.grounded_claims)


def test_grounding_invalid_block_ref_out_of_range():
    verifier = HallucinationVerifier()
    answer = """### 📋 Chi tiết
- Đồng phạm là trường hợp có từ hai người trở lên cùng cố ý thực hiện tội phạm (Điều 17 BLHS 2015) [1].
- Một nội dung không có thật được trích dẫn từ nguồn không tồn tại [5]."""

    result = _run_grounding(verifier, answer, CONTEXT_2_BLOCKS)

    assert 5 in result.invalid_block_refs
    # The claim citing only [5] (out of range for a 2-block context) must not
    # be counted as grounded.
    assert result.grounded_claims[1] is False
    assert result.grounded_claims[0] is True


def test_grounding_article_not_in_cited_block():
    verifier = HallucinationVerifier()
    # Cites block [1] but references Điều 99, which only exists in block [2]
    # in this fabricated context (used purely to exercise the mismatch path).
    context = (
        "[1] Bộ luật Hình sự 2015 — Điều 17 (Chương IV)\n"
        "Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm."
        "\n\n---\n\n"
        "[2] Bộ luật Hình sự 2015 — Điều 99 (Chương IV)\n"
        "Một quy định khác không liên quan đến đồng phạm."
    )
    answer = """### 📋 Chi tiết
- Đồng phạm còn được quy định tại Điều 99 Bộ luật Hình sự 2015 [1]."""

    result = _run_grounding(verifier, answer, context)

    assert "Điều 99" in result.ungrounded_article_citations


def test_grounding_no_block_refs_at_all_backward_compat():
    """Old-style answers (or the refusal template) carry no `[n]` markers at
    all — grounding must degrade gracefully, not crash or spuriously warn."""
    verifier = HallucinationVerifier()
    answer = """### 📋 Chi tiết
- Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm (Điều 17 BLHS 2015).
- Người tổ chức là người chủ mưu, cầm đầu việc thực hiện tội phạm."""

    result = _run_grounding(verifier, answer, CONTEXT_2_BLOCKS)

    assert result.grounding_ratio == 0.0
    assert result.invalid_block_refs == []
    assert result.ungrounded_article_citations == []
    assert all(g is False for g in result.grounded_claims)


def test_grounding_questions_trailer_never_counted():
    """The '### Câu hỏi liên quan' trailer must never be treated as a claim,
    grounded or otherwise — even if it (incorrectly) contained `[n]` markers."""
    verifier = HallucinationVerifier()
    answer = """### 📋 Chi tiết
- Đồng phạm là trường hợp có hai người trở lên cố ý cùng thực hiện tội phạm (Điều 17 BLHS 2015) [1].

### 💡 Câu hỏi liên quan
- Trách nhiệm hình sự của đồng phạm được quy định như thế nào? [1]
- Có phân biệt mức hình phạt giữa các loại đồng phạm không? [99]"""

    result = _run_grounding(verifier, answer, CONTEXT_2_BLOCKS)

    # Only the single real "Chi tiết" bullet should be counted as a claim.
    assert len(result.grounded_claims) == 1
    assert result.grounding_ratio == 1.0
    # The bogus [99] ref lives only in the trailer, which must be excluded —
    # so it must never surface as an invalid ref.
    assert 99 not in result.invalid_block_refs


def test_grounding_ignores_ungrounded_summary_and_label_lines():
    """`grounding_ratio` must be measured only over the "### 📋 Chi tiết"
    bullets. A "Trả lời ngắn" summary sentence with no `[n]` (the prompt
    never asks it to carry one) and a bare "**Label**:" sub-heading inside
    Chi tiết must NOT drag the ratio below 1.0 when every real bullet is
    grounded — this was the false-negative this fix addresses."""
    verifier = HallucinationVerifier()
    answer = """### 💬 Trả lời ngắn
**Đồng phạm** là những người tham gia thực hiện tội phạm với vai trò khác nhau, theo Điều 17 BLHS 2015.

### 📋 Chi tiết
**Phương thức thực hiện**:
- Đồng phạm là trường hợp có từ hai người trở lên cùng cố ý thực hiện tội phạm (Điều 17 BLHS 2015) [1].
- Người tổ chức, người thực hành đều được coi là đồng phạm (Điều 20 BLHS 2015) [2]."""

    result = _run_grounding(verifier, answer, CONTEXT_2_BLOCKS)

    assert result.grounding_ratio == 1.0
    assert len(result.grounded_claims) == 2
    assert all(result.grounded_claims)
    assert result.invalid_block_refs == []
    assert result.ungrounded_article_citations == []


def test_groundable_claims_excludes_summary_trailer_and_labels():
    """Direct unit test of `_groundable_claims` (not just the ratio it
    feeds): only the two substantive Chi-tiết bullets should survive."""
    verifier = HallucinationVerifier()
    answer = """### 💬 Trả lời ngắn
**Đồng phạm** là những người tham gia thực hiện tội phạm với vai trò khác nhau, theo Điều 17 BLHS 2015.

### 📋 Chi tiết
**Phương thức thực hiện**:
- Đồng phạm là trường hợp có từ hai người trở lên cùng cố ý thực hiện tội phạm (Điều 17 BLHS 2015) [1].
- Người tổ chức, người thực hành đều được coi là đồng phạm (Điều 20 BLHS 2015) [2].

### 💡 Câu hỏi liên quan
- Trách nhiệm hình sự của đồng phạm được quy định như thế nào?"""

    claims = verifier._groundable_claims(answer)

    assert len(claims) == 2
    assert not any("Đồng phạm là những người tham gia" in c for c in claims)  # Trả lời ngắn excluded
    assert not any("Phương thức thực hiện" in c for c in claims)  # bare label excluded
    assert not any("Trách nhiệm hình sự" in c for c in claims)  # trailer excluded
    assert any("hai người trở lên" in c for c in claims)
    assert any("Người tổ chức, người thực hành" in c for c in claims)


def test_split_claims_unchanged_guard_against_nli_drift():
    """Regression guard: `_split_claims` (Tier 2 NLI's claim set, which
    feeds entailment_ratio — a signal in the calibrated REFUSAL_THRESHOLD
    confidence pipeline) must NOT be affected by the addition of
    `_groundable_claims`. Pins down the exact claim list for a
    representative answer so any accidental change to `_split_claims`
    itself is caught here."""
    verifier = HallucinationVerifier()
    claims = verifier._split_claims(ANSWER_WITH_RELATED_QUESTIONS)

    assert claims == [
        "Đồng phạm là những người tham gia thực hiện tội phạm với vai trò khác nhau, "
        "được quy định tại Điều 17 Bộ luật Hình sự 2015.",
        "Người tổ chức: Chủ mưu, cầm đầu, chỉ huy việc thực hiện tội phạm (Điều 17 BLHS 2015).",
        "Người thực hành: Trực tiếp thực hiện hành vi phạm tội (Điều 17 BLHS 2015).",
    ]


def _run_grounding(verifier, answer, context):
    """Helper: exercise `_verify_grounding` directly (pure string logic, no
    model load) via a fresh VerificationResult, mirroring how `verify()`
    wires it in."""
    from src.rag.verifier import VerificationResult
    result = VerificationResult()
    verifier._verify_grounding(answer, context, result)
    return result


def test_normalize_for_match_collapses_and_casefolds():
    v = HallucinationVerifier()
    a = v._normalize_for_match('  “Bị PHẠT   tù từ 03 năm…”  ')
    b = v._normalize_for_match('bị phạt tù từ 03 năm')
    assert a == b


def test_normalize_for_match_keeps_diacritics():
    v = HallucinationVerifier()
    # "phạt" and "phat" must NOT normalize to the same string.
    assert v._normalize_for_match("phạt") != v._normalize_for_match("phat")


def test_parse_bullet_spans_pairs_bullet_with_following_blockquote():
    v = HallucinationVerifier()
    answer = (
        "### 📋 Chi tiết\n"
        "- Ý một (Điều 5) [1]\n"
        "  > \"span một\"\n"
        "- Ý hai (Điều 6) [2][3]\n"
        "  > \"span hai dòng một\"\n"
        "  > \"span hai dòng hai\"\n"
        "- Ý ba không có span [4]"
    )
    pairs = v._parse_bullet_spans(answer)
    assert pairs == [
        ([1], "\"span một\""),
        ([2, 3], "\"span hai dòng một\" \"span hai dòng hai\""),
        ([4], None),
    ]


def test_parse_bullet_spans_ignores_tra_loi_ngan_and_trailer():
    v = HallucinationVerifier()
    answer = (
        "### 💬 Trả lời ngắn\n"
        "Tóm tắt.\n"
        "> \"không phải span của Chi tiết\"\n"
        "### 📋 Chi tiết\n"
        "- Ý một [1]\n"
        "  > \"span một\"\n"
        "### 💡 Câu hỏi liên quan\n"
        "- Câu hỏi 1?"
    )
    pairs = v._parse_bullet_spans(answer)
    assert pairs == [([1], "\"span một\"")]
