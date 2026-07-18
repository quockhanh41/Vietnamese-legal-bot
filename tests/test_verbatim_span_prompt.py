"""Prompt contract test for M3.6 verbatim spans. Pure string assertions."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.prompts import SYSTEM_PROMPT


def test_system_prompt_requires_verbatim_blockquote_span():
    assert "nguyên văn" in SYSTEM_PROMPT
    # A blockquote example is present in the format template.
    assert "\n  > " in SYSTEM_PROMPT
    # The omit-if-not-verbatim guardrail must be spelled out.
    assert "BỎ dòng trích" in SYSTEM_PROMPT
