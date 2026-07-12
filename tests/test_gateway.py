"""Tests for LLMGateway provider/key selection — no network calls.

M1.7 follow-up: Groq TPD is per-organization, so key rotation inside one
org shares a single daily bucket. Cross-provider fallback (Gemini free +
OpenRouter free) is the $0 fix. These tests pin the provider routing so a
missing key skips the model instead of sending an empty api_key.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.llm.gateway import LLMGateway


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gk1,gk2")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "gem1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or1")
    # Hermetic against a real .env leaked into os.environ by load_dotenv()
    # in other test modules.
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    return LLMGateway()


class TestKeysFor:
    def test_groq_models_rotate_all_groq_keys(self, gateway):
        assert gateway._keys_for("groq/qwen/qwen3-32b") == ["gk1", "gk2"]

    def test_gemini_model_uses_gemini_key(self, gateway):
        assert gateway._keys_for("gemini/gemini-2.5-flash") == ["gem1"]

    def test_openrouter_model_uses_openrouter_key(self, gateway):
        assert gateway._keys_for(
            "openrouter/qwen/qwen3-next-80b-a3b-instruct:free") == ["or1"]

    def test_openai_model_uses_openai_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEYS", "gk1")
        monkeypatch.setenv("OPENAI_API_KEY", "oa1")
        gw = LLMGateway()
        assert gw._keys_for("openai/gpt-4o-mini") == ["oa1"]


class TestMultiKeyRotation:
    """GEMINI_API_KEYS / OPENROUTER_API_KEYS mirror GROQ_API_KEYS: a
    comma-separated list rotated on rate-limit; plural wins over singular."""

    def test_gemini_plural_keys_rotate(self, gateway, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEYS", "gem1,gem2,gem3")
        gw = LLMGateway()
        assert gw._keys_for("gemini/gemini-2.5-flash") == ["gem1", "gem2", "gem3"]

    def test_openrouter_plural_keys_rotate(self, gateway, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEYS", "or1,or2")
        gw = LLMGateway()
        assert gw._keys_for(
            "openrouter/meta-llama/llama-3.3-70b-instruct:free") == ["or1", "or2"]

    def test_plural_takes_precedence_over_singular(self, gateway, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gem-single")
        monkeypatch.setenv("GEMINI_API_KEYS", "gem-a,gem-b")
        gw = LLMGateway()
        assert gw._keys_for("gemini/gemini-2.5-flash") == ["gem-a", "gem-b"]

    def test_singular_still_works_when_plural_unset(self, gateway):
        assert gateway._keys_for("gemini/gemini-2.5-flash") == ["gem1"]
        assert gateway._keys_for(
            "openrouter/qwen/qwen3-next-80b-a3b-instruct:free") == ["or1"]

    def test_whitespace_and_empty_entries_are_dropped(self, gateway, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEYS", " or1 , ,or2,")
        gw = LLMGateway()
        assert gw._keys_for(
            "openrouter/meta-llama/llama-3.3-70b-instruct:free") == ["or1", "or2"]


class TestUsableModels:
    def test_keyless_providers_are_skipped(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEYS", "gk1")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        gw = LLMGateway()
        chain = [
            "groq/qwen/qwen3-32b",
            "gemini/gemini-2.5-flash",
            "openrouter/meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-4o-mini",
        ]
        assert gw._usable_models(chain) == ["groq/qwen/qwen3-32b"]

    def test_all_providers_keyed_keeps_full_chain(self, gateway):
        chain = list(LLMGateway.MODEL_CHAIN)
        assert gateway._usable_models(chain) == [
            m for m in chain if not m.startswith("openai/")
        ]

    def test_openrouter_model_names_do_not_false_match_other_providers(self, gateway):
        # "openrouter/openai/gpt-oss-120b:free" is an OPENROUTER model — the
        # substring "openai" must not route it to the (empty) OpenAI key.
        chain = ["openrouter/openai/gpt-oss-120b:free"]
        assert gateway._usable_models(chain) == chain
        assert gateway._keys_for("openrouter/openai/gpt-oss-120b:free") == ["or1"]


class TestThinkingModels:
    def test_groq_qwen3_is_thinking(self, gateway):
        assert gateway._is_thinking_model("groq/qwen/qwen3-32b") is True

    def test_gemini_is_thinking(self, gateway):
        # gemini-3.5-flash thinks by default and returns EMPTY content on
        # small max_tokens budgets (classifier/self-eval use 150) unless
        # reasoning_effort="none" is sent — verified live 2026-07-12.
        assert gateway._is_thinking_model("gemini/gemini-3.5-flash") is True

    def test_openrouter_qwen3_instruct_is_not_thinking(self, gateway):
        # Instruct variant has no thinking mode; sending reasoning_effort
        # to OpenRouter for it would 400 every fallback request.
        assert gateway._is_thinking_model(
            "openrouter/qwen/qwen3-next-80b-a3b-instruct:free") is False
