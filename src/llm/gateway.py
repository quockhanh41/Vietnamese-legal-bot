"""
LiteLLM Gateway wrapper for unified LLM access with auto-fallback.

Strategy:
- Primary: Qwen 2.5 72B (Groq) — fast, free, good Vietnamese
- Fallback: GPT-4o mini → GPT-4o (OpenAI)
"""

import os
from typing import Optional, Generator, Tuple, Union, List
import re
import litellm

# Configure LiteLLM
litellm.drop_params = True
litellm.telemetry = False


class LLMGateway:
    """Unified LLM access through LiteLLM with automatic fallback."""

    # Model fallback chain: Groq primary + fallback, then cross-provider $0
    # fallbacks. CORRECTED 2026-07-19: every key in GROQ_API_KEYS/GEMINI_API_KEY/
    # OPENROUTER_API_KEY belongs to a SEPARATE account — rotation multiplies
    # capacity for all three providers, not just Gemini/OpenRouter (supersedes
    # the M1.7 note below __init__, which assumed shared-org Groq keys). Order:
    # same-day Groq capacity first, then Gemini free (biggest external quota),
    # then OpenRouter free (~50 req/day account-wide across ALL :free models
    # per key — a mop-up tail, not a workhorse, even with multiple keys).
    MODEL_CHAIN = [
        "groq/llama-3.3-70b-versatile",  # Primary: Llama 70B, excellent Vietnamese
                                         # (qwen/qwen3-32b was retired by Groq —
                                         # 404 as of 2026-07-19, promoted this
                                         # from Fallback 1)
        "groq/openai/gpt-oss-120b",       # Fallback 1: larger/stronger reasoning,
                                         # live-verified 2026-07-19 (own TPD bucket)
        "gemini/gemini-3.5-flash",       # Fallback 2: Google AI Studio free tier
                                         # (2.5-flash is closed to new projects
                                         # — 404 on every key created after the
                                         # cutoff; 3.5-flash verified on all
                                         # keys 2026-07-12. NOTE: only flash is
                                         # free — every gemini *-pro model has a
                                         # hard 0 free-tier quota, confirmed
                                         # 2026-07-19)
        "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",   # same family as primary
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",  # same weights as Fallback 1
        "openai/gpt-4o-mini",            # last resort (only if OPENAI_API_KEY is set)
    ]

    def __init__(self):
        # Per-provider key rotation: *_API_KEYS is a comma-separated list
        # rotated on rate-limit errors; falls back to the singular *_API_KEY.
        # CONFIRMED 2026-07-19: every key across all three providers is
        # provisioned under a separate account — rotation adds real capacity
        # for Groq too (Groq TPD is per-organization, but each key here is its
        # own org), not just Gemini/OpenRouter. Supersedes the older M1.7
        # assumption that Groq keys shared one org's bucket.
        self.groq_keys = self._env_keys("GROQ_API_KEYS", "GROQ_API_KEY")
        self.gemini_keys = self._env_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
        self.openrouter_keys = self._env_keys("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")

    @staticmethod
    def _env_keys(plural_var: str, singular_var: str) -> list:
        """Comma-separated key list from plural_var, else [singular_var]."""
        raw = os.getenv(plural_var, "") or os.getenv(singular_var, "")
        return [k.strip() for k in raw.split(",") if k.strip()]

    @staticmethod
    def _provider(model: str) -> str:
        """LiteLLM provider prefix — 'openrouter/openai/gpt-oss:free' is an
        openrouter model, so route on the prefix, never on substrings."""
        return model.split("/", 1)[0]

    def _is_thinking_model(self, model: str) -> bool:
        """Models that think by default and need reasoning_effort controlled on
        small token budgets: Groq Qwen3 emits <think> blocks; Gemini
        2.5+/3.x burns the whole budget thinking and returns EMPTY content
        (verified live on gemini-3.5-flash, max_tokens=100 → finish=length,
        content=''); Groq gpt-oss burns the budget the same way (verified
        live, max_tokens=150 → content='' with no reasoning_effort forwarded).
        NOT the OpenRouter Qwen3 in the chain — it's an instruct (non-thinking)
        variant that would 400 on the param."""
        provider = self._provider(model)
        if provider == "groq":
            return "qwen3" in model or "gpt-oss" in model
        return provider == "gemini"

    @staticmethod
    def _normalize_reasoning_effort(model: str, effort: str) -> str:
        """Groq gpt-oss rejects reasoning_effort='none' (only low/medium/high
        are valid) — map the "disable thinking" request to its weakest tier."""
        if "gpt-oss" in model and effort == "none":
            return "low"
        return effort

    def _usable_models(self, models_to_try: list) -> list:
        """Skip models whose provider has no API key configured."""
        return [m for m in models_to_try if any(self._keys_for(m))]

    def _keys_for(self, model: str) -> list:
        """API keys to try for a model — the provider's full rotation list."""
        provider = self._provider(model)
        if provider == "openai":
            return [self.openai_key]
        if provider == "gemini":
            return self.gemini_keys
        if provider == "openrouter":
            return self.openrouter_keys
        return self.groq_keys or [""]

    @staticmethod
    def _is_rate_limit(err: Exception) -> bool:
        """Rate-limit / quota errors are worth retrying on ANOTHER KEY;
        other errors mean the model itself failed — move to the next model."""
        if isinstance(err, litellm.RateLimitError):
            return True
        msg = str(err).lower()
        return "rate limit" in msg or "rate_limit" in msg or "429" in msg or "quota" in msg

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """
        Generate a response.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            model: Override model (default: uses MODEL_CHAIN)
            temperature: Generation temperature
            max_tokens: Max output tokens
            json_mode: Request JSON output format
            reasoning_effort: If set, forwarded as a provider-specific reasoning
                control (e.g. "none" hides Groq Qwen3 <think> blocks).

        Returns:
            Generated text
        """
        text, _ = self._generate_impl(
            prompt, system_prompt, model, temperature, max_tokens, json_mode, reasoning_effort,
        )
        return text

    def generate_with_model(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Same as `generate()`, but also returns which model in the chain
        actually produced the answer — e.g. so a caller can tell
        `ConfidenceScorer` which model to EXCLUDE from LLM self-eval, avoiding
        a model grading its own generation.

        Returns:
            (generated_text, model_name)
        """
        return self._generate_impl(
            prompt, system_prompt, model, temperature, max_tokens, json_mode, reasoning_effort,
        )

    def _generate_impl(
        self,
        prompt: str,
        system_prompt: Optional[str],
        model: Optional[Union[str, List[str]]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        reasoning_effort: Optional[str],
    ) -> Tuple[str, str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build fallback chain: user-specified model first, then MODEL_CHAIN
        if isinstance(model, list):
            models_to_try = list(model)
        elif model:
            models_to_try = [model] + [m for m in self.MODEL_CHAIN if m != model]
        else:
            models_to_try = list(self.MODEL_CHAIN)
        models_to_try = self._usable_models(models_to_try)

        if json_mode:
            extra_kwargs = {"response_format": {"type": "json_object"}}
        else:
            extra_kwargs = {}

        # Try each model in chain; within a model, rotate API keys on rate-limit
        last_error = None
        for attempt_model in models_to_try:
            model_kwargs = dict(extra_kwargs)
            if reasoning_effort is not None and self._is_thinking_model(attempt_model):
                # reasoning_effort="none" disables Groq Qwen3 thinking entirely.
                # (reasoning_format="hidden" only hides the <think> block — the
                # model still burns the token budget thinking → empty output.)
                model_kwargs["reasoning_effort"] = self._normalize_reasoning_effort(attempt_model, reasoning_effort)

            for api_key in self._keys_for(attempt_model):
                try:
                    response = litellm.completion(
                        model=attempt_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        **model_kwargs,
                    )
                    return self._clean_thinking(response.choices[0].message.content), attempt_model
                except Exception as e:
                    last_error = e
                    if self._is_rate_limit(e):
                        continue  # rotate to the next key for the same model
                    break  # non-rate-limit failure — try the next model

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _clean_thinking(self, text: str) -> str:
        """
        Strip Qwen 3 `<think>...</think>` tags from output.
        Qwen 3 models embed their reasoning in these tags — clean before returning.
        """
        if text is None:
            # Thinking models (Gemini, Qwen3) return content=None when they burn
            # the whole max_tokens budget on internal reasoning before answering.
            return ""
        # Remove complete <think>...</think> blocks
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        # Remove trailing incomplete <think> tag (edge case)
        text = re.sub(r'<think>[\s\S]*$', '', text, flags=re.IGNORECASE)
        return text.strip()

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[Union[str, List[str]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = None,
        used_model: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token.

        Args:
            reasoning_effort: If set, forwarded as a provider-specific reasoning
                control (e.g. "none" hides Groq Qwen3 <think> blocks).
            used_model: If a dict is passed, populated with {"model": name} once
                a chunk successfully streams — lets the caller learn which model
                in the fallback chain actually produced the answer (e.g. to
                exclude it from `ConfidenceScorer` self-eval). A plain out-param
                dict rather than a return value, since generators can't return
                a value AND be driven by a simple `for` loop. Fresh dict per
                call — safe with the single shared `LLMGateway` instance used
                across concurrent API requests (no state stored on `self`).

        Yields:
            Text tokens one at a time

        Raises:
            RuntimeError: every model in the chain failed ("All models failed…"),
                recognised by src.rag.degraded.is_exhaustion.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build fallback chain: user-specified model first, then MODEL_CHAIN
        if isinstance(model, list):
            models_to_try = list(model)
        elif model:
            models_to_try = [model] + [m for m in self.MODEL_CHAIN if m != model]
        else:
            models_to_try = list(self.MODEL_CHAIN)
        models_to_try = self._usable_models(models_to_try)

        # Try each model; within a model, rotate API keys on rate-limit
        last_error = None
        for attempt_model in models_to_try:
            model_kwargs = {}
            if reasoning_effort is not None and self._is_thinking_model(attempt_model):
                model_kwargs["reasoning_effort"] = self._normalize_reasoning_effort(attempt_model, reasoning_effort)

            for api_key in self._keys_for(attempt_model):
                started = False
                try:
                    response = litellm.completion(
                        model=attempt_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=api_key,
                        stream=True,
                        **model_kwargs,
                    )
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            started = True
                            if used_model is not None:
                                used_model["model"] = attempt_model
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    last_error = e
                    if started:
                        # Mid-stream failure: tokens already reached the client,
                        # so retrying would duplicate output. Task 3 gives this
                        # its own exception type.
                        raise
                    if self._is_rate_limit(e):
                        continue  # rotate to the next key for the same model
                    break  # non-rate-limit failure — try the next model

        # Signal exhaustion the SAME way _generate_impl does: src/rag/degraded.py
        # matches on this message prefix, and both API front-ends already have an
        # is_exhaustion branch waiting for it.
        raise RuntimeError(f"All models failed. Last error: {last_error}")
