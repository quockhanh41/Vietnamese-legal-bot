"""
LiteLLM Gateway wrapper for unified LLM access with auto-fallback.

Strategy:
- Primary: Qwen 2.5 72B (Groq) — fast, free, good Vietnamese
- Fallback: GPT-4o mini → GPT-4o (OpenAI)
"""

import os
from typing import Optional, Generator
import re
import litellm

# Configure LiteLLM
litellm.drop_params = True
litellm.telemetry = False


class LLMGateway:
    """Unified LLM access through LiteLLM with automatic fallback."""

    # Model fallback chain: Groq primary + fallback, then cross-provider $0
    # fallbacks (M1.7: Groq TPD is per-ORGANIZATION — key rotation inside one
    # org shares a single 100K/day bucket, so a second provider is the only
    # real headroom). Order: same-day Groq capacity first, then Gemini free
    # (biggest external quota), then OpenRouter free (~50 req/day account-wide
    # across ALL :free models — a mop-up tail, not a workhorse).
    MODEL_CHAIN = [
        "groq/qwen/qwen3-32b",           # Primary: Qwen 3 32B, excellent Vietnamese
        "groq/llama-3.3-70b-versatile",  # Fallback 1: Llama 70B (own TPD bucket)
        "gemini/gemini-3.5-flash",       # Fallback 2: Google AI Studio free tier
                                         # (2.5-flash is closed to new projects
                                         # — 404 on every key created after the
                                         # cutoff; 3.5-flash verified on all
                                         # keys 2026-07-12)
        "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",   # same family as primary
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",  # same weights as Fallback 1
        "openai/gpt-4o-mini",            # last resort (only if OPENAI_API_KEY is set)
    ]

    def __init__(self):
        # Per-provider key rotation: *_API_KEYS is a comma-separated list
        # rotated on rate-limit errors; falls back to the singular *_API_KEY.
        # (NOTE 2026-07-11: Groq TPD is per-ORG, not per-key — rotation only
        # helps if the keys belong to different organizations. Gemini and
        # OpenRouter free quotas are per-account, so extra accounts' keys DO
        # add capacity there.)
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
        """Models that think by default and need reasoning_effort='none' on
        small token budgets: Groq Qwen3 emits <think> blocks; Gemini
        2.5+/3.x burns the whole budget thinking and returns EMPTY content
        (verified live on gemini-3.5-flash, max_tokens=100 → finish=length,
        content=''). NOT the OpenRouter Qwen3 in the chain — it's an
        instruct (non-thinking) variant that would 400 on the param."""
        provider = self._provider(model)
        if provider == "groq":
            return "qwen3" in model
        return provider == "gemini"

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
        model: Optional[str] = None,
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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build fallback chain: user-specified model first, then MODEL_CHAIN
        if model:
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
                model_kwargs["reasoning_effort"] = reasoning_effort

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
                    return self._clean_thinking(response.choices[0].message.content)
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
        # Remove complete <think>...</think> blocks
        text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
        # Remove trailing incomplete <think> tag (edge case)
        text = re.sub(r'<think>[\s\S]*$', '', text, flags=re.IGNORECASE)
        return text.strip()

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response token by token.

        Args:
            reasoning_effort: If set, forwarded as a provider-specific reasoning
                control (e.g. "none" hides Groq Qwen3 <think> blocks).

        Yields:
            Text tokens one at a time
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build fallback chain: user-specified model first, then MODEL_CHAIN
        if model:
            models_to_try = [model] + [m for m in self.MODEL_CHAIN if m != model]
        else:
            models_to_try = list(self.MODEL_CHAIN)
        models_to_try = self._usable_models(models_to_try)

        # Try each model; within a model, rotate API keys on rate-limit
        for attempt_model in models_to_try:
            model_kwargs = {}
            if reasoning_effort is not None and self._is_thinking_model(attempt_model):
                model_kwargs["reasoning_effort"] = reasoning_effort

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
                            yield chunk.choices[0].delta.content
                    return
                except Exception as e:
                    if started:
                        # Mid-stream failure: tokens already reached the client —
                        # retrying would duplicate output. Surface and stop.
                        yield "\n\n[Lỗi] Kết nối bị gián đoạn giữa chừng. Vui lòng thử lại."
                        return
                    if self._is_rate_limit(e):
                        continue  # rotate to the next key for the same model
                    break  # non-rate-limit failure — try the next model

        yield "[Lỗi] Không thể kết nối đến LLM. Vui lòng thử lại sau."
