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

    # Model fallback chain: Groq primary + fallback, then cross-provider (OpenAI) fallback
    MODEL_CHAIN = [
        "groq/qwen/qwen3-32b",           # Primary: Qwen 3 32B, excellent Vietnamese
        "groq/llama-3.3-70b-versatile",  # Fallback 1: Llama 70B
        "openai/gpt-4o-mini",            # Fallback 2: cross-provider (only if OPENAI_API_KEY is set)
    ]

    def __init__(self):
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.openai_key = os.getenv("OPENAI_API_KEY", "")

    def _is_thinking_model(self, model: str) -> bool:
        """Groq Qwen3 models emit <think> blocks; reasoning_effort='none' disables them."""
        return "qwen3" in model

    def _usable_models(self, models_to_try: list) -> list:
        """Skip models whose provider has no API key configured."""
        usable = []
        for m in models_to_try:
            if ("openai" in m or "gpt" in m) and not self.openai_key:
                continue
            usable.append(m)
        return usable

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

        # Try each model in chain
        last_error = None
        for attempt_model in models_to_try:
            try:
                if "groq" in attempt_model:
                    api_key = self.groq_key
                elif "openai" in attempt_model or "gpt" in attempt_model:
                    api_key = self.openai_key
                else:
                    api_key = self.groq_key  # default to groq

                model_kwargs = dict(extra_kwargs)
                if reasoning_effort is not None and self._is_thinking_model(attempt_model):
                    # reasoning_effort="none" disables Groq Qwen3 thinking entirely.
                    # (reasoning_format="hidden" only hides the <think> block — the
                    # model still burns the token budget thinking → empty output.)
                    model_kwargs["reasoning_effort"] = reasoning_effort

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
                continue

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

        # Try each model
        for attempt_model in models_to_try:
            try:
                if "groq" in attempt_model:
                    api_key = self.groq_key
                elif "openai" in attempt_model or "gpt" in attempt_model:
                    api_key = self.openai_key
                else:
                    api_key = self.groq_key  # default to groq

                model_kwargs = {}
                if reasoning_effort is not None and self._is_thinking_model(attempt_model):
                    model_kwargs["reasoning_effort"] = reasoning_effort

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
                        yield chunk.choices[0].delta.content
                return
            except Exception:
                continue

        yield "[Lỗi] Không thể kết nối đến LLM. Vui lòng thử lại sau."
