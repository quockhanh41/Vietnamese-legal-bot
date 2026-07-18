"""Env-gated Langfuse tracing (M4.3).

Single import point for observability. When LANGFUSE keys are present and the
official kill-switch (LANGFUSE_TRACING_ENABLED=false) is not set, re-exports
the real langfuse @observe decorator and registers LiteLLM's langfuse_otel
callback (OTel-based integration recommended for SDK v3 — generations
auto-nest under the active @observe span via the shared OpenTelemetry
context, including failed 429/fallback attempts). Otherwise @observe is a
zero-overhead identity decorator and langfuse is never imported.

Related env vars (documented, not read here): LANGFUSE_HOST,
LANGFUSE_TRACING_ENVIRONMENT (tagging only — never a kill-switch).
"""

import os


def _tracing_enabled() -> bool:
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower() == "false":
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


TRACING_ENABLED = _tracing_enabled()

if TRACING_ENABLED:
    import litellm
    from langfuse import observe  # noqa: F401  (re-export)

    if "langfuse_otel" not in litellm.callbacks:
        litellm.callbacks.append("langfuse_otel")
else:
    def observe(*args, **kwargs):
        """Identity no-op decorator: supports @observe and @observe(...)."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _identity(fn):
            return fn

        return _identity
