"""Shared quota-exhaustion signal + user-facing message for graceful
degradation (M5). Single source of truth so src/api/main.py (FastAPI path)
and src/ui/app_gradio.py (Gradio+ZeroGPU path) show identical behavior:
degraded replies are never cached/persisted (same rule as refusals)."""

DEGRADED_MSG = ("Xin lỗi, hệ thống đang quá tải hạn mức miễn phí. "
                "Vui lòng thử lại sau ít phút.")


def is_exhaustion(e: Exception) -> bool:
    """True for the LLM gateway's 'all providers exhausted' signal
    (src/llm/gateway.py raises RuntimeError("All models failed...") when
    every provider in the fallback chain is rate-limited/out of budget)."""
    return isinstance(e, RuntimeError) and str(e).startswith("All models failed")


INTERRUPTED_MSG = ("\n\n[Lỗi] Kết nối bị gián đoạn giữa chừng. "
                   "Vui lòng thử lại.")


class StreamInterrupted(RuntimeError):
    """Streaming failed AFTER tokens already reached the client.

    Distinct from exhaustion: the turn cannot be retried (retrying would
    duplicate output) and the partial answer must never be cached or written to
    session history. Subclasses RuntimeError but does NOT use the "All models
    failed" prefix, so is_exhaustion() correctly returns False for it.
    """
