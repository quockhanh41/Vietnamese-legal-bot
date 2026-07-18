"""Global test config — deterministic observability baseline (M4.3).

Runs before any test-module import. Forces the official Langfuse kill-switch
so tests never enable live tracing: `src/api/main.py` calls `load_dotenv()`
at import, which would otherwise leak the developer's real LANGFUSE keys from
`.env` into the test session. `load_dotenv()` never overrides existing env
vars, so the setting below survives it.
"""
import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
os.environ.pop("LANGFUSE_SECRET_KEY", None)
