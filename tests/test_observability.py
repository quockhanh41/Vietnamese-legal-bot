"""M4.3 — env-gated Langfuse tracing setup."""
import importlib

import litellm
import pytest

import src.observability as obs


@pytest.fixture
def reload_obs(monkeypatch):
    """Reload src.observability under a controlled env; restore module + litellm
    state afterwards (finalizer runs before monkeypatch teardown, so it resets
    env explicitly before the final reload)."""
    saved_callbacks = list(litellm.callbacks)

    def _reload(env: dict):
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
                    "LANGFUSE_TRACING_ENABLED"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(obs)

    yield _reload
    litellm.callbacks = saved_callbacks
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
    importlib.reload(obs)


DUMMY_KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-lf-test", "LANGFUSE_SECRET_KEY": "sk-lf-test"}


class TestDisabledMode:
    def test_disabled_without_keys(self, reload_obs):
        module = reload_obs({})
        assert module.TRACING_ENABLED is False

    def test_kill_switch_beats_keys(self, reload_obs):
        module = reload_obs({**DUMMY_KEYS, "LANGFUSE_TRACING_ENABLED": "false"})
        assert module.TRACING_ENABLED is False

    def test_observe_is_identity_both_forms(self, reload_obs):
        module = reload_obs({})

        def fn():
            return 42

        assert module.observe(fn) is fn
        assert module.observe(name="x", capture_output=False)(fn) is fn
        assert module.observe(fn).__name__ == "fn"

    def test_litellm_callbacks_untouched(self, reload_obs):
        before = list(litellm.callbacks)
        reload_obs({})
        assert list(litellm.callbacks) == before


class TestEnabledMode:
    def test_enabled_with_keys(self, reload_obs):
        module = reload_obs(DUMMY_KEYS)
        assert module.TRACING_ENABLED is True

    def test_langfuse_otel_callback_registered(self, reload_obs):
        reload_obs(DUMMY_KEYS)
        assert "langfuse_otel" in litellm.callbacks

    def test_callback_not_duplicated_on_reload(self, reload_obs):
        reload_obs(DUMMY_KEYS)
        reload_obs(DUMMY_KEYS)
        assert litellm.callbacks.count("langfuse_otel") == 1

    def test_observe_is_langfuse_decorator(self, reload_obs):
        module = reload_obs(DUMMY_KEYS)
        from langfuse import observe as langfuse_observe

        assert module.observe is langfuse_observe
