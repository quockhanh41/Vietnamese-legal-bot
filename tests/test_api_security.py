from fastapi.testclient import TestClient
import src.api.main as main


def test_question_over_2000_chars_rejected():
    client = TestClient(main.app)  # no `with` → lifespan does not run
    r = client.post("/api/chat", json={"question": "x" * 2001, "session_id": "s"})
    assert r.status_code == 422


def test_parse_origins():
    assert main._parse_origins(None) == []
    assert main._parse_origins("") == []
    assert main._parse_origins("https://a.hf.space") == ["https://a.hf.space"]
    assert main._parse_origins("https://a, https://b") == ["https://a", "https://b"]


def test_default_cors_is_not_wildcard():
    assert main.ALLOWED_ORIGINS != ["*"]


import pytest
from fastapi import HTTPException


def test_require_admin_forbidden_when_key_unset(monkeypatch):
    monkeypatch.setattr(main, "ADMIN_KEY", None)
    with pytest.raises(HTTPException) as e:
        main.require_admin(x_admin_key="anything")
    assert e.value.status_code == 403


def test_require_admin_forbidden_on_wrong_key(monkeypatch):
    monkeypatch.setattr(main, "ADMIN_KEY", "secret")
    with pytest.raises(HTTPException) as e:
        main.require_admin(x_admin_key="nope")
    assert e.value.status_code == 403


def test_require_admin_ok_on_correct_key(monkeypatch):
    monkeypatch.setattr(main, "ADMIN_KEY", "secret")
    main.require_admin(x_admin_key="secret")  # no raise


from unittest.mock import MagicMock


def _req(ip="127.0.0.1"):
    r = MagicMock()
    r.client.host = ip
    return r


def test_rate_limit_blocks_after_limit(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 2)
    main._rate_buckets.clear()
    body = main.ChatRequest(question="q", session_id="s1")
    main.rate_limit_session(_req(), body)
    main.rate_limit_session(_req(), body)
    with pytest.raises(HTTPException) as e:
        main.rate_limit_session(_req(), body)
    assert e.value.status_code == 429


def test_rate_limit_sessions_are_independent(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    main._rate_buckets.clear()
    main.rate_limit_session(_req(), main.ChatRequest(question="q", session_id="A"))
    main.rate_limit_session(_req(), main.ChatRequest(question="q", session_id="B"))  # ok


def test_rate_limit_falls_back_to_ip_without_session(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    main._rate_buckets.clear()
    main.rate_limit_session(_req("1.2.3.4"), main.ChatRequest(question="q"))
    with pytest.raises(HTTPException):
        main.rate_limit_session(_req("1.2.3.4"), main.ChatRequest(question="q"))


def test_rate_limit_sweeps_idle_buckets_past_cap(monkeypatch):
    import time
    from collections import deque
    monkeypatch.setattr(main, "RATE_LIMIT", 30)
    monkeypatch.setattr(main, "_RATE_MAX_KEYS", 3)
    main._rate_buckets.clear()
    stale = time.time() - main._RATE_WINDOW - 10
    for i in range(5):  # 5 idle sessions, all outside the window → leaked deques
        main._rate_buckets[f"idle-{i}"] = deque([stale])
    # A fresh request past the cap triggers the sweep of stale buckets.
    main.rate_limit_session(_req(), main.ChatRequest(question="q", session_id="live"))
    assert "live" in main._rate_buckets
    assert not any(k.startswith("idle-") for k in main._rate_buckets)
