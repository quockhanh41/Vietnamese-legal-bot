import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.ui.pages.chat as chat


def test_backend_state():
    assert chat.backend_state(ready_ok=True, health_ok=True) == "ready"
    assert chat.backend_state(ready_ok=False, health_ok=True) == "warming"
    assert chat.backend_state(ready_ok=False, health_ok=False) == "offline"


def test_is_degraded():
    assert chat.is_degraded({"degraded": True}) is True
    assert chat.is_degraded({"degraded": False}) is False
    assert chat.is_degraded({}) is False
