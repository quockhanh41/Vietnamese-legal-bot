from src.rag.degraded import DEGRADED_MSG, is_exhaustion


def test_is_exhaustion_true_for_gateway_signal():
    assert is_exhaustion(RuntimeError("All models failed. Last error: 429")) is True


def test_is_exhaustion_false_for_other_errors():
    assert is_exhaustion(ValueError("something else")) is False
    assert is_exhaustion(RuntimeError("unrelated runtime error")) is False


def test_degraded_msg_is_vietnamese_and_nonempty():
    assert "quá tải" in DEGRADED_MSG
