import src.data.mixed_embedder as mod
from scripts.ingest import _make_embedder


def test_make_embedder_bge(monkeypatch):
    calls = {}
    monkeypatch.setattr(mod, "BGEEmbedder", lambda **k: calls.update(k) or "BGE")
    out = _make_embedder("bge-m3", embed_model=None, model_path="/local/bge", device="cpu")
    assert out == "BGE"
    assert calls["model_name"] == "/local/bge"  # --model-path routed as the bge model


def test_make_embedder_mixed(monkeypatch):
    seen = {}

    class _Fake(mod.MixedHybridEmbedder):
        def __init__(self, **k):
            seen.update(k)

    monkeypatch.setattr(mod, "MixedHybridEmbedder", _Fake)
    out = _make_embedder("mixed", embed_model="AITeamVN/Vietnamese_Embedding",
                         model_path=None, device="cuda")
    assert isinstance(out, _Fake)
    assert seen["dense_model"] == "AITeamVN/Vietnamese_Embedding"
    assert seen["device"] == "cuda"
