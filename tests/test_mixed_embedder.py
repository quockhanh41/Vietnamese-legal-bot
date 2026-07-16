import numpy as np
import pytest

import src.data.mixed_embedder as mod


class _FakeST:
    """Stand-in for SentenceTransformer — no weights loaded."""
    def __init__(self, *a, **k):
        self.max_seq_length = 8192  # inherited-large config, to test the cap

    def get_sentence_embedding_dimension(self):
        return 1024

    def half(self):
        return self  # fp16 no-op for the fake

    def encode(self, texts, batch_size=32, normalize_embeddings=False):
        assert normalize_embeddings is True  # dot-product model must be normalized
        return np.ones((len(texts), 1024), dtype="float32")


class _FakeBGE:
    """Stand-in for BGEEmbedder — returns sparse only."""
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, batch_size=32, return_dense=True, return_sparse=True):
        assert return_dense is False  # mixed embedder only wants BGE's sparse
        return {"lexical_weights": [{1: 0.5} for _ in texts]}


@pytest.fixture
def emb(monkeypatch):
    monkeypatch.setattr(mod, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(mod, "BGEEmbedder", _FakeBGE)
    return mod.MixedHybridEmbedder(device="cpu")


def test_encode_returns_dense_and_sparse(emb):
    out = emb.encode(["a", "b"])
    assert out["dense_vecs"].shape == (2, 1024)
    assert out["lexical_weights"] == [{1: 0.5}, {1: 0.5}]


def test_encode_documents_returns_tuple(emb):
    dense, sparse = emb.encode_documents(["x"])
    assert dense.shape == (1, 1024)
    assert sparse == [{1: 0.5}]


def test_encode_queries_dense_only(emb):
    q = emb.encode_queries(["hello"])
    assert q.shape == (1, 1024)


def test_vector_size(emb):
    assert emb.vector_size == 1024


def test_dense_seq_len_capped(emb):
    assert emb._dense.max_seq_length == 2048


def test_build_embedder_mixed(monkeypatch):
    monkeypatch.setattr(mod, "SentenceTransformer", _FakeST)
    monkeypatch.setattr(mod, "BGEEmbedder", _FakeBGE)
    e = mod.build_embedder("mixed", device="cpu")
    assert isinstance(e, mod.MixedHybridEmbedder)


def test_build_embedder_unknown_raises():
    with pytest.raises(ValueError):
        mod.build_embedder("nope")
