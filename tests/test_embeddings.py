import numpy as np

from weft.embeddings import Embedder, FakeEmbedder


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder(dim=16)
    a = e.embed(["hello world"])
    b = e.embed(["hello world"])
    assert a.shape == (1, 16)
    assert np.allclose(a, b)


def test_fake_embedder_distinguishes_text():
    e = FakeEmbedder(dim=16)
    v = e.embed(["coffee", "coffee", "tea"])
    assert v.shape == (3, 16)
    assert np.allclose(v[0], v[1])       # same text -> same vector
    assert not np.allclose(v[0], v[2])   # different text -> different vector


def test_fake_embedder_vectors_are_unit_norm():
    e = FakeEmbedder(dim=16)
    v = e.embed(["anything"])
    assert np.isclose(np.linalg.norm(v[0]), 1.0)


def test_fake_embedder_satisfies_protocol():
    assert isinstance(FakeEmbedder(dim=8), Embedder)
