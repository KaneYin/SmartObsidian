import numpy as np

from weft.agent import ask, graph_aware_retrieve
from weft.graph import LinkGraph
from weft.embeddings import FakeEmbedder
from weft.index import build_index
from weft.llm import FakeLLM
from weft.store import VectorStore


def test_build_index_empty_vault_writes_loadable_store(tmp_path):
    vault = tmp_path / "empty"
    vault.mkdir()
    n_chunks, n_edges = build_index(vault, FakeEmbedder(dim=16), tmp_path / "idx")
    assert n_chunks == 0
    assert n_edges == 0
    store = VectorStore.load(tmp_path / "idx")
    assert len(store) == 0
    assert store.search(np.zeros(16), k=3) == []


def test_ask_on_empty_store_reports_nothing_found(tmp_path):
    vault = tmp_path / "empty"
    vault.mkdir()
    build_index(vault, FakeEmbedder(dim=16), tmp_path / "idx")
    store = VectorStore.load(tmp_path / "idx")
    result = ask("anything?", FakeEmbedder(dim=16), store, FakeLLM(response="unused"), k=3)
    assert "couldn't find" in result.answer.lower()
    assert result.sources == []


def test_graph_retrieval_on_empty_store_returns_no_hits():
    store = VectorStore(dim=16)
    assert graph_aware_retrieve(
        "anything?", FakeEmbedder(dim=16), store, LinkGraph(), k=3
    ) == []


def test_add_batch_matches_repeated_add_and_is_unit_norm():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((5, 4)).astype(np.float32)
    metas = [{"i": i} for i in range(5)]

    batch = VectorStore(dim=4)
    batch.add_batch(vecs, metas)

    single = VectorStore(dim=4)
    for v, m in zip(vecs, metas):
        single.add(v, m)

    assert np.allclose(batch._vectors, single._vectors, atol=1e-6)
    norms = np.linalg.norm(batch._vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_index_metadata_includes_tags_and_wikilinks(sample_vault, tmp_path):
    build_index(sample_vault, FakeEmbedder(dim=16), tmp_path / "idx")
    store = VectorStore.load(tmp_path / "idx")
    coffee = next(m for m in store._metadata if m["rel_path"] == "coffee.md")
    assert "drinks" in coffee["tags"]
    assert "water" in coffee["wikilinks"]
