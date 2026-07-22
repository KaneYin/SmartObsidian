from weft.embeddings import FakeEmbedder
from weft.index import build_index
from weft.store import VectorStore


def test_build_index_populates_and_persists(sample_vault, tmp_path):
    store_path = tmp_path / "weft_index"
    n_chunks, n_edges = build_index(sample_vault, FakeEmbedder(dim=16), store_path)
    assert n_chunks > 0

    store = VectorStore.load(store_path)
    assert len(store) == n_chunks
    rel_paths = {m["rel_path"] for m in store._metadata}
    assert {"coffee.md", "water.md", "notes/tea.md"} <= rel_paths
    # every chunk carries the fields the agent needs to cite
    for m in store._metadata:
        assert {"rel_path", "heading", "text"} <= set(m)


def test_build_index_dim_matches_embedder(sample_vault, tmp_path):
    build_index(sample_vault, FakeEmbedder(dim=16), tmp_path / "idx")
    assert VectorStore.load(tmp_path / "idx").dim == 16
