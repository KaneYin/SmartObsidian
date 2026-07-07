import numpy as np

from weft.store import VectorStore, SearchHit


def test_add_and_search_returns_nearest_first():
    store = VectorStore(dim=3)
    store.add(np.array([1.0, 0.0, 0.0]), {"rel_path": "a.md", "heading": "A", "text": "a"})
    store.add(np.array([0.0, 1.0, 0.0]), {"rel_path": "b.md", "heading": "B", "text": "b"})
    hits = store.search(np.array([0.9, 0.1, 0.0]), k=2)
    assert [h.metadata["rel_path"] for h in hits] == ["a.md", "b.md"]
    assert isinstance(hits[0], SearchHit)
    assert hits[0].score > hits[1].score


def test_search_k_caps_at_store_size():
    store = VectorStore(dim=3)
    store.add(np.array([1.0, 0.0, 0.0]), {"rel_path": "a.md", "heading": "A", "text": "a"})
    hits = store.search(np.array([1.0, 0.0, 0.0]), k=5)
    assert len(hits) == 1


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore(dim=3)
    store.add(np.array([1.0, 0.0, 0.0]), {"rel_path": "a.md", "heading": "A", "text": "alpha"})
    path = tmp_path / "index"
    store.save(path)

    loaded = VectorStore.load(path)
    assert loaded.dim == 3
    assert len(loaded) == 1
    hits = loaded.search(np.array([1.0, 0.0, 0.0]), k=1)
    assert hits[0].metadata["text"] == "alpha"


def test_search_empty_store_returns_empty():
    store = VectorStore(dim=3)
    assert store.search(np.array([1.0, 0.0, 0.0]), k=3) == []
