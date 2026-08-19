import numpy as np
import pytest

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


def test_search_rejects_non_positive_k():
    store = VectorStore(dim=3)
    store.add(np.array([1.0, 0.0, 0.0]), {"rel_path": "a.md"})
    with pytest.raises(ValueError, match="greater than zero"):
        store.search(np.array([1.0, 0.0, 0.0]), k=-1)


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
    assert (path.with_suffix(".npz").stat().st_mode & 0o777) == 0o600
    assert (path.with_suffix(".json").stat().st_mode & 0o777) == 0o600


def test_search_empty_store_returns_empty():
    store = VectorStore(dim=3)
    assert store.search(np.array([1.0, 0.0, 0.0]), k=3) == []


def test_vectors_by_note_groups_chunk_vectors_per_note():
    store = VectorStore(dim=3)
    store.add(np.array([1.0, 0.0, 0.0]), {"rel_path": "a.md", "text": "a1"})
    store.add(np.array([0.0, 1.0, 0.0]), {"rel_path": "a.md", "text": "a2"})
    store.add(np.array([0.0, 0.0, 1.0]), {"rel_path": "b.md", "text": "b1"})

    grouped = store.vectors_by_note()

    assert set(grouped) == {"a.md", "b.md"}
    assert grouped["a.md"].shape == (2, 3)
    assert grouped["b.md"].shape == (1, 3)


def test_metadata_by_note_groups_chunk_metadata_in_order():
    store = VectorStore(dim=3)
    store.add(np.array([1.0, 0.0, 0.0]), {"rel_path": "a.md", "text": "a1", "tags": ["t"]})
    store.add(np.array([0.0, 1.0, 0.0]), {"rel_path": "a.md", "text": "a2", "tags": ["t"]})

    grouped = store.metadata_by_note()

    assert list(grouped) == ["a.md"]
    assert [m["text"] for m in grouped["a.md"]] == ["a1", "a2"]


def test_vectors_by_note_empty_store():
    store = VectorStore(dim=3)
    assert store.vectors_by_note() == {}
