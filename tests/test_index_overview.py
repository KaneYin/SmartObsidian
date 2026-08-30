from weft.embeddings import FakeEmbedder
from weft.index import build_index, build_overview_chunk
from weft.parser import Note
from weft.store import VectorStore


def test_overview_chunk_folder_and_tag_breakdown():
    notes = [
        Note(rel_path="coffee.md", title="Coffee", frontmatter={},
             tags=["drinks"], wikilinks=[], body=""),
        Note(rel_path="notes/tea.md", title="Tea", frontmatter={},
             tags=[], wikilinks=[], body=""),
    ]
    chunk = build_overview_chunk(notes)
    assert chunk.rel_path == "(vault overview)"
    assert chunk.heading == "Vault overview"
    assert "2 notes across 2 top-level folders" in chunk.text
    assert "#drinks (1)" in chunk.text


def test_overview_chunk_none_for_empty_notes():
    assert build_overview_chunk([]) is None


def test_build_index_adds_overview_chunk_to_store(sample_vault, tmp_path):
    store_path = tmp_path / "idx"
    build_index(sample_vault, FakeEmbedder(dim=16), store_path)
    store = VectorStore.load(store_path)
    overview = [m for m in store._metadata if m["rel_path"] == "(vault overview)"]
    assert len(overview) == 1
    text = overview[0]["text"]
    assert "3 notes across 2 top-level folders" in text
    assert "notes/ (1 notes)" in text
    assert "#strong (1)" in text
