import pytest

from weft.chunking import chunk_strategy
from weft.parser import parse_note


def _note(tmp_path, body):
    p = tmp_path / "n.md"
    p.write_text(body, encoding="utf-8")
    note = parse_note(p)
    note.rel_path = "n.md"
    return note


def test_heading_strategy_matches_chunk_note(tmp_path):
    from weft.parser import chunk_note
    note = _note(tmp_path, "# A\npara one\n\npara two\n\n# B\nmore")
    assert chunk_strategy("heading")(note) == chunk_note(note)


def test_parent_child_splits_paragraphs_sharing_parent(tmp_path):
    note = _note(tmp_path, "# A\npara one\n\npara two\n")
    chunks = chunk_strategy("parent_child")(note)
    texts = [c.text for c in chunks]
    assert texts == ["para one", "para two"]
    assert chunks[0].parent_id == chunks[1].parent_id
    assert "para one" in chunks[0].parent_text and "para two" in chunks[0].parent_text


def test_parent_child_single_paragraph_is_one_child(tmp_path):
    note = _note(tmp_path, "# A\njust one paragraph\n")
    chunks = chunk_strategy("parent_child")(note)
    assert len(chunks) == 1
    assert chunks[0].text == "just one paragraph"
    assert chunks[0].parent_text.strip() == "just one paragraph"


def test_parent_child_keeps_fence_whole(tmp_path):
    note = _note(tmp_path, "# A\nintro\n\n```\ncode\n\nstill code\n```\n")
    chunks = chunk_strategy("parent_child")(note)
    fence_children = [c.text for c in chunks if "code" in c.text]
    assert any("still code" in t for t in fence_children)


def test_unknown_strategy_raises(tmp_path):
    with pytest.raises(ValueError):
        chunk_strategy("bogus")
