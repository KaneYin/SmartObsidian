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


def test_sliding_produces_overlapping_windows(tmp_path):
    from weft.chunking import SLIDING_SIZE
    body = "x" * (SLIDING_SIZE * 2)
    note = _note(tmp_path, body)
    chunks = chunk_strategy("sliding")(note)
    assert len(chunks) >= 2
    assert all(len(c.text) <= SLIDING_SIZE for c in chunks)
    assert chunks[1].ordinal == 1


def test_sliding_short_note_single_window(tmp_path):
    note = _note(tmp_path, "just a little text")
    chunks = chunk_strategy("sliding")(note)
    assert len(chunks) == 1
    assert "little text" in chunks[0].text


def test_contextualize_sets_embed_text(tmp_path):
    from weft.chunking import contextualize
    from weft.llm import FakeLLM
    note = _note(tmp_path, "# A\npara one\n\npara two\n")
    chunks = chunk_strategy("heading")(note)
    contextualize(note, chunks, FakeLLM(response="This is section A."))
    assert chunks[0].embed_text.startswith("This is section A.")
    assert chunks[0].text in chunks[0].embed_text


def test_contextualize_falls_back_on_error(tmp_path):
    from weft.chunking import contextualize
    from weft.llm import FakeLLM
    note = _note(tmp_path, "# A\nonly para\n")
    chunks = chunk_strategy("heading")(note)

    class Boom(FakeLLM):
        def complete(self, system, prompt):
            raise RuntimeError("no provider")

    contextualize(note, chunks, Boom())
    assert chunks[0].embed_text == chunks[0].text
