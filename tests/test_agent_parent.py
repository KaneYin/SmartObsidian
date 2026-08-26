import json

from weft.agent import build_prompt
from weft.store import SearchHit


def _hit(rel, ordinal, text, parent_id=None, parent_text=None):
    meta = {"rel_path": rel, "heading": "H", "text": text, "ordinal": ordinal}
    if parent_id is not None:
        meta["parent_id"] = parent_id
        meta["parent_text"] = parent_text
    return SearchHit(score=1.0, metadata=meta)


def test_build_prompt_collapses_children_of_one_parent():
    hits = [
        _hit("n.md", 0, "child a", parent_id="par_1", parent_text="the full section"),
        _hit("n.md", 1, "child b", parent_id="par_1", parent_text="the full section"),
    ]
    payload = json.loads(build_prompt("q", hits))
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["content"] == "the full section"
    assert payload["sources"][0]["id"] == 1


def test_build_prompt_heading_mode_unchanged():
    hits = [_hit("a.md", 0, "alpha"), _hit("b.md", 0, "beta")]
    payload = json.loads(build_prompt("q", hits))
    assert [s["content"] for s in payload["sources"]] == ["alpha", "beta"]
    assert [s["id"] for s in payload["sources"]] == [1, 2]
