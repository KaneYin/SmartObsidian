import json

from weft.ledger import load_seen, record
from weft.suggest import LinkSuggestion, pair_id


def _sugg(a, b, score=0.9):
    return LinkSuggestion(id=pair_id(a, b), note_a=a, note_b=b, score=score)


def test_load_seen_missing_file_returns_empty_set(tmp_path):
    assert load_seen(tmp_path / "nope.jsonl") == set()


def test_record_then_load_seen_roundtrip(tmp_path):
    path = tmp_path / "suggestions.jsonl"
    record(path, [_sugg("a.md", "b.md"), _sugg("a.md", "c.md")])

    seen = load_seen(path)
    assert seen == {("a.md", "b.md"), ("a.md", "c.md")}


def test_record_writes_proposed_status_and_pair_fields(tmp_path):
    path = tmp_path / "suggestions.jsonl"
    record(path, [_sugg("a.md", "b.md", score=0.87)])

    line = path.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["status"] == "proposed"
    assert rec["a"] == "a.md"
    assert rec["b"] == "b.md"
    assert rec["id"] == pair_id("a.md", "b.md")
    assert rec["score"] == 0.87
    assert "first_seen" in rec


def test_load_seen_returns_canonical_pairs(tmp_path):
    path = tmp_path / "suggestions.jsonl"
    # a record stored with a > b should still load as canonical (sorted) pair
    path.write_text(
        json.dumps({"id": "x", "a": "z.md", "b": "a.md", "status": "proposed"}) + "\n",
        encoding="utf-8",
    )
    assert load_seen(path) == {("a.md", "z.md")}


def test_record_does_not_duplicate_existing_pairs(tmp_path):
    path = tmp_path / "suggestions.jsonl"
    record(path, [_sugg("a.md", "b.md")])
    record(path, [_sugg("a.md", "b.md"), _sugg("a.md", "c.md")])

    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2  # b and c, but a.md<->b.md not written twice
    assert load_seen(path) == {("a.md", "b.md"), ("a.md", "c.md")}
