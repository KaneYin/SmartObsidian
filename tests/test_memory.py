import os
import stat

from weft.memory import MemoryStore


def _store(tmp_path):
    return MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl")


def test_remember_then_active_semantic(tmp_path):
    ms = _store(tmp_path)
    item = ms.remember("preference", "Answer concisely")
    assert item.id.startswith("mem_")
    assert item.type == "preference"
    assert item.status == "active"
    assert item.provenance == "explicit"
    active = ms.active_semantic()
    assert [i.text for i in active] == ["Answer concisely"]


def test_active_semantic_filters_by_type(tmp_path):
    ms = _store(tmp_path)
    ms.remember("preference", "Concise")
    ms.remember("fact", "Writing a thesis on X")
    assert [i.text for i in ms.active_semantic(type="fact")] == ["Writing a thesis on X"]


def test_persistence_reloads_and_is_private(tmp_path):
    ms = _store(tmp_path)
    ms.remember("decision", "Chose LanceDB")
    reloaded = _store(tmp_path)
    assert [i.text for i in reloaded.active_semantic()] == ["Chose LanceDB"]
    mode = stat.S_IMODE(os.stat(tmp_path / "memory.jsonl").st_mode)
    assert mode == 0o600


def test_supersede_replaces_active(tmp_path):
    ms = _store(tmp_path)
    old = ms.remember("decision", "Chose Neo4j")
    updated = ms.supersede(old.id, "Chose LanceDB")
    texts = [i.text for i in ms.active_semantic()]
    assert texts == ["Chose LanceDB"]
    assert updated.supersedes == old.id
    assert updated.type == "decision"


def test_reject_tombstones_item(tmp_path):
    ms = _store(tmp_path)
    item = ms.remember("fact", "secret detail")
    ms.reject(item.id)
    assert ms.active_semantic() == []
    assert ms.get(item.id).status == "rejected"


def test_compact_collapses_and_drops_rejected_text(tmp_path):
    ms = _store(tmp_path)
    a = ms.remember("preference", "keep me")
    b = ms.remember("fact", "forget me")
    ms.reject(b.id)
    ms.compact()
    reloaded = _store(tmp_path)
    assert [i.text for i in reloaded.active_semantic()] == ["keep me"]
    assert reloaded.get(b.id).status == "rejected"
    assert reloaded.get(b.id).text == ""   # rejected text dropped on compaction
    lines = (tmp_path / "memory.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
