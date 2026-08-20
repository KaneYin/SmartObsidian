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
