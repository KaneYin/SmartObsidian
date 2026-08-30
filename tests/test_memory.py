import json
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


def test_log_episode_appends_and_reloads(tmp_path):
    ms = _store(tmp_path)
    ep = ms.log_episode("what did I decide?", "You chose LanceDB [1]", ["d.md"])
    assert ep.id.startswith("ep_")
    reloaded = _store(tmp_path)
    eps = reloaded.episodes()
    assert len(eps) == 1
    assert eps[0].question == "what did I decide?"
    assert eps[0].sources == ["d.md"]


def test_episode_stamped_with_vault_root(tmp_path):
    ms = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                      vault_root="/vault/a")
    ms.log_episode("q", "a", ["n.md"])
    eps = ms.episodes()
    assert len(eps) == 1
    assert eps[0].vault_root == "/vault/a"


def test_episodes_exclude_different_vault_root(tmp_path):
    # Same store path, two different vaults -- the exact bug scenario.
    ms_a = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/a")
    ms_a.log_episode("what is this vault about", "vault A content", ["a.md"])
    ms_b = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/b")
    assert ms_b.episodes() == []


def test_episodes_exclude_missing_vault_root_field(tmp_path):
    # A pre-fix record with no vault_root key at all must never match, even
    # if the current store also happens to be unscoped.
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps({
        "id": "ep_legacy", "ts": "2026-08-28T00:00:00+00:00",
        "question": "old q", "answer": "old a", "sources": [],
    }) + "\n")
    os.chmod(episodes_path, 0o600)
    ms = MemoryStore(tmp_path / "memory.jsonl", episodes_path, vault_root="")
    assert ms.episodes() == []


def test_recall_log_kind_excludes_other_vault(tmp_path):
    from weft.embeddings import FakeEmbedder
    ms_a = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/a")
    ms_a.log_episode("what is this vault about", "vault A is about coffee", ["a.md"])
    ms_b = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/b")
    hits = ms_b.recall(FakeEmbedder(dim=16), "what is this vault about", k=5, kinds={"log"})
    assert hits == []
