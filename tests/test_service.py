from weft.config import ResolvedConfig, config_path_for, save_config
from weft.embeddings import FakeEmbedder
from weft.service import service_config, service_health, service_models
from weft.store import VectorStore


def _index(store_path):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed(["hello"])[0],
              {"rel_path": "n.md", "heading": "H", "text": "hello",
               "ordinal": 0, "tags": [], "wikilinks": []})
    store.save(store_path)


def test_service_health_reports_index(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    assert service_health(store)["index"] is False
    _index(store)
    h = service_health(store)
    assert h["index"] is True and h["chunks"] == 1 and h["provider"] == "fake"


def test_service_config_has_no_secrets(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="anthropic", model="claude-opus-4-8"))
    cfg = service_config(store)
    assert cfg == {"provider": "anthropic", "model": "claude-opus-4-8",
                   "mode": "balanced", "endpoint": "http://localhost:11434",
                   "fallback": []}


def test_service_models_shape(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="ollama"))
    m = service_models(store)
    assert "recommended" in m and isinstance(m["tiers"], list)
    assert {"name", "default", "installed"} <= set(m["tiers"][0])


def test_service_memory_flow(tmp_path):
    from weft.service import (service_memory_accept, service_memory_list,
                              service_memory_pending, service_remember, make_proposals)
    from weft.proposals import Candidate
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True, exist_ok=True)

    item = service_remember(store, "preference", "Answer concisely")
    assert item["id"].startswith("mem_")
    assert [i["text"] for i in service_memory_list(store)["items"]] == ["Answer concisely"]

    prop = make_proposals(store).add([Candidate("fact", "Recurring interest: x", "heuristic")])[0]
    assert [p["id"] for p in service_memory_pending(store)["proposals"]] == [prop.id]
    assert service_memory_accept(store, prop.id) == {"accepted": prop.id}
    assert any(i["text"] == "Recurring interest: x" for i in service_memory_list(store)["items"])


def test_service_remember_validates(tmp_path):
    import pytest
    from weft.service import service_remember
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        service_remember(store, "bogus", "x")
    with pytest.raises(ValueError):
        service_remember(store, "fact", "x" * 3000)


def test_service_ask_returns_answer_and_logs_episode(tmp_path, monkeypatch):
    import weft.service as S
    from weft.llm import FakeLLM
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    _index(store)
    monkeypatch.setattr(S, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(S, "make_llm", lambda *a, **k: FakeLLM(response="answer [1]"))
    data = S.service_ask(store, "hello?", k=3)
    assert data["answer"] == "answer [1]"
    assert data["sources"] == ["n.md"]
    assert len(S.make_memory(store).episodes()) == 1


def test_service_ask_validates_k(tmp_path):
    import pytest
    from weft.service import service_ask
    store = tmp_path / ".weft" / "index"
    _index(store)
    with pytest.raises(ValueError):
        service_ask(store, "q", k=0)


def test_service_ask_memory_query_runs(tmp_path, monkeypatch):
    import weft.service as S
    from weft.llm import FakeLLM
    from weft.memory import MemoryStore
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    _index(store)
    mem = MemoryStore(store.parent / "memory.jsonl", store.parent / "episodes.jsonl")
    mem.remember("fact", "hello")
    monkeypatch.setattr(S, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(S, "make_llm", lambda *a, **k: FakeLLM(response="answer [1]"))
    monkeypatch.setattr(S, "make_memory", lambda sp: mem)
    data = S.service_ask(store, "world", k=3, use_memory_query=True)
    assert data["answer"] == "answer [1]"


def test_service_chat_session_uses_same_mode_resolver(tmp_path, monkeypatch):
    import weft.service as S
    from weft.llm import FakeLLM
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    _index(store)
    monkeypatch.setattr(S, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(S, "make_llm", lambda *a, **k: FakeLLM(response="answer"))
    session = S.service_chat_session(store, mode="fast")
    assert session._k == 5
    assert session._bm25 is None
    assert session._memory_query is False


def test_make_memory_stamps_episodes_with_manifest_vault_root(tmp_path):
    import weft.service as S
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("# N\n\nhello\n")
    store = tmp_path / ".weft" / "index"
    S.service_index(vault, store, embedder=FakeEmbedder(dim=16))

    mem = S.make_memory(store)
    mem.log_episode("q", "a", ["n.md"])
    assert mem.episodes()[0].vault_root == str(vault.resolve())


def test_make_memory_defaults_to_empty_vault_root_before_indexing(tmp_path):
    import weft.service as S
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True)
    mem = S.make_memory(store)
    mem.log_episode("q", "a", [])
    assert mem.episodes()[0].vault_root == ""


def test_service_index_refuses_different_vault_without_force(tmp_path):
    import pytest
    from weft.service import IndexVaultMismatchError, service_index
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    (vault_a / "a.md").write_text("# A\n\nhello\n")
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    (vault_b / "b.md").write_text("# B\n\nhello\n")
    store = tmp_path / ".weft" / "index"

    service_index(vault_a, store, embedder=FakeEmbedder(dim=16))
    with pytest.raises(IndexVaultMismatchError):
        service_index(vault_b, store, embedder=FakeEmbedder(dim=16))


def test_service_index_force_overwrites_different_vault(tmp_path):
    from weft.service import service_index
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    (vault_a / "a.md").write_text("# A\n\nhello\n")
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    (vault_b / "b.md").write_text("# B\n\nhello\n")
    store = tmp_path / ".weft" / "index"

    service_index(vault_a, store, embedder=FakeEmbedder(dim=16))
    result = service_index(vault_b, store, embedder=FakeEmbedder(dim=16), force=True)
    assert result["vault"] == str(vault_b)


def test_service_index_same_vault_reindex_needs_no_force(tmp_path):
    from weft.service import service_index
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nhello\n")
    store = tmp_path / ".weft" / "index"
    service_index(vault, store, embedder=FakeEmbedder(dim=16))
    result = service_index(vault, store, embedder=FakeEmbedder(dim=16))
    assert result["chunks"] >= 1
