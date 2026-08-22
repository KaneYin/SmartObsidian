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
                   "endpoint": "http://localhost:11434", "fallback": []}


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
