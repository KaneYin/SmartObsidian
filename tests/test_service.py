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
