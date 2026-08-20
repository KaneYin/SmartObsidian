import pytest

from weft.config import ResolvedConfig, save_config
from weft.providers import ProviderUnavailable, resolve_llm


def test_resolve_ollama_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.providers.ollama_ping", lambda endpoint: True)
    monkeypatch.setattr("weft.providers.ollama_installed", lambda endpoint: ["llama3.1:8b"])
    save_config(tmp_path / "config.toml",
                ResolvedConfig(provider="ollama", model="llama3.1:8b"))
    llm = resolve_llm({}, store_path=tmp_path / "index", env={})
    assert llm.provider == "ollama" and llm.model == "llama3.1:8b"


def test_ollama_daemon_down_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.providers.ollama_ping", lambda endpoint: False)
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="ollama"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "ollama serve" in str(exc.value)


def test_missing_model_fails_with_pull_remedy(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.providers.ollama_ping", lambda endpoint: True)
    monkeypatch.setattr("weft.providers.ollama_installed", lambda endpoint: [])
    save_config(tmp_path / "config.toml",
                ResolvedConfig(provider="ollama", model="llama3.1:8b"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "weft models pull" in str(exc.value)


def test_anthropic_requires_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="anthropic"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_cli_override_selects_provider(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="anthropic"))
    llm = resolve_llm({"provider": "fake"}, store_path=tmp_path / "index", env={})
    assert llm.provider == "fake"
