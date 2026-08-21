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


def test_openai_available_with_model_and_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="gpt-4o-mini", endpoint="https://api.openai.com/v1"))
    llm = resolve_llm({}, store_path=tmp_path / "index", env={"OPENAI_API_KEY": "sk-x"})
    assert llm.provider == "openai" and llm.model == "gpt-4o-mini"


def test_openai_auto_model_fails_fast(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="auto", endpoint="https://api.openai.com/v1"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={"OPENAI_API_KEY": "sk-x"})
    assert "explicit model" in str(exc.value)


def test_openai_remote_requires_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="gpt-4o-mini", endpoint="https://api.openai.com/v1"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "OPENAI_API_KEY" in str(exc.value)


def test_openai_local_needs_no_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="local-model", endpoint="http://localhost:1234/v1"))
    llm = resolve_llm({}, store_path=tmp_path / "index", env={})
    assert llm.provider == "openai" and llm.left_machine is False
