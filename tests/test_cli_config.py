import tomllib

from weft.cli import main


def test_config_set_and_show(tmp_path, capsys):
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "provider", "ollama", "--store", str(store)]) == 0
    data = tomllib.loads((tmp_path / ".weft" / "config.toml").read_text())
    assert data["provider"] == "ollama"
    assert main(["config", "show", "--store", str(store)]) == 0
    assert "ollama" in capsys.readouterr().out


def test_config_set_rejects_unknown_provider(tmp_path):
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "provider", "bogus", "--store", str(store)]) == 2


def test_config_set_fallback_list(tmp_path):
    import tomllib
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "fallback", "anthropic,openai", "--store", str(store)]) == 0
    data = tomllib.loads((tmp_path / ".weft" / "config.toml").read_text())
    assert data["fallback"] == ["anthropic", "openai"]


def test_config_set_fallback_rejects_unknown(tmp_path):
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "fallback", "anthropic,bogus", "--store", str(store)]) == 2
