from weft.config import (
    ResolvedConfig,
    config_path_for,
    env_overrides,
    load_config,
    merge,
    save_config,
)


def test_missing_file_yields_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.provider == "ollama"
    assert cfg.model == "auto"
    assert cfg.mode == "balanced"
    assert cfg.endpoint == "http://localhost:11434"


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, ResolvedConfig(provider="anthropic", model="claude-opus-4-8"))
    cfg = load_config(path)
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-opus-4-8"


def test_boolean_param_roundtrips_as_valid_toml(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, ResolvedConfig(params={"think": False}))
    assert "think = false" in path.read_text(encoding="utf-8")
    assert load_config(path).params["think"] is False


def test_precedence_cli_over_env_over_file():
    base = ResolvedConfig(provider="ollama", model="llama3.1:8b")
    env = env_overrides({"WEFT_PROVIDER": "anthropic", "WEFT_MODEL": "env-model"})
    merged = merge(base, env, {"model": "cli-model"})
    assert merged.provider == "anthropic"   # from env (no CLI override)
    assert merged.model == "cli-model"      # CLI wins over env


def test_mode_roundtrip_and_environment_precedence(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, ResolvedConfig(mode="fast"))
    base = load_config(path)
    assert base.mode == "fast"
    merged = merge(base, env_overrides({"WEFT_MODE": "best"}), {"mode": "balanced"})
    assert merged.mode == "balanced"


def test_config_path_is_beside_store():
    assert config_path_for("/x/.weft/index").name == "config.toml"
    assert config_path_for("/x/.weft/index").parent.name == ".weft"


def test_fallback_defaults_empty(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.fallback == []


def test_fallback_roundtrips(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, ResolvedConfig(provider="ollama", fallback=["anthropic", "openai"]))
    cfg = load_config(path)
    assert cfg.fallback == ["anthropic", "openai"]


def test_merge_preserves_fallback():
    base = ResolvedConfig(provider="ollama", fallback=["anthropic"])
    merged = merge(base, {"model": "x"})
    assert merged.fallback == ["anthropic"]
