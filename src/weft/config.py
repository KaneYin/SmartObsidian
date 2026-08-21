"""Provider/model selection config: a non-secret `.weft/config.toml` merged with
env and CLI overrides. API keys never live here — they stay in the environment.

Precedence (lowest to highest): built-in defaults < file < env < CLI overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from weft.security import secure_write_text

VALID_PROVIDERS = {"ollama", "anthropic", "openai", "fake"}
DEFAULT_ENDPOINT = "http://localhost:11434"


@dataclass
class ResolvedConfig:
    provider: str = "ollama"
    model: str = "auto"
    endpoint: str = DEFAULT_ENDPOINT
    params: dict = field(default_factory=lambda: {"temperature": 0.2, "num_ctx": 8192})
    fallback: list[str] = field(default_factory=list)


def config_path_for(store_path: Path) -> Path:
    """`.weft/index` -> `.weft/config.toml` (config sits beside the index)."""
    return Path(store_path).parent / "config.toml"


def load_config(path: Path) -> ResolvedConfig:
    path = Path(path)
    if not path.exists():
        return ResolvedConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    default = ResolvedConfig()
    return ResolvedConfig(
        provider=str(data.get("provider", default.provider)),
        model=str(data.get("model", default.model)),
        endpoint=str(data.get("endpoint", default.endpoint)),
        params=dict(data.get("params", default.params)),
        fallback=list(data.get("fallback", [])),
    )


def env_overrides(env: dict) -> dict:
    """Map WEFT_* environment variables to config keys (only those present)."""
    out: dict = {}
    if env.get("WEFT_PROVIDER"):
        out["provider"] = env["WEFT_PROVIDER"]
    if env.get("WEFT_MODEL"):
        out["model"] = env["WEFT_MODEL"]
    if env.get("WEFT_ENDPOINT"):
        out["endpoint"] = env["WEFT_ENDPOINT"]
    return out


def merge(cfg: ResolvedConfig, *overrides: dict) -> ResolvedConfig:
    """Apply override dicts in order (later wins), skipping falsy values."""
    provider, model, endpoint = cfg.provider, cfg.model, cfg.endpoint
    params = dict(cfg.params)
    fallback = list(cfg.fallback)
    for layer in overrides:
        if layer.get("provider"):
            provider = layer["provider"]
        if layer.get("model"):
            model = layer["model"]
        if layer.get("endpoint"):
            endpoint = layer["endpoint"]
        if layer.get("params"):
            params.update(layer["params"])
        if layer.get("fallback"):
            fallback = list(layer["fallback"])
    return ResolvedConfig(provider=provider, model=model, endpoint=endpoint,
                          params=params, fallback=fallback)


def _to_toml(cfg: ResolvedConfig) -> str:
    fallback_rendered = ", ".join(f'"{p}"' for p in cfg.fallback)
    lines = [
        f'provider = "{cfg.provider}"',
        f'model = "{cfg.model}"',
        f'endpoint = "{cfg.endpoint}"',
        f"fallback = [{fallback_rendered}]",
        "",
        "[params]",
    ]
    for key, value in cfg.params.items():
        rendered = value if isinstance(value, (int, float)) else f'"{value}"'
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def save_config(path: Path, cfg: ResolvedConfig) -> Path:
    return secure_write_text(Path(path), _to_toml(cfg), overwrite=True)
