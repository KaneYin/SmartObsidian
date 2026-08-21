"""Provider registry: resolve config (file + env + CLI) into a concrete
LLMClient, or fail fast with an actionable remedy. resolve_llm is pure — it never
prompts or downloads; provisioning is an explicit `weft models pull`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from weft.config import (
    ResolvedConfig,
    config_path_for,
    env_overrides,
    load_config,
    merge,
)
from weft.hardware import detect_gpu
from weft.llm import ClaudeClient, FakeLLM, LLMClient
from weft.models import pick_default
from weft.ollama_client import OllamaClient
from weft.ollama_client import list_models as ollama_installed
from weft.ollama_client import ping as ollama_ping
from weft.openai_client import OpenAIClient, is_local_endpoint


class ProviderUnavailable(RuntimeError):
    """The selected provider cannot serve; the message is the user's remedy."""


@dataclass
class Availability:
    ok: bool
    remedy: str = ""


class Provider(Protocol):
    name: str

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability: ...
    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient: ...


def _resolve_model(cfg: ResolvedConfig) -> str:
    if cfg.model and cfg.model != "auto":
        return cfg.model
    return pick_default(detect_gpu())


class OllamaProvider:
    name = "ollama"

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        if not ollama_ping(cfg.endpoint):
            return Availability(False,
                f"Ollama not reachable at {cfg.endpoint} — start it with "
                f"`ollama serve`, or `weft config set provider anthropic`.")
        model = _resolve_model(cfg)
        if model not in ollama_installed(cfg.endpoint):
            return Availability(False,
                f"Model {model!r} is not installed — run `weft models pull {model}`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        return OllamaClient(cfg.endpoint, _resolve_model(cfg), params=cfg.params)


class AnthropicProvider:
    name = "anthropic"

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        if not env.get("ANTHROPIC_API_KEY"):
            return Availability(False,
                "ANTHROPIC_API_KEY is not set — export it, or switch provider with "
                "`weft config set provider ollama`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        if cfg.model and cfg.model != "auto":
            return ClaudeClient(model=cfg.model)
        return ClaudeClient()


class OpenAIProvider:
    name = "openai"

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        if not cfg.model or cfg.model == "auto":
            return Availability(False,
                "openai needs an explicit model — run `weft config set model <name>`.")
        if not is_local_endpoint(cfg.endpoint) and not env.get("OPENAI_API_KEY"):
            return Availability(False,
                f"OPENAI_API_KEY is not set for remote endpoint {cfg.endpoint} — export "
                "it, use a local endpoint, or switch provider.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        return OpenAIClient(cfg.endpoint, cfg.model,
                            api_key=env.get("OPENAI_API_KEY"), params=cfg.params)


class FakeProvider:
    name = "fake"

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        client = FakeLLM(response="[fake]")
        client.provider = "fake"          # metadata for the audit log
        client.model = cfg.model
        client.left_machine = False
        return client


REGISTRY: dict[str, Provider] = {
    "ollama": OllamaProvider(),
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    "fake": FakeProvider(),
}


_LOCAL_PROVIDERS = {"ollama", "fake"}


def _is_local_provider(name: str, cfg: ResolvedConfig) -> bool:
    if name == "openai":
        return is_local_endpoint(cfg.endpoint)
    return name in _LOCAL_PROVIDERS


def _fallback_cfg(cfg: ResolvedConfig, name: str) -> ResolvedConfig:
    """A fallback provider uses its own defaults, not the primary's model/endpoint."""
    from weft.config import DEFAULT_ENDPOINT
    return ResolvedConfig(provider=name, model="auto", endpoint=DEFAULT_ENDPOINT,
                          params=dict(cfg.params))


def resolve_llm(overrides: dict, *, store_path: Path, env: dict,
                on_fallback=None) -> LLMClient:
    cfg = load_config(config_path_for(store_path))
    cfg = merge(cfg, env_overrides(env), overrides)
    order = [cfg.provider] + [p for p in cfg.fallback if p != cfg.provider]

    remedies: list[str] = []
    for i, name in enumerate(order):
        provider = REGISTRY.get(name)
        if provider is None:
            remedies.append(f"unknown provider {name!r}")
            continue
        use_cfg = cfg if i == 0 else _fallback_cfg(cfg, name)
        avail = provider.available(use_cfg, env)
        if not avail.ok:
            remedies.append(avail.remedy)
            continue
        client = provider.build(use_cfg, env)
        if i > 0 and on_fallback is not None:
            crossed = _is_local_provider(order[0], cfg) and bool(
                getattr(client, "left_machine", True))
            on_fallback(order[0], name, crossed)
        return client
    raise ProviderUnavailable(" ; ".join(remedies))
