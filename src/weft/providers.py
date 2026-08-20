"""Provider registry: resolve config (file + env + CLI) into a concrete
LLMClient, or fail fast with an actionable remedy. resolve_llm is pure — it never
prompts or downloads; provisioning is an explicit `weft models pull`."""

from __future__ import annotations

import os
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


class ProviderUnavailable(RuntimeError):
    """The selected provider cannot serve; the message is the user's remedy."""


@dataclass
class Availability:
    ok: bool
    remedy: str = ""


class Provider(Protocol):
    name: str

    def available(self, cfg: ResolvedConfig) -> Availability: ...
    def build(self, cfg: ResolvedConfig) -> LLMClient: ...


def _resolve_model(cfg: ResolvedConfig) -> str:
    if cfg.model and cfg.model != "auto":
        return cfg.model
    return pick_default(detect_gpu())


class OllamaProvider:
    name = "ollama"

    def available(self, cfg: ResolvedConfig) -> Availability:
        if not ollama_ping(cfg.endpoint):
            return Availability(False,
                f"Ollama not reachable at {cfg.endpoint} — start it with "
                f"`ollama serve`, or `weft config set provider anthropic`.")
        model = _resolve_model(cfg)
        if model not in ollama_installed(cfg.endpoint):
            return Availability(False,
                f"Model {model!r} is not installed — run `weft models pull {model}`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig) -> LLMClient:
        return OllamaClient(cfg.endpoint, _resolve_model(cfg), params=cfg.params)


class AnthropicProvider:
    name = "anthropic"

    def available(self, cfg: ResolvedConfig, env: dict | None = None) -> Availability:
        key = (env if env is not None else os.environ).get("ANTHROPIC_API_KEY")
        if not key:
            return Availability(False,
                "ANTHROPIC_API_KEY is not set — export it, or switch provider with "
                "`weft config set provider ollama`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig) -> LLMClient:
        if cfg.model and cfg.model != "auto":
            return ClaudeClient(model=cfg.model)
        return ClaudeClient()


class FakeProvider:
    name = "fake"

    def available(self, cfg: ResolvedConfig) -> Availability:
        return Availability(True)

    def build(self, cfg: ResolvedConfig) -> LLMClient:
        client = FakeLLM(response="[fake]")
        client.provider = "fake"          # metadata for the audit log
        client.model = cfg.model
        client.left_machine = False
        return client


REGISTRY: dict[str, Provider] = {
    "ollama": OllamaProvider(),
    "anthropic": AnthropicProvider(),
    "fake": FakeProvider(),
}


def resolve_llm(overrides: dict, *, store_path: Path, env: dict) -> LLMClient:
    cfg = load_config(config_path_for(store_path))
    cfg = merge(cfg, env_overrides(env), overrides)
    provider = REGISTRY.get(cfg.provider)
    if provider is None:
        raise ProviderUnavailable(
            f"Unknown provider {cfg.provider!r}. Choose one of: "
            f"{', '.join(sorted(REGISTRY))}.")
    # AnthropicProvider.available needs env; pass it when supported.
    if isinstance(provider, AnthropicProvider):
        avail = provider.available(cfg, env=env)
    else:
        avail = provider.available(cfg)
    if not avail.ok:
        raise ProviderUnavailable(avail.remedy)
    return provider.build(cfg)
