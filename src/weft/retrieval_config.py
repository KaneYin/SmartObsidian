"""User-facing retrieval modes resolved into explicit pipeline settings.

Interfaces pass intent (``fast``, ``balanced``, or ``best``) plus sparse
overrides.  Retrieval code receives only the fully resolved configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from weft.config import VALID_MODES, config_path_for, env_overrides, load_config, merge

MAX_K = 50
MAX_RERANK_POOL = 500


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str
    k: int
    hybrid: bool
    graph: bool
    rerank: bool
    rerank_pool: int
    memory_query: bool
    query_rewrite: bool = False


@dataclass(frozen=True)
class RetrievalOverrides:
    k: int | None = None
    hybrid: bool | None = None
    graph: bool | None = None
    rerank: bool | None = None
    rerank_pool: int | None = None
    memory_query: bool | None = None
    query_rewrite: bool | None = None

    def merged(self, **values) -> "RetrievalOverrides":
        """Return a copy with non-None values applied last."""
        updates = {key: value for key, value in values.items() if value is not None}
        return replace(self, **updates)


_PRESETS = {
    "fast": RetrievalConfig(
        mode="fast",
        k=5,
        hybrid=False,
        graph=True,
        rerank=False,
        rerank_pool=20,
        memory_query=False,
    ),
    "balanced": RetrievalConfig(
        mode="balanced",
        k=8,
        hybrid=True,
        graph=True,
        rerank=False,
        rerank_pool=20,
        memory_query=True,
    ),
    "best": RetrievalConfig(
        mode="best",
        k=8,
        hybrid=True,
        graph=True,
        rerank=True,
        rerank_pool=30,
        memory_query=True,
    ),
}


def preset_for(mode: str) -> RetrievalConfig:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_MODES:
        raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
    return _PRESETS[normalized]


def build_retrieval_config(
    mode: str,
    overrides: RetrievalOverrides | None = None,
) -> RetrievalConfig:
    """Apply explicit advanced overrides on top of a user-facing preset."""
    config = preset_for(mode)
    if overrides is not None:
        updates = {
            name: value
            for name, value in vars(overrides).items()
            if value is not None
        }
        config = replace(config, **updates)
    if not isinstance(config.k, int) or not 1 <= config.k <= MAX_K:
        raise ValueError(f"k must be between 1 and {MAX_K}")
    if not isinstance(config.rerank_pool, int) or not 1 <= config.rerank_pool <= MAX_RERANK_POOL:
        raise ValueError(f"rerank pool must be between 1 and {MAX_RERANK_POOL}")
    return config


def resolve_retrieval_config(
    store_path: str | Path,
    *,
    mode: str | None = None,
    overrides: RetrievalOverrides | None = None,
    env: dict[str, str] | None = None,
) -> RetrievalConfig:
    """Resolve mode as defaults < config file < environment < interface.

    Once the mode is selected, explicit advanced options override its preset.
    The same function is used by CLI, TUI, and REST service entry points.
    """
    file_config = load_config(config_path_for(Path(store_path)))
    layers = [env_overrides(dict(os.environ) if env is None else env)]
    if mode is not None:
        layers.append({"mode": mode})
    resolved = merge(file_config, *layers)
    return build_retrieval_config(resolved.mode, overrides)


__all__ = [
    "RetrievalConfig",
    "RetrievalOverrides",
    "build_retrieval_config",
    "preset_for",
    "resolve_retrieval_config",
]
