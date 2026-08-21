"""Shared service core: the operations behind both the CLI and the HTTP API.
Functions take a store path plus arguments and return plain JSON-able dicts. This
is the single source of truth; the API never reimplements CLI logic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from weft.config import config_path_for, load_config
from weft.embeddings import Embedder, SentenceTransformerEmbedder
from weft.graph import LinkGraph
from weft.hardware import detect_gpu
from weft.index import graph_path_for
from weft.llm import AuditedLLM, LLMClient
from weft.memory import SEMANTIC_TYPES, MemoryStore
from weft.models import TIERS, pick_default
from weft.ollama_client import list_models as ollama_installed
from weft.proposals import ProposalStore
from weft.providers import resolve_llm
from weft.store import VectorStore

MAX_K = 50
MAX_MEMORY_TEXT = 2000


def make_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


def _fallback_notice(primary: str, chosen: str, crossed: bool) -> None:
    if crossed:
        print(f"falling back to {chosen} — this sends note content off your machine "
              f"(configured in fallback).", file=sys.stderr)
    else:
        print(f"primary {primary} unavailable; using fallback {chosen}.", file=sys.stderr)


def make_llm(overrides: dict, store_path: Path) -> LLMClient:
    return resolve_llm(overrides, store_path=store_path, env=dict(os.environ),
                       on_fallback=_fallback_notice)


def make_memory(store_path: Path) -> MemoryStore:
    return MemoryStore(store_path.parent / "memory.jsonl",
                       store_path.parent / "episodes.jsonl")


def make_proposals(store_path: Path) -> ProposalStore:
    return ProposalStore(store_path.parent / "memory-proposals.jsonl")


def service_health(store_path: Path) -> dict:
    store_path = Path(store_path)
    exists = store_path.with_suffix(".npz").exists()
    chunks = len(VectorStore.load(store_path)) if exists else 0
    cfg = load_config(config_path_for(store_path))
    return {"index": exists, "chunks": chunks, "provider": cfg.provider}


def service_config(store_path: Path) -> dict:
    cfg = load_config(config_path_for(Path(store_path)))
    return {"provider": cfg.provider, "model": cfg.model,
            "endpoint": cfg.endpoint, "fallback": list(cfg.fallback)}


def service_models(store_path: Path) -> dict:
    cfg = load_config(config_path_for(Path(store_path)))
    gpu = detect_gpu()
    try:
        installed = set(ollama_installed(cfg.endpoint))
    except Exception:
        installed = set()
    tiers = [{"name": t.name, "default": t.default, "installed": t.default in installed}
             for t in TIERS]
    return {"gpu": {"backend": gpu.backend, "budget_mb": gpu.budget_mb},
            "recommended": pick_default(gpu), "tiers": tiers}
