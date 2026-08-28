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


def service_memory_list(store_path: Path) -> dict:
    ms = make_memory(Path(store_path))
    return {"items": [{"id": i.id, "type": i.type, "text": i.text}
                      for i in ms.active_semantic()]}


def service_memory_pending(store_path: Path) -> dict:
    ps = make_proposals(Path(store_path))
    return {"proposals": [{"id": p.id, "type": p.type, "text": p.text}
                          for p in ps.pending()]}


def service_remember(store_path: Path, type: str, text: str) -> dict:
    if type not in SEMANTIC_TYPES:
        raise ValueError(f"type must be one of: {', '.join(sorted(SEMANTIC_TYPES))}")
    if not text or len(text) > MAX_MEMORY_TEXT:
        raise ValueError(f"text must be 1-{MAX_MEMORY_TEXT} characters")
    item = make_memory(Path(store_path)).remember(type, text)
    return {"id": item.id, "type": item.type, "text": item.text}


def service_memory_accept(store_path: Path, prop_id: str) -> dict:
    sp = Path(store_path)
    proposals, memory = make_proposals(sp), make_memory(sp)
    prop = proposals.get(prop_id)   # raises KeyError if absent
    memory.remember(prop.type, prop.text, provenance="inferred", source=prop.source)
    proposals.mark(prop.id, "accepted")
    return {"accepted": prop.id}


def service_memory_reject(store_path: Path, prop_id: str) -> dict:
    make_proposals(Path(store_path)).mark(prop_id, "rejected")  # KeyError if absent
    return {"rejected": prop_id}


def load_bm25(store_path: Path):
    from weft.bm25 import BM25Index
    from weft.index import bm25_path_for
    path = bm25_path_for(Path(store_path))
    return BM25Index.load(path) if path.exists() else None


def make_reranker():
    from weft.rerank import CrossEncoderReranker
    return CrossEncoderReranker()


def service_ask(store_path: Path, question: str, k: int = 5, *,
                use_graph: bool = True, use_memory: bool = True,
                overrides: dict | None = None, use_hybrid: bool = True,
                rerank: bool = False, use_memory_query: bool = False) -> dict:
    from weft.agent import ask, build_memory_query  # local import keeps langgraph off the read path
    if not isinstance(k, int) or not (1 <= k <= MAX_K):
        raise ValueError(f"k must be between 1 and {MAX_K}")
    if not question or not str(question).strip():
        raise ValueError("question must not be empty")
    sp = Path(store_path)
    if not sp.with_suffix(".npz").exists():
        raise ValueError("no index; run `weft index <vault>` first")
    store = VectorStore.load(sp)
    graph = None
    if use_graph:
        gpath = graph_path_for(sp)
        graph = LinkGraph.load(gpath) if gpath.exists() else None
    bm25 = load_bm25(sp) if use_hybrid else None
    raw = make_llm(overrides or {}, sp)         # may raise ProviderUnavailable
    llm = AuditedLLM(raw, sp.parent / "api-log.jsonl", "ask")
    memory = make_memory(sp) if use_memory else None
    reranker = make_reranker() if rerank else None
    mq = build_memory_query(memory) if use_memory_query else None
    result = ask(question, make_embedder(), store, llm, k=k, graph=graph,
                 memory=memory, bm25=bm25, reranker=reranker, memory_query=mq)
    return {"answer": result.answer, "sources": list(result.sources)}
