"""Weft command-line interface: secure `index`, `ask`, and `suggest` flows.

Backends are created via make_embedder()/make_llm() so tests can monkeypatch in
fakes. The default store path keeps private artifacts next to where you run."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from weft.config import (
    config_path_for,
    load_config,
)
from weft.embeddings import Embedder, SentenceTransformerEmbedder
from weft.hardware import detect_gpu
from weft.index import read_vault_root
from weft.llm import AuditedLLM, LLMClient, LLMRequestError
from weft.memory import SEMANTIC_TYPES, MemoryStore
from weft.memory_infer import infer_candidates
from weft.memory_mirror import render_mirror, write_mirror
from weft.models import TIERS, pick_default
from weft.ollama_client import list_models as ollama_installed
from weft.ollama_client import pull as ollama_pull
from weft.privacy import DEFAULT_EXCLUDES, PrivacyPolicy
from weft.proposals import ProposalStore
from weft.providers import ProviderUnavailable, resolve_llm
from weft.retrieval_config import RetrievalOverrides
from weft.security import WeftSecurityError, terminal_safe
from weft import service

DEFAULT_STORE = ".weft/index"
DEFAULT_THRESHOLD = 0.80
DEFAULT_LIMIT = 10
MAX_K = 50
MAX_LIMIT = 100
MAX_MEMORY_TEXT = 2000
DEFAULT_MEMORY_LIMIT = 10


def make_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


def _fallback_notice(primary: str, chosen: str, crossed: bool) -> None:
    if crossed:
        print(
            f"falling back to {chosen} — this sends note content off your machine "
            f"(configured in fallback).",
            file=sys.stderr,
        )
    else:
        print(f"primary {primary} unavailable; using fallback {chosen}.", file=sys.stderr)


def make_llm(overrides: dict, store_path: Path) -> LLMClient:
    """Resolve the configured provider into a raw LLM client. The caller wraps
    it in AuditedLLM. Raises ProviderUnavailable with an actionable remedy."""
    return resolve_llm(overrides, store_path=store_path, env=dict(os.environ),
                       on_fallback=_fallback_notice)


def make_memory(store_path: Path) -> MemoryStore:
    return MemoryStore(
        store_path.parent / "memory.jsonl",
        store_path.parent / "episodes.jsonl",
        vault_root=read_vault_root(store_path),
    )


def make_proposals(store_path: Path) -> ProposalStore:
    return ProposalStore(store_path.parent / "memory-proposals.jsonl")


def _llm_overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if getattr(args, "provider", None):
        out["provider"] = args.provider
    if getattr(args, "model", None):
        out["model"] = args.model
    return out


def _retrieval_overrides(args: argparse.Namespace) -> RetrievalOverrides:
    return RetrievalOverrides(
        k=getattr(args, "k", None),
        graph=getattr(args, "graph", None),
        hybrid=getattr(args, "hybrid", None),
        rerank=getattr(args, "rerank", None),
        rerank_pool=getattr(args, "rerank_pool", None),
        memory_query=getattr(args, "memory_query", None),
        query_rewrite=getattr(args, "rewrite_llm", None),
    )


def _bounded_int(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _cosine_threshold(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number") from exc
    if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be finite and between -1 and 1")
    return parsed


def _redaction_pattern(value: str) -> str:
    try:
        re.compile(value)
    except re.error as exc:
        raise argparse.ArgumentTypeError(
            f"invalid redaction regular expression: {exc}"
        ) from exc
    return value


def _cmd_index(args: argparse.Namespace) -> int:
    defaults = tuple(
        p for p in DEFAULT_EXCLUDES if not (args.include_private and p == "Private")
    )
    policy = PrivacyPolicy(
        includes=tuple(args.include),
        excludes=defaults + tuple(args.exclude),
        redaction_patterns=tuple(args.redact),
    )
    try:
        result = service.service_index(
            args.vault,
            args.store,
            policy=policy,
            chunking=args.chunking,
            no_bm25=args.no_bm25,
            contextual=args.contextual,
            force=args.force,
            overrides=_llm_overrides(args),
            embedder=make_embedder(),
            on_fallback=_fallback_notice,
        )
    except (ProviderUnavailable, service.IndexVaultMismatchError) as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
    print(
        f"Indexed {result['chunks']} chunks and {result['edges']} link edges from "
        f"{terminal_safe(args.vault)} -> {terminal_safe(args.store)}"
    )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        data = service.service_ask(
            Path(args.store),
            args.question,
            mode=args.mode,
            retrieval_overrides=_retrieval_overrides(args),
            use_memory=not args.no_memory,
            overrides=_llm_overrides(args),
            on_fallback=_fallback_notice,
        )
    except service.IndexNotFoundError:
        print(
            f"No index at {terminal_safe(args.store)}. Run `weft index <vault>` first.",
            file=sys.stderr,
        )
        return 1
    except (ProviderUnavailable, ValueError) as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1

    print(terminal_safe(data["answer"]))
    if data["sources"]:
        print("\nSources:")
        for s in data["sources"]:
            print(f"  - {terminal_safe(s)}")
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    from weft.chat import run_repl
    try:
        session = service.service_chat_session(
            args.store,
            mode=args.mode,
            retrieval_overrides=_retrieval_overrides(args),
            use_memory=not args.no_memory,
            overrides=_llm_overrides(args),
            on_fallback=_fallback_notice,
        )
    except service.IndexNotFoundError:
        print(
            f"No index at {terminal_safe(args.store)}. Run `weft index <vault>` first.",
            file=sys.stderr,
        )
        return 1
    except ProviderUnavailable as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
    return run_repl(session)


def _cmd_suggest(args: argparse.Namespace) -> int:
    try:
        result = service.service_suggest(
            args.vault,
            args.store,
            threshold=args.threshold,
            limit=args.limit,
            rationale=args.rationale,
            overwrite_inbox=args.overwrite_inbox,
            overrides=_llm_overrides(args),
            llm_factory=make_llm,
        )
    except service.IndexNotFoundError:
        print(
            f"No index at {terminal_safe(args.store)}. Run `weft index <vault>` first.",
            file=sys.stderr,
        )
        return 1
    except service.InboxExistsError:
        print(
            "Inbox already exists; review or move it, or pass --overwrite-inbox explicitly.",
            file=sys.stderr,
        )
        return 1
    except (ProviderUnavailable, service.IndexVaultMismatchError, ValueError) as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
    if result["warning"]:
        print(terminal_safe(result["warning"]), file=sys.stderr)
    n = result["count"]
    if not n:
        print("0 suggestions; existing inbox left unchanged")
        return 0
    plural = "suggestion" if n == 1 else "suggestions"
    print(f"{n} {plural} -> {terminal_safe(result['inbox'])}")
    return 0


def _cmd_remember(args: argparse.Namespace) -> int:
    try:
        item = service.service_remember(Path(args.store), args.type, args.text)
    except ValueError as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 2
    print(f"remembered [{item['type']}] {item['id']}: {terminal_safe(item['text'])}")
    return 0


def _cmd_memory(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    memory = make_memory(store_path)
    proposals = make_proposals(store_path)

    if args.action == "list":
        for item in service.service_memory_list(store_path)["items"]:
            print(f"{item['id']}  [{item['type']}]  {terminal_safe(item['text'])}")
        return 0
    if args.action == "show":
        try:
            item = memory.get(args.id)
        except KeyError:
            print(f"no memory with id {terminal_safe(str(args.id))}", file=sys.stderr)
            return 1
        print(f"{item.id} [{item.type}] status={item.status}")
        print(terminal_safe(item.text))
        return 0
    if args.action == "forget":
        try:
            memory.reject(args.id)
        except KeyError:
            print(f"no memory with id {terminal_safe(str(args.id))}", file=sys.stderr)
            return 1
        print(f"forgot {args.id}")
        return 0
    if args.action == "compact":
        memory.compact()
        print("compacted memory")
        return 0
    if args.action == "suggest":
        llm = None
        if args.llm:
            try:
                llm = AuditedLLM(
                    make_llm(_llm_overrides(args), store_path),
                    store_path.parent / "api-log.jsonl", "memory_suggest",
                )
            except ProviderUnavailable as exc:
                print(f"--llm unavailable ({terminal_safe(exc)}); using heuristic.",
                      file=sys.stderr)
        existing = {i.text for i in memory.active_semantic()}
        candidates = infer_candidates(
            memory.episodes(), existing_texts=existing,
            seen_ids=proposals.known_ids(), limit=args.limit, llm=llm,
        )
        added = proposals.add(candidates)
        n = len(added)
        print(f"{n} {'proposal' if n == 1 else 'proposals'} "
              f"(run `weft memory pending` to review)")
        return 0
    if args.action == "pending":
        for prop in service.service_memory_pending(store_path)["proposals"]:
            print(f"{prop['id']}  [{prop['type']}]  {terminal_safe(prop['text'])}")
        return 0
    if args.action == "accept":
        try:
            service.service_memory_accept(store_path, args.id)
        except KeyError:
            print(f"no proposal with id {terminal_safe(str(args.id))}", file=sys.stderr)
            return 1
        print(f"accepted {args.id} -> memory")
        return 0
    if args.action == "reject":
        try:
            service.service_memory_reject(store_path, args.id)
        except KeyError:
            print(f"no proposal with id {terminal_safe(str(args.id))}", file=sys.stderr)
            return 1
        print(f"rejected {args.id}")
        return 0
    # mirror
    if not args.vault:
        print("memory mirror requires --vault <path>", file=sys.stderr)
        return 2
    text = render_mirror(memory.active_semantic(), proposals.pending(), datetime.now())
    path = write_mirror(Path(args.vault), text)
    print(f"wrote {terminal_safe(path)}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    path = config_path_for(Path(args.store))
    if args.action == "show":
        cfg = service.service_config(Path(args.store))
        print(f"provider = {cfg['provider']}\nmodel = {cfg['model']}\n"
              f"mode = {cfg['mode']}\nendpoint = {cfg['endpoint']}")
        print(f"fallback = {', '.join(cfg['fallback']) or '(none)'}")
        return 0
    if args.action == "path":
        print(path)
        return 0
    try:
        result = service.service_config_set(
            Path(args.store), str(args.key or ""), str(args.value or "")
        )
    except ValueError as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 2
    rendered = result["value"]
    if isinstance(rendered, list):
        rendered = ", ".join(rendered) or "(none)"
    print(f"{result['key']} = {terminal_safe(str(rendered))}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    cfg = load_config(config_path_for(Path(args.store)))
    if args.action in ("list", "show"):
        data = service.service_models(Path(args.store))
    if args.action == "list":
        installed = {t["default"] for t in data["tiers"] if t["installed"]}
        for tier in data["tiers"]:
            flags = []
            if tier["default"] == data["recommended"]:
                flags.append("recommended")
            if tier["default"] in installed:
                flags.append("installed")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(f"{tier['name']:6} {tier['default']}{suffix}")
        return 0
    if args.action == "show":
        print(f"gpu: {data['gpu']['backend']} budget={data['gpu']['budget_mb']}MB")
        print(f"recommended: {data['recommended']}")
        print(f"configured model: {cfg.model}")
        return 0
    # pull
    tag = args.tag or pick_default(detect_gpu())
    if not args.yes:
        answer = input(f"Pull {tag!r} via Ollama? This downloads several GB. [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print(f"Skipped. To pull manually: ollama pull {tag}")
            return 0
    ollama_pull(cfg.endpoint, tag)
    print(f"Pulled {terminal_safe(tag)}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from weft.api import WeftHTTPServer, load_or_create_token
    store_path = Path(args.store)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        print("--host must be a loopback address (127.0.0.1, ::1, localhost)",
              file=sys.stderr)
        return 2
    token = load_or_create_token(store_path)
    server = WeftHTTPServer((args.host, args.port), store_path, token)
    host, port = server.server_address[0], server.server_address[1]
    print(f"Weft API on http://{host}:{port}  (token in "
          f"{terminal_safe(store_path.parent / 'api-token')})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from weft.benchmarks.crag import CRAGModel
    from weft.benchmarks.crag_runner import run_crag

    try:
        model = CRAGModel(
            store_path=args.store,
            batch_size=args.batch_size,
            k=args.k,
            use_hybrid=not args.no_hybrid,
            use_rewrite=args.rewrite_llm,
            use_reranker=args.reranker,
            reranker_model=args.reranker_model,
            chunk_chars=args.chunk_chars,
            overlap=args.overlap,
            rerank_pool=args.rerank_pool,
            system_prompt=args.system_prompt,
        )
        result = run_crag(
            args.dataset,
            model,
            output_path=args.output,
            limit=args.limit,
            judge=not args.generation_only,
            on_progress=lambda count: (
                print(f"generated {count} CRAG answers", file=sys.stderr)
                if count % 10 == 0 else None
            ),
        )
    except (OSError, ProviderUnavailable, ValueError) as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
    print(f"CRAG examples: {result['total']}")
    print(f"Results: {terminal_safe(result['output'])}")
    if result["judge"]:
        print("Judge: Ollama local approximation (not an official comparable score)")
        print(
            f"Score: {result['score']:.4f}  accuracy: {result['accuracy']:.4f}  "
            f"hallucination: {result['hallucination']:.4f}  "
            f"missing: {result['missing']:.4f}"
        )
    else:
        print("Generation only; no score computed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weft", description="Local-first agent over your Obsidian vault."
    )
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Index a vault into a local vector store.")
    p_index.add_argument("vault", help="Path to the Obsidian vault directory.")
    p_index.add_argument(
        "--store", default=DEFAULT_STORE, help="Index path (default: .weft/index)."
    )
    p_index.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATH",
        help="Allow only this relative vault path (repeatable).",
    )
    p_index.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATH",
        help="Exclude this relative vault path in addition to secure defaults (repeatable).",
    )
    p_index.add_argument(
        "--include-private",
        action="store_true",
        help="Explicitly include Private/; its retrieved content may be sent to Claude.",
    )
    p_index.add_argument(
        "--redact",
        action="append",
        default=[],
        type=_redaction_pattern,
        metavar="REGEX",
        help="Replace matching content with [REDACTED] before embedding (repeatable).",
    )
    p_index.add_argument(
        "--chunking", choices=["heading", "parent-child", "sliding"], default="heading",
        help="Chunking strategy: heading (default), parent-child, or sliding.",
    )
    p_index.add_argument(
        "--contextual", action="store_true",
        help="Prepend an LLM-written context sentence to each chunk's embedding "
             "(payload-logged; free/offline on a local provider).",
    )
    p_index.add_argument("--provider", help="Provider override for --contextual.")
    p_index.add_argument("--model", help="Model override for --contextual.")
    p_index.add_argument(
        "--no-bm25", action="store_true",
        help="Skip building the BM25 lexical index (disables hybrid search).",
    )
    p_index.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing store even if it belongs to a different vault.",
    )
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question over the indexed vault.")
    p_ask.add_argument("question", help="Your question, in quotes.")
    p_ask.add_argument(
        "--store", default=DEFAULT_STORE, help="Index path (default: .weft/index)."
    )
    p_ask.add_argument(
        "--k",
        type=_bounded_int("k", 1, MAX_K),
        default=None,
        help=f"Override the mode's chunk count (1-{MAX_K}).",
    )
    p_ask.add_argument(
        "--mode", choices=["fast", "balanced", "best"], default=None,
        help="Retrieval preset (default: configured mode, initially balanced).",
    )
    p_ask.add_argument(
        "--no-memory", action="store_true",
        help="Do not read or write agent memory for this question.",
    )
    p_ask.set_defaults(func=_cmd_ask)

    for provider_sub in (p_ask,):
        provider_sub.add_argument(
            "--provider", help="Override the configured provider (ollama|anthropic|fake)."
        )
        provider_sub.add_argument(
            "--model", help="Override the configured model tag."
        )

    p_suggest = sub.add_parser(
        "suggest",
        help="Suggest inferred links between semantically-close, unlinked notes -> _inbox.md.",
    )
    p_suggest.add_argument("vault", help="Path to the vault; `_inbox.md` is written here.")
    p_suggest.add_argument(
        "--store", default=DEFAULT_STORE, help="Index path (default: .weft/index)."
    )
    p_suggest.add_argument(
        "--threshold", type=_cosine_threshold, default=DEFAULT_THRESHOLD,
        help="Minimum cosine for a pair to be a candidate (default: 0.80).",
    )
    p_suggest.add_argument(
        "--limit", type=_bounded_int("limit", 0, MAX_LIMIT), default=DEFAULT_LIMIT,
        help=f"Attention budget: max suggestions per run (0-{MAX_LIMIT}; default: 10).",
    )
    p_suggest.add_argument(
        "--rationale", action="store_true",
        help="Opt-in: batched Claude rationale (else a local template). Payload-logged.",
    )
    p_suggest.add_argument(
        "--overwrite-inbox",
        action="store_true",
        help="Explicitly replace an existing regular _inbox.md; symlinks are always refused.",
    )
    p_suggest.add_argument(
        "--provider", help="Override the configured provider for --rationale."
    )
    p_suggest.add_argument(
        "--model", help="Override the configured model tag for --rationale."
    )
    p_suggest.set_defaults(func=_cmd_suggest)

    p_config = sub.add_parser("config", help="Show or set provider/model config.")
    p_config.add_argument("action", choices=["show", "set", "path"])
    p_config.add_argument("key", nargs="?", help="Config key for `set`.")
    p_config.add_argument("value", nargs="?", help="Config value for `set`.")
    p_config.add_argument("--store", default=DEFAULT_STORE)
    p_config.set_defaults(func=_cmd_config)

    p_models = sub.add_parser("models", help="List/inspect/pull local models.")
    p_models.add_argument("action", choices=["list", "show", "pull"])
    p_models.add_argument("tag", nargs="?", help="Model tag for `pull`.")
    p_models.add_argument(
        "--yes", action="store_true", help="Skip the pull confirmation."
    )
    p_models.add_argument("--store", default=DEFAULT_STORE)
    p_models.set_defaults(func=_cmd_models)

    p_remember = sub.add_parser("remember", help="Store a durable memory item.")
    p_remember.add_argument("text", help="What to remember, in quotes.")
    p_remember.add_argument(
        "--type", default="fact",
        help="preference | fact | decision | task (default: fact).",
    )
    p_remember.add_argument("--store", default=DEFAULT_STORE)
    p_remember.set_defaults(func=_cmd_remember)

    p_memory = sub.add_parser("memory", help="Inspect/curate/propose memory.")
    p_memory.add_argument(
        "action",
        choices=["list", "show", "forget", "compact",
                 "suggest", "pending", "accept", "reject", "mirror"],
    )
    p_memory.add_argument(
        "id", nargs="?", help="Memory/proposal id for show/forget/accept/reject."
    )
    p_memory.add_argument("--vault", help="Vault path for `mirror`.")
    p_memory.add_argument(
        "--llm", action="store_true",
        help="Opt-in LLM extraction for `suggest` (payload-logged; falls back to heuristic).",
    )
    p_memory.add_argument(
        "--limit", type=_bounded_int("limit", 0, MAX_LIMIT), default=DEFAULT_MEMORY_LIMIT,
        help=f"Max proposals per suggest run (0-{MAX_LIMIT}; default: 10).",
    )
    p_memory.add_argument("--provider", help="Provider override for --llm.")
    p_memory.add_argument("--model", help="Model override for --llm.")
    p_memory.add_argument("--store", default=DEFAULT_STORE)
    p_memory.set_defaults(func=_cmd_memory)

    p_chat = sub.add_parser("chat", help="Interactive multi-turn chat over your vault.")
    p_chat.add_argument("--store", default=DEFAULT_STORE)
    p_chat.add_argument("--k", type=_bounded_int("k", 1, MAX_K), default=None,
                        help=f"Override the mode's chunk count (1-{MAX_K}).")
    p_chat.add_argument(
        "--mode", choices=["fast", "balanced", "best"], default=None,
        help="Retrieval preset (default: configured mode, initially balanced).",
    )
    p_chat.add_argument("--no-memory", action="store_true",
                        help="Do not read or write agent memory.")
    p_chat.add_argument("--provider", help="Override the configured provider.")
    p_chat.add_argument("--model", help="Override the configured model tag.")
    p_chat.set_defaults(func=_cmd_chat)

    for hybrid_sub in (p_ask, p_chat):
        graph_group = hybrid_sub.add_mutually_exclusive_group()
        graph_group.add_argument(
            "--graph", dest="graph", action="store_true", default=None,
            help="Enable graph-aware retrieval (overrides mode).",
        )
        graph_group.add_argument(
            "--no-graph", dest="graph", action="store_false",
            help="Disable graph-aware retrieval (overrides mode).",
        )
        hybrid_group = hybrid_sub.add_mutually_exclusive_group()
        hybrid_group.add_argument(
            "--hybrid", dest="hybrid", action="store_true", default=None,
            help="Enable BM25 hybrid fusion (overrides mode).",
        )
        hybrid_group.add_argument(
            "--no-hybrid", dest="hybrid", action="store_false",
            help="Disable BM25 hybrid fusion (overrides mode).",
        )
        rerank_group = hybrid_sub.add_mutually_exclusive_group()
        rerank_group.add_argument(
            "--rerank", dest="rerank", action="store_true", default=None,
            help="Enable cross-encoder reranking (overrides mode; may download a model).",
        )
        rerank_group.add_argument(
            "--no-rerank", dest="rerank", action="store_false",
            help="Disable cross-encoder reranking (overrides mode).",
        )
        memory_query_group = hybrid_sub.add_mutually_exclusive_group()
        memory_query_group.add_argument(
            "--memory-query", dest="memory_query", action="store_true", default=None,
            help="Enable durable-memory query fusion (overrides mode).",
        )
        memory_query_group.add_argument(
            "--no-memory-query", dest="memory_query", action="store_false",
            help="Disable durable-memory query fusion (overrides mode).",
        )
        hybrid_sub.add_argument(
            "--rerank-pool", type=_bounded_int("rerank-pool", 1, 500), default=None,
            help="Override candidates fetched before reranking (1-500).",
        )

    rewrite_group = p_chat.add_mutually_exclusive_group()
    rewrite_group.add_argument(
        "--rewrite-llm", dest="rewrite_llm", action="store_true", default=None,
        help="Enable conversation-aware LLM query rewriting.",
    )
    rewrite_group.add_argument(
        "--no-rewrite-llm", dest="rewrite_llm", action="store_false",
        help="Disable conversation-aware LLM query rewriting.",
    )

    p_serve = sub.add_parser("serve", help="Run the local HTTP API for a GUI.")
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Loopback host (default 127.0.0.1).")
    p_serve.add_argument("--port", type=_bounded_int("port", 1, 65535), default=8765,
                         help="Port (default 8765).")
    p_serve.add_argument("--store", default=DEFAULT_STORE)
    p_serve.set_defaults(func=_cmd_serve)

    p_benchmark = sub.add_parser(
        "benchmark", help="Run an external RAG benchmark with local Ollama."
    )
    p_benchmark.add_argument("benchmark", choices=["crag"])
    p_benchmark.add_argument("dataset", help="CRAG .jsonl or .jsonl.bz2 dataset path.")
    p_benchmark.add_argument("--output", default=".weft/crag-results.jsonl")
    p_benchmark.add_argument("--store", default=DEFAULT_STORE,
                             help="Path used to read Weft's Ollama configuration.")
    p_benchmark.add_argument("--limit", type=_bounded_int("limit", 1, 1_000_000))
    p_benchmark.add_argument("--batch-size", type=_bounded_int("batch-size", 1, 16))
    p_benchmark.add_argument("--k", type=_bounded_int("k", 1, MAX_K), default=8)
    p_benchmark.add_argument(
        "--generation-only", action="store_true",
        help="Generate predictions without the approximate local Ollama judge.",
    )
    p_benchmark.add_argument(
        "--no-hybrid", action="store_true",
        help="Disable BM25 hybrid fusion in benchmark retrieval.",
    )
    p_benchmark.add_argument(
        "--reranker", action="store_true",
        help="Enable CrossEncoder reranking stage in benchmark retrieval.",
    )
    p_benchmark.add_argument(
        "--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="CrossEncoder model name for reranking.",
    )
    p_benchmark.add_argument(
        "--rewrite-llm", action="store_true",
        help="Enable LLM query rewrite with RRF in benchmark retrieval.",
    )
    p_benchmark.add_argument(
        "--chunk-chars", type=_bounded_int("chunk-chars", 100, 10000),
        help="Chunk character limit for web page chunking.",
    )
    p_benchmark.add_argument(
        "--overlap", type=_bounded_int("overlap", 0, 5000),
        help="Chunk character overlap for web page chunking.",
    )
    p_benchmark.add_argument(
        "--rerank-pool", type=_bounded_int("rerank-pool", 1, 500),
        help="Number of top candidates fetched prior to reranking.",
    )
    p_benchmark.add_argument(
        "--system-prompt",
        help="Custom system prompt override for CRAG generation.",
    )
    p_benchmark.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        from weft.tui import run_tui
        return run_tui(store_path=DEFAULT_STORE)
    try:
        return args.func(args)
    except LLMRequestError as exc:
        print(f"Model request error: {terminal_safe(exc)}", file=sys.stderr)
        return 1
    except (WeftSecurityError, ValueError) as exc:
        print(f"Security or validation error: {terminal_safe(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
