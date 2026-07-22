"""Weft command-line interface: `weft index <vault>` and `weft ask "<q>"`.

Backends are created via make_embedder()/make_llm() so tests can monkeypatch
in the fakes. Default store path keeps the index next to where you run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weft.embeddings import Embedder, SentenceTransformerEmbedder
from weft.graph import LinkGraph
from weft.index import build_index, graph_path_for
from weft.llm import ClaudeClient, LLMClient
from weft.store import VectorStore

DEFAULT_STORE = ".weft/index"


def make_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


def make_llm() -> LLMClient:
    return ClaudeClient()


def _cmd_index(args: argparse.Namespace) -> int:
    n_chunks, n_edges = build_index(Path(args.vault), make_embedder(), Path(args.store))
    print(f"Indexed {n_chunks} chunks and {n_edges} link edges from {args.vault} -> {args.store}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from weft.agent import ask  # local import: keeps langgraph out of `index` path

    store_path = Path(args.store)
    if not store_path.with_suffix(".npz").exists():
        print(f"No index at {args.store}. Run `weft index <vault>` first.", file=sys.stderr)
        return 1

    store = VectorStore.load(store_path)

    link_graph = None
    if not args.no_graph:
        gpath = graph_path_for(store_path)
        if gpath.exists():
            link_graph = LinkGraph.load(gpath)

    result = ask(args.question, make_embedder(), store, make_llm(), k=args.k, graph=link_graph)

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for s in result.sources:
            print(f"  - {s}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weft", description="Local-first agent over your Obsidian vault.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index a vault into a local vector store.")
    p_index.add_argument("vault", help="Path to the Obsidian vault directory.")
    p_index.add_argument("--store", default=DEFAULT_STORE, help="Index path (default: .weft/index).")
    p_index.set_defaults(func=_cmd_index)

    p_ask = sub.add_parser("ask", help="Ask a question over the indexed vault.")
    p_ask.add_argument("question", help="Your question, in quotes.")
    p_ask.add_argument("--store", default=DEFAULT_STORE, help="Index path (default: .weft/index).")
    p_ask.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve.")
    p_ask.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph-aware retrieval; use pure vector search (for A/B comparison).",
    )
    p_ask.set_defaults(func=_cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
