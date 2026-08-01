"""Weft command-line interface: `weft index <vault>` and `weft ask "<q>"`.

Backends are created via make_embedder()/make_llm() so tests can monkeypatch
in the fakes. Default store path keeps the index next to where you run."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from weft.embeddings import Embedder, SentenceTransformerEmbedder
from weft.graph import LinkGraph
from weft.index import build_index, graph_path_for
from weft.inbox import render_inbox, write_inbox
from weft.ledger import load_seen, record
from weft.llm import ClaudeClient, LLMClient
from weft.store import VectorStore
from weft.suggest import LinkSuggestion, infer_links

DEFAULT_STORE = ".weft/index"
DEFAULT_THRESHOLD = 0.80
DEFAULT_LIMIT = 10


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


def _apply_claude_rationale(
    suggestions: list[LinkSuggestion], llm: LLMClient, log_path: Path
) -> None:
    """Opt-in: replace each suggestion's local rationale with a Claude-written
    one-liner, in a single batched pass. Logs the payload sent (privacy is
    auditable) and degrades gracefully to the local rationale on any failure."""
    system = (
        "You explain why two notes might be worth linking. For each numbered "
        "pair, reply with exactly one line: the pair number, a colon, then a "
        "short reason. Keep each reason under 20 words."
    )
    lines = [
        f"{i}. {s.note_a} <-> {s.note_b} (cosine {s.score:.2f}, "
        f"shared tags: {', '.join(s.shared_tags) or 'none'})"
        for i, s in enumerate(suggestions, start=1)
    ]
    prompt = "Note pairs:\n" + "\n".join(lines)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "purpose": "suggest_rationale",
                    "system": system,
                    "prompt": prompt,
                    "n_pairs": len(suggestions),
                }
            )
            + "\n"
        )

    try:
        response = llm.complete(system=system, prompt=prompt)
    except Exception as e:  # no API key, network, etc. — keep local rationale
        print(f"--rationale unavailable ({e}); using local rationale.", file=sys.stderr)
        return

    # Positional pairing: reply line i is assumed to be suggestion i (the
    # system prompt asks for exactly one ordered line per pair). A short/reordered
    # reply just leaves the affected suggestions on their local rationale.
    reply_lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    for s, line in zip(suggestions, reply_lines):
        # Strip an optional leading "1." / "1:" numbering the model may echo.
        _, sep, rest = line.partition(":")
        s.rationale = (rest.strip() if sep else line).strip() or s.rationale


def _cmd_suggest(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    if not store_path.with_suffix(".npz").exists():
        print(f"No index at {args.store}. Run `weft index <vault>` first.", file=sys.stderr)
        return 1

    store = VectorStore.load(store_path)

    gpath = graph_path_for(store_path)
    graph = LinkGraph.load(gpath) if gpath.exists() else LinkGraph()

    ledger_path = store_path.parent / "suggestions.jsonl"
    seen = load_seen(ledger_path)

    suggestions = infer_links(
        store,
        graph,
        threshold=args.threshold,
        limit=args.limit,
        seen_pairs=seen,
    )

    if suggestions and args.rationale:
        _apply_claude_rationale(suggestions, make_llm(), store_path.parent / "api-log.jsonl")

    text = render_inbox(suggestions, datetime.now())
    inbox_path = write_inbox(Path(args.vault), text)
    record(ledger_path, suggestions)  # no-op when empty; leaves ledger untouched

    n = len(suggestions)
    plural = "suggestion" if n == 1 else "suggestions"
    print(f"{n} {plural} -> {inbox_path}")
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

    p_suggest = sub.add_parser(
        "suggest",
        help="Suggest inferred links between semantically-close, unlinked notes -> _inbox.md.",
    )
    p_suggest.add_argument("vault", help="Path to the vault; `_inbox.md` is written here.")
    p_suggest.add_argument("--store", default=DEFAULT_STORE, help="Index path (default: .weft/index).")
    p_suggest.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Minimum cosine for a pair to be a candidate (default: 0.80).",
    )
    p_suggest.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help="Attention budget: max suggestions written per run (default: 10).",
    )
    p_suggest.add_argument(
        "--rationale", action="store_true",
        help="Opt-in: batched Claude rationale (else a local template). Payload-logged.",
    )
    p_suggest.set_defaults(func=_cmd_suggest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
