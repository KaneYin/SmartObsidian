"""Build the vector index AND the link graph from a vault:
parse -> chunk -> embed -> store; parse -> resolve links -> graph. Both persist
beside each other so a single `weft index` writes .npz/.json/.graph.json."""

from __future__ import annotations

import json
from pathlib import Path

from weft.embeddings import Embedder
from weft.graph import LinkGraph
from weft.parser import chunk_note, parse_vault
from weft.privacy import PrivacyPolicy
from weft.security import secure_write_text, vault_root
from weft.store import VectorStore


def graph_path_for(store_path: Path) -> Path:
    """The graph file sits next to the vector index: `.weft/index` ->
    `.weft/index.graph.json`."""
    return Path(str(store_path) + ".graph.json")


def manifest_path_for(store_path: Path) -> Path:
    """Security metadata bound to an index, including its canonical vault."""
    return Path(str(store_path) + ".manifest.json")


def build_index(
    vault_path: Path,
    embedder: Embedder,
    store_path: Path,
    policy: PrivacyPolicy | None = None,
) -> tuple[int, int]:
    """Index every chunk and build the link graph.
    Returns (n_chunks, n_edges)."""
    root = vault_root(vault_path)
    policy = policy or PrivacyPolicy()
    notes = parse_vault(root, policy=policy)
    pairs = [(note, c) for note in notes for c in chunk_note(note)]

    store = VectorStore(dim=embedder.dim)
    if pairs:
        vectors = embedder.embed([c.text for _, c in pairs])
        metadatas = [
            {
                "rel_path": c.rel_path,
                "heading": c.heading,
                "text": c.text,
                "ordinal": c.ordinal,
                "tags": note.tags,
                "wikilinks": note.wikilinks,
            }
            for note, c in pairs
        ]
        store.add_batch(vectors, metadatas)
    store.save(Path(store_path))

    graph = LinkGraph.from_notes(notes)
    graph.save(graph_path_for(store_path))

    manifest = {
        "version": 1,
        "vault_root": str(root),
        "privacy": policy.as_dict(),
    }
    secure_write_text(
        manifest_path_for(store_path),
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    return len(pairs), graph.edge_count()
