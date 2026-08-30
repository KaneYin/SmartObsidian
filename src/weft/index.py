"""Build the vector index AND the link graph from a vault:
parse -> chunk -> embed -> store; parse -> resolve links -> graph. Both persist
beside each other so a single `weft index` writes .npz/.json/.graph.json."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from weft.bm25 import BM25Index
from weft.chunking import chunk_strategy, contextualize
from weft.embeddings import Embedder
from weft.graph import LinkGraph
from weft.parser import Chunk, Note, parse_vault
from weft.privacy import PrivacyPolicy
from weft.security import secure_write_text, vault_root
from weft.store import VectorStore


def graph_path_for(store_path: Path) -> Path:
    """The graph file sits next to the vector index: `.weft/index` ->
    `.weft/index.graph.json`."""
    return Path(str(store_path) + ".graph.json")


def bm25_path_for(store_path: Path) -> Path:
    """`.weft/index` -> `.weft/index.bm25.json` (lexical index beside the vectors)."""
    return Path(str(store_path) + ".bm25.json")


def manifest_path_for(store_path: Path) -> Path:
    """Security metadata bound to an index, including its canonical vault."""
    return Path(str(store_path) + ".manifest.json")


def read_vault_root(store_path: Path) -> str:
    """The vault_root recorded in this store's manifest, or "" if the store
    has no manifest yet (fresh index, or a pre-manifest one)."""
    path = manifest_path_for(store_path)
    if not path.exists():
        return ""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("vault_root") or ""


# Real rel_paths always end in ".md" (see parse_vault below), so this
# sentinel can never collide with an actual vault note.
OVERVIEW_REL_PATH = "(vault overview)"


def build_overview_chunk(notes: list[Note]) -> Chunk | None:
    """A synthetic chunk summarizing vault composition -- folder and tag
    breakdown -- so a broad question like 'what is this vault about' has a
    real target to retrieve instead of relying on nearest-neighbor luck
    over individual note chunks. Built from already privacy-filtered notes,
    so it never surfaces excluded content."""
    if not notes:
        return None
    folder_counts = Counter(
        note.rel_path.split("/", 1)[0] if "/" in note.rel_path else "(root)"
        for note in notes
    )
    tag_counts = Counter(tag for note in notes for tag in note.tags)
    folders = ", ".join(
        f"{name}/ ({n} notes)" for name, n in folder_counts.most_common()
    )
    lines = [
        f"Vault overview -- {len(notes)} notes across {len(folder_counts)} "
        f"top-level folders.",
        f"Folders: {folders}.",
    ]
    if tag_counts:
        tags = ", ".join(f"#{tag} ({n})" for tag, n in tag_counts.most_common(15))
        lines.append(f"Most common tags: {tags}.")
    return Chunk(
        rel_path=OVERVIEW_REL_PATH,
        heading="Vault overview",
        text="\n".join(lines),
        ordinal=-1,
    )


def build_index(
    vault_path: Path,
    embedder: Embedder,
    store_path: Path,
    policy: PrivacyPolicy | None = None,
    chunking: str = "heading",
    no_bm25: bool = False,
    contextual_llm=None,
) -> tuple[int, int]:
    """Index every chunk and build the link graph.
    Returns (n_chunks, n_edges)."""
    root = vault_root(vault_path)
    policy = policy or PrivacyPolicy()
    notes = parse_vault(root, policy=policy)
    strategy = chunk_strategy(chunking)
    pairs = []
    for note in notes:
        chunks = strategy(note)
        if contextual_llm is not None:
            chunks = contextualize(note, chunks, contextual_llm)
        pairs.extend((note, c) for c in chunks)

    overview_chunk = build_overview_chunk(notes)
    if overview_chunk is not None:
        overview_note = Note(
            rel_path=OVERVIEW_REL_PATH, title="Vault overview",
            frontmatter={}, tags=[], wikilinks=[], body="",
        )
        pairs.append((overview_note, overview_chunk))

    store = VectorStore(dim=embedder.dim)
    if pairs:
        vectors = embedder.embed([(c.embed_text or c.text) for _, c in pairs])
        metadatas = []
        for note, c in pairs:
            meta = {
                "rel_path": c.rel_path,
                "heading": c.heading,
                "text": c.text,
                "ordinal": c.ordinal,
                "tags": note.tags,
                "wikilinks": note.wikilinks,
            }
            if c.parent_id is not None:
                meta["parent_id"] = c.parent_id
            if c.parent_text is not None:
                meta["parent_text"] = c.parent_text
            metadatas.append(meta)
        store.add_batch(vectors, metadatas)
    store.save(Path(store_path))

    if not no_bm25 and pairs:
        BM25Index.build([c.text for _, c in pairs]).save(bm25_path_for(store_path))

    graph = LinkGraph.from_notes(notes)
    graph.save(graph_path_for(store_path))

    manifest = {
        "version": 1,
        "vault_root": str(root),
        "privacy": policy.as_dict(),
        "chunking": chunking,
        "contextual": contextual_llm is not None,
    }
    secure_write_text(
        manifest_path_for(store_path),
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    return len(pairs), graph.edge_count()
