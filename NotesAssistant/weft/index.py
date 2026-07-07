"""Build the vector index from a vault: parse -> chunk -> embed -> store -> save."""

from __future__ import annotations

from pathlib import Path

from weft.embeddings import Embedder
from weft.parser import chunk_note, parse_vault
from weft.store import VectorStore


def build_index(vault_path: Path, embedder: Embedder, store_path: Path) -> int:
    """Index every chunk of every note. Returns the number of chunks indexed."""
    notes = parse_vault(Path(vault_path))
    # Keep each chunk paired with its note so note-level tags/wikilinks (the
    # seed of the M1 link graph) land in the chunk metadata.
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

    chunks = pairs

    store.save(Path(store_path))
    return len(chunks)
