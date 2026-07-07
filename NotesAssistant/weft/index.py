"""Build the vector index from a vault: parse -> chunk -> embed -> store -> save."""

from __future__ import annotations

from pathlib import Path

from weft.embeddings import Embedder
from weft.parser import chunk_note, parse_vault
from weft.store import VectorStore


def build_index(vault_path: Path, embedder: Embedder, store_path: Path) -> int:
    """Index every chunk of every note. Returns the number of chunks indexed."""
    notes = parse_vault(Path(vault_path))
    chunks = [c for note in notes for c in chunk_note(note)]

    store = VectorStore(dim=embedder.dim)
    if chunks:
        vectors = embedder.embed([c.text for c in chunks])
        metadatas = [
            {"rel_path": c.rel_path, "heading": c.heading, "text": c.text, "ordinal": c.ordinal}
            for c in chunks
        ]
        store.add_batch(vectors, metadatas)

    store.save(Path(store_path))
    return len(chunks)
