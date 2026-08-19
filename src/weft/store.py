"""A minimal embedded vector store: numpy matrix + parallel metadata list,
cosine similarity for search, persisted as .npz + .json. Zero external
services — a single local file pair. Graduate to sqlite-vec/LanceDB later."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weft.security import secure_write_binary, secure_write_text


@dataclass
class SearchHit:
    score: float
    metadata: dict


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self._metadata)

    def add(self, vector: np.ndarray, metadata: dict) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(1, self.dim)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        self._vectors = np.vstack([self._vectors, v])
        self._metadata.append(metadata)

    def add_batch(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        if not metadatas:
            return
        v = np.asarray(vectors, dtype=np.float32).reshape(len(metadatas), self.dim)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        v = v / norms
        self._vectors = np.vstack([self._vectors, v]) if len(self) else v
        self._metadata.extend(metadatas)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        if len(self) == 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(self.dim)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self._vectors @ q  # cosine, both sides unit-norm
        k = min(k, len(self))
        top = np.argsort(-scores)[:k]
        return [SearchHit(score=float(scores[i]), metadata=self._metadata[i]) for i in top]

    def vectors_by_note(self) -> dict[str, np.ndarray]:
        """Chunk vectors grouped by note `rel_path`, each an (n_chunks, dim)
        array. Lets callers (e.g. suggest.note_vectors) mean-pool per note
        without reaching into the store's internal arrays."""
        grouped: dict[str, list[np.ndarray]] = {}
        for vec, meta in zip(self._vectors, self._metadata):
            grouped.setdefault(meta["rel_path"], []).append(vec)
        return {rp: np.vstack(vecs) for rp, vecs in grouped.items()}

    def metadata_by_note(self) -> dict[str, list[dict]]:
        """Chunk metadata grouped by note `rel_path`, preserving insertion
        order so callers can read tags/headings/text per note."""
        grouped: dict[str, list[dict]] = {}
        for meta in self._metadata:
            grouped.setdefault(meta["rel_path"], []).append(meta)
        return grouped

    def save(self, path: Path) -> None:
        path = Path(path)

        def write_npz(handle) -> None:
            np.savez(handle, vectors=self._vectors, dim=self.dim)

        secure_write_binary(path.with_suffix(".npz"), write_npz)
        secure_write_text(
            path.with_suffix(".json"),
            json.dumps(self._metadata, ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        path = Path(path)
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            dim = int(data["dim"])
            vectors = data["vectors"].astype(np.float32)
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if dim <= 0 or vectors.ndim != 2 or vectors.shape[1] != dim:
            raise ValueError("Invalid vector index dimensions")
        if not isinstance(metadata, list) or len(metadata) != len(vectors):
            raise ValueError("Vector index metadata does not match its vectors")
        store = cls(dim=dim)
        store._vectors = vectors
        store._metadata = metadata
        return store
