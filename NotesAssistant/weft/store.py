"""A minimal embedded vector store: numpy matrix + parallel metadata list,
cosine similarity for search, persisted as .npz + .json. Zero external
services — a single local file pair. Graduate to sqlite-vec/LanceDB later."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        for vec, meta in zip(vectors, metadatas):
            self.add(vec, meta)

    def search(self, query: np.ndarray, k: int = 5) -> list[SearchHit]:
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

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path.with_suffix(".npz"), vectors=self._vectors, dim=self.dim)
        path.with_suffix(".json").write_text(json.dumps(self._metadata), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        path = Path(path)
        data = np.load(path.with_suffix(".npz"))
        store = cls(dim=int(data["dim"]))
        store._vectors = data["vectors"].astype(np.float32)
        store._metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return store
