"""Local embedding backends. The real one downloads a sentence-transformers
model; the fake one is deterministic and dependency-free for tests."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of unit-norm vectors."""
        ...


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (v / norms).astype(np.float32)


class FakeEmbedder:
    """Deterministic hash-based embedder — no model, no network. Same text
    always maps to the same unit vector; different text maps elsewhere."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim)
        return _normalize(out)


class SentenceTransformerEmbedder:
    """Real local embedder using sentence-transformers (default all-MiniLM-L6-v2,
    384-dim). Imported lazily so tests never load torch."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)
