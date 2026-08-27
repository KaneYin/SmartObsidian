"""Cross-encoder reranking: a second stage that re-scores retrieval candidates by
running a transformer over each (query, chunk) pair. Opt-in; the real backend loads a
sentence-transformers CrossEncoder lazily, the fake one is deterministic for tests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from weft.store import SearchHit

RERANK_POOL = 20  # candidates fetched before reranking


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        """Return the top-k hits re-scored against the query."""
        ...


class CrossEncoderReranker:
    """Real backend. Lazily loads a cross-encoder; scores (query, chunk text) pairs."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy: tests never load it

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.metadata.get("text", "")) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        return [SearchHit(score=float(scores[i]), metadata=hits[i].metadata)
                for i in order[:k]]


class FakeReranker:
    """Deterministic lexical-overlap reranker for tests; no model, no network."""

    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]:
        from weft.bm25 import _tokenize
        qt = set(_tokenize(query))
        scored = sorted(
            hits,
            key=lambda h: -len(qt & set(_tokenize(h.metadata.get("text", "")))),
        )
        return scored[:k]
