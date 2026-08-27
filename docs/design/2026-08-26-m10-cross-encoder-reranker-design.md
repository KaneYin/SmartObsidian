# Weft M10 — Cross-Encoder Reranker — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-26.
**Scope:** An opt-in cross-encoder reranking stage that re-scores the fused retrieval
candidates for precision. Final stage of the retrieval pipeline (M8 rewrite → M9
hybrid → M10 rerank).

## 1. Problem & goal

Bi-encoder retrieval (vector + BM25) is fast but approximate. A cross-encoder scores
each `(query, chunk)` pair jointly and is far more accurate, at the cost of a
transformer pass per candidate — so it is a reranking stage: retrieve a larger pool
cheaply, then re-score to the final top-k. Opt-in with `--rerank`.

## 2. Decisions (locked during brainstorming)

- **Pluggable `Reranker` (real + fake), opt-in `--rerank`.** Mirrors the
  `Embedder`/`LLMClient` split: `CrossEncoderReranker` (lazy sentence-transformers
  CrossEncoder, offline after first download) and `FakeReranker` (deterministic, no
  model) for network-free tests. Default off.
- **Fixed candidate pool of 20**, capped by store size; rerank to `k`.
- **Rerank query = full conversational intent.** `question` for `ask`; for chat,
  `f"{context}\n{question}"` when there is history (concatenation is fine — a
  cross-encoder reads the text, no centroid to blur).

## 3. Components

```
src/weft/
  rerank.py  NEW   — Reranker protocol; CrossEncoderReranker; FakeReranker.
  agent.py   TOUCH — build_graph/ask gain reranker + rerank_pool; _retrieve fetches
                     the pool then reranks.
  chat.py    TOUCH — ChatSession carries reranker; reranks against the conversational
                     query.
  service.py TOUCH — make_reranker(); service_ask(rerank=False).
  cli.py     TOUCH — `weft ask --rerank`; `weft chat --rerank`.
```

## 4. Reranker (`rerank.py`)

```python
from typing import Protocol, runtime_checkable

RERANK_POOL = 20  # candidates fetched before reranking

@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], k: int) -> list[SearchHit]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # lazy: tests never load it
        self._model = CrossEncoder(model_name)

    def rerank(self, query, hits, k):
        if not hits:
            return []
        scores = self._model.predict([(query, h.metadata.get("text", "")) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        return [SearchHit(score=float(scores[i]), metadata=hits[i].metadata)
                for i in order[:k]]


class FakeReranker:
    """Deterministic lexical-overlap reranker for tests; no model."""
    def rerank(self, query, hits, k):
        from weft.bm25 import _tokenize
        qt = set(_tokenize(query))
        scored = sorted(hits, key=lambda h: -len(qt & set(_tokenize(h.metadata.get("text", "")))))
        return scored[:k]
```

The cross-encoder scores each candidate's own chunk text (`metadata["text"]`) — the
precise unit that matched. M7 parent-dedup still runs later in `build_prompt`, so the
model reads the parent section for the surviving top-k.

## 5. Agent integration (`agent.py`)

`build_graph` and `ask` gain `reranker=None` and `rerank_pool: int = RERANK_POOL`. The
retrieve node:

```python
def _retrieve(state):
    q, k = state["question"], state.get("k", 5)
    if reranker is not None:
        pool = fused_retrieve([q], embedder, store, bm25=bm25, graph=link_graph,
                              k=rerank_pool)
        return {"hits": reranker.rerank(q, pool, k)}
    return {"hits": fused_retrieve([q], embedder, store, bm25=bm25, graph=link_graph, k=k)}
```

When `reranker` is None, behavior is identical to M9.

## 6. Chat integration (`chat.py`)

`ChatSession` gains `reranker=None`. `_retrieve`:

```python
def _retrieve(self, question):
    ctx = self._context_query(question)
    if self._reranker is not None:
        pool = dual_query_retrieve(question, ctx, self._embedder, self._store,
                                   graph=self._graph, k=RERANK_POOL, bm25=self._bm25)
        rerank_query = f"{ctx}\n{question}" if ctx else question
        return self._reranker.rerank(rerank_query, pool, self._k)
    return dual_query_retrieve(question, ctx, self._embedder, self._store,
                               graph=self._graph, k=self._k, bm25=self._bm25)
```

## 7. Wiring (`service.py`, `cli.py`)

- `service.make_reranker()` returns a `CrossEncoderReranker` (constructs the model,
  so it is only called when `--rerank` is set). `service_ask(..., rerank=False)` builds
  it and passes it to `ask`.
- `cli`: `weft ask --rerank` → `service_ask(rerank=True)`. `weft chat --rerank` →
  `ChatSession(reranker=service.make_reranker())`. Both default off.

## 8. Security & invariants

- Offline after the one-time model download (~80 MB), like the sentence-transformer
  embedder; no network at query time.
- The reranker only reorders/selects among already-retrieved, already-filtered
  candidates — it introduces no new content and no boundary crossing.
- Pure reordering; the audited LLM call and privacy posture are unchanged.

## 9. Testing (offline, FakeReranker/FakeEmbedder/FakeLLM)

- `test_rerank.py` — `FakeReranker` reorders candidates so the one with the most
  query-term overlap comes first; empty hits → `[]`; returns at most `k`.
- `test_agent_rerank.py` — with a `FakeReranker`, `build_graph`/`ask`'s retrieve step
  fetches the pool via `fused_retrieve` then reranks to `k`; `reranker=None` is
  identical to M9.
- `test_chat` addition — a `ChatSession(reranker=FakeReranker())` reranks against the
  conversational query; no reranker path unchanged.
- `test_cli` additions — `weft ask --rerank` / `weft chat --rerank` run end-to-end
  with `make_reranker` patched to return a `FakeReranker`.

## 10. Milestone

This is **M10**, completing the three-stage retrieval pipeline (query rewrite → BM25
hybrid → cross-encoder rerank). Future: expose the pool size / cross-encoder model via
config; add memory-injection as a third RRF ranking.
