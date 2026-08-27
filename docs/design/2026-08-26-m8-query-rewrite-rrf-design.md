# Weft M8 — Query Rewrite via Dual-Query + Reciprocal Rank Fusion — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-26.
**Scope:** Fix multi-turn retrieval by contextualizing the query, using dual-query
retrieval fused with Reciprocal Rank Fusion (RRF). First stage of the retrieval
pipeline; RRF is the shared primitive for M9 (BM25 hybrid) and M10 (reranker).

## 1. Problem & goal

`ChatSession._retrieve` embeds the raw current message, so a follow-up like "why?"
searches for "why?" and loses the conversation context. Concatenating the history
into one embedding blurs two clean questions into a mushy centroid. M8 keeps each
query clean and fuses at the rank layer: retrieve for the current message and for a
reconstructed context separately, then merge with RRF.

## 2. Decisions (locked during brainstorming)

- **Build order for the pipeline:** query rewrite (M8) → BM25 hybrid (M9) →
  cross-encoder reranker (M10).
- **Dual-query + RRF, not concatenation.** Embed the current message and the
  reconstructed context as two clean queries; fuse their ranked hit lists with RRF.
  Avoids the blurred-centroid problem and reuses the RRF code M9 needs anyway.
- **Reconstructed context = the conversation window's prior user turns** (the sliding
  window). No history (first turn, or `ask`) → single query, unchanged behavior.
- **Heuristic default + opt-in LLM rewrite.** Heuristic: join prior user turns as the
  second query. `--rewrite-llm`: the configured provider produces a standalone query
  (local/offline capable, audited), degrading to the heuristic on failure.
- **Memory-injection deferred.** RRF takes N rankings, so a memory-derived query can
  be added as a third ranking later without touching the primitive.

## 3. Components

```
src/weft/
  fusion.py  NEW   — reciprocal_rank_fusion(rankings, *, k=60, key) -> list[SearchHit].
                     General N-ranking RRF. Reused verbatim by M9 hybrid search.
  agent.py   TOUCH — dual_query_retrieve(question, context_query, ...) fuses two
                     single-query retrievals; context_query=None -> single query.
  chat.py    TOUCH — ChatSession dual-query retrieval + reconstructed context +
                     opt-in llm_rewrite_query; `rewrite_llm` flag.
  cli.py     TOUCH — `weft chat --rewrite-llm`.
```

## 4. RRF primitive (`fusion.py`)

```python
def reciprocal_rank_fusion(rankings, *, k: int = 60, key) -> list:
    """Fuse ranked lists by summed reciprocal rank. `key(hit)` identifies a hit's
    identity for dedup. Returns unique hits sorted by fused score (desc)."""
    scores: dict = {}
    representative: dict = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            kk = key(hit)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank + 1)
            representative.setdefault(kk, hit)
    ordered = sorted(scores, key=lambda kk: -scores[kk])
    return [representative[kk] for kk in ordered]
```

- `k=60` is the standard RRF constant (dampens the weight of deep ranks).
- Identity `key` = `parent_id` if present, else `(rel_path, ordinal)` — the same key
  M7's parent-dedup uses, so fusion and prompt-dedup agree on chunk identity.
- Stable: earlier rankings' representative hit is kept for a shared identity.

## 5. Dual-query retrieval (`agent.py`)

```python
def _hit_key(h) -> object:
    m = h.metadata
    return m.get("parent_id") or (m["rel_path"], m.get("ordinal", 0))

def dual_query_retrieve(question, context_query, embedder, store, *, graph=None,
                        k=5) -> list[SearchHit]:
    def one(q):
        if graph is not None:
            return graph_aware_retrieve(q, embedder, store, graph, k=k)
        return retrieve(q, embedder, store, k=k)
    hits_main = one(question)
    if not context_query:
        return hits_main
    hits_ctx = one(context_query)
    return reciprocal_rank_fusion([hits_main, hits_ctx], key=_hit_key)[:k]
```

Retrieval work doubles when a context query exists — a second local matrix multiply,
negligible at vault scale. Existing single-query callers (`ask` via `agent.ask`) are
untouched.

## 6. Chat integration (`chat.py`)

`ChatSession` gains a `rewrite_llm: bool = False` constructor flag. `_retrieve`:

```python
def _retrieve(self, question):
    ctx = self._context_query(question)
    return dual_query_retrieve(question, ctx, self._embedder, self._store,
                               graph=self._graph, k=self._k)

def _context_query(self, question):
    users = [t["text"] for t in self.history if t["role"] == "user"]
    if not users:
        return None
    if self._rewrite_llm:
        try:
            return llm_rewrite_query(self.history, question, self._llm)
        except Exception:
            pass
    return " ".join(users)  # heuristic: prior user turns as one clean query
```

`llm_rewrite_query(history, question, llm)` sends the window + latest message to the
provider asking for a standalone search query (system prompt frames the conversation
as untrusted data), returns the stripped reply, and is audited by the existing
`AuditedLLM` the session already wraps. Any failure falls back to the heuristic.

`send()` is otherwise unchanged — it still builds the prompt (with M7 parent dedup),
answers, appends the turn, and logs an episode.

## 7. CLI

`weft chat --rewrite-llm` sets `ChatSession(rewrite_llm=True)`. Off by default, so
chat stays heuristic/offline unless opted in. `ask` gets no flag (single-turn).

## 8. Security & invariants

- No new files or network by default; the LLM rewrite is opt-in and audited exactly
  like the answer call (`provider`/`model`/`left_machine`).
- Conversation text used to build the context query is note-adjacent user input; the
  LLM-rewrite prompt frames it as untrusted data, consistent with the rest of Weft.
- RRF is pure and deterministic; retrieval remains local.

## 9. Testing (offline, FakeEmbedder/FakeLLM)

- `test_fusion.py` — two rankings fuse so an item ranked high in both beats an item
  ranked high in only one; a single ranking passes through unchanged; three rankings
  fuse; identity `key` dedups the same chunk across rankings.
- `test_chat_rewrite.py` — with history, `_retrieve` issues two retrievals and returns
  a fused list; no history → single query (identical to today); heuristic context is
  the prior user turns; `rewrite_llm=True` uses `FakeLLM` and falls back to heuristic
  when it raises.
- `test_cli_chat.py` addition — `weft chat --rewrite-llm` runs a turn end-to-end with
  fakes.

## 10. Milestone & future

This is **M8**. `fusion.reciprocal_rank_fusion` is the join point for:
- **M9 BM25 hybrid** — build a lexical index beside the vector store; fuse the vector
  ranking and the BM25 ranking with the same RRF.
- **M10 cross-encoder reranker** — re-score the fused top-N with a cross-encoder before
  reasoning.
- **Memory-injection** — add a durable-memory-derived query as a third RRF ranking.
