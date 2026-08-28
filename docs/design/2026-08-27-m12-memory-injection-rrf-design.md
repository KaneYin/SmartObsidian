# Weft M12 — Memory-Injection as a Third RRF Query — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-27.
**Scope:** An opt-in retrieval enhancement: fuse a durable-memory-derived query as an
additional RRF ranking, biasing retrieval toward what Weft knows the user cares about.

## 1. Problem & goal

M8/M9 fuse rankings for the current message, the conversation context, and vector +
BM25 — all through one N-ary `reciprocal_rank_fusion`. The specs noted memory-injection
as the natural third query. M12 builds a query from durable *facts* and *decisions* and
adds it as one more ranking, so retrieval can surface chunks relevant to the user's
standing context ("Chose LanceDB", "Writing a thesis on X").

## 2. Decisions (locked during brainstorming)

- **Opt-in `--memory-query`** on `ask`/`chat`, off by default (mirrors
  `--rewrite-llm`/`--rerank`).
- **Query built from active `fact` + `decision` items only.** Preferences are excluded
  — behavioral instructions like "answer concisely" make poor search queries.
- **Fused as one more ranking** through the existing RRF; the primitive is unchanged.

## 3. Components

```
src/weft/
  agent.py   TOUCH — build_memory_query(memory); build_graph/ask thread memory_query;
                     dual_query_retrieve accepts memory_query.
  chat.py    TOUCH — ChatSession memory_query flag; _retrieve adds the memory query.
  service.py TOUCH — service_ask(use_memory_query=False) builds + passes it.
  cli.py     TOUCH — `weft ask --memory-query`; `weft chat --memory-query`.
```

## 4. Memory query (`agent.py`)

```python
def build_memory_query(memory) -> str | None:
    """A search query from durable facts + decisions (topical memory). Preferences
    are excluded; returns None when there is nothing usable."""
    if memory is None:
        return None
    texts = [i.text for i in memory.active_semantic()
             if i.type in ("fact", "decision")]
    return " ".join(texts) or None
```

## 5. Retrieval integration (`agent.py`, `chat.py`)

Add an optional `memory_query` that becomes an extra query string in the fused set.

`dual_query_retrieve(..., memory_query=None)`:

```python
    queries = [question]
    if context_query:
        queries.append(context_query)
    if memory_query:
        queries.append(memory_query)
    return fused_retrieve(queries, embedder, store, bm25=bm25, graph=graph, k=k)
```

`build_graph`/`ask` gain `memory_query=None`; the retrieve node builds
`[question] + ([memory_query] if memory_query else [])` and passes it to
`fused_retrieve`. Each extra query still produces a vector ranking (and a BM25 ranking
when hybrid is on), all fused in one RRF pass. `memory_query=None` → today's behavior.

`ChatSession` gains a `memory_query: bool = False` flag; `_retrieve` computes
`build_memory_query(self._memory)` when enabled and passes it into
`dual_query_retrieve`, composing with the M8 context query and M10 rerank pool.

## 6. Wiring (`service.py`, `cli.py`)

- `service_ask(..., use_memory_query: bool = False)`: when set and memory is enabled,
  `mq = build_memory_query(memory)`; pass `memory_query=mq` to `ask`.
- `cli`: `weft ask --memory-query` → `service_ask(use_memory_query=True)`;
  `weft chat --memory-query` → `ChatSession(memory_query=True)`. Off by default.

## 7. Security & invariants

- No new files or network. The memory query is assembled locally from already-stored
  memory text; retrieval stays local and offline.
- Facts/decisions are the user's own durable statements; they flow only into the local
  vector/BM25 search, never off-machine.
- RRF is unchanged; an off-target memory ranking is naturally down-weighted.

## 8. Testing (offline)

- `test_agent` addition — `build_memory_query` joins facts + decisions, excludes
  preferences, returns None when empty; `dual_query_retrieve` with a `memory_query`
  fuses a third ranking (RRF of all three).
- `test_service` addition — `service_ask(use_memory_query=True)` with a seeded
  fact runs and returns an answer (fakes).
- `test_cli` additions — `weft ask --memory-query` and `weft chat --memory-query`
  run end-to-end with fakes.

## 9. Milestone

This is **M12**, the final reserved retrieval extension — the third RRF query the
M8/M9 designs anticipated. It composes with query rewrite (M8), hybrid (M9), and rerank
(M10) through the single fusion primitive.
