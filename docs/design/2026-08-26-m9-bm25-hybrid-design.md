# Weft M9 — BM25 Hybrid Search — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-26.
**Scope:** Add a hand-rolled BM25 lexical ranking beside the vector ranking, fused
with the M8 RRF primitive. Second stage of the retrieval pipeline (M8 rewrite → M9
hybrid → M10 reranker).

## 1. Problem & goal

Vector retrieval blurs exact terms — names, IDs, rare tokens — that lexical matching
nails. M9 builds a BM25 index beside the vector store and fuses the two rankings with
Reciprocal Rank Fusion (already introduced in M8), improving recall without a new
dependency.

## 2. Decisions (locked during brainstorming)

- **Hand-rolled BM25**, no new dependency (consistent with the hand-rolled `store.py`).
- **Fuse all rankings in one RRF call.** Per query: a vector ranking and a BM25
  ranking; chat's two queries yield up to four rankings, fused together.
- **Default-on with escapes.** `weft index` builds the BM25 index by default; retrieval
  auto-fuses when it exists. `weft index --no-bm25` skips building; `ask`/`chat
  --no-hybrid` disables per query. Backward-compatible: no BM25 file → vector-only.

## 3. Components

```
src/weft/
  bm25.py    NEW   — BM25Index (tokenize, Okapi scoring, save/load).
  store.py   TOUCH — public metadata_rows() so BM25 rows map to SearchHits.
  index.py   TOUCH — build + save .weft/index.bm25.json (unless --no-bm25); bm25_path_for.
  agent.py   TOUCH — fused_retrieve(queries, ..., bm25=None); dual_query_retrieve wraps it;
                     build_graph/ask thread bm25.
  service.py TOUCH — load the BM25 index; thread into service_ask and chat sessions.
  chat.py    TOUCH — ChatSession carries bm25; passes it through.
  cli.py     TOUCH — `weft index --no-bm25`; `ask`/`chat --no-hybrid`.
```

## 4. BM25Index (`bm25.py`)

```python
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

@dataclass
class BM25Index:
    docs: list[list[str]]        # tokens per store row (row i <-> store metadata i)
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        return cls(docs=[_tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        # Okapi BM25 over self.docs; returns top-k (row_index, score), score desc.
        ...

    def save(self, path) -> None: ...       # secure_write_text 0600, JSON {k1,b,docs}
    @classmethod
    def load(cls, path) -> "BM25Index": ...
```

Scoring: `N = len(docs)`; `avgdl = mean(len(d))`; `df[t]` = number of docs containing
`t`; `idf(t) = ln((N - df + 0.5)/(df + 0.5) + 1)` (the BM25+ non-negative idf);
`score(d) = Σ_t∈query idf(t) · f(t,d)·(k1+1) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))`.
Docs with zero score are dropped. Empty index or empty query → `[]`.

The index is built from `[c.text for _, c in pairs]` in the exact order `build_index`
adds vectors, so BM25 row `i` is `store.metadata_rows()[i]`.

## 5. Store accessor (`store.py`)

Add a small public method so BM25 can turn row indices into `SearchHit`s without
touching internals:

```python
def metadata_rows(self) -> list[dict]:
    """Chunk metadata in vector-row order (row i is the i-th added chunk)."""
    return list(self._metadata)
```

## 6. Fused retrieval (`agent.py`)

```python
def _bm25_ranking(bm25, query, store, k) -> list[SearchHit]:
    rows = store.metadata_rows()
    return [SearchHit(score=s, metadata=rows[i]) for i, s in bm25.search(query, k)]

def _vector_ranking(query, embedder, store, graph, k) -> list[SearchHit]:
    if graph is not None:
        return graph_aware_retrieve(query, embedder, store, graph, k=k)
    return retrieve(query, embedder, store, k=k)

def fused_retrieve(queries, embedder, store, *, bm25=None, graph=None, k=5):
    rankings = []
    for q in queries:
        rankings.append(_vector_ranking(q, embedder, store, graph, k))
        if bm25 is not None:
            rankings.append(_bm25_ranking(bm25, q, store, k))
    if len(rankings) == 1:
        return rankings[0]
    return reciprocal_rank_fusion(rankings, key=_hit_key)[:k]
```

`dual_query_retrieve` becomes a thin wrapper that assembles the query list and passes
`bm25` through, so M8's chat path and single-query `ask` share one fusion routine.
`build_graph`/`ask` gain a `bm25=None` parameter and route the retrieve step through
`fused_retrieve([question], …, bm25=bm25)`. When `bm25` is None (no index or
`--no-hybrid`), behavior is byte-identical to M8.

## 7. Wiring (`service.py`, `chat.py`, `cli.py`)

- `bm25_path_for(store_path)` = `.weft/index.bm25.json`. `service.load_bm25(store_path)`
  returns a `BM25Index` if the file exists, else `None`.
- `service_ask` and the chat command load the BM25 index (unless `--no-hybrid`) and
  pass it into `ask` / `ChatSession`.
- `build_index` builds and saves the BM25 index unless `chunking`-style `--no-bm25` is
  set; the index build already has the chunk texts.
- CLI: `weft index --no-bm25`; `weft ask --no-hybrid`; `weft chat --no-hybrid`.

## 8. Security & invariants

- `index.bm25.json` holds tokens derived from redacted note text — written with
  `secure_write_text` at `0600`, treated as sensitive like `index.json`.
- Pure-local, deterministic; no network. Retrieval stays offline.
- Backward-compatible: an index built before M9 has no BM25 file, so retrieval is
  vector-only until re-indexed.

## 9. Testing (offline)

- `test_bm25.py` — `_tokenize`; a query containing a rare exact term ranks the doc
  that contains it first (where a vector embedding would not); save/load round-trip
  preserves ranking; empty index and empty query return `[]`.
- `test_agent_hybrid.py` — `fused_retrieve` with a BM25 index fuses vector + BM25
  rankings (RRF of both); `bm25=None` equals the M8 vector-only path; `_bm25_ranking`
  maps rows to `SearchHit`s with store metadata.
- `test_index_bm25.py` — `build_index` writes `index.bm25.json` by default and omits
  it with `no_bm25=True`; the BM25 rows align to store rows.
- `test_cli` additions — `weft index --no-bm25` skips the file; `weft ask --no-hybrid`
  runs vector-only end-to-end.

## 10. Milestone & future

This is **M9**. **M10** adds a cross-encoder reranker that re-scores the fused top-N
before reasoning. Memory-injection can still enter as an additional RRF ranking.
