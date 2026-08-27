# M9 BM25 Hybrid Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hand-rolled BM25 lexical index beside the vector store and fuse the vector and BM25 rankings with the M8 RRF primitive.

**Architecture:** `bm25.py` builds/scoring; `agent.fused_retrieve` runs a vector ranking and a BM25 ranking per query and RRF-fuses all; `build_index` builds the BM25 index by default; `ask`/`chat` auto-fuse when the index is present. `--no-bm25` skips building, `--no-hybrid` disables per query.

**Tech Stack:** Python ≥3.11 stdlib (`math`, `re`, `collections`), existing `agent`/`store`/`fusion`, `pytest`.

**Scope:** M9 — BM25 + hybrid. M10 adds the cross-encoder reranker.

**Run tests with (verify as a separate step — do not pipe pytest, it masks the exit code):** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Create `src/weft/bm25.py` — `BM25Index`, `_tokenize`.
- Modify `src/weft/store.py` — `metadata_rows()`.
- Modify `src/weft/agent.py` — `_bm25_ranking`, `_vector_ranking`, `fused_retrieve`; `dual_query_retrieve` wrapper; `build_graph`/`ask` `bm25` param.
- Modify `src/weft/index.py` — build + save BM25; `bm25_path_for`; `no_bm25`.
- Modify `src/weft/service.py`, `src/weft/chat.py`, `src/weft/cli.py` — wiring + flags.
- Tests: `test_bm25.py`, `test_agent_hybrid.py`, `test_index_bm25.py`, `test_cli.py`/`test_cli_chat.py` additions.

---

## Task 1: BM25Index (`bm25.py`)

**Files:**
- Create: `src/weft/bm25.py`
- Test: `tests/test_bm25.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bm25.py
from weft.bm25 import BM25Index, _tokenize


def test_tokenize():
    assert _tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_bm25_ranks_exact_term_first():
    idx = BM25Index.build(["the quick brown fox", "lazy dog sleeps", "zebra stripes here"])
    ranked = idx.search("zebra", k=3)
    assert ranked[0][0] == 2  # only doc 2 contains 'zebra'


def test_bm25_empty_index_and_query():
    assert BM25Index.build([]).search("x", 3) == []
    assert BM25Index.build(["a b c"]).search("", 3) == []
    assert BM25Index.build(["a b c"]).search("nomatch", 3) == []


def test_bm25_roundtrip(tmp_path):
    idx = BM25Index.build(["alpha beta", "gamma delta", "gamma gamma"])
    p = tmp_path / "index.bm25.json"
    idx.save(p)
    loaded = BM25Index.load(p)
    assert loaded.search("gamma", 3) == idx.search("gamma", 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_bm25.py -v`
Expected: FAIL — `weft.bm25` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/bm25.py
"""A small hand-rolled Okapi BM25 lexical index, aligned to the vector store's rows
(row i corresponds to the i-th chunk added to the store). The lexical half of hybrid
retrieval; its ranking is fused with the vector ranking via reciprocal rank fusion."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from weft.security import secure_write_text

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    docs: list[list[str]]        # tokens per store row
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        return cls(docs=[_tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = _tokenize(query)
        N = len(self.docs)
        if N == 0 or not q:
            return []
        lengths = [len(d) for d in self.docs]
        avgdl = sum(lengths) / N or 1.0
        df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        query_terms = [t for t in dict.fromkeys(q) if t in df]
        scores: list[float] = []
        for i, d in enumerate(self.docs):
            tf = Counter(d)
            dl = lengths[i]
            s = 0.0
            for t in query_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                )
            scores.append(s)
        ranked = sorted(range(N), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked if scores[i] > 0.0][:k]

    def save(self, path: Path) -> None:
        secure_write_text(
            Path(path),
            json.dumps({"k1": self.k1, "b": self.b, "docs": self.docs},
                       ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(docs=data["docs"], k1=data.get("k1", 1.5), b=data.get("b", 0.75))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_bm25.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/bm25.py tests/test_bm25.py
git commit -m "feat(weft): M9 hand-rolled BM25 lexical index"
```

---

## Task 2: Fused retrieval with BM25 (`store.py`, `agent.py`)

**Files:**
- Modify: `src/weft/store.py` (add `metadata_rows`)
- Modify: `src/weft/agent.py`
- Test: `tests/test_agent_hybrid.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_hybrid.py
from weft.agent import _hit_key, fused_retrieve, retrieve
from weft.bm25 import BM25Index
from weft.embeddings import FakeEmbedder
from weft.fusion import reciprocal_rank_fusion
from weft.store import VectorStore


def _store():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    texts = ["alpha coffee", "beta tea", "gamma water", "zebra delta"]
    for i, text in enumerate(texts):
        store.add(emb.embed([text])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": text, "ordinal": i,
                   "tags": [], "wikilinks": []})
    return emb, store, texts


def test_fused_none_bm25_is_vector():
    emb, store, _ = _store()
    assert fused_retrieve(["coffee"], emb, store, bm25=None, k=3) == \
        retrieve("coffee", emb, store, k=3)


def test_fused_with_bm25_fuses_both():
    emb, store, texts = _store()
    bm25 = BM25Index.build(texts)
    got = fused_retrieve(["coffee"], emb, store, bm25=bm25, k=3)
    vec = retrieve("coffee", emb, store, k=3)
    lex = [type("H", (), {"score": s, "metadata": store.metadata_rows()[i]})()
           for i, s in bm25.search("coffee", 3)]
    expect = reciprocal_rank_fusion([vec, lex], key=_hit_key)[:3]
    assert [_hit_key(h) for h in got] == [_hit_key(h) for h in expect]


def test_bm25_surfaces_exact_term(tmp_path):
    emb, store, texts = _store()
    bm25 = BM25Index.build(texts)
    got = fused_retrieve(["zebra"], emb, store, bm25=bm25, k=4)
    assert any(h.metadata["rel_path"] == "n3.md" for h in got)  # the 'zebra delta' doc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_agent_hybrid.py -v`
Expected: FAIL — `fused_retrieve` missing.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/store.py`, add a public accessor to `VectorStore` (next to
`metadata_by_note`):

```python
    def metadata_rows(self) -> list[dict]:
        """Chunk metadata in vector-row order (row i is the i-th added chunk)."""
        return list(self._metadata)
```

In `src/weft/agent.py`, replace `dual_query_retrieve` (lines 104-118) with the
generalized fusion plus a thin wrapper:

```python
def _vector_ranking(query: str, embedder: Embedder, store: VectorStore,
                    graph: LinkGraph | None, k: int) -> list[SearchHit]:
    if graph is not None:
        return graph_aware_retrieve(query, embedder, store, graph, k=k)
    return retrieve(query, embedder, store, k=k)


def _bm25_ranking(bm25, query: str, store: VectorStore, k: int) -> list[SearchHit]:
    rows = store.metadata_rows()
    return [SearchHit(score=s, metadata=rows[i]) for i, s in bm25.search(query, k)]


def fused_retrieve(queries: list[str], embedder: Embedder, store: VectorStore, *,
                   bm25=None, graph: LinkGraph | None = None,
                   k: int = 5) -> list[SearchHit]:
    """Run a vector ranking (and a BM25 ranking, when bm25 is given) for each query,
    then RRF-fuse all rankings. One ranking total -> returned as-is."""
    rankings: list[list[SearchHit]] = []
    for q in queries:
        rankings.append(_vector_ranking(q, embedder, store, graph, k))
        if bm25 is not None:
            rankings.append(_bm25_ranking(bm25, q, store, k))
    if len(rankings) == 1:
        return rankings[0]
    return reciprocal_rank_fusion(rankings, key=_hit_key)[:k]


def dual_query_retrieve(question: str, context_query, embedder: Embedder,
                        store: VectorStore, *, graph: LinkGraph | None = None,
                        k: int = 5, bm25=None) -> list[SearchHit]:
    """Fuse the current question and the reconstructed context query (M8), each
    optionally hybridized with BM25 (M9)."""
    queries = [question] + ([context_query] if context_query else [])
    return fused_retrieve(queries, embedder, store, bm25=bm25, graph=graph, k=k)
```

Thread `bm25` through `build_graph` and `ask`. Replace `build_graph`'s signature and
`_retrieve` node (lines 176-188):

```python
def build_graph(embedder: Embedder, store: VectorStore, llm: LLMClient,
                link_graph: LinkGraph | None = None, memory: dict | None = None,
                bm25=None):
    """Compile the retrieve -> reason graph. Retrieval fuses vector (and BM25 when
    provided) rankings via RRF."""

    def _retrieve(state: AgentState) -> AgentState:
        hits = fused_retrieve([state["question"]], embedder, store,
                              bm25=bm25, graph=link_graph, k=state.get("k", 5))
        return {"hits": hits}
```

Add `bm25=None` to `ask` and forward it (lines 202-212):

```python
def ask(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    k: int = 5,
    graph: LinkGraph | None = None,
    memory=None,
    bm25=None,
) -> AskResult:
    mem_obj = collect_memory(memory, embedder, question, k=k) if memory is not None else None
    app = build_graph(embedder, store, llm, link_graph=graph, memory=mem_obj, bm25=bm25)
```

(The rest of `ask` is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_agent_hybrid.py tests/test_agent_dual.py tests/test_agent.py tests/test_agent_graph.py tests/test_agent_memory.py tests/test_chat.py tests/test_chat_rewrite.py -v`
Expected: PASS (new hybrid tests + all existing agent/chat tests — `bm25=None` keeps M8 behavior identical).

- [ ] **Step 5: Commit**

```bash
git add src/weft/store.py src/weft/agent.py tests/test_agent_hybrid.py
git commit -m "feat(weft): M9 fused vector+BM25 retrieval via RRF"
```

---

## Task 3: build_index writes the BM25 index (`index.py`)

**Files:**
- Modify: `src/weft/index.py`
- Test: `tests/test_index_bm25.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_bm25.py
from weft.bm25 import BM25Index
from weft.embeddings import FakeEmbedder
from weft.index import build_index, bm25_path_for
from weft.store import VectorStore


def _vault(tmp_path):
    (tmp_path / "n.md").write_text("# A\nzebra alpha\n\nbeta gamma\n", encoding="utf-8")
    return tmp_path


def test_build_writes_bm25_by_default(tmp_path):
    store_path = tmp_path / "idx"
    build_index(_vault(tmp_path), FakeEmbedder(dim=16), store_path)
    path = bm25_path_for(store_path)
    assert path.exists()
    bm25 = BM25Index.load(path)
    store = VectorStore.load(store_path)
    assert len(bm25.docs) == len(store)  # one BM25 doc per store row
    # 'zebra' occurs only in the first chunk -> its row ranks first
    assert bm25.search("zebra", 3)[0][0] == 0


def test_no_bm25_skips_file(tmp_path):
    store_path = tmp_path / "idx"
    build_index(_vault(tmp_path), FakeEmbedder(dim=16), store_path, no_bm25=True)
    assert not bm25_path_for(store_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_index_bm25.py -v`
Expected: FAIL — `bm25_path_for`/`no_bm25` missing.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/index.py`, add the import and path helper:

```python
from weft.bm25 import BM25Index
```

```python
def bm25_path_for(store_path: Path) -> Path:
    """`.weft/index` -> `.weft/index.bm25.json` (lexical index beside the vectors)."""
    return Path(str(store_path) + ".bm25.json")
```

Add `no_bm25: bool = False` to `build_index`'s signature, and after `store.save(...)`
build and save the BM25 index from the same chunk texts:

```python
    store.save(Path(store_path))

    if not no_bm25 and pairs:
        BM25Index.build([c.text for _, c in pairs]).save(bm25_path_for(store_path))
```

(Chunk texts are already computed as `pairs`; the BM25 rows therefore align to the
store rows in the same order.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_index_bm25.py tests/test_index.py tests/test_index_chunking.py tests/test_empty_and_batch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/weft/index.py tests/test_index_bm25.py
git commit -m "feat(weft): M9 build BM25 index beside the vector store"
```

---

## Task 4: Wire hybrid into ask/chat + CLI flags + docs

**Files:**
- Modify: `src/weft/service.py`, `src/weft/chat.py`, `src/weft/cli.py`, `docs/how-to/configure-env-and-use-cli.md`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_cli.py)**

```python
def test_cli_ask_no_hybrid_runs(sample_vault, tmp_path, monkeypatch, capsys):
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="ans [1]"))
    idx = tmp_path / "idx"
    cli.main(["index", str(sample_vault), "--store", str(idx)])
    capsys.readouterr()
    rc = cli.main(["ask", "coffee", "--store", str(idx), "--no-hybrid"])
    assert rc == 0
    assert "ans [1]" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli.py -k no_hybrid -v`
Expected: FAIL — `ask` has no `--no-hybrid` option.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/service.py`, add a loader and thread BM25 into `service_ask`. Add near
the top helpers:

```python
def load_bm25(store_path: Path):
    from weft.bm25 import BM25Index
    from weft.index import bm25_path_for
    path = bm25_path_for(Path(store_path))
    return BM25Index.load(path) if path.exists() else None
```

In `service_ask`, add a `use_hybrid: bool = True` keyword and pass BM25 into `ask`.
Change the signature and the `ask(...)` call:

```python
def service_ask(store_path: Path, question: str, k: int = 5, *,
                use_graph: bool = True, use_memory: bool = True,
                overrides: dict | None = None, use_hybrid: bool = True) -> dict:
    ...
    bm25 = load_bm25(sp) if use_hybrid else None
    result = ask(question, make_embedder(), store, llm, k=k, graph=graph,
                 memory=memory, bm25=bm25)
```

In `src/weft/chat.py`, add `bm25=None` to `ChatSession.__init__` (after `rewrite_llm`)
and pass it through `_retrieve`:

```python
    def __init__(self, embedder, store, llm, *, graph=None, memory=None,
                 k: int = 5, window: int = 6, rewrite_llm: bool = False, bm25=None):
        ...
        self._bm25 = bm25
        ...

    def _retrieve(self, question: str):
        return dual_query_retrieve(question, self._context_query(question),
                                   self._embedder, self._store, graph=self._graph,
                                   k=self._k, bm25=self._bm25)
```

In `src/weft/cli.py`:
- `_cmd_ask`: pass `use_hybrid=not args.no_hybrid` into `service.service_ask(...)`.
- `_cmd_chat`: load BM25 and pass it into `ChatSession`:

```python
    bm25 = None if args.no_hybrid else service.load_bm25(store_path)
    session = ChatSession(service.make_embedder(), store, llm,
                          graph=link_graph, memory=memory, k=args.k,
                          rewrite_llm=args.rewrite_llm, bm25=bm25)
```

- Add flags in `build_parser`: `p_index.add_argument("--no-bm25", action="store_true",
  help="Skip building the BM25 lexical index.")` and pass
  `no_bm25=args.no_bm25` in `_cmd_index`'s `build_index(...)` call; add
  `--no-hybrid` to both `p_ask` and `p_chat`:

```python
    for hy in (p_ask, p_chat):
        hy.add_argument("--no-hybrid", action="store_true",
                        help="Disable BM25 hybrid fusion; vector-only retrieval.")
```

- [ ] **Step 4: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS across all modules. If a prior CLI ask/chat test now also builds a BM25
file, that is expected and harmless.

- [ ] **Step 5: Document + commit**

Add to the indexing / ask sections of `docs/how-to/configure-env-and-use-cli.md`:

```markdown
### Hybrid search (BM25 + vectors)

`weft index` also builds a small BM25 lexical index (`.weft/index.bm25.json`).
Retrieval then fuses the vector ranking and the BM25 ranking (reciprocal rank
fusion), so exact terms, names, and IDs that embeddings blur are still found.
Skip building it with `weft index --no-bm25`; disable it for a single query with
`weft ask --no-hybrid` or `weft chat --no-hybrid`. Indexes built before this
feature stay vector-only until re-indexed.
```

```bash
git add src/weft/service.py src/weft/chat.py src/weft/cli.py tests/test_cli.py docs/how-to/configure-env-and-use-cli.md
git commit -m "feat(weft): M9 wire hybrid into ask/chat + --no-bm25/--no-hybrid + docs"
```

---

## Self-Review Notes (author)

- **Spec coverage:** BM25 index + scoring + save/load (Task 1) · `metadata_rows` +
  fused vector+BM25 with one RRF + single-ranking passthrough (Task 2) · build/save
  the index by default + `no_bm25` (Task 3) · load + thread into ask/chat + `--no-bm25`
  /`--no-hybrid` + docs (Task 4). BM25 row alignment to store rows is guaranteed by
  building from the same `pairs` order (Task 3).
- **Type consistency:** `_tokenize`, `BM25Index.build/search/save/load`,
  `bm25_path_for`, `metadata_rows()`, `fused_retrieve(queries, embedder, store, *,
  bm25=None, graph=None, k=5)`, `dual_query_retrieve(..., bm25=None)`,
  `ask(..., bm25=None)`, `build_graph(..., bm25=None)`, `service_ask(..., use_hybrid=True)`,
  `ChatSession(..., bm25=None)`, `load_bm25(store_path)`. Consistent across tasks.
- **Regression control:** `bm25=None` (no index or `--no-hybrid`) reproduces M8/M7
  behavior exactly; existing agent/chat/index tests pass because the fusion collapses
  to a single ranking when there's one. RRF `_hit_key` unchanged.
```
