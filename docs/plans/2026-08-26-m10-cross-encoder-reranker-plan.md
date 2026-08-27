# M10 Cross-Encoder Reranker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in cross-encoder reranking stage that re-scores the fused retrieval pool to the final top-k.

**Architecture:** A `Reranker` protocol (real `CrossEncoderReranker` + `FakeReranker`); `ask`/`chat` fetch a pool via the M8/M9 fused path when a reranker is set, then rerank to `k`. Opt-in via `--rerank`; off by default.

**Tech Stack:** Python ≥3.11, sentence-transformers (already present, lazy CrossEncoder), existing `agent`/`chat`, `pytest` with `FakeReranker`.

**Scope:** M10 — completes the retrieval pipeline (rewrite → hybrid → rerank).

**Run tests with (verify as a separate step — do not pipe pytest, it masks the exit code):** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Create `src/weft/rerank.py` — `Reranker`, `CrossEncoderReranker`, `FakeReranker`, `RERANK_POOL`.
- Modify `src/weft/agent.py` — `build_graph`/`ask` reranker + rerank_pool.
- Modify `src/weft/chat.py` — `ChatSession` reranker.
- Modify `src/weft/service.py`, `src/weft/cli.py` — `make_reranker`, `--rerank`.
- Tests: `test_rerank.py`, `test_agent_rerank.py`, `test_chat_rewrite.py`/`test_cli` additions.

---

## Task 1: Reranker (`rerank.py`)

**Files:**
- Create: `src/weft/rerank.py`
- Test: `tests/test_rerank.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank.py
from weft.rerank import FakeReranker
from weft.store import SearchHit


def _h(text):
    return SearchHit(score=0.0, metadata={"rel_path": "n.md", "text": text, "ordinal": 0})


def test_fake_reranker_orders_by_overlap():
    hits = [_h("cats and dogs"), _h("quantum physics"), _h("zebra stripes pattern")]
    out = FakeReranker().rerank("zebra stripes", hits, 2)
    assert out[0].metadata["text"] == "zebra stripes pattern"
    assert len(out) == 2


def test_fake_reranker_empty():
    assert FakeReranker().rerank("q", [], 5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_rerank.py -v`
Expected: FAIL — `weft.rerank` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/rerank.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_rerank.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/rerank.py tests/test_rerank.py
git commit -m "feat(weft): M10 pluggable cross-encoder reranker (+ fake)"
```

---

## Task 2: Agent rerank stage (`agent.py`)

**Files:**
- Modify: `src/weft/agent.py`
- Test: `tests/test_agent_rerank.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_rerank.py
from weft.agent import ask, retrieve
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.rerank import FakeReranker
from weft.store import VectorStore


def _store():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    for i, t in enumerate(["zebra stripes here", "coffee beans", "random text", "zebra pattern"]):
        store.add(emb.embed([t])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": t, "ordinal": i,
                   "tags": [], "wikilinks": []})
    return emb, store


def test_ask_reranker_selects_overlap_docs():
    emb, store = _store()
    res = ask("zebra", emb, store, FakeLLM(response="a"), k=2, reranker=FakeReranker())
    assert set(res.sources) == {"n0.md", "n3.md"}  # the two 'zebra' docs


def test_ask_no_reranker_unchanged():
    emb, store = _store()
    res = ask("coffee", emb, store, FakeLLM(response="a"), k=2)
    vec = retrieve("coffee", emb, store, k=2)
    assert res.sources == list(dict.fromkeys(h.metadata["rel_path"] for h in vec))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_agent_rerank.py -v`
Expected: FAIL — `ask` takes no `reranker`.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/agent.py`, add the import near the top:

```python
from weft.rerank import RERANK_POOL
```

Change `build_graph`'s signature and `_retrieve` node to add reranking:

```python
def build_graph(embedder: Embedder, store: VectorStore, llm: LLMClient,
                link_graph: LinkGraph | None = None, memory: dict | None = None,
                bm25=None, reranker=None, rerank_pool: int = RERANK_POOL):
    """Compile the retrieve -> reason graph. Retrieval fuses vector (and BM25 when
    provided) rankings via RRF, then optionally reranks with a cross-encoder."""

    def _retrieve(state: AgentState) -> AgentState:
        q, k = state["question"], state.get("k", 5)
        if reranker is not None:
            pool = fused_retrieve([q], embedder, store, bm25=bm25,
                                  graph=link_graph, k=rerank_pool)
            return {"hits": reranker.rerank(q, pool, k)}
        return {"hits": fused_retrieve([q], embedder, store, bm25=bm25,
                                       graph=link_graph, k=k)}
```

Add `reranker=None` and `rerank_pool=RERANK_POOL` to `ask` and forward them:

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
    reranker=None,
    rerank_pool: int = RERANK_POOL,
) -> AskResult:
    mem_obj = collect_memory(memory, embedder, question, k=k) if memory is not None else None
    app = build_graph(embedder, store, llm, link_graph=graph, memory=mem_obj,
                      bm25=bm25, reranker=reranker, rerank_pool=rerank_pool)
```

(The rest of `ask` is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_agent_rerank.py tests/test_agent.py tests/test_agent_hybrid.py tests/test_agent_memory.py -v`
Expected: PASS (new rerank tests + existing agent tests — `reranker=None` unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/weft/agent.py tests/test_agent_rerank.py
git commit -m "feat(weft): M10 cross-encoder rerank stage in ask"
```

---

## Task 3: Chat rerank (`chat.py`)

**Files:**
- Modify: `src/weft/chat.py`
- Test: `tests/test_chat_rewrite.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_chat_rewrite.py)**

```python
def test_chat_reranks_when_set():
    from weft.rerank import FakeReranker
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    for i, t in enumerate(["zebra stripes", "coffee drip", "zebra herd"]):
        store.add(emb.embed([t])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": t, "ordinal": i,
                   "tags": [], "wikilinks": []})
    s = ChatSession(emb, store, FakeLLM(response="a"), reranker=FakeReranker(), k=2)
    turn = s.send("zebra")
    assert set(turn.sources) == {"n0.md", "n2.md"}  # the two 'zebra' docs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_chat_rewrite.py -k reranks -v`
Expected: FAIL — `ChatSession` takes no `reranker`.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/chat.py`, add the import:

```python
from weft.rerank import RERANK_POOL
```

Add `reranker=None` to `ChatSession.__init__` (after `bm25`), store it:

```python
    def __init__(self, embedder, store, llm, *, graph=None, memory=None,
                 k: int = 5, window: int = 6, rewrite_llm: bool = False, bm25=None,
                 reranker=None):
        ...
        self._bm25 = bm25
        self._reranker = reranker
        ...
```

Replace `_retrieve` to rerank when a reranker is present:

```python
    def _retrieve(self, question: str):
        ctx = self._context_query(question)
        if self._reranker is not None:
            pool = dual_query_retrieve(question, ctx, self._embedder, self._store,
                                       graph=self._graph, k=RERANK_POOL, bm25=self._bm25)
            rerank_query = f"{ctx}\n{question}" if ctx else question
            return self._reranker.rerank(rerank_query, pool, self._k)
        return dual_query_retrieve(question, ctx, self._embedder, self._store,
                                   graph=self._graph, k=self._k, bm25=self._bm25)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_chat_rewrite.py tests/test_chat.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/weft/chat.py tests/test_chat_rewrite.py
git commit -m "feat(weft): M10 chat rerank stage"
```

---

## Task 4: `--rerank` wiring + docs

**Files:**
- Modify: `src/weft/service.py`, `src/weft/cli.py`, `docs/how-to/configure-env-and-use-cli.md`
- Test: `tests/test_cli_chat.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_cli_chat.py)**

```python
def test_chat_rerank_flag(tmp_path, monkeypatch, capsys):
    from weft.rerank import FakeReranker
    store = tmp_path / ".weft" / "index"
    _index(store)
    monkeypatch.setattr(service, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(service, "make_llm", lambda *a, **k: FakeLLM(response="chat answer"))
    monkeypatch.setattr(service, "make_reranker", lambda: FakeReranker())
    monkeypatch.setattr(builtins, "input", _script(["zebra", "/exit"]))
    rc = cli.main(["chat", "--store", str(store), "--rerank"])
    assert rc == 0
    assert "chat answer" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli_chat.py -k rerank -v`
Expected: FAIL — `chat` has no `--rerank`, no `service.make_reranker`.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/service.py`, add the factory and thread rerank into `service_ask`:

```python
def make_reranker():
    from weft.rerank import CrossEncoderReranker
    return CrossEncoderReranker()
```

In `service_ask`, add a `rerank: bool = False` keyword and build/pass the reranker:

```python
def service_ask(store_path: Path, question: str, k: int = 5, *,
                use_graph: bool = True, use_memory: bool = True,
                overrides: dict | None = None, use_hybrid: bool = True,
                rerank: bool = False) -> dict:
    ...
    reranker = make_reranker() if rerank else None
    result = ask(question, make_embedder(), store, llm, k=k, graph=graph,
                 memory=memory, bm25=bm25, reranker=reranker)
```

In `src/weft/cli.py`:
- `_cmd_ask`: pass `rerank=args.rerank` into `service.service_ask(...)`.
- `_cmd_chat`: build the reranker and pass it:

```python
    reranker = service.make_reranker() if args.rerank else None
    session = ChatSession(service.make_embedder(), store, llm,
                          graph=link_graph, memory=memory, k=args.k,
                          rewrite_llm=args.rewrite_llm, bm25=bm25, reranker=reranker)
```

- Add `--rerank` to both `p_ask` and `p_chat` (extend the existing hybrid loop):

```python
    for hybrid_sub in (p_ask, p_chat):
        hybrid_sub.add_argument(
            "--no-hybrid", action="store_true",
            help="Disable BM25 hybrid fusion; vector-only retrieval.",
        )
        hybrid_sub.add_argument(
            "--rerank", action="store_true",
            help="Rerank the retrieval pool with a cross-encoder (downloads a model).",
        )
```

- [ ] **Step 4: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS across all modules.

- [ ] **Step 5: Document + commit**

Add to `docs/how-to/configure-env-and-use-cli.md` after the hybrid-search section:

```markdown
### Rerank with a cross-encoder

For maximum precision, add `--rerank` to `ask` or `chat`:

    weft ask "..." --rerank
    weft chat --rerank

Weft retrieves a larger candidate pool (hybrid + query rewrite) and a cross-encoder
re-scores each `(query, chunk)` pair, keeping the best `--k`. It downloads a small
reranker model on first use, then runs offline. It is slower than plain retrieval,
so it is opt-in.
```

```bash
git add src/weft/service.py src/weft/cli.py tests/test_cli_chat.py docs/how-to/configure-env-and-use-cli.md
git commit -m "feat(weft): M10 weft ask/chat --rerank + docs"
```

---

## Self-Review Notes (author)

- **Spec coverage:** Reranker protocol + CrossEncoder + Fake (Task 1) · pool→rerank in
  ask/build_graph, no-reranker unchanged (Task 2) · chat rerank against conversational
  query (Task 3) · `make_reranker` + `--rerank` + docs (Task 4). Cross-encoder scores
  `metadata["text"]`; M7 parent dedup still runs in build_prompt (unchanged).
- **Type consistency:** `Reranker.rerank(query, hits, k)`; `CrossEncoderReranker`,
  `FakeReranker`; `RERANK_POOL`; `build_graph(..., reranker=None, rerank_pool=RERANK_POOL)`;
  `ask(..., reranker=None, rerank_pool=RERANK_POOL)`; `ChatSession(..., reranker=None)`;
  `service_ask(..., rerank=False)`; `make_reranker()`. Consistent across tasks.
- **Regression control:** `reranker=None` (default, no `--rerank`) reproduces M9
  behavior; the real CrossEncoder is only constructed when `--rerank` is set, so the
  test suite (FakeReranker) stays offline. `FakeReranker`'s lexical overlap uses the
  BM25 tokenizer, so a query term present in a chunk ranks it up deterministically.
```
