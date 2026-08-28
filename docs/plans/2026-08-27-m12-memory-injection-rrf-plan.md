# M12 Memory-Injection as a Third RRF Query — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--memory-query` that fuses a durable-memory-derived query (active facts + decisions) as one more RRF ranking on `ask`/`chat`.

**Architecture:** `build_memory_query(memory)` joins active fact+decision texts into a query string. It threads through `dual_query_retrieve`/`build_graph`/`ask` (library), `ChatSession` (REPL), and `service_ask` (shared core) as an extra query, fused by the existing `reciprocal_rank_fusion`. Off by default; `None` → today's behavior.

**Tech Stack:** Python ≥3.11 stdlib, existing `agent`/`chat`/`service`/`memory`, `pytest` with `FakeEmbedder`/`FakeLLM`/`MemoryStore`.

**Scope:** M12 — the third RRF query anticipated by M8/M9. Composes with query rewrite (M8), hybrid (M9), rerank (M10).

**Run tests with (verify as a separate step — do not pipe pytest):** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Modify `src/weft/agent.py` — `build_memory_query`; `memory_query` on `dual_query_retrieve`/`build_graph`/`ask`.
- Modify `src/weft/chat.py` — `ChatSession(memory_query=False)`.
- Modify `src/weft/service.py` — `service_ask(use_memory_query=False)`.
- Modify `src/weft/cli.py` — `ask/chat --memory-query`.
- Tests: `test_agent.py`, `test_chat.py`, `test_service.py`, `test_cli.py`.

---

## Task 1: `build_memory_query` + retrieval threading (`agent.py`)

**Files:**
- Modify: `src/weft/agent.py`
- Test: `tests/test_agent.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_agent.py)**

```python
def test_build_memory_query_facts_and_decisions_only(tmp_path):
    from weft.agent import build_memory_query
    from weft.memory import MemoryStore
    m = MemoryStore(tmp_path / ".weft")
    m.remember("Uses LanceDB for vectors", type="decision")
    m.remember("Thesis is on retrieval", type="fact")
    m.remember("Answer concisely", type="preference")
    q = build_memory_query(m)
    assert "LanceDB" in q and "Thesis" in q
    assert "concisely" not in q  # preferences excluded


def test_build_memory_query_none_when_empty(tmp_path):
    from weft.agent import build_memory_query
    from weft.memory import MemoryStore
    assert build_memory_query(None) is None
    m = MemoryStore(tmp_path / ".weft")
    m.remember("Answer concisely", type="preference")
    assert build_memory_query(m) is None  # only a preference → nothing usable


def test_dual_query_retrieve_fuses_memory_query(tmp_path):
    from weft.agent import dual_query_retrieve
    from weft.embeddings import FakeEmbedder
    from weft.store import VectorStore
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=16)
    store.add_batch(emb.embed(["alpha note", "beta note"]),
                    [{"rel_path": "a.md", "heading": "A", "text": "alpha note", "ordinal": 0},
                     {"rel_path": "b.md", "heading": "B", "text": "beta note", "ordinal": 1}])
    hits = dual_query_retrieve("alpha", None, emb, store, k=2, memory_query="beta")
    assert {h.metadata["rel_path"] for h in hits} == {"a.md", "b.md"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_agent.py -k "memory_query" -v`
Expected: FAIL — `build_memory_query` undefined; `dual_query_retrieve` has no `memory_query`.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/agent.py`, add `build_memory_query` (place it just above `collect_memory`):

```python
def build_memory_query(memory) -> str | None:
    """A search query from durable facts + decisions (topical memory). Preferences
    are excluded — behavioral instructions make poor search queries. Returns None
    when there is nothing usable."""
    if memory is None:
        return None
    texts = [i.text for i in memory.active_semantic()
             if i.type in ("fact", "decision")]
    return " ".join(texts) or None
```

Update `dual_query_retrieve` to accept and append `memory_query`:

```python
def dual_query_retrieve(question: str, context_query, embedder: Embedder,
                        store: VectorStore, *, graph: LinkGraph | None = None,
                        k: int = 5, bm25=None, memory_query=None) -> list[SearchHit]:
    """Fuse the current question, the reconstructed context query (M8), and an
    optional durable-memory query (M12), each optionally hybridized with BM25 (M9)."""
    queries = [question]
    if context_query:
        queries.append(context_query)
    if memory_query:
        queries.append(memory_query)
    return fused_retrieve(queries, embedder, store, bm25=bm25, graph=graph, k=k)
```

Add `memory_query=None` to `build_graph`'s signature and use it in both `_retrieve`
branches:

```python
def build_graph(embedder: Embedder, store: VectorStore, llm: LLMClient,
                link_graph: LinkGraph | None = None, memory: dict | None = None,
                bm25=None, reranker=None, rerank_pool: int = RERANK_POOL,
                memory_query=None):
    """Compile the retrieve -> reason graph. Retrieval fuses vector (and BM25 when
    provided) rankings via RRF, then optionally reranks with a cross-encoder."""

    def _retrieve(state: AgentState) -> AgentState:
        q, k = state["question"], state.get("k", 5)
        queries = [q] + ([memory_query] if memory_query else [])
        if reranker is not None:
            pool = fused_retrieve(queries, embedder, store, bm25=bm25,
                                  graph=link_graph, k=rerank_pool)
            return {"hits": reranker.rerank(q, pool, k)}
        return {"hits": fused_retrieve(queries, embedder, store, bm25=bm25,
                                       graph=link_graph, k=k)}
```

(The `_reason` node and the rest of `build_graph` are unchanged.)

Add `memory_query=None` to `ask`'s signature and pass it to `build_graph`:

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
    memory_query=None,
) -> AskResult:
    mem_obj = collect_memory(memory, embedder, question, k=k) if memory is not None else None
    app = build_graph(embedder, store, llm, link_graph=graph, memory=mem_obj,
                      bm25=bm25, reranker=reranker, rerank_pool=rerank_pool,
                      memory_query=memory_query)
```

(The rest of `ask` — invoke, sources, episode log, return — is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/weft/agent.py tests/test_agent.py
git commit -m "feat(weft): M12 build_memory_query + memory_query retrieval threading"
```

---

## Task 2: `ChatSession` memory-query flag (`chat.py`)

**Files:**
- Modify: `src/weft/chat.py`
- Test: `tests/test_chat.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_chat.py)**

```python
def test_chat_memory_query_adds_ranking(tmp_path):
    from weft.chat import ChatSession
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    from weft.memory import MemoryStore
    from weft.store import VectorStore
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=16)
    store.add_batch(emb.embed(["alpha note", "beta note"]),
                    [{"rel_path": "a.md", "heading": "A", "text": "alpha note", "ordinal": 0},
                     {"rel_path": "b.md", "heading": "B", "text": "beta note", "ordinal": 1}])
    mem = MemoryStore(tmp_path / ".weft")
    mem.remember("beta", type="fact")
    s = ChatSession(emb, store, FakeLLM(response="ok"), memory=mem,
                    k=2, memory_query=True)
    turn = s.send("alpha")
    assert turn.answer == "ok"
    assert set(s.last_sources) == {"a.md", "b.md"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_chat.py -k memory_query -v`
Expected: FAIL — `ChatSession` has no `memory_query` kwarg.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/chat.py`, add the import of `build_memory_query` (extend the existing
`from weft.agent import ...` line):

```python
from weft.agent import (
    SYSTEM, build_memory_query, build_prompt, collect_memory, dual_query_retrieve,
)
```

Add `memory_query: bool = False` to `__init__` and store it:

```python
    def __init__(self, embedder, store, llm, *, graph=None, memory=None,
                 k: int = 5, window: int = 6, rewrite_llm: bool = False, bm25=None,
                 reranker=None, memory_query: bool = False):
        ...
        self._reranker = reranker
        self._memory_query = memory_query
```

In `_retrieve`, compute the memory query and pass it into both `dual_query_retrieve`
calls:

```python
    def _retrieve(self, question: str):
        ctx = self._context_query(question)
        mq = build_memory_query(self._memory) if self._memory_query else None
        if self._reranker is not None:
            pool = dual_query_retrieve(question, ctx, self._embedder, self._store,
                                       graph=self._graph, k=RERANK_POOL,
                                       bm25=self._bm25, memory_query=mq)
            rerank_query = f"{ctx}\n{question}" if ctx else question
            return self._reranker.rerank(rerank_query, pool, self._k)
        return dual_query_retrieve(question, ctx, self._embedder, self._store,
                                   graph=self._graph, k=self._k, bm25=self._bm25,
                                   memory_query=mq)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_chat.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/weft/chat.py tests/test_chat.py
git commit -m "feat(weft): M12 ChatSession memory-query flag"
```

---

## Task 3: `service_ask(use_memory_query=False)` (`service.py`)

**Files:**
- Modify: `src/weft/service.py`
- Test: `tests/test_service.py` (append)

- [ ] **Step 1: Read the current service_ask body**

Read `src/weft/service.py` lines 128-160 to see how `memory`/`ask` are wired before editing.

- [ ] **Step 2: Write the failing test (append to tests/test_service.py)**

```python
def test_service_ask_memory_query_runs(tmp_path, monkeypatch):
    import weft.service as service
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    from weft.memory import MemoryStore
    from weft.index import build_index

    vault = tmp_path / "v"
    vault.mkdir()
    (vault / "n.md").write_text("# A\nalpha beta\n", encoding="utf-8")
    store_path = tmp_path / ".weft" / "index"
    monkeypatch.setattr(service, "make_embedder", lambda: FakeEmbedder(dim=16))
    build_index(vault, FakeEmbedder(dim=16), store_path)

    mem = MemoryStore(store_path.parent)
    mem.remember("alpha", type="fact")
    monkeypatch.setattr(service, "make_memory", lambda sp: mem)
    monkeypatch.setattr(service, "make_llm", lambda *a, **k: FakeLLM(response="answer [1]"))

    result = service_ask(store_path, "beta", k=3, use_memory_query=True)
    assert result["answer"] == "answer [1]"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_service.py -k memory_query -v`
Expected: FAIL — `service_ask` has no `use_memory_query`.

- [ ] **Step 4: Write minimal implementation**

In `src/weft/service.py`, add `use_memory_query: bool = False` to `service_ask`'s
keyword-only args (alongside `rerank`), import `build_memory_query`, build the query
when memory is enabled, and pass it to `ask`. Extend the existing agent import:

```python
from weft.agent import ask, build_memory_query
```

Then, where `memory` is resolved and `ask(...)` is called:

```python
    memory = make_memory(sp) if use_memory else None
    mq = build_memory_query(memory) if use_memory_query else None
    result = ask(question, make_embedder(), store, llm, k=k, graph=graph,
                 memory=memory, bm25=bm25, reranker=reranker, memory_query=mq)
```

(Keep every other argument of the existing `ask(...)` call exactly as it is; only add
`memory_query=mq`. If the existing call spans multiple lines with `rerank`/`bm25`, add
`memory_query=mq` to it without changing the others.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_service.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/weft/service.py tests/test_service.py
git commit -m "feat(weft): M12 service_ask use_memory_query"
```

---

## Task 4: CLI `--memory-query` on ask/chat + docs

**Files:**
- Modify: `src/weft/cli.py`, `docs/how-to/configure-env-and-use-cli.md`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_cli.py)**

```python
def test_cli_ask_and_chat_memory_query(sample_vault, tmp_path, monkeypatch, capsys):
    from weft.embeddings import FakeEmbedder
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="A [1]"))
    idx = tmp_path / "idx"
    cli.main(["index", str(sample_vault), "--store", str(idx)])
    rc = cli.main(["ask", "hello", "--store", str(idx), "--memory-query"])
    assert rc == 0
    # chat parses the flag (feed an immediate /exit so the REPL returns)
    monkeypatch.setattr(cli, "run_repl", lambda session: 0)
    rc2 = cli.main(["chat", "--store", str(idx), "--memory-query"])
    assert rc2 == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli.py -k cli_ask_and_chat_memory_query -v`
Expected: FAIL — `ask`/`chat` reject `--memory-query`.

- [ ] **Step 3: Write minimal implementation**

In `cli.py` `_cmd_ask`, pass the flag into `service_ask` (add `use_memory_query=args.memory_query`
to the existing call). In `_cmd_chat`, pass `memory_query=args.memory_query` into the
`ChatSession(...)` constructor.

In `build_parser`, add the flag to both `p_ask` and `p_chat`. There is an existing
loop that adds shared hybrid/rerank flags to `(p_ask, p_chat)`; add `--memory-query`
in that same loop:

```python
    for hybrid_sub in (p_ask, p_chat):
        ...  # existing --no-hybrid / --rerank
        hybrid_sub.add_argument(
            "--memory-query", action="store_true",
            help="Fuse a query built from durable facts + decisions into retrieval.",
        )
```

- [ ] **Step 4: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS across all modules.

- [ ] **Step 5: Document + commit**

Add a subsection after the rerank how-to in `docs/how-to/configure-env-and-use-cli.md`:

```markdown
### Bias retrieval toward what Weft remembers

`--memory-query` (on `ask` and `chat`) builds an extra search query from your active
durable **facts and decisions** (preferences are excluded) and fuses it as one more
RRF ranking, so answers lean toward your standing context:

    uv run weft ask "what did I decide?" --memory-query
    uv run weft chat --memory-query

It stays fully local — the memory query is assembled from already-stored memory text
and never leaves the machine. Off by default; composes with `--rewrite-llm`,
hybrid search, and `--rerank`.
```

```bash
git add src/weft/cli.py tests/test_cli.py docs/how-to/configure-env-and-use-cli.md
git commit -m "feat(weft): M12 weft ask/chat --memory-query + docs"
```

---

## Self-Review Notes (author)

- **Spec coverage:** `build_memory_query` (Task 1, §4) · `memory_query` on
  `dual_query_retrieve`/`build_graph`/`ask` (Task 1, §5) · `ChatSession` flag
  (Task 2, §5) · `service_ask(use_memory_query=)` (Task 3, §6) · CLI `--memory-query`
  on ask + chat + docs (Task 4, §6).
- **Type consistency:** `build_memory_query(memory) -> str | None`; every retrieval
  hop uses the keyword `memory_query=`; the ChatSession/service/CLI toggles are
  booleans (`memory_query: bool`, `use_memory_query: bool`, `--memory-query`).
- **Regression control:** all new params default to `None`/`False`, so every existing
  ask/chat/service/CLI path is byte-identical; only the opt-in flag changes behavior.
```
