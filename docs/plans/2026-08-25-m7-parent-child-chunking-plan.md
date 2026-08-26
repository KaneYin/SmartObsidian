# M7 Parent/Child Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable chunking-strategy layer and a `parent_child` strategy that embeds paragraph children for precise matching but returns the parent heading-section for context.

**Architecture:** `chunking.py` maps a strategy name to a `Note -> list[Chunk]` function. `parent_child` splits each heading section into paragraph children carrying the parent section's id + text. `build_index` persists those fields; `build_prompt` collapses hits by parent and shows the parent text. Heading mode is unchanged and default.

**Tech Stack:** Python ≥3.11 stdlib (`hashlib`), existing `parser`/`index`/`agent`, `pytest` with `FakeEmbedder`.

**Scope:** M7 — pluggable layer + parent/child. Contextual Retrieval (A) and sliding-window (C) register in `chunking.py` later.

**Run tests with:** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Modify `src/weft/parser.py` — `Chunk` gains optional `parent_id`/`parent_text`.
- Create `src/weft/chunking.py` — `chunk_strategy`, `heading`, `parent_child`, paragraph splitter.
- Modify `src/weft/index.py` — `build_index(chunking=...)`; persist parent fields; manifest.
- Modify `src/weft/agent.py` — `build_prompt` dedup by parent.
- Modify `src/weft/cli.py` — `weft index --chunking {heading,parent-child}`.
- Tests: `test_chunking.py`, `test_index_chunking.py`, `test_agent_parent.py`, `test_cli.py` addition.

---

## Task 1: `Chunk` gains parent fields (`parser.py`)

**Files:**
- Modify: `src/weft/parser.py:32-37`
- Test: `tests/test_parser.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_parser.py)**

```python
def test_chunk_parent_fields_default_none():
    from weft.parser import Chunk
    c = Chunk(rel_path="a.md", heading="H", text="t")
    assert c.parent_id is None and c.parent_text is None
    c2 = Chunk(rel_path="a.md", heading="H", text="t", parent_id="par_x", parent_text="section")
    assert c2.parent_id == "par_x" and c2.parent_text == "section"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_parser.py -k parent_fields -v`
Expected: FAIL — `Chunk` has no `parent_id`.

- [ ] **Step 3: Write minimal implementation**

Replace the `Chunk` dataclass in `src/weft/parser.py`:

```python
@dataclass
class Chunk:
    rel_path: str
    heading: str
    text: str
    ordinal: int = field(default=0)
    parent_id: str | None = None
    parent_text: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_parser.py -v`
Expected: PASS (existing parser tests + new one).

- [ ] **Step 5: Commit**

```bash
git add src/weft/parser.py tests/test_parser.py
git commit -m "feat(weft): M7 optional parent fields on Chunk"
```

---

## Task 2: Chunking strategies (`chunking.py`)

**Files:**
- Create: `src/weft/chunking.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from weft.chunking import chunk_strategy
from weft.parser import parse_note


def _note(tmp_path, body):
    p = tmp_path / "n.md"
    p.write_text(body, encoding="utf-8")
    note = parse_note(p)
    note.rel_path = "n.md"
    return note


def test_heading_strategy_matches_chunk_note(tmp_path):
    from weft.parser import chunk_note
    note = _note(tmp_path, "# A\npara one\n\npara two\n\n# B\nmore")
    assert chunk_strategy("heading")(note) == chunk_note(note)


def test_parent_child_splits_paragraphs_sharing_parent(tmp_path):
    note = _note(tmp_path, "# A\npara one\n\npara two\n")
    chunks = chunk_strategy("parent_child")(note)
    texts = [c.text for c in chunks]
    assert texts == ["para one", "para two"]
    # both children share one parent id and the full section text
    assert chunks[0].parent_id == chunks[1].parent_id
    assert "para one" in chunks[0].parent_text and "para two" in chunks[0].parent_text


def test_parent_child_single_paragraph_is_one_child(tmp_path):
    note = _note(tmp_path, "# A\njust one paragraph\n")
    chunks = chunk_strategy("parent_child")(note)
    assert len(chunks) == 1
    assert chunks[0].text == "just one paragraph"
    assert chunks[0].parent_text.strip() == "just one paragraph"


def test_parent_child_keeps_fence_whole(tmp_path):
    note = _note(tmp_path, "# A\nintro\n\n```\ncode\n\nstill code\n```\n")
    chunks = chunk_strategy("parent_child")(note)
    fence_children = [c.text for c in chunks if "code" in c.text]
    assert any("still code" in t for t in fence_children)  # blank line inside fence not split


def test_unknown_strategy_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        chunk_strategy("bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_chunking.py -v`
Expected: FAIL — `weft.chunking` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/chunking.py
"""Pluggable chunking strategies. `heading` reproduces the original heading-boundary
chunks; `parent_child` embeds paragraph children but carries the parent heading
section for context. Contextual Retrieval and sliding-window register here later."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from weft.parser import Chunk, Note, chunk_note


def _parent_id(rel_path: str, section_ordinal: int) -> str:
    digest = hashlib.sha1(f"{rel_path}#{section_ordinal}".encode("utf-8")).hexdigest()
    return "par_" + digest[:8]


def _split_paragraphs(text: str) -> list[str]:
    """Split a section into paragraphs on blank lines, keeping fenced code blocks
    whole. A section with no blank line yields a single paragraph."""
    paras: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buf
        joined = "\n".join(buf).strip()
        if joined:
            paras.append(joined)
        buf = []

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not in_fence and line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return paras


def _parent_child_chunks(note: Note) -> list[Chunk]:
    out: list[Chunk] = []
    ordinal = 0
    for section in chunk_note(note):
        pid = _parent_id(note.rel_path, section.ordinal)
        for para in _split_paragraphs(section.text):
            out.append(Chunk(
                rel_path=note.rel_path, heading=section.heading, text=para,
                ordinal=ordinal, parent_id=pid, parent_text=section.text,
            ))
            ordinal += 1
    return out


def chunk_strategy(name: str) -> Callable[[Note], list[Chunk]]:
    if name == "heading":
        return chunk_note
    if name == "parent_child":
        return _parent_child_chunks
    raise ValueError(f"unknown chunking strategy: {name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_chunking.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/chunking.py tests/test_chunking.py
git commit -m "feat(weft): M7 pluggable chunking + parent_child strategy"
```

---

## Task 3: build_index writes parent fields + manifest (`index.py`)

**Files:**
- Modify: `src/weft/index.py`
- Test: `tests/test_index_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_index_chunking.py
import json

from weft.embeddings import FakeEmbedder
from weft.index import build_index, manifest_path_for
from weft.store import VectorStore


def _vault(tmp_path):
    (tmp_path / "n.md").write_text("# A\npara one\n\npara two\n", encoding="utf-8")
    return tmp_path


def test_parent_child_persists_parent_fields(tmp_path):
    vault = _vault(tmp_path)
    store_path = tmp_path / "idx"
    build_index(vault, FakeEmbedder(dim=16), store_path, chunking="parent_child")
    store = VectorStore.load(store_path)
    metas = store.metadata_by_note()["n.md"]
    assert [m["text"] for m in metas] == ["para one", "para two"]
    assert all(m["parent_id"] == metas[0]["parent_id"] for m in metas)
    assert all("para one" in m["parent_text"] for m in metas)
    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["chunking"] == "parent_child"


def test_default_heading_omits_parent_fields(tmp_path):
    vault = _vault(tmp_path)
    store_path = tmp_path / "idx"
    build_index(vault, FakeEmbedder(dim=16), store_path)
    metas = VectorStore.load(store_path).metadata_by_note()["n.md"]
    assert "parent_id" not in metas[0]
    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["chunking"] == "heading"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_index_chunking.py -v`
Expected: FAIL — `build_index` has no `chunking` param.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/index.py`, add the import:

```python
from weft.chunking import chunk_strategy
```

Change the `build_index` signature and the chunk/metadata/manifest sections. Replace
the body from the `pairs = ...` line through the `manifest = {...}` block:

```python
def build_index(
    vault_path: Path,
    embedder: Embedder,
    store_path: Path,
    policy: PrivacyPolicy | None = None,
    chunking: str = "heading",
) -> tuple[int, int]:
    """Index every chunk and build the link graph.
    Returns (n_chunks, n_edges)."""
    root = vault_root(vault_path)
    policy = policy or PrivacyPolicy()
    notes = parse_vault(root, policy=policy)
    strategy = chunk_strategy(chunking)
    pairs = [(note, c) for note in notes for c in strategy(note)]

    store = VectorStore(dim=embedder.dim)
    if pairs:
        vectors = embedder.embed([c.text for _, c in pairs])
        metadatas = []
        for note, c in pairs:
            meta = {
                "rel_path": c.rel_path,
                "heading": c.heading,
                "text": c.text,
                "ordinal": c.ordinal,
                "tags": note.tags,
                "wikilinks": note.wikilinks,
            }
            if c.parent_id is not None:
                meta["parent_id"] = c.parent_id
            if c.parent_text is not None:
                meta["parent_text"] = c.parent_text
            metadatas.append(meta)
        store.add_batch(vectors, metadatas)
    store.save(Path(store_path))

    graph = LinkGraph.from_notes(notes)
    graph.save(graph_path_for(store_path))

    manifest = {
        "version": 1,
        "vault_root": str(root),
        "privacy": policy.as_dict(),
        "chunking": chunking,
    }
```

(Leave the `secure_write_text(manifest_path_for(...))` and `return` lines below
unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_index_chunking.py tests/test_index.py tests/test_empty_and_batch.py -v`
Expected: PASS (new tests + existing index tests — the manifest now has an extra key,
which the existing tests don't forbid).

- [ ] **Step 5: Commit**

```bash
git add src/weft/index.py tests/test_index_chunking.py
git commit -m "feat(weft): M7 build_index chunking strategy + parent metadata"
```

---

## Task 4: build_prompt collapses hits by parent (`agent.py`)

**Files:**
- Modify: `src/weft/agent.py` (`build_prompt`)
- Test: `tests/test_agent_parent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_parent.py
import json

from weft.agent import build_prompt
from weft.store import SearchHit


def _hit(rel, ordinal, text, parent_id=None, parent_text=None):
    meta = {"rel_path": rel, "heading": "H", "text": text, "ordinal": ordinal}
    if parent_id is not None:
        meta["parent_id"] = parent_id
        meta["parent_text"] = parent_text
    return SearchHit(score=1.0, metadata=meta)


def test_build_prompt_collapses_children_of_one_parent():
    hits = [
        _hit("n.md", 0, "child a", parent_id="par_1", parent_text="the full section"),
        _hit("n.md", 1, "child b", parent_id="par_1", parent_text="the full section"),
    ]
    payload = json.loads(build_prompt("q", hits))
    assert len(payload["sources"]) == 1
    assert payload["sources"][0]["content"] == "the full section"
    assert payload["sources"][0]["id"] == 1


def test_build_prompt_heading_mode_unchanged():
    hits = [_hit("a.md", 0, "alpha"), _hit("b.md", 0, "beta")]
    payload = json.loads(build_prompt("q", hits))
    assert [s["content"] for s in payload["sources"]] == ["alpha", "beta"]
    assert [s["id"] for s in payload["sources"]] == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_agent_parent.py -v`
Expected: FAIL — `build_prompt` emits one source per hit (2, not 1) and uses `text`.

- [ ] **Step 3: Write minimal implementation**

Replace the source-building loop in `build_prompt` (`src/weft/agent.py`) — the block
that builds the `sources` list — with a parent-collapsing version:

```python
    sources: list[dict] = []
    seen: set = set()
    for h in hits:
        m = h.metadata
        key = m.get("parent_id") or (m["rel_path"], m["ordinal"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "id": len(sources) + 1,
                "path": m["rel_path"],
                "heading": m["heading"],
                "content": m.get("parent_text") or m["text"],
            }
        )
```

(The rest of `build_prompt` — the `payload` dict, the `memory`/`conversation`
additions, and the `json.dumps` return — stays exactly as is.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_agent_parent.py tests/test_agent.py tests/test_agent_graph.py tests/test_agent_memory.py -v`
Expected: PASS (new tests + all existing agent tests — heading-mode hits each have a
unique `(rel_path, ordinal)` so nothing collapses).

- [ ] **Step 5: Commit**

```bash
git add src/weft/agent.py tests/test_agent_parent.py
git commit -m "feat(weft): M7 collapse retrieval hits by parent in build_prompt"
```

---

## Task 5: `weft index --chunking` + docs

**Files:**
- Modify: `src/weft/cli.py` (`_cmd_index`, `build_parser`), `docs/how-to/configure-env-and-use-cli.md`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing test (append to tests/test_cli.py)**

```python
def test_cli_index_parent_child(sample_vault, tmp_path, monkeypatch):
    from weft.embeddings import FakeEmbedder
    from weft.store import VectorStore
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    idx = tmp_path / "idx"
    rc = cli.main(["index", str(sample_vault), "--store", str(idx), "--chunking", "parent-child"])
    assert rc == 0
    metas = next(iter(VectorStore.load(idx).metadata_by_note().values()))
    assert "parent_id" in metas[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli.py -k parent_child -v`
Expected: FAIL — `index` has no `--chunking` option (argparse SystemExit 2).

- [ ] **Step 3: Write minimal implementation**

In `cli.py` `_cmd_index`, pass the strategy (mapping the CLI hyphen form to the
underscore strategy name). Replace the `build_index(...)` call:

```python
    n_chunks, n_edges = build_index(
        Path(args.vault), make_embedder(), Path(args.store), policy=policy,
        chunking=args.chunking.replace("-", "_"),
    )
```

In `build_parser`, add the flag to the `p_index` subparser (near the other index
options):

```python
    p_index.add_argument(
        "--chunking", choices=["heading", "parent-child"], default="heading",
        help="Chunking strategy: heading (default) or parent-child.",
    )
```

- [ ] **Step 4: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS across all modules.

- [ ] **Step 5: Document + commit**

Add to `docs/how-to/configure-env-and-use-cli.md`, in the indexing section:

```markdown
### Parent/child chunking

For more precise retrieval on long notes, index with parent/child chunking:

    weft index "/path/to/Vault" --chunking parent-child

This embeds paragraph-sized *children* for precise matching but hands the model the
whole heading *section* (the parent) for context, de-duplicated so one section is
sent once even when several of its paragraphs match. The default `heading` strategy
is unchanged. The chosen strategy is recorded in `index.manifest.json`; `ask` reads
the parent context straight from the stored metadata (no extra flags).
```

```bash
git add src/weft/cli.py tests/test_cli.py docs/how-to/configure-env-and-use-cli.md
git commit -m "feat(weft): M7 weft index --chunking parent-child + docs"
```

---

## Self-Review Notes (author)

- **Spec coverage:** pluggable layer + heading/parent_child (Task 2) · Chunk parent
  fields (Task 1) · build_index chunking + parent metadata + manifest (Task 3) ·
  build_prompt dedup-by-parent, heading no-op (Task 4) · `--chunking` flag + manifest
  record + docs (Task 5). Redaction-before-chunk and untrusted-text discipline are
  inherited unchanged (parse_vault redacts before chunking; build_prompt already
  treats content as data). A (contextual) and C (sliding) are out of scope, noted.
- **Type consistency:** `Chunk(..., parent_id=None, parent_text=None)`;
  `chunk_strategy(name) -> Callable[[Note], list[Chunk]]`; `_parent_id`,
  `_split_paragraphs`, `_parent_child_chunks`; metadata keys `parent_id`/`parent_text`;
  `build_index(..., chunking="heading")`; CLI hyphen `parent-child` → strategy
  `parent_child`. Consistent across tasks.
- **Regression control:** heading mode omits parent fields, so old indexes and every
  existing agent/index/cli test behave identically; the new `chunking` manifest key is
  additive.
```
