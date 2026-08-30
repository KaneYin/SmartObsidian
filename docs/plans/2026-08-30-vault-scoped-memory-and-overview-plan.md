# Vault-scoped memory + vault-overview grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop episodic agent memory from bleeding across vaults, guard `weft index` against silently overwriting a store bound to a different vault, and give broad "what is this vault about"-style questions a real chunk to retrieve.

**Architecture:** `Episode` records gain a `vault_root` stamp read from the index manifest at `MemoryStore` construction time, and `MemoryStore.episodes()` filters to the current vault before anything reaches recall. `service_index()` reuses the existing `IndexVaultMismatchError` pattern (already used by `service_suggest()`) to refuse overwriting a differently-bound store unless `--force` is passed. `build_index()` appends one synthetic "(vault overview)" chunk — folder/tag statistics computed from already privacy-filtered notes — into the normal chunk/embed/BM25 pipeline, so it competes for retrieval like any other chunk with no new query-classification code.

**Tech Stack:** Python 3.12+, pytest, `weft.memory`/`weft.index`/`weft.service`/`weft.cli` (existing modules), `FakeEmbedder`/`FakeLLM` test doubles already in the codebase.

Spec: `docs/design/2026-08-30-vault-scoped-memory-and-overview-design.md`

---

### Task 1: Stamp and filter episodic memory by vault root

**Files:**
- Modify: `src/weft/memory.py` (`Episode` dataclass, `MemoryStore.__init__`, `log_episode`, `episodes`)
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory.py` (keep the existing `_store` helper as-is — new tests construct `MemoryStore` directly so they don't disturb the other tests in this file):

```python
def test_episode_stamped_with_vault_root(tmp_path):
    ms = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                      vault_root="/vault/a")
    ms.log_episode("q", "a", ["n.md"])
    eps = ms.episodes()
    assert len(eps) == 1
    assert eps[0].vault_root == "/vault/a"


def test_episodes_exclude_different_vault_root(tmp_path):
    # Same store path, two different vaults -- the exact bug scenario.
    ms_a = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/a")
    ms_a.log_episode("what is this vault about", "vault A content", ["a.md"])
    ms_b = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/b")
    assert ms_b.episodes() == []


def test_episodes_exclude_missing_vault_root_field(tmp_path):
    # A pre-fix record with no vault_root key at all must never match, even
    # if the current store also happens to be unscoped.
    episodes_path = tmp_path / "episodes.jsonl"
    episodes_path.write_text(json.dumps({
        "id": "ep_legacy", "ts": "2026-08-28T00:00:00+00:00",
        "question": "old q", "answer": "old a", "sources": [],
    }) + "\n")
    os.chmod(episodes_path, 0o600)
    ms = MemoryStore(tmp_path / "memory.jsonl", episodes_path, vault_root="")
    assert ms.episodes() == []


def test_recall_log_kind_excludes_other_vault(tmp_path):
    from weft.embeddings import FakeEmbedder
    ms_a = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/a")
    ms_a.log_episode("what is this vault about", "vault A is about coffee", ["a.md"])
    ms_b = MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl",
                        vault_root="/vault/b")
    hits = ms_b.recall(FakeEmbedder(dim=16), "what is this vault about", k=5, kinds={"log"})
    assert hits == []
```

Add `import json` to the top of `tests/test_memory.py` (it already imports `os` and `stat`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_memory.py -v`
Expected: the four new tests fail — `test_episode_stamped_with_vault_root` and
`test_episodes_exclude_different_vault_root` with `TypeError: __init__() got an
unexpected keyword argument 'vault_root'`; the "missing field" and "recall"
tests either fail the same way or (once the signature issue is worked around)
because nothing filters by vault yet.

- [ ] **Step 3: Implement vault scoping in `memory.py`**

In `src/weft/memory.py`, change the `Episode` dataclass:

```python
@dataclass
class Episode:
    id: str
    ts: str
    question: str
    answer: str
    sources: list[str]
    vault_root: str | None = None
```

Change `MemoryStore.__init__`:

```python
class MemoryStore:
    def __init__(self, memory_path: Path, episodes_path: Path, vault_root: str = ""):
        self._memory_path = Path(memory_path)
        self._episodes_path = Path(episodes_path)
        self._vault_root = vault_root
```

Change `log_episode`:

```python
    def log_episode(self, question: str, answer: str, sources: list[str]) -> Episode:
        ep = Episode(
            id="ep_" + secrets.token_hex(4),
            ts=_now(),
            question=question,
            answer=answer,
            sources=list(sources),
            vault_root=self._vault_root,
        )
        secure_append_json(self._episodes_path, asdict(ep))
        return ep
```

Change `episodes`:

```python
    def episodes(self) -> list[Episode]:
        """Only episodes logged against this exact vault_root. A missing or
        mismatched vault_root (including every pre-scoping record) is
        excluded -- fail closed rather than assume it matches."""
        return [
            ep for ep in (Episode(**rec) for rec in _read_jsonl(self._episodes_path))
            if ep.vault_root == self._vault_root
        ]
```

`recall()` needs no change: its `"log"` branch already calls `self.episodes()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_memory.py -v`
Expected: all tests in the file PASS, including the pre-existing ones (they
never pass `vault_root`, so both the write side and read side default to
`""` and keep matching each other).

- [ ] **Step 5: Run the full existing test suite to check for other breakage**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS. `Episode(**rec)` is used in `tests/test_agent_memory.py`,
`tests/test_memory_recall.py`, `tests/test_chat.py`, `tests/test_agent.py`,
and `tests/test_service.py` via `MemoryStore(...)` with no `vault_root`
argument — all default to `""` on both write and read, so behavior is
unchanged for them.

- [ ] **Step 6: Commit**

```bash
git add src/weft/memory.py tests/test_memory.py
git commit -m "$(cat <<'EOF'
fix(weft): scope episodic memory recall to the vault it was logged against

Episode gained a vault_root stamp; MemoryStore.episodes() now excludes any
episode whose vault_root doesn't match the current store, including
pre-fix records with no vault_root at all (fail closed). This stops a
stale Q&A from an unrelated vault winning memory-query recall purely by
text similarity to a new question.
EOF
)"
```

---

### Task 2: Thread the vault root from the index manifest into `MemoryStore`

**Files:**
- Modify: `src/weft/index.py` (add `read_vault_root`)
- Modify: `src/weft/service.py` (`make_memory`)
- Modify: `src/weft/cli.py` (`make_memory`)
- Test: `tests/test_service.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_service.py`:

```python
def test_make_memory_stamps_episodes_with_manifest_vault_root(tmp_path):
    import weft.service as S
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "n.md").write_text("# N\n\nhello\n")
    store = tmp_path / ".weft" / "index"
    S.service_index(vault, store, embedder=FakeEmbedder(dim=16))

    mem = S.make_memory(store)
    mem.log_episode("q", "a", ["n.md"])
    assert mem.episodes()[0].vault_root == str(vault.resolve())


def test_make_memory_defaults_to_empty_vault_root_before_indexing(tmp_path):
    import weft.service as S
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True)
    mem = S.make_memory(store)
    mem.log_episode("q", "a", [])
    assert mem.episodes()[0].vault_root == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_service.py -k make_memory -v`
Expected: `test_make_memory_stamps_episodes_with_manifest_vault_root` FAILS —
`mem.episodes()[0].vault_root == ""`, not the vault path, because
`make_memory` doesn't read the manifest yet. The second test currently
passes already (both sides default to `""`); it's here to lock in that
behavior going forward.

- [ ] **Step 3: Add `read_vault_root` to `index.py`**

In `src/weft/index.py`, right after `manifest_path_for`:

```python
def read_vault_root(store_path: Path) -> str:
    """The vault_root recorded in this store's manifest, or "" if the store
    has no manifest yet (fresh index, or a pre-manifest one)."""
    path = manifest_path_for(store_path)
    if not path.exists():
        return ""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return manifest.get("vault_root") or ""
```

- [ ] **Step 4: Wire it into `service.py`'s `make_memory`**

In `src/weft/service.py`, change the import line:

```python
from weft.index import build_index, graph_path_for, manifest_path_for, read_vault_root
```

And change `make_memory`:

```python
def make_memory(store_path: Path) -> MemoryStore:
    return MemoryStore(store_path.parent / "memory.jsonl",
                       store_path.parent / "episodes.jsonl",
                       vault_root=read_vault_root(store_path))
```

- [ ] **Step 5: Wire it into `cli.py`'s `make_memory`**

In `src/weft/cli.py`, add an import:

```python
from weft.index import read_vault_root
```

And change `make_memory`:

```python
def make_memory(store_path: Path) -> MemoryStore:
    return MemoryStore(
        store_path.parent / "memory.jsonl",
        store_path.parent / "episodes.jsonl",
        vault_root=read_vault_root(store_path),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_service.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/weft/index.py src/weft/service.py src/weft/cli.py tests/test_service.py
git commit -m "$(cat <<'EOF'
fix(weft): read vault_root from the index manifest for MemoryStore

Both CLI and service make_memory() now stamp new episodes with the
vault_root recorded in the current index's manifest, so Task 1's
filtering actually reflects which vault is currently indexed at a given
store path.
EOF
)"
```

---

### Task 3: Refuse to silently overwrite a differently-bound index

**Files:**
- Modify: `src/weft/service.py` (`service_index`)
- Modify: `src/weft/cli.py` (`--force` flag, `_cmd_index`)
- Test: `tests/test_service.py`, `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_service.py`:

```python
def test_service_index_refuses_different_vault_without_force(tmp_path):
    import pytest
    from weft.service import IndexVaultMismatchError, service_index
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    (vault_a / "a.md").write_text("# A\n\nhello\n")
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    (vault_b / "b.md").write_text("# B\n\nhello\n")
    store = tmp_path / ".weft" / "index"

    service_index(vault_a, store, embedder=FakeEmbedder(dim=16))
    with pytest.raises(IndexVaultMismatchError):
        service_index(vault_b, store, embedder=FakeEmbedder(dim=16))


def test_service_index_force_overwrites_different_vault(tmp_path):
    from weft.service import service_index
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    (vault_a / "a.md").write_text("# A\n\nhello\n")
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    (vault_b / "b.md").write_text("# B\n\nhello\n")
    store = tmp_path / ".weft" / "index"

    service_index(vault_a, store, embedder=FakeEmbedder(dim=16))
    result = service_index(vault_b, store, embedder=FakeEmbedder(dim=16), force=True)
    assert result["vault"] == str(vault_b)


def test_service_index_same_vault_reindex_needs_no_force(tmp_path):
    from weft.service import service_index
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# A\n\nhello\n")
    store = tmp_path / ".weft" / "index"
    service_index(vault, store, embedder=FakeEmbedder(dim=16))
    result = service_index(vault, store, embedder=FakeEmbedder(dim=16))
    assert result["chunks"] >= 1
```

Add to `tests/test_cli.py`:

```python
def test_cli_index_refuses_different_vault_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    vault_a = tmp_path / "vault_a"
    vault_a.mkdir()
    (vault_a / "a.md").write_text("# A\n\nhello\n")
    vault_b = tmp_path / "vault_b"
    vault_b.mkdir()
    (vault_b / "b.md").write_text("# B\n\nhello\n")
    store = tmp_path / "idx"

    assert cli.main(["index", str(vault_a), "--store", str(store)]) == 0

    rc = cli.main(["index", str(vault_b), "--store", str(store)])
    assert rc == 1
    assert "different vault" in capsys.readouterr().err

    rc = cli.main(["index", str(vault_b), "--store", str(store), "--force"])
    assert rc == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_service.py tests/test_cli.py -k "different_vault or force" -v`
Expected: the `service_index` tests fail because `service_index()` accepts
no `force` keyword yet and never raises `IndexVaultMismatchError`; the CLI
test fails with `SystemExit`/argparse error on the unrecognized `--force`
flag.

- [ ] **Step 3: Add the guard to `service_index()`**

In `src/weft/service.py`, change the `service_index` signature and add the
check at the top of the function body (it already imports `json`,
`manifest_path_for`, and `vault_root`):

```python
def service_index(
    vault_path: str | Path,
    store_path: str | Path,
    *,
    policy: PrivacyPolicy | None = None,
    chunking: str = "heading",
    no_bm25: bool = False,
    contextual: bool = False,
    force: bool = False,
    overrides: dict | None = None,
    embedder: Embedder | None = None,
    on_fallback=None,
) -> dict:
    """Build an index without coupling the workflow to a terminal interface."""
    sp = Path(store_path)
    manifest_path = manifest_path_for(sp)
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_root = existing.get("vault_root")
        requested_root = str(vault_root(Path(vault_path)))
        if existing_root is not None and existing_root != requested_root:
            raise IndexVaultMismatchError(
                f"store at {sp} belongs to a different vault ({existing_root}); "
                "pass --force to overwrite it"
            )
    contextual_llm = None
    if contextual:
        raw = make_llm(overrides or {}, sp, on_fallback=on_fallback)
        contextual_llm = AuditedLLM(
            raw, sp.parent / "api-log.jsonl", "contextualize"
        )
    chunks, edges = build_index(
        Path(vault_path),
        embedder or make_embedder(),
        sp,
        policy=policy or PrivacyPolicy(),
        chunking=chunking.replace("-", "_"),
        no_bm25=no_bm25,
        contextual_llm=contextual_llm,
    )
    return {
        "chunks": chunks,
        "edges": edges,
        "vault": str(vault_path),
        "store": str(sp),
    }
```

- [ ] **Step 4: Add `--force` and wire it through `cli.py`**

In `src/weft/cli.py`, in `_cmd_index`, add `force=args.force` to the
`service.service_index(...)` call, and catch the new error:

```python
    try:
        result = service.service_index(
            args.vault,
            args.store,
            policy=policy,
            chunking=args.chunking,
            no_bm25=args.no_bm25,
            contextual=args.contextual,
            force=args.force,
            overrides=_llm_overrides(args),
            embedder=make_embedder(),
            on_fallback=_fallback_notice,
        )
    except (ProviderUnavailable, service.IndexVaultMismatchError) as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
```

In the `p_index` argparse block, right before `p_index.set_defaults(func=_cmd_index)`:

```python
    p_index.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing store even if it belongs to a different vault.",
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_service.py tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/weft/service.py src/weft/cli.py tests/test_service.py tests/test_cli.py
git commit -m "$(cat <<'EOF'
fix(weft): refuse to silently overwrite an index bound to another vault

weft index now raises IndexVaultMismatchError (reusing the same check
service_suggest already performs) when the store's existing manifest
names a different vault_root, unless --force is passed. Prevents the
class of accident where a shared default store path clobbers an
unrelated vault's index.
EOF
)"
```

---

### Task 4: Synthetic vault-overview chunk for corpus-level questions

**Files:**
- Modify: `src/weft/index.py` (`build_overview_chunk`, wiring into `build_index`)
- Modify: `tests/test_agent.py` (fix an assertion invalidated by the new chunk)
- Test: `tests/test_index_overview.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_index_overview.py`:

```python
from weft.embeddings import FakeEmbedder
from weft.index import build_index, build_overview_chunk
from weft.parser import Note
from weft.store import VectorStore


def test_overview_chunk_folder_and_tag_breakdown():
    notes = [
        Note(rel_path="coffee.md", title="Coffee", frontmatter={},
             tags=["drinks"], wikilinks=[], body=""),
        Note(rel_path="notes/tea.md", title="Tea", frontmatter={},
             tags=[], wikilinks=[], body=""),
    ]
    chunk = build_overview_chunk(notes)
    assert chunk.rel_path == "(vault overview)"
    assert chunk.heading == "Vault overview"
    assert "2 notes across 2 top-level folders" in chunk.text
    assert "#drinks (1)" in chunk.text


def test_overview_chunk_none_for_empty_notes():
    assert build_overview_chunk([]) is None


def test_build_index_adds_overview_chunk_to_store(sample_vault, tmp_path):
    store_path = tmp_path / "idx"
    build_index(sample_vault, FakeEmbedder(dim=16), store_path)
    store = VectorStore.load(store_path)
    overview = [m for m in store._metadata if m["rel_path"] == "(vault overview)"]
    assert len(overview) == 1
    text = overview[0]["text"]
    assert "3 notes across 2 top-level folders" in text
    assert "notes/ (1 notes)" in text
    assert "#strong (1)" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_index_overview.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_overview_chunk'`.

- [ ] **Step 3: Implement `build_overview_chunk` and wire it into `build_index`**

In `src/weft/index.py`, add the import and function:

```python
from collections import Counter
```

(add alongside the existing `from pathlib import Path` import block)

```python
from weft.parser import Note, parse_vault
```

(replaces the existing `from weft.parser import parse_vault` line)

```python
def build_overview_chunk(notes: list[Note]) -> Chunk | None:
    """A synthetic chunk summarizing vault composition -- folder and tag
    breakdown -- so a broad question like 'what is this vault about' has a
    real target to retrieve instead of relying on nearest-neighbor luck
    over individual note chunks. Built from already privacy-filtered notes,
    so it never surfaces excluded content."""
    if not notes:
        return None
    folder_counts = Counter(
        note.rel_path.split("/", 1)[0] if "/" in note.rel_path else "(root)"
        for note in notes
    )
    tag_counts = Counter(tag for note in notes for tag in note.tags)
    folders = ", ".join(
        f"{name}/ ({n} notes)" for name, n in folder_counts.most_common()
    )
    lines = [
        f"Vault overview -- {len(notes)} notes across {len(folder_counts)} "
        f"top-level folders.",
        f"Folders: {folders}.",
    ]
    if tag_counts:
        tags = ", ".join(f"#{tag} ({n})" for tag, n in tag_counts.most_common(15))
        lines.append(f"Most common tags: {tags}.")
    return Chunk(
        rel_path="(vault overview)",
        heading="Vault overview",
        text="\n".join(lines),
        ordinal=-1,
    )
```

`Chunk` needs importing too -- change the parser import line to:

```python
from weft.parser import Chunk, Note, parse_vault
```

Then in `build_index`, right after the per-note chunking loop and before
`store = VectorStore(dim=embedder.dim)`:

```python
    overview_chunk = build_overview_chunk(notes)
    if overview_chunk is not None:
        overview_note = Note(
            rel_path="(vault overview)", title="Vault overview",
            frontmatter={}, tags=[], wikilinks=[], body="",
        )
        pairs.append((overview_note, overview_chunk))
```

(`graph = LinkGraph.from_notes(notes)` further down keeps using the original
`notes` list, so the synthetic note never enters the wikilink graph.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest tests/test_index_overview.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite and fix the one known collateral failure**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

Expected failure: `tests/test_agent.py::test_ask_end_to_end_with_fakes`. This
test indexes `sample_vault` with `FakeEmbedder` and asks "tell me about
coffee" with `k=3`; `FakeEmbedder` is hash-based with no real semantics, and
for this exact fixture text the new "(vault overview)" chunk's hash happens
to rank above every real note chunk, so `result.sources[0]` is no longer
guaranteed to end in `.md`. Fix the assertion in `tests/test_agent.py` to
check the source list rather than assume position 0:

```python
def test_ask_end_to_end_with_fakes(sample_vault, tmp_path):
    store = _indexed_store(sample_vault, tmp_path)
    llm = FakeLLM(response="Answer citing [1].")
    result = ask("tell me about coffee", FakeEmbedder(dim=16), store, llm, k=3)
    assert result.answer == "Answer citing [1]."
    assert len(result.sources) >= 1
    assert any(s.endswith(".md") for s in result.sources)
```

Re-run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS, no other failures. (Every other place that builds an index
from `sample_vault` either checks for a specific `rel_path` by name, checks
`Sources:`/text presence rather than position, or uses a `k` large enough
to include every chunk regardless of ranking -- see the design spec's
"Testing" section for the full audit of why nothing else is affected.)

- [ ] **Step 6: Commit**

```bash
git add src/weft/index.py tests/test_index_overview.py tests/test_agent.py
git commit -m "$(cat <<'EOF'
feat(weft): add synthetic vault-overview chunk for corpus-level questions

build_index() now appends one synthetic "(vault overview)" chunk with
folder/tag statistics computed from already privacy-filtered notes. It
flows through the existing embed/store/BM25 pipeline unchanged and
surfaces via ordinary retrieval for broad questions like "what is this
vault about", instead of leaving them to nearest-neighbor luck over
individual note chunks.
EOF
)"
```

---

### Task 5: Security documentation addendum

**Files:**
- Create: `docs/security/2026-08-30-vault-scoped-memory.md`

- [ ] **Step 1: Write the record**

```markdown
# 2026-08-30 — Vault-scoped episodic memory

## What changed

`Episode` records (the agent's episodic Q&A log) are now stamped with the
`vault_root` of the index they were logged against, read from
`<store>.manifest.json`. `MemoryStore.episodes()` filters out any episode
whose `vault_root` doesn't match the current store's vault -- including
records with no `vault_root` field at all, which fail closed rather than
being assumed to match.

`weft index` also now refuses to overwrite a store whose manifest names a
different vault unless `--force` is passed (`IndexVaultMismatchError`,
reusing the check already used by `weft suggest`).

## Why

Investigating a real `weft ask` run found that reusing a store path across
two different vaults let a stale, unrelated episode win memory-query recall
by text similarity alone and get pasted into the prompt as context,
producing a confusing, unrepresentative answer. The same missing vault
binding let `weft index` silently clobber an unrelated vault's index with
no warning.

## Regression coverage

- `tests/test_memory.py`: episodes are stamped with vault_root; recall
  excludes episodes from a different vault_root and episodes with a
  missing vault_root field.
- `tests/test_service.py`: `service_index()` raises `IndexVaultMismatchError`
  on a vault-mismatched re-index without `force=True`; `make_memory()`
  reads vault_root from the manifest.
- `tests/test_cli.py`: `weft index` without `--force` exits 1 against a
  mismatched store; `--force` proceeds.

## Known limitation

Semantic memory items (`preference`/`fact`/`decision`/`task`) are
intentionally NOT vault-scoped -- they're documented as durable facts about
the user, meant to persist across vaults. Only the episodic log is scoped.
```

- [ ] **Step 2: Commit**

```bash
git add docs/security/2026-08-30-vault-scoped-memory.md
git commit -m "$(cat <<'EOF'
docs(weft): record the vault-scoped-memory security fix

Per CLAUDE.md's security-boundary-change requirement: documents why
episodic memory is now vault-scoped and where the regression coverage
lives.
EOF
)"
```
