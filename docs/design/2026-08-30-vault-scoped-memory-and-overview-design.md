# Vault-scoped memory + vault-overview grounding

## Problem

Investigating a real `weft ask "what is this vault about"` run turned up two
separate defects:

1. **Cross-vault memory bleed.** `MemoryStore` reads every episode ever logged
   to `episodes.jsonl` at a given store path and recalls by pure cosine
   similarity, with no record of which vault/index an episode was logged
   against. Reusing (or accidentally colliding on) a store path across two
   different vaults lets a stale, unrelated episode win recall and get pasted
   into the prompt as "memory context," actively confusing the model. The
   same root cause — a store path with no enforced vault binding — also let
   `weft index` silently overwrite an index belonging to a different vault
   with no warning, which is what happened during the investigation
   (`AgentDevelopment/.weft` held an index for `examples/SampleVault` and was
   silently clobbered by indexing an unrelated vault into the same default
   path).

2. **No grounding for corpus-level questions.** Top-k dense+BM25 chunk
   retrieval has no correct target for a question like "what is this vault
   about" — no single note chunk describes the vault holistically. Worse,
   because the query contains the word "vault," retrieval gets hijacked by
   notes that discuss Obsidian mechanics (plugin lists, vault configuration)
   rather than the vault's actual subject matter, producing a confident but
   badly unrepresentative answer.

`service.py` already has the right pattern for vault binding —
`service_suggest()` compares the index manifest's `vault_root` against the
requested vault and raises `IndexVaultMismatchError` on mismatch. That
pattern just isn't applied anywhere else.

## Fix 1 — Vault-scoped episodic memory

**Scope decision:** only the episodic Q&A log (`Episode`, the `"log"` recall
kind) is vault-scoped. Semantic memory items (`preference` / `fact` /
`decision` / `task`) stay global/cross-vault, per their existing documented
role as durable facts about the *user*, not the vault.

### Data model

`Episode` (`src/weft/memory.py`) gains a required field:

```python
@dataclass
class Episode:
    id: str
    ts: str
    question: str
    answer: str
    sources: list[str]
    vault_root: str
```

`MemoryStore.__init__` takes the resolved vault root string:

```python
def __init__(self, memory_path: Path, episodes_path: Path, vault_root: str):
    ...
    self._vault_root = vault_root
```

- `log_episode()` stamps the new episode with `self._vault_root`.
- `episodes()` filters to `ep.vault_root == self._vault_root` before
  returning. An episode with a missing/mismatched `vault_root` (including
  every episode logged before this change) is excluded — fail closed rather
  than assume it matches.
- `recall()` is unaffected structurally; it already calls `self.episodes()`
  for the `"log"` kind, so the filtering happens transparently upstream.

No migration of existing `episodes.jsonl` files: old records simply age out
of recall going forward. They remain on disk (append-only, as today).

### Threading the vault root through

Every call site that builds a `MemoryStore` needs the current index's vault
root. It's already sitting in `<store_path>.manifest.json` (written by
`build_index()`), the same file `service_suggest()` already reads. Add a
small helper next to `manifest_path_for()` in `index.py`:

```python
def read_vault_root(store_path: Path) -> str | None:
    """None if the store has no manifest yet (fresh/pre-manifest index)."""
    path = manifest_path_for(store_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("vault_root")
```

`make_memory()` in both `cli.py` and `service.py` becomes:

```python
def make_memory(store_path: Path) -> MemoryStore:
    return MemoryStore(
        store_path.parent / "memory.jsonl",
        store_path.parent / "episodes.jsonl",
        read_vault_root(store_path) or "",
    )
```

A fresh index with no manifest yet resolves to `""`, which simply can't match
any real vault_root — new episodes logged before an index exists are scoped
to that sentinel and won't leak either.

### Reindex guard

`service_index()` gains the same check `service_suggest()` already performs,
plus a `force: bool = False` escape hatch:

```python
manifest_path = manifest_path_for(sp)
if manifest_path.exists() and not force:
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_vault = vault_root(Path(vault_path))
    if existing.get("vault_root") not in (None, str(requested_vault)):
        raise IndexVaultMismatchError(
            f"store at {sp} belongs to a different vault "
            f"({existing['vault_root']}); pass --force to overwrite it"
        )
```

CLI: `weft index` gets a new `--force` flag. `_cmd_index` catches
`IndexVaultMismatchError` alongside the `ProviderUnavailable` it already
catches and prints the message to stderr with exit code 1 — same shape as
`_cmd_suggest`'s existing handling of the same exception type.

## Fix 2 — Vault-overview grounding chunk

### Content

A single synthetic chunk built in `index.py`, after `parse_vault()` and
before chunking/embedding, from data already produced by parsing (so it
inherits the privacy policy for free — it only ever sees notes that survived
`PrivacyPolicy` filtering):

```python
def build_overview_chunk(notes: list[Note]) -> Chunk | None:
    """A synthetic chunk summarizing vault composition, so broad questions
    ('what is this vault about') have a real target to retrieve."""
    if not notes:
        return None
    folder_counts = Counter(
        note.rel_path.split("/", 1)[0] if "/" in note.rel_path else "(root)"
        for note in notes
    )
    tag_counts = Counter(tag for note in notes for tag in note.tags)
    folders = ", ".join(f"{name}/ ({n} notes)" for name, n in folder_counts.most_common())
    tags = ", ".join(f"#{tag} ({n})" for tag, n in tag_counts.most_common(15))
    text = (
        f"Vault overview — {len(notes)} notes across {len(folder_counts)} "
        f"top-level folders.\n{folders}."
        + (f"\nMost common tags: {tags}." if tags else "")
    )
    return Chunk(rel_path="(vault overview)", heading="Vault overview", text=text, ordinal=-1)
```

### Wiring into `build_index()`

Paired with a matching synthetic `Note` so it flows through the *existing*
metadata-construction loop unchanged:

```python
overview_chunk = build_overview_chunk(notes)
if overview_chunk is not None:
    overview_note = Note(
        rel_path="(vault overview)", title="Vault overview",
        frontmatter={}, tags=[], wikilinks=[], body="",
    )
    pairs.append((overview_note, overview_chunk))
```

This is appended after the normal per-note chunking loop, before
`contextualize()`/embedding — it goes through embedding, `store.add_batch`,
and the BM25 build exactly like any real chunk. No changes to `agent.py`,
retrieval, reranking, or fusion: it competes for top-k the same way any other
chunk does, and naturally surfaces for queries whose vocabulary matches its
content (vault, about, overview, summarize, folder names, common tags).

### Citation

`rel_path="(vault overview)"` flows unchanged through `SearchHit` into the
printed `Sources:` list — parenthesized and distinct from any real path, so
it reads as synthetic without needing separate filtering logic.

### Regeneration & flags

Rebuilt from scratch on every `weft index` run — pure aggregation over
already-parsed notes, no caching. No new opt-out flag (e.g. `--no-overview`):
unlike BM25, this exposes no information beyond what's already visible via
existing tags/rel_path citations, so keeping the flag surface minimal per
YAGNI.

### Ranking-competition tradeoff (evaluated post-implementation)

A final review flagged that the overview chunk could in principle outrank a
genuinely relevant note for an ordinary, specific question (not just broad
"about this vault" questions) if it shares vocabulary with the query — and
this was in fact observed against `FakeEmbedder` (the test suite's
deterministic hash-based stand-in, used because it needs no model download).
Checked against the real embedder
(`SentenceTransformerEmbedder`/all-MiniLM-L6-v2) on the same fixture and
query ("tell me about coffee" over the `sample_vault` fixture), the overview
chunk scored **0.109** cosine similarity versus **0.63/0.57/0.40/0.39** for
the four real note chunks — nowhere close to competitive. `FakeEmbedder`
produces content-independent random unit vectors from a text hash, so its
similarity scores carry no real semantic signal; the ranking swap seen under
it is an artifact of the test double, not a production risk. No mitigation
was added; `tests/test_agent.py`'s assertion was correctly loosened (see
Task 4 in the implementation plan) to not assume a specific rank under that
fake embedder, since asserting an exact tie-break order there was never a
meaningful invariant to protect.

### Suggest-command interaction (caught in final review, fixed)

The overview chunk is a real chunk in the `VectorStore`, and `weft suggest`'s
`note_vectors()` originally mean-pooled *every* `rel_path` in the store with
no filtering — so the synthetic chunk was being treated as a linkable "note"
and could be recommended as a link target that doesn't exist in the vault.
Fixed by excluding `OVERVIEW_REL_PATH` (a shared constant, replacing the
previously-duplicated `"(vault overview)"` string literal) in
`note_vectors()`. This is the one cross-cutting interaction between Fix 2 and
the rest of the indexing pipeline that per-file review didn't catch; a
final holistic review across all touched files did.

## Testing

- `tests/test_config.py` / a new `tests/test_memory.py` (or extend existing
  memory tests if present): `log_episode` stamps `vault_root`; `episodes()`
  and `recall(..., kinds={"log"})` exclude episodes from a different
  `vault_root` and episodes with a missing `vault_root` field (simulating a
  pre-fix record).
- `tests/test_service.py`: `service_index()` raises `IndexVaultMismatchError`
  when re-indexing a different vault into an existing store without
  `force=True`, and succeeds (overwriting) with `force=True`. A same-vault
  re-index (no manifest change) still succeeds without `force`.
- `tests/test_cli.py`: `weft index` without `--force` against a
  vault-mismatched store exits 1 with a message naming the existing vault;
  `--force` proceeds.
- `tests/test_service.py` / index tests: `build_index()` on a small fixture
  vault produces one extra chunk with `rel_path == "(vault overview)"`
  containing the expected folder names and counts; confirm it's absent when
  `notes` is empty (empty vault).
- Regression test for the original bug: index two different fixture vaults
  into two different store paths, log an episode against vault A, assert
  `ask`-level memory recall against vault B's store never surfaces it.

## Security note

Per `CLAUDE.md`, this touches a security boundary (memory is documented as
"the most sensitive surface Weft has," and the reindex guard is a fail-closed
change to `weft index`). This spec's testing section adds regression
coverage; `docs/security/2026-08-08-security-hardening.md` (or a new dated
record) should get a short addendum noting the vault-scoping invariant once
implemented.
