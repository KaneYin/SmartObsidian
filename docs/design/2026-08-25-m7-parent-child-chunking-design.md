# Weft M7 — Parent/Child Chunking & Pluggable Chunking Strategies — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-25.
**Scope:** A pluggable chunking-strategy layer plus the first advanced strategy —
parent/child retrieval. Contextual Retrieval and sliding-window are future siblings.

## 1. Problem & goal

Weft chunks a note at heading boundaries, embeds each chunk's text, and hands the
matched chunk text to the model. A heading section is often too coarse to match a
narrow question precisely, and there is no separation between the small piece that
matched and the surrounding context the model needs. M7 introduces a pluggable
chunking layer and implements **parent/child**: embed small children for precise
matching, return the larger parent section for context.

## 2. Decisions (locked during brainstorming)

- **First increment:** parent/child (LLM-free, structural). Contextual Retrieval (A)
  and sliding-window (C) are follow-on strategies in the same layer.
- **Parent = heading section; child = paragraph** (blank-line split within a section).
  A single-paragraph section yields one child equal to its parent.
- **Storage:** inline `parent_id` + `parent_text` in each child's metadata; dedup by
  parent happens at prompt-build time. No new persisted file. A parent-registry file
  is the documented graduation for index-size at scale.
- **Selection:** a `weft index --chunking {heading,parent-child}` flag, default
  `heading`, recorded in the index manifest. Existing behavior is unchanged.

## 3. Components

```
src/weft/
  chunking.py  NEW   — chunk_strategy(name) -> callable(Note) -> list[Chunk];
                       registers "heading" and "parent_child". Home for A/C later.
  parser.py    TOUCH — Chunk gains optional parent_id/parent_text; paragraph splitter.
  index.py     TOUCH — build_index(chunking=...) writes parent_id/parent_text into
                       metadata; records the strategy in the manifest.
  agent.py     TOUCH — build_prompt collapses hits by parent; ask() source dedup.
  cli.py       TOUCH — `weft index --chunking {heading,parent-child}`.
```

## 4. Data model

`Chunk` (parser.py) gains two optional fields:

```python
@dataclass
class Chunk:
    rel_path: str
    heading: str
    text: str                       # embedded + matched text (child in parent_child)
    ordinal: int = 0
    parent_id: str | None = None    # stable id of the parent section
    parent_text: str | None = None  # full section text shown to the LLM
```

Child metadata persisted by `build_index`:

```json
{
  "rel_path": "coffee.md", "heading": "Espresso", "text": "<paragraph>",
  "ordinal": 3, "parent_id": "par_9f2a1b3c", "parent_text": "<full Espresso section>",
  "tags": [...], "wikilinks": [...]
}
```

`parent_id = "par_" + sha1(f"{rel_path}#{section_ordinal}")[:8]`. Heading mode omits
`parent_id`/`parent_text` entirely, so existing indexes and retrieval are untouched.

## 5. Chunking strategies (`chunking.py`)

```python
def chunk_strategy(name: str) -> Callable[[Note], list[Chunk]]:
    if name == "heading":
        return chunk_note                     # existing behavior, unchanged
    if name == "parent_child":
        return _parent_child_chunks
    raise ValueError(f"unknown chunking strategy: {name}")
```

`_parent_child_chunks(note)`:
1. Run `chunk_note(note)` to get heading **sections** (the parents), each with its
   `heading`, `text`, and section `ordinal`.
2. For each section, compute `parent_id` and split `text` into paragraph children on
   blank lines (a section with no blank line yields a single child == the section).
3. Emit one `Chunk` per child, carrying the child paragraph as `text`, and the
   section's `parent_id` + full `text` as `parent_text`. Child `ordinal` is a running
   index across the note for stable ordering.

Fenced code blocks are treated as opaque (never split mid-fence), reusing the fence
handling already in `chunk_note`.

## 6. Retrieval & prompt (`agent.py`)

Retrieval mechanics are unchanged — the query matches child embeddings and returns
`SearchHit`s. The one new step is in `build_prompt` (and the source list in `ask`):

- Group hits by **parent key** = `parent_id` if present, else `(rel_path, ordinal)`.
- Keep first-occurrence order (hits are already score-ranked).
- Emit each parent once, using `parent_text` when present, else `text`, as the source
  `content`. Citations continue to reference `rel_path`.

This collapses two matching children of one section into a single source, and no-ops
exactly to today's behavior when `parent_id`/`parent_text` are absent (heading mode,
or any pre-M7 index).

## 7. CLI & manifest

`weft index --chunking {heading,parent-child}` (default `heading`). The chosen
strategy is written into `index.manifest.json` (`"chunking": "parent_child"`) alongside
the vault binding and privacy policy, so `ask`/`suggest` need no new flags — retrieval
reads `parent_id`/`parent_text` straight from the stored metadata.

## 8. Security & invariants

- No new network or LLM use (parent/child is purely local and structural).
- Redaction/privacy still run before chunking (unchanged in `parse_vault`), so parent
  and child text are both post-redaction.
- `parent_text` is note content and is treated as untrusted data with the same
  escaping discipline as `text` in the structured prompt.
- Index-size note: inline `parent_text` duplicates section text across a section's
  children. Acceptable at vault scale; the parent-registry file is the graduation.

## 9. Testing (offline, FakeEmbedder/FakeLLM)

- `test_chunking.py` — `chunk_strategy("heading")` reproduces existing chunks;
  `parent_child` splits a multi-paragraph section into children sharing one
  `parent_id`/`parent_text`; a single-paragraph section yields one child equal to its
  parent; a fenced code block is never split mid-fence.
- `test_index_chunking.py` — `build_index(..., chunking="parent_child")` persists
  `parent_id`/`parent_text` in metadata and records `"chunking"` in the manifest;
  default build omits them.
- `test_agent_parent.py` — `build_prompt` collapses two children of one parent into a
  single source carrying `parent_text`; heading-mode hits are unchanged.
- `test_cli.py` addition — `weft index --chunking parent-child` end-to-end builds an
  index whose metadata carries parent fields.

## 10. Milestone & future

This is **M7**. Follow-on strategies register in `chunking.py`:
- **A) Contextual Retrieval** — a `contextual` strategy that prepends an LLM-generated,
  document-aware blurb to each chunk's embedded text (raw text still shown). Runs
  through the M5 provider layer; free/offline on local Ollama, prompt-cached on the
  Anthropic provider.
- **C) Sliding-window** — a `sliding` strategy producing fixed-size overlapping
  windows (size/overlap parameters), usable as a child-splitter for parent/child too.

Graduation: move inline `parent_text` to an `index.parents.json` registry keyed by
`parent_id` if index size becomes a concern at large vault scale.
