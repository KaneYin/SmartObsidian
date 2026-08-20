# Weft M3.1 — Memory Mirror & Inferred-with-Confirmation Capture — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-19.
**Scope:** The two pieces M3.0 deferred: a read-only `_memory.md` vault mirror, and
inferred-with-confirmation memory capture mined from the episodic log.

## 1. Problem & current state

M3.0 shipped explicit memory (`weft remember`) and a memory-aware `ask`. Two gaps
remain from the memory design: (1) there is no way to *see* what Weft remembers
from inside Obsidian, and (2) all capture is manual — Weft never proposes memories.
M2 built a propose→review→ledger pattern for link suggestions, but its accept/reject
was explicitly deferred; M3.1 delivers real accept/reject for memory.

## 2. Decisions (locked during brainstorming)

- **Candidate source:** the episodic log (`.weft/episodes.jsonl`) — data the user
  already generated, mined locally. Not the vault (overlaps retrieval).
- **Confirmation:** explicit CLI accept/reject by id, backed by a proposals store.
  Not fragile markdown checkbox round-trips.
- **Extraction:** local heuristic default + opt-in `--llm` through the configured
  provider (local via Ollama post-M5), audited, degrading to heuristic on failure.
  Mirrors `suggest` / `suggest --rationale`.
- **Proposed types:** `preference` and `fact` only (decisions/tasks stay explicit).
- **Mirror:** read-only `_memory.md`, reusing the `_inbox.md` secure-write pattern;
  it also surfaces pending proposals as read-only info (accept happens via CLI).

## 3. Components

```
src/weft/
  proposals.py      NEW  — ProposalStore over .weft/memory-proposals.jsonl
  memory_infer.py   NEW  — infer_candidates(episodes, ...) heuristic + opt-in LLM
  memory_mirror.py  NEW  — render/write read-only _memory.md (reuses security helpers)
  parser.py         TOUCH — add "_memory.md" to GENERATED_NOTES
  cli.py            TOUCH — memory suggest/pending/accept/reject/mirror subcommands
```

`MemoryStore` (M3.0) is the sink: accepted proposals call
`remember(type, text, provenance="inferred", source=...)`. The link-suggestion
`_inbox.md` surface is untouched and separate.

## 4. Data model

**Proposal** (one JSON object per line in `memory-proposals.jsonl`, append-only,
latest-record-per-id wins on load):

```json
{
  "id": "prop_9f2a1b3c4d5e",
  "type": "preference | fact",
  "text": "Recurring interest: retrieval",
  "status": "pending | accepted | rejected",
  "signature": "9f2a1b3c4d5e...",
  "source": "heuristic | inferred:llm",
  "created_at": "2026-08-19T14:00:00Z",
  "updated_at": "2026-08-19T14:00:00Z"
}
```

- `id = "prop_" + sha1(normalized_text)[:12]`, so re-running `suggest` maps a
  repeated candidate to the same id (idempotent); any id already present in the
  store (pending/accepted/rejected) is skipped — rejected candidates stay dead
  forever, reusing the M2 ledger's tombstone-as-negative-memory idea.
- `normalized_text` = lowercased, whitespace-collapsed candidate text.

`ProposalStore` interface:

```python
class ProposalStore:
    def add(candidates: list[Candidate]) -> list[Proposal]   # skips known signatures
    def pending() -> list[Proposal]
    def get(prop_id) -> Proposal
    def mark(prop_id, status) -> Proposal                     # accepted | rejected
    def known_ids() -> set[str]                               # all statuses (dedup)
```

## 5. Data flow

**`weft memory suggest [--llm] [--limit N]`:**

1. `episodes = MemoryStore.episodes()`; `existing = {i.text for i in active_semantic()}`;
   `seen = ProposalStore.known_ids()`.
2. `candidates = infer_candidates(episodes, existing, seen, llm=optional, limit=N)`:
   - **Heuristic (default):** count salient terms across question texts (reuse the
     stopword/term approach from `suggest.py`); a term appearing in ≥ `MIN_EPISODES`
     (3) distinct episodes and not already in `existing` becomes a `fact` candidate
     `"Recurring interest: <term>"`. Ranked by frequency, capped at `limit`.
   - **`--llm` (opt-in):** send the episode questions/answers through the configured
     provider (audited to `api-log.jsonl`), asking for structured `preference`/`fact`
     candidates; on any error, fall back to the heuristic result.
   - Every candidate whose `id` is in `seen` is dropped.
3. New candidates are written to `ProposalStore` as `pending`. `suggest` operates
   only on `.weft/` (no vault path needed); it prints the count and points the user
   to `weft memory pending` and `weft memory mirror <vault>`.

**`weft memory pending`** — list pending proposals with ids and text.

**`weft memory accept <id>`** — `MemoryStore.remember(prop.type, prop.text,
provenance="inferred", source=prop.source)`, then `ProposalStore.mark(id, "accepted")`.

**`weft memory reject <id>`** — `ProposalStore.mark(id, "rejected")`. Never re-proposed.

**`weft memory mirror <vault>`** — render active semantic memory plus a read-only
"Pending (run `weft memory accept <id>`)" section into `_memory.md`.

## 6. The `_memory.md` mirror

`memory_mirror.py` renders deterministic Markdown and writes it with the exact
`_inbox.md` guarantees: refuse a symlink or non-regular target, no silent replace of
a foreign file, `0600`, atomic write. Dynamic text is Markdown/terminal-escaped
(reuse `inbox.py` helpers). `_memory.md` is added to `parser.GENERATED_NOTES` so it
is never indexed (mirroring `_inbox.md`), preventing memory from feeding back into
retrieval.

The mirror is regenerated only by `weft memory mirror <vault>` (which needs the
vault path); it is never read back (read-only), so there is no untrusted round-trip.

## 7. Security & invariants

- Proposals and episodes stay in `.weft/` at `0600`; `_read_jsonl` refuses symlinks.
- `_memory.md` write reuses the audited inbox path (symlink refusal, explicit
  overwrite only). It contains memory text, so it inherits the "treat as sensitive"
  posture; it lives in the vault deliberately for visibility and is git-ignored by
  the user's vault as they see fit.
- Candidate/episode text is untrusted: the `--llm` prompt wraps it as untrusted data
  with the same boundary discipline as `suggest --rationale`; the payload is logged.
- `accept` is the only path that writes memory, and only for an explicit id — nothing
  is auto-remembered. `--limit` bounds proposals per run (attention budget).

## 8. Testing (offline, FakeEmbedder/FakeLLM)

- `test_proposals.py` — add/get/mark/pending; latest-per-id; `known_ids` dedup;
  idempotent id from text; `0600`; symlink refusal.
- `test_memory_infer.py` — recurring-term heuristic; dedup vs existing memory and
  seen ids; `--limit` cap; LLM path via `FakeLLM`; graceful fallback on LLM error.
- `test_memory_mirror.py` — renders active items + pending section; secure write;
  symlink refusal; `_memory.md` excluded from indexing.
- `test_cli_memory_infer.py` — `suggest`/`pending`/`accept`/`reject` end-to-end;
  accept promotes into memory with `provenance="inferred"`; reject prevents
  re-proposal on the next `suggest`.

## 9. Milestone

This is **M3.1**. It completes the memory feature's review workflow and delivers
the real accept/reject that M2 deferred. Remaining memory work: **M4** (`weft chat`
REPL on the MemoryStore). Provider work continues separately at **M5.1**.
