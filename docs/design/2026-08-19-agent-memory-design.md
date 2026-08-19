# Weft Agent Memory — Design (M3)

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-19.
**Scope:** The first cross-session/within-session memory layer for Weft.

## 1. Problem & current state

Weft today has **no agent-memory layer**. `weft ask` is a stateless one-shot
process: it loads the vector store, retrieves chunks, calls Claude once, prints,
and exits. The LangGraph in `agent.py` compiles with **no checkpointer**, so its
`AgentState` lives only for a single `.invoke()`. Ask a question, then a
follow-up, and the second call knows nothing of the first.

The only durable state on disk is *derived index state* (`.weft/index.*`,
`.weft/index.graph.json`) plus the `suggestions.jsonl` **ledger**, which remembers
only which note-pairs `suggest` already proposed. Nothing stores user facts,
preferences, past questions/answers, decisions, or agent commitments.

**Goal:** users interact with Weft knowing it remembers them — within a session
and across sessions — without violating the project's "never silent writes",
"auditable", and "local-first" invariants.

## 2. Decisions (locked during brainstorming)

| Axis | First increment (M3.0) | Target (M3.1 / M4) |
|---|---|---|
| Interaction | Cross-invocation memory on one-shot commands | `weft chat` REPL on the same foundation (M4) |
| Taxonomy | episodic `log` + semantic `{preference, fact, decision, task}` with status + provenance | (same) |
| Capture | Explicit (`weft remember …`) | + inferred-with-confirmation via inbox (M3.1) |
| Storage | `.weft/` private sidecar | + read-only `_memory.md` vault mirror (M3.1) |
| Substrate | Flat JSONL + small embedding index (mirrors `store.py`) | graduate to `sqlite-vec` when volume/query grows |

`task` = **agent-side commitments** (follow-ups the agent owes the user), not user
to-dos (those live in the vault as checkboxes, read via the index).

Silent auto-write capture is **ruled out** by the project's non-negotiable
"never silent writes" invariant.

## 3. Taxonomy: two layers

- **Episodic layer** — `log`: append-only raw interaction records (question,
  answer, cited sources, timestamp). Complete, cheap, high-volume.
- **Semantic layer** — curated, deduplicated, updatable items with
  `type ∈ {preference, fact, decision, task}`, each carrying a lifecycle
  `status ∈ {active, superseded, rejected}` and provenance.

Rejection is a **status (tombstone)**, never a parallel category — avoiding
`rejected_preference`, `rejected_fact`, etc. A tombstone doubles as negative
memory: future inferred capture (M3.1) must not re-propose something the user
killed, mirroring how `suggestions.jsonl` suppresses repeats.

## 4. Architecture & module boundaries

One new module plus small, well-bounded touch-points. The memory subsystem stays
parallel to — never entangled with — the vault index.

```
src/weft/
  memory.py      NEW  — MemoryStore: the whole subsystem behind one interface
  agent.py       TOUCH — recall step added; reason injects memory; record writes episode
  cli.py         TOUCH — `weft remember` + `weft memory`; ask wires memory in
  security.py    REUSE — secure_append_json / secure_write_text / 0600
  embeddings.py  REUSE — same Embedder for memory recall (no new model)
```

New files under `.weft/` (private, `0600`, already excluded from indexing):

- `.weft/memory.jsonl` — semantic items
- `.weft/episodes.jsonl` — append-only interaction log
- `.weft/memory.npz` + `.weft/memory.meta.json` — derived embedding index for recall

`MemoryStore` public interface (nothing depends on its internals):

```python
class MemoryStore:
    def remember(type, text, *, provenance, confidence=None) -> MemoryItem
    def supersede(item_id, new_text) -> MemoryItem
    def reject(item_id) -> None                              # tombstone
    def log_episode(question, answer, sources) -> None
    def active_semantic(type=None) -> list[MemoryItem]       # always-inject set
    def recall(query_vec, k, *, kinds) -> list[MemoryHit]    # semantic retrieval
    @classmethod
    def load(path) -> "MemoryStore"
    def save(path) -> None
```

- **What it does:** owns all memory persistence + retrieval.
- **How you use it:** the methods above.
- **What it depends on:** `Embedder`, `security.py`, stdlib.

## 5. Data model

**Semantic item** (one JSON object per line in `memory.jsonl`):

```json
{
  "id": "mem_a1b2c3d4",
  "type": "preference | fact | decision | task",
  "text": "Prefer LanceDB over Neo4j for the graph store",
  "status": "active | superseded | rejected",
  "provenance": "explicit | inferred",
  "confidence": null,
  "source": "weft remember",
  "supersedes": null,
  "created_at": "2026-08-19T14:00:00Z",
  "updated_at": "2026-08-19T14:00:00Z"
}
```

- `id` — stable, content-independent (`mem_` + short token), so text can change on supersede.
- `status` — the lifecycle axis. `active` items are live; `superseded`/`rejected` are tombstones, not physically removed on write.
- `supersedes` — id of the record this replaces, so history is reconstructable.
- `confidence` — `null` for explicit (M3.0); a float for inferred (M3.1).
- `source` — human-readable origin (`"weft remember"`, later `"inferred:suggest"`).

**Episode** (one per line in `episodes.jsonl`):

```json
{
  "id": "ep_...",
  "ts": "2026-08-19T14:00:00Z",
  "question": "what did I decide about the graph store?",
  "answer": "You decided on LanceDB… [1]",
  "sources": ["Decisions/graph-store.md"]
}
```

**Lifecycle (append-only + compaction):**

- `remember` → append `active`.
- `supersede(id, new_text)` → append a new `active` record with `supersedes: id`,
  plus a status-flip record marking the old one `superseded`. Latest record per
  `id` wins on load.
- `reject(id)` → append a status-flip to `rejected`; becomes a tombstone checked
  by future inference.
- **Compaction** (`weft memory compact`, or periodic): collapse to the latest
  record per `id`, rewrite the file, preserve tombstones but **drop rejected
  text** (keep only id + status for negative-memory dedup).

The embedding index (`memory.npz`) is **derived and disposable** — embeds each
`active` semantic item's text and each episode, rebuilt when memory changes. Same
pattern as `index.npz`.

## 6. Data flow

`ask` grows from `retrieve → reason` to **`recall → retrieve → reason → record`**,
staying a plain LangGraph line so each step is independently testable.

On `weft ask "<q>"`:

1. **recall** — embed the question once (reused across steps).
   - `active_semantic(preference|fact)` → **always-inject** set (few, cheap; applied
     regardless of similarity because e.g. "answer concisely" must always hold).
   - `recall(query_vec, k, kinds={decision, task, log})` → **similarity-gated**:
     decisions, open tasks, past episodes surface only when relevant.
2. **retrieve** — unchanged graph-aware chunk retrieval from the vault index.
3. **reason** — `build_prompt` gains a **structured, clearly-bounded memory block**
   above the vault sources, separately labeled so memory is distinguishable from
   notes and citations still point only at real notes:

   ```
   What I remember about you (durable):
   - [preference] Prefer concise answers
   - [decision] Chose LanceDB for the graph store (2026-08-01)
   Relevant from our past conversations:
   - On 2026-08-08 you asked "…", I answered "…"

   Sources:
   [1] Decisions/graph-store.md — …
   ```

4. **record** — after the answer, append one **episode**. This is the only
   automatic write, to a private local log under `.weft/` (like the API audit
   log), not vault content — so it does not violate "never silent *vault* writes".
   Embedding index marked stale / incrementally updated.

**`weft remember "<text>" [--type preference|fact|decision|task]`** — validate
bounds → `remember(...)` → append + update index → print stored item + id.
Default `--type` = `fact`.

**`weft memory [list|show <id>|forget <id>|compact]`** — read-only views +
`forget` (→ `reject` tombstone) + `compact`. The "see what it remembers" surface
until the `_memory.md` mirror lands in M3.1.

**Prompt-budget guard:** always-inject capped (top-N most-recently-used active
preferences/facts); similarity recall `k`-bounded and threshold-gated — memory
cannot blow the context window as it grows.

## 7. Privacy, security & failure modes

- Files never leave the machine; live under `.weft/` (excluded from indexing, dir
  `0700`); created `0600` via `secure_append_json` / `secure_write_text`; symlinks
  refused (same as ledger/inbox).
- **Never silent vault writes.** Memory writes to `.weft/` only. The `_memory.md`
  mirror (M3.1) reuses the `_inbox.md` guard: refuse symlink / non-regular file,
  no silent overwrite.
- **Memory text is untrusted data** — escaped and kept inside labeled prompt
  boundaries so a stored string cannot spoof a citation or source block.
- **Explicit-only capture in M3.0** → zero inference risk; the auditable payload
  log already covers anything sent to Claude.
- **Right to be forgotten:** `weft memory forget <id>` + `compact` give real
  deletion; compaction drops rejected text, keeping only id + status.
- **Bounds validation:** `--type` restricted to the enum, `text` length-capped,
  `k`/limits validated in code — never trusting argparse alone.

**Failure modes:** missing/corrupt memory files → treat as empty; `ask` still
works (memory is additive, never required). Stale embedding index → rebuild on
next write. Malformed JSONL line → skipped, not fatal (same tolerance as the
ledger).

## 8. Testing

Mirrors the one-test-file-per-module convention; all offline via
`FakeEmbedder` / `FakeLLM`.

- `test_memory.py` — remember/supersede/reject lifecycle; latest-record-wins load;
  compaction preserves tombstones + drops rejected text; recall ranking;
  `0600`/symlink refusal.
- `test_agent_memory.py` — recall→retrieve→reason→record wiring; always-inject vs.
  similarity-gated; memory block labeled & escaped; episode written after answer;
  empty-memory path unchanged.
- `test_cli_memory.py` — `remember`, `memory list/show/forget/compact`; bounds
  validation; ask-with-memory end-to-end.

## 9. Milestones

- **M3.0 (first increment, this build):** `MemoryStore`, explicit `weft remember`,
  `.weft/` sidecar, memory-aware `ask` (recall + inject + episode log),
  `weft memory` read views. Shippable; delivers cross-session recall.
- **M3.1 (target):** inferred-with-confirmation capture routed through the
  inbox/ledger; read-only `_memory.md` vault mirror.
- **M4 (foundation pays off):** `weft chat` REPL — multi-turn working memory in
  the LangGraph state, persisting salient turns into the same `MemoryStore`.
  Specced separately.

## 10. Graduation path

Flat JSONL + numpy embedding index mirrors `store.py`'s deliberate
"zero external services… graduate later" stance. When memory volume or query
complexity grows, migrate to `sqlite-vec` (structured tables + vector column;
`UPDATE status` makes lifecycle natural) — the same escalation story already used
for the vector store and graph.
