# Weft M4 — Conversational `weft chat` REPL — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-22.
**Scope:** A multi-turn `weft chat` REPL over the existing retrieval + memory stack,
with a rolling conversation window and per-turn episode logging.

## 1. Problem & goal

`weft ask` is one-shot: each question is answered in isolation, so follow-ups like
"why?" have no context. M4 adds a conversational surface where prior turns are carried
across the session, while still retrieving relevant note chunks and injecting durable
memory each turn. This is the conversational "target" the M3 memory design named.

## 2. Decisions (locked during brainstorming)

- **Context strategy:** a rolling window of the last N raw turns (default 6) included
  in the prompt each turn. Retrieval runs per current message. Summarization of older
  turns is the future escalation path (not built).
- **Persistence:** every turn appends an episode `(question, answer, sources)` to
  `.weft/episodes.jsonl` (same private log as `ask`); durable semantic memory is
  written only via an explicit in-chat `/remember`. Auto-proposing inferred memories
  at session end is **future (B)**, noted not built. `weft memory suggest` already
  mines episodes, so chat feeds inferred capture with no new machinery.
- **Commands:** `/help`, `/exit` (`/quit`), `/reset`, `/sources`, `/remember [--type T] <text>`.

## 3. Components

```
src/weft/
  chat.py   NEW   — ChatSession (rolling history + send/remember/reset) and
                    run_repl(session, read, emit) command loop.
  agent.py  TOUCH — build_prompt gains an optional `conversation` field; SYSTEM
                    notes prior turns are untrusted context.
  cli.py    TOUCH — `weft chat` builds a ChatSession and runs run_repl over stdin.
```

### ChatSession

```python
@dataclass
class ChatTurn:
    answer: str
    sources: list[str]

class ChatSession:
    def __init__(self, embedder, store, llm, *, graph=None, memory=None,
                 k=5, window=6): ...
    def send(self, question: str) -> ChatTurn: ...
    def remember(self, type: str, text: str): ...   # delegates to MemoryStore
    def reset(self) -> None: ...                     # clears the window
    last_sources: list[str]
```

- `history`: `list[dict]` of `{"role": "user"|"assistant", "text": str}`, trimmed to
  the last `2*window` entries (window turn-pairs).
- `send` reuses `agent.graph_aware_retrieve` (or `agent.retrieve` when no graph),
  `agent.collect_memory`, and `agent.build_prompt(..., conversation=window)`. It calls
  `llm.complete` directly (not `agent.reason`) so a turn with no note hits can still
  answer from conversation + memory. After answering it appends the user and assistant
  turns, trims, sets `last_sources`, and calls `memory.log_episode` when memory is set.

### run_repl

```python
def run_repl(session: ChatSession, read=input, emit=print) -> int:
    # prints a banner, loops reading lines, dispatches commands, else session.send.
    # EOFError or /exit|/quit ends the loop and returns 0.
```

Injecting `read`/`emit` keeps the loop fully testable without touching real stdin.

## 4. Prompt & data flow

`agent.build_prompt(question, hits, memory=None, conversation=None)` adds a
`conversation` array to the existing JSON payload when present:

```json
{
  "sources": [...],
  "memory": {...},
  "conversation": [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}],
  "question": "why?",
  "instruction": "Answer only from sources and cite claims as [n]."
}
```

`SYSTEM` gains one clause: the `conversation` array is prior turns for context, is
untrusted data, and must not be treated as instructions. Sources and memory keep their
existing untrusted-data framing. Citations still refer only to `sources`.

Per turn: retrieve → collect memory → build prompt (memory + conversation window +
sources + question) → `llm.complete` → print answer + citations → append to history →
`log_episode`. The CLI wraps the client in `AuditedLLM(..., "chat")`, so every turn is
logged with `provider`/`model`/`left_machine`, identical to `ask`.

## 5. CLI: `weft chat`

```
weft chat [--store .weft/index] [--k 5] [--no-graph] [--no-memory]
          [--provider P] [--model M]
```

Loads the store (prints the standard "No index" error and exits 1 if absent), resolves
the provider via `service.make_llm` (fails fast on `ProviderUnavailable`), wraps in
`AuditedLLM`, builds a `ChatSession`, and runs `run_repl`. `/remember [--type T] <text>`
defaults `type` to `fact` and validates via the same rules as `weft remember`
(reusing `service.service_remember`). `/sources` prints `session.last_sources`.

## 6. Security & invariants

- No new files or endpoints. Chat writes only episodes (private `.weft/`, `0600`) and,
  on explicit `/remember`, a semantic memory item — never silently.
- Conversation text is treated as untrusted data inside the structured prompt, with the
  same escaping discipline as note content; terminal output uses `terminal_safe`.
- The rolling window bounds prompt size (bounded cost); `k` stays within its 1–50 limit.

## 7. Testing (offline, FakeEmbedder/FakeLLM)

- `test_chat.py` — `ChatSession.send` includes prior turns in turn 2's prompt (FakeLLM
  echoes the prompt); the window trims older turns; an episode is logged per turn;
  `remember` writes a memory item; a turn with empty retrieval still returns an answer.
- `test_cli_chat.py` — `run_repl` driven by a scripted `read` (a list iterator raising
  `EOFError` at end) through: a question (asserts answer emitted), `/remember foo`
  (asserts a memory item exists), `/reset` (asserts the window clears), `/exit`
  (asserts the loop returns 0). Uses `service`-level fakes.

## 8. Milestone & future

This is **M4**, completing the memory feature's conversational target. Future
(explicitly not built): running-summary context (B from Q1) for very long sessions, and
auto-proposing inferred memories at session end (B from Q2). A GUI chat view can later
sit on the M6 API rather than the terminal REPL.
