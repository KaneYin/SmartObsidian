# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**Weft** — a local-first, agent-first "second brain" over an Obsidian markdown vault. The vision: a proactive background agent that reasons over the *graph* of your notes (not just their text), surfaces inferred connections and contradictions, and takes real actions (create notes, wire links, schedule) with a human in the loop.

**Status: pre-implementation.** As of now there is *no source code* — only design artifacts under `AgentsOut/`. The first task in a coding session is likely to scaffold the project, not modify existing code. Read `AgentsOut/2026-07-07-notes-agent-brainstorm.md` in full before proposing architecture; it is the source of truth for scope, the roadmap (M0–M6), and hard problems already thought through.

## Locked decisions (do not relitigate without asking)

- **Source of truth:** an Obsidian markdown vault (`.md` + frontmatter + `[[wikilinks]]` + tags).
- **LLM strategy:** hybrid — *local* embeddings for retrieval, *Claude* (via API) for reasoning. A `--local-only` switch (Claude → Ollama) is a planned escape hatch, not the default.
- **Core framework:** LangGraph (stateful loop, checkpointer, interrupt-for-approval).
- **Novel wedge (the whole point):** proactive background daemon **+** hybrid knowledge-graph reasoning (human `[[links]]` fused with inferred semantic edges) **+** human-in-the-loop action-taking. Plain "RAG chat over my vault" is considered a solved, commoditized problem — do not build only that.

## Non-negotiable design constraints

These come from the brainstorm's risk analysis and should shape any implementation:

- **Never silent writes.** Every proactive action is a reviewable diff / inbox item. The MVP "inbox" is literally a generated `_inbox.md` note written into the vault (dogfooding).
- **Attention budget.** Cap proactive suggestions per day, ranked by expected usefulness; accept/reject feedback must be logged (later trains a ranker).
- **Privacy is auditable.** Raw vault, embeddings, graph, and vector store never leave the machine. Only agent-selected context chunks go to Claude — and every payload sent to the API must be logged. Honor a folder allowlist/redaction (e.g. `Private/` never leaves).
- **Revertible edits.** Assume the vault is (or becomes) a git repo so every agent edit is a revertible commit; dry-run is the default.
- **Local-first cost discipline.** Keep embeddings + graph traversal local; escalate to Claude only for genuine reasoning; cache aggressively; incremental re-index on file events, full semantic-edge inference only nightly.

## Intended stack (when code lands)

Chosen in the brainstorm but not yet installed — treat as the default unless a session decides otherwise:

- **Orchestration:** LangGraph. **Reasoning:** Claude API. **Embeddings:** `nomic-embed-text`/`bge` via Ollama, or in-process `sentence-transformers`.
- **Vector store:** LanceDB or `sqlite-vec` (embedded, local file). **Graph store:** start with NetworkX, graduate to Kùzu if it grows.
- **Daemon:** long-running Python process — `watchdog` (FS events) + `APScheduler` (ticks).
- **Package manager:** `uv` (consistent with the rest of `AgentDevelopment/`). The Python virtualenv and LangChain deps live one level up in `AgentDevelopment/.venv` / `AgentDevelopment/LangChainLearning`.

## Working convention: AgentsOut/

This project runs on committed session artifacts. Each brainstorm/design/research session produces a dated markdown doc in `AgentsOut/`, and `AgentsOut/README.md` is a hand-maintained index plus the running list of "Locked decisions" and "Next session" pointer.

When you finish a design/brainstorm session: write `AgentsOut/YYYY-MM-DD-<topic>.md`, add a row to the index table in `AgentsOut/README.md`, update "Locked decisions" if any changed, and update "Next session". Keep new decisions in sync between this file and `AgentsOut/README.md`.

## Repo layout note

The git root is `AgentDevelopment/` (one level up), not this directory — commits and `git status` span sibling projects like `LangChainLearning/`. Scope changes to `NotesAssistant/` unless intentionally working across the workspace.
