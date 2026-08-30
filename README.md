# Weft

Weft is a local-first assistant over an Obsidian Markdown vault. Index a vault,
then ask questions from the CLI or use a small interactive terminal menu. Weft
keeps advanced retrieval controls available without requiring normal users to
understand them.

Implemented milestones:

- M0: local indexing and retrieval-augmented questions;
- M1: one-hop graph-aware retrieval, with `--no-graph` for comparison;
- M2: local inferred-link suggestions, an append-only proposal ledger, and an
  optional payload-logged Claude rationale.
- M3.0: durable agent memory — explicit `remember` of preferences/facts/
  decisions/agent-tasks plus an episodic log, recalled into `ask` across sessions.
- M3.1: inferred-with-confirmation memory — `memory suggest` mines the episodic
  log into proposals you `accept`/`reject`, plus a read-only `_memory.md` mirror.
- M5.0: a pluggable provider layer so reasoning can run on a local open-weight
  model (Ollama) fully offline, or on a remote API, chosen by GPU capability.
- CRAG Task 1 benchmark adapter: ephemeral retrieval over CRAG's cached web pages
  with local Ollama generation and no remote fallback.
- intent-based `fast`, `balanced`, and `best` retrieval modes shared by CLI,
  TUI, and REST adapters;
- a dependency-free interactive terminal menu for ask, chat, index, suggestions,
  and settings.

The background daemon, automatic note edits, and scheduling remain future work.

## Setup

```bash
uv sync --extra dev
```

Set the API key used by `weft ask` and optional suggestion rationales:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

To run fully offline instead, install [Ollama](https://ollama.com), then select
the local provider and pull a model sized for your GPU:

```bash
uv run weft config set provider ollama   # persists to .weft/config.toml
uv run weft models list                  # tiers + the pick for your GPU
uv run weft models pull                  # pulls the recommended model (confirms first)
```

Selection precedence is CLI flags > `WEFT_*` env vars > `.weft/config.toml` >
GPU default. API keys are read from the environment and never written to config.

## Quick start

```bash
uv run weft index "/path/to/your/Vault"
uv run weft
```

Bare `weft` opens the interactive menu:

```text
Ask vault
Chat
Index vault
Discover links
Settings
Exit
```

For automation, the primary commands remain:

```bash
uv run weft ask "What architectural decisions did I make?"
uv run weft chat
uv run weft suggest "/path/to/your/Vault"
```

If the key is stored in `.env`, protect it before sourcing it:

```bash
chmod 600 .env
set -a
source .env
set +a
```

## Index with a privacy policy

```bash
uv run weft index "/path/to/your/Vault"
```

Secure defaults exclude `Private/`, `.obsidian/`, `.trash/`, and `.weft/` before
any file is read or embedded. Vault symlinks are rejected rather than followed.

Privacy options are repeatable:

```bash
uv run weft index "/path/to/Vault" \
  --include Projects \
  --exclude Projects/Client-A \
  --redact 'sk-ant-[A-Za-z0-9_-]+'
```

- `--include PATH` creates an allowlist of relative Vault paths.
- `--exclude PATH` adds a relative path to the default exclusions.
- `--redact REGEX` replaces matches with `[REDACTED]` before parsing,
  embedding, or persistence.
- `--include-private` is an explicit opt-in that removes the default `Private/`
  exclusion. Retrieved Private content may then be sent to Claude.

Re-index after changing the privacy policy. The effective policy and canonical
Vault path are stored in the private `index.manifest.json` sidecar.

## Ask

```bash
uv run weft ask "What did I decide about X?"
uv run weft ask "What are my priorities?" --mode fast
uv run weft ask "What are my priorities?" --mode best
```

`balanced` is the initial default. `fast` uses a smaller vector-first pipeline,
`balanced` adds hybrid and memory-query retrieval, and `best` adds cross-encoder
reranking. The configured default can be changed with:

```bash
uv run weft config set mode best
```

Advanced options still override a mode when needed:

```bash
uv run weft ask "What are my priorities?" --mode best --k 12 --no-rerank
```

Only retrieved, already-filtered chunks are sent to the selected provider. The
exact outbound system prompt and payload are appended to
`.weft/api-log.jsonl` with mode `0600`; treat that audit log as sensitive.
See the [advanced CLI reference](docs/how-to/cli-command-reference.md) for
retrieval tuning, benchmarking, memory administration, and server commands.

## Suggest links

```bash
uv run weft suggest "/path/to/your/Vault"
uv run weft suggest "/path/to/your/Vault" --rationale
```

Suggestions are generated locally unless `--rationale` is used. The command:

- verifies that the index was built from the same canonical Vault;
- leaves an existing `_inbox.md` unchanged by default;
- leaves the inbox unchanged when there are no new suggestions;
- never follows an `_inbox.md` symlink;
- requires `--overwrite-inbox` to replace an existing regular inbox.

Review or move the existing inbox before another run whenever possible. The
checkboxes are still display-only; accept/reject state is planned for M3.

## Local artifacts

The default `.weft/` directory contains:

```text
.weft/index.npz
.weft/index.json
.weft/index.graph.json
.weft/index.manifest.json
.weft/suggestions.jsonl
.weft/api-log.jsonl
.weft/config.toml
.weft/memory.jsonl
.weft/episodes.jsonl
.weft/memory-proposals.jsonl
```

The JSON index contains redacted note text and the API log contains exact remote
payloads. Weft creates or refreshes these files as `0600` and `.weft/` as `0700`.
They are ignored by Git, but filesystem permissions and backups still matter.

## Test

```bash
UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest
```

To evaluate general RAG behavior against CRAG Task 1, see
[Run CRAG Task 1 with local Ollama](docs/how-to/run-crag-benchmark.md). The
adapter implements CRAG's `get_batch_size()` and `batch_generate_answer()` model
contract without mixing benchmark pages into the user's Obsidian index. A
lightweight `weft benchmark crag <dataset>` runner can use Ollama for both answer
generation and an approximate local judge.

See [the August 2026 security hardening record](docs/security/2026-08-08-security-hardening.md)
for the threat model, root causes, behavior changes, dependency advisories, and
remaining limitations.
