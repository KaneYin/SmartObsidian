# Configure Weft and Use the CLI Safely

Weft indexes an Obsidian Markdown Vault locally, retrieves allowed note chunks,
and sends only selected context to Claude for an answer. This guide covers the
current M0–M2 commands and their security behavior.

## Prerequisites

You need Python 3.11 or newer, `uv`, an Obsidian Vault, and an Anthropic API key
for `ask` or `suggest --rationale`.

From the current repository root:

```bash
cd /Users/kane/Dev/AgentDevelopment
uv sync --extra dev
```

## Configure the API key

Weft reads `ANTHROPIC_API_KEY` from the process environment. It does not parse
`.env` itself.

If using a local `.env`:

```dotenv
ANTHROPIC_API_KEY=replace-with-your-anthropic-api-key
```

Protect and load it:

```bash
chmod 600 .env
set -a
source .env
set +a
```

Confirm presence without printing the value:

```bash
if [[ -n "$ANTHROPIC_API_KEY" ]]; then
  echo "ANTHROPIC_API_KEY is configured"
else
  echo "ANTHROPIC_API_KEY is missing"
fi
```

`.env` is ignored by Git, but ignore rules do not prevent another local account
from reading an overly permissive file. Keep mode `0600`.

## Index the Vault

```bash
uv run weft index "/path/to/your/Obsidian Vault"
```

Indexing performs the following work locally:

1. canonicalizes and validates the Vault root;
2. rejects Vault symlinks;
3. applies the include/exclude privacy policy before reading files;
4. redacts configured patterns before parsing or embedding;
5. creates local sentence-transformer embeddings;
6. persists private vector, metadata, graph, and manifest files.

The first run may download the sentence-transformer model. Note text is not sent
to Claude during indexing.

### Secure defaults

The following relative directories are excluded case-insensitively:

```text
Private/
.obsidian/
.trash/
.weft/
```

The generated `_inbox.md` is also excluded from indexing.

### Use an allowlist

Only index one or more approved Vault areas:

```bash
uv run weft index "/path/to/Vault" \
  --include Projects \
  --include Reference
```

Each value is a relative Vault path. Absolute paths and `..` traversal are
rejected. When at least one `--include` is present, notes outside all included
paths are skipped.

### Add exclusions

```bash
uv run weft index "/path/to/Vault" \
  --exclude Projects/Client-A \
  --exclude Journal
```

Additional exclusions are combined with the secure defaults and take precedence
over the allowlist.

### Redact sensitive content

`--redact` accepts a Python regular expression and is repeatable:

```bash
uv run weft index "/path/to/Vault" \
  --redact 'sk-ant-[A-Za-z0-9_-]+' \
  --redact '(?im)^password\s*:\s*.*$'
```

Matches become `[REDACTED]` before parsing, embeddings, and persistence. Quote
regular expressions so the shell does not expand them. Test project-specific
patterns on non-sensitive sample notes before relying on them; an overly broad
expression can remove useful context.

### Explicitly include `Private/`

```bash
uv run weft index "/path/to/Vault" --include-private
```

This removes only the default `Private/` exclusion. It is deliberately explicit:
if a Private chunk is later retrieved by `ask`, that chunk may be sent to Claude.
Prefer a narrow `--include` policy and redaction whenever possible.

### Custom store location

```bash
uv run weft index "/path/to/Vault" \
  --store "/path/to/weft-data/my-index"
```

The store value is a base path; do not add `.npz` or `.json`. The same base path
must be passed to later `ask` and `suggest` commands.

## Ask questions

```bash
uv run weft ask "What did I decide about the project architecture?"
```

Vector retrieval runs locally. If a graph sidecar exists, one-hop linked notes
may add a bounded number of source chunks. The structured prompt labels source
objects as untrusted note data and asks Claude to cite them as `[n]`.

Change the initial result count within its enforced range:

```bash
uv run weft ask "What are my current priorities?" --k 8
```

`--k` accepts 1 through 50. Zero, negative, non-integer, and oversized values are
rejected so slicing cannot accidentally send most of the index.

Disable graph expansion for comparison:

```bash
uv run weft ask "What are my current priorities?" --no-graph
```

Use a custom index:

```bash
uv run weft ask "What did I decide?" \
  --store "/path/to/weft-data/my-index"
```

### API audit log

Before every supported CLI LLM request, Weft appends the exact system prompt and
payload to the store directory's `api-log.jsonl`. The write occurs before the
network request, so failed attempts are still visible.

The log uses mode `0600`, but it may contain note text, paths, and the question.
Do not paste it into issues or support chats without reviewing and redacting it.

## Suggest inferred links

```bash
uv run weft suggest "/path/to/Vault"
```

The default candidate search and rationale are local. It considers semantically
close pairs not already joined by an explicit wikilink and respects the proposal
ledger.

Optionally request one batched Claude rationale:

```bash
uv run weft suggest "/path/to/Vault" --rationale
```

Only note-path/tag/score metadata is included in that rationale request, and the
payload is recorded in the same API audit log.

### Inbox preservation

`suggest` now follows these rules:

- the index manifest must identify the same canonical Vault;
- an index created before the manifest feature must be rebuilt;
- zero new suggestions leave `_inbox.md` unchanged;
- an existing regular inbox is not replaced by default;
- an inbox symlink or non-regular target is always rejected;
- the ledger is updated only after the inbox is written successfully.

After deliberately reviewing the current inbox, explicit replacement is
available:

```bash
uv run weft suggest "/path/to/Vault" --overwrite-inbox
```

Prefer moving the reviewed inbox instead. `--overwrite-inbox` is an escape hatch,
not the normal workflow.

Threshold and attention-budget controls are bounded:

```bash
uv run weft suggest "/path/to/Vault" --threshold 0.85 --limit 5
```

- threshold must be finite and between -1 and 1;
- limit must be between 0 and 100.

## Generated files and permissions

The default store contains:

```text
.weft/index.npz             # local vectors
.weft/index.json            # redacted chunk text and metadata
.weft/index.graph.json      # explicit wikilink graph
.weft/index.manifest.json   # Vault binding and effective privacy policy
.weft/suggestions.jsonl     # proposed-pair ledger
.weft/api-log.jsonl         # exact outbound LLM payloads
```

Weft creates these files with mode `0600` and the dedicated `.weft/` directory
with mode `0700`. Existing files are corrected when Weft next writes them.

To inspect permissions on macOS without printing contents:

```bash
stat -f '%Sp %N' .env .weft .weft/*
```

## Typical secure session

```bash
cd /Users/kane/Dev/AgentDevelopment

chmod 600 .env
set -a
source .env
set +a

uv run weft index "/path/to/Vault" \
  --include Projects \
  --exclude Projects/Confidential \
  --redact 'sk-ant-[A-Za-z0-9_-]+'

uv run weft ask "Summarize decisions about the next release."
uv run weft suggest "/path/to/Vault"
```

Re-index whenever notes or privacy rules change materially.

## Troubleshooting

### `Index has no security manifest`

The index predates privacy-policy and Vault-binding metadata. Re-run `weft index`
with the intended privacy options.

### `Index belongs to a different vault`

The `suggest` Vault does not match the canonical path stored at index time. Use
the correct Vault or build a separate index for this one.

### `Vault symlinks are not allowed`

Remove or replace the reported symlink with a regular file. Weft intentionally
does not offer a follow-symlink override because it would weaken the Vault
confidentiality boundary.

### `Inbox already exists`

Review and move `_inbox.md`, then rerun. Use `--overwrite-inbox` only when losing
the existing generated review state is intentional.

### Authentication error

Check presence without printing the key:

```bash
[[ -n "$ANTHROPIC_API_KEY" ]] && echo configured || echo missing
```

If missing, source `.env` again. Never use `echo $ANTHROPIC_API_KEY`.

### Answers do not include recent notes

The current milestone does not watch the Vault. Re-run `weft index` after edits.

### Inspect command help

```bash
uv run weft --help
uv run weft index --help
uv run weft ask --help
uv run weft suggest --help
```

For root causes, regression coverage, dependency advisories, and residual risks,
read [the security hardening record](../security/2026-08-08-security-hardening.md).
