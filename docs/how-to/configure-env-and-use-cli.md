# Configure Weft and Use the MVP from the CLI

Weft is a local-first assistant for an Obsidian Markdown vault. It builds the
search index on your machine, retrieves relevant note chunks locally, and sends
only the retrieved context to Claude when you ask a question.

This tutorial covers how to:

1. Install the project dependencies.
2. Configure the Anthropic API key in a `.env` file.
3. Load that configuration into your shell.
4. Index an Obsidian vault.
5. Ask questions through the CLI.

## Prerequisites

You need:

- Python 3.11 or newer;
- [`uv`](https://docs.astral.sh/uv/);
- an Obsidian vault containing Markdown files; and
- an Anthropic API key for the `ask` command.

Open a terminal and move into the Weft project directory:

```bash
cd /Users/kane/Dev/AgentDevelopment/NotesAssistant
```

Install the application and its development dependencies:

```bash
uv sync --extra dev
```

## Configure the `.env` file

Create a file named `.env` in the `NotesAssistant` directory. Its contents
should be:

```dotenv
ANTHROPIC_API_KEY=replace-with-your-anthropic-api-key
```

Replace the placeholder with your real key. Do not add quotes or spaces around
the `=` unless they are part of the value.

> [!IMPORTANT]
> Treat `.env` as a secret. Do not commit it, paste it into documentation, or
> share it in terminal output. Ensure `.env` is listed in `.gitignore` before
> committing changes.

### Load `.env` in zsh

The current MVP reads `ANTHROPIC_API_KEY` from the process environment, but it
does not automatically read the `.env` file. Load the file into the current
terminal session with:

```bash
set -a
source .env
set +a
```

`set -a` tells zsh to export variables while `.env` is sourced. `set +a` turns
that behavior off afterward. You must repeat these commands in each new terminal
session before running `weft ask`.

Confirm that the variable is available without printing the secret:

```bash
if [[ -n "$ANTHROPIC_API_KEY" ]]; then
  echo "ANTHROPIC_API_KEY is configured"
else
  echo "ANTHROPIC_API_KEY is missing"
fi
```

## Index an Obsidian vault

Indexing parses the vault, creates local embeddings, stores note chunks in a
local vector index, and builds a graph from the vault's `[[wikilinks]]`.
Indexing does not call Claude, so it does not require the API key.

Run:

```bash
uv run weft index "/path/to/your/Obsidian Vault"
```

Use quotes when the vault path contains spaces. For example:

```bash
uv run weft index "$HOME/Documents/My Vault"
```

By default, Weft writes the following local index files under `.weft/`:

```text
.weft/index.npz
.weft/index.json
.weft/index.graph.json
```

The command reports how many chunks and wikilink edges it indexed. Run it again
after adding or changing notes so the index reflects the latest vault contents.

### Use a custom index location

Pass `--store` if you do not want to use `.weft/index`:

```bash
uv run weft index "/path/to/your/Obsidian Vault" --store "/path/to/weft-data/my-index"
```

The store path is a base path, so do not add `.npz` or `.json` yourself.

## Ask questions from the CLI

After loading `.env` and building the index, ask a question:

```bash
uv run weft ask "What did I decide about the project architecture?"
```

Weft retrieves relevant chunks, expands the results using linked notes when a
link graph is available, and sends that context to Claude. The response includes
a source list containing the relevant note paths.

### Change the number of initial search results

The default is five vector-search results. Change it with `--k`:

```bash
uv run weft ask "What are my current priorities?" --k 8
```

Graph expansion may add related notes beyond those initial results.

### Disable graph-aware retrieval

Use pure vector retrieval for comparison:

```bash
uv run weft ask "What are my current priorities?" --no-graph
```

### Ask against a custom index

When indexing with `--store`, pass the same base path when asking:

```bash
uv run weft ask "What did I decide?" --store "/path/to/weft-data/my-index"
```

## Typical session

The complete workflow in a new terminal is:

```bash
cd /Users/kane/Dev/AgentDevelopment/NotesAssistant

set -a
source .env
set +a

uv run weft index "/path/to/your/Obsidian Vault"
uv run weft ask "Summarize my notes about the next release."
```

You only need to run `uv sync --extra dev` again when the project dependencies
change. Re-index whenever the vault changes materially.

## Troubleshooting

### `No index at .weft/index`

Run the index command first from the same directory:

```bash
uv run weft index "/path/to/your/Obsidian Vault"
```

If you used `--store` while indexing, provide the same value to `weft ask`.

### Authentication or missing API-key error

Check whether the variable is loaded:

```bash
[[ -n "$ANTHROPIC_API_KEY" ]] && echo "configured" || echo "missing"
```

If it is missing, source `.env` again. Do not use `echo $ANTHROPIC_API_KEY`,
because that prints the secret to the terminal.

### The first indexing or question command is slow

The local sentence-transformer model may need to be downloaded on first use.
Later runs can reuse the cached model.

### Answers do not include recent notes

The MVP does not watch the vault continuously. Run `weft index` again after
editing notes.

### Inspect the available commands

Use the built-in help:

```bash
uv run weft --help
uv run weft index --help
uv run weft ask --help
```
