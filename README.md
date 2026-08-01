# Weft

Local-first agent over an Obsidian markdown vault: local embeddings for
retrieval, Claude for reasoning. M0 MVP.

## Setup

    uv sync --extra dev

Set your key for real answers (retrieval + indexing need no key):

    export ANTHROPIC_API_KEY=sk-ant-...

## Use

    uv run weft index /path/to/your/Vault
    uv run weft ask "what did I decide about X?"

The `ask` command retrieves the most relevant note chunks locally and sends
only those to Claude, which answers citing the source files.

## Test

    uv run pytest
