# Weft CLI & Performance Configuration Reference

A complete reference of all command-line interface (CLI) subcommands, performance add-ons, environment variables, configuration parameters, and execution recipes for **Weft**.

---

## 1. Overview & Configuration Precedence

Weft uses a precedence hierarchy for configuration settings:
$$\text{Built-in Defaults} < \text{Config File } (\text{\texttt{.weft/config.toml}}) < \text{Environment Variables } (\text{\texttt{WEFT\_*}}) < \text{CLI Arguments}$$

Normal usage should prefer the `fast`, `balanced`, and `best` modes. The flags
in this document remain supported for experiments and precise automation.

* **Config File Location**: `.weft/config.toml` (resides beside `.weft/index`).
* **Audit Logging**: All prompt payloads sent to external/local LLM providers are logged to mode-`0600` audit logs.

---

## 2. Environment Variables Reference

### Core System & LLM Provider Settings
| Variable | Description | Allowed Values / Format | Default |
| :--- | :--- | :--- | :--- |
| `WEFT_PROVIDER` | Default LLM provider | `ollama`, `anthropic`, `openai`, `fake` | `ollama` |
| `WEFT_MODEL` | Default model tag | e.g. `qwen3.5:9b`, `claude-3-5-sonnet-20241022`, `auto` | `auto` |
| `WEFT_MODE` | Default retrieval preset | `fast`, `balanced`, `best` | `balanced` |
| `WEFT_ENDPOINT` | Provider HTTP endpoint URL | Must be loopback `http://localhost:...` for Ollama/CRAG | `http://localhost:11434` |
| `WEFT_STORE` | Path to vector store directory | Directory path | `.weft/index` |
| `ANTHROPIC_API_KEY` | API key for Anthropic provider | Secret string | *(None)* |
| `OPENAI_API_KEY` | API key for OpenAI provider | Secret string | *(None)* |
| `HF_TOKEN` | Hugging Face token for faster downloads | Bearer token string | *(None)* |

### CRAG Benchmark Optimization Settings
| Variable | Description | Range / Values | Default |
| :--- | :--- | :--- | :--- |
| `WEFT_CRAG_BATCH_SIZE` | Batch size for CRAG model execution | `1` to `16` | `1` |
| `WEFT_CRAG_HYBRID` | Enable sparse BM25 + dense vector RRF hybrid search | `1` (enabled) or `0` (disabled) | `1` |
| `WEFT_CRAG_REWRITE` | Enable LLM search query rewriting | `1` (enabled) or `0` (disabled) | `0` |
| `WEFT_CRAG_RERANKER` | Enable CrossEncoder reranking stage | `1` (enabled) or `0` (disabled) | `0` |
| `WEFT_CRAG_RERANKER_MODEL` | HuggingFace CrossEncoder model tag | e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `WEFT_CRAG_K` | Number of evidence chunks passed to LLM prompt | Integer `> 0` | `8` |
| `WEFT_CRAG_CHUNK_CHARS` | Character size for web page text chunking | Integer `> 0` | `900` |
| `WEFT_CRAG_CHUNK_OVERLAP` | Character overlap between consecutive chunks | Integer `>= 0` | `120` |
| `WEFT_CRAG_RERANK_POOL` | Number of candidate chunks retrieved before reranking | Integer `> 0` | `20` |
| `WEFT_CRAG_SYSTEM_PROMPT` | Custom system prompt override for CRAG answer generation | Text string | Default CRAG system prompt |

---

## 3. CLI Subcommands & Options

### `weft`
*Open the dependency-free interactive menu for Ask, Chat, Index, Suggest, and Settings.*

```bash
weft
```

The menu calls the same application services as the CLI and REST adapter. It
does not execute `weft` commands through subprocesses.

### `weft index`
*Index an Obsidian Markdown vault into a local vector store.*

```bash
weft index <vault_path> [options]
```

* **Arguments**:
  * `<vault_path>` *(required)*: Path to the Obsidian vault directory.
* **Options**:
  * `--store <path>`: Vector store directory path (default: `.weft/index`).
  * `--include <rel_path>`: Allow only specific relative paths in the vault (repeatable).
  * `--exclude <rel_path>`: Exclude relative path in addition to defaults (`Private/`, `.obsidian/`, `.trash/`, `.weft/`).
  * `--include-private`: Explicitly include `Private/` notes in the index.
  * `--redact <regex>`: Redact matching text patterns before embedding (repeatable).
  * `--chunking {heading|parent-child|sliding}`: Chunking strategy (default: `heading`).
  * `--contextual`: Prepend an LLM-written context sentence to each chunk's embedding for enhanced semantic retrieval.
  * `--provider <name>` / `--model <tag>`: Provider/model override for `--contextual`.
  * `--no-bm25`: Skip building the BM25 lexical index (disables sparse hybrid retrieval).

---

### `weft ask`
*Ask a single-turn factual question over the indexed vault.*

```bash
weft ask "<question>" [options]
```

* **Arguments**:
  * `<question>` *(required)*: Question string wrapped in quotes.
* **Options**:
  * `--store <path>`: Vector store directory path.
  * `--mode {fast|balanced|best}`: Retrieval preset (configured default: `balanced`).
  * `--k <1-50>`: Override the mode's number of final chunks.
  * `--no-graph`: Disable wikilink graph-aware retrieval (pure vector search).
  * `--no-memory`: Skip reading or writing agent memory.
  * `--provider <name>` / `--model <tag>`: Provider and model overrides.
* **Performance Add-ons**:
  * `--hybrid` / `--no-hybrid`: Override the mode's BM25 hybrid setting.
  * `--graph` / `--no-graph`: Override graph expansion.
  * `--rerank` / `--no-rerank`: Override CrossEncoder reranking.
  * `--memory-query` / `--no-memory-query`: Override memory-query fusion.
  * `--rerank-pool <1-500>`: Override the reranking candidate pool.

---

### `weft chat`
*Interactive multi-turn conversation over the indexed vault.*

```bash
weft chat [options]
```

* **Options**:
  * `--store <path>`: Vector store directory path.
  * `--mode {fast|balanced|best}`: Retrieval preset.
  * `--k <1-50>`: Override chunks retrieved per turn.
  * `--no-graph`: Disable wikilink graph traversal.
  * `--no-memory`: Disable memory read/write.
  * `--rewrite-llm` / `--no-rewrite-llm`: Override conversation-aware query rewriting.
  * `--provider <name>` / `--model <tag>`: Provider and model overrides.
  * The same hybrid, graph, rerank, memory-query, and rerank-pool overrides as `ask`.

---

### `weft suggest`
*Infer semantic links between unlinked notes and write proposals to `_inbox.md`.*

```bash
weft suggest <vault_path> [options]
```

* **Arguments**:
  * `<vault_path>` *(required)*: Path to the target vault.
* **Options**:
  * `--store <path>`: Vector store directory path.
  * `--threshold <0.5-1.0>`: Minimum cosine similarity threshold for proposals (default: `0.80`).
  * `--limit <0-100>`: Attention budget; max suggestions per run (default: `10`).
  * `--rationale`: Opt-in to generate LLM-written rationale explanations (default: local template).
  * `--overwrite-inbox`: Explicitly replace existing `_inbox.md` (refuses symlinks).
  * `--provider <name>` / `--model <tag>`: Provider/model override for `--rationale`.

---

### `weft benchmark crag`
*Run an external CRAG Task 1 benchmark evaluation with local Ollama.*

```bash
weft benchmark crag <dataset_path> [options]
```

* **Arguments**:
  * `<dataset_path>` *(required)*: Path to `.jsonl` or `.jsonl.bz2` CRAG dataset file.
* **Options**:
  * `--output <path>`: Output JSONL path (default: `.weft/crag-results.jsonl`).
  * `--store <path>`: Vector store path for config resolution.
  * `--limit <n>`: Limit execution to first `n` dataset items.
  * `--batch-size <1-16>`: Batch size (default: `1`).
  * `--k <1-20>`: Evidence chunks retrieved per question (default: `8`).
  * `--generation-only`: Generate predictions without running local approximate LLM judging.
* **Score Optimization Add-ons**:
  * `--no-hybrid`: Disable BM25 hybrid search.
  * `--reranker`: Enable CrossEncoder candidate reranking stage.
  * `--reranker-model <tag>`: CrossEncoder model tag (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`).
  * `--rewrite-llm`: Enable LLM query rewriting with RRF fusion.
  * `--chunk-chars <100-10000>`: Web page text chunk size.
  * `--overlap <0-5000>`: Character overlap between web page chunks.
  * `--rerank-pool <1-500>`: Candidate pool size retrieved prior to reranking.
  * `--system-prompt "<text>"`: System prompt override for answer generation.

---

### Auxiliary Commands

#### `weft config`
*Show or set configuration parameters in `.weft/config.toml`.*
```bash
weft config show                          # Display active configuration
weft config set provider ollama           # Set default provider
weft config set model qwen3.5:9b          # Set default model tag
weft config set mode balanced             # Set default retrieval mode
weft config path                          # Print resolved config.toml file path
```

#### `weft models`
*Inspect and pull local Ollama open-weight models.*
```bash
weft models list                          # List installed Ollama models
weft models show                          # Show GPU memory budget & tier default
weft models pull qwen3.5:9b [--yes]       # Pull a model tag from Ollama registry
```

#### `weft remember` & `weft memory`
*Store, inspect, curate, and mirror agent memory.*
```bash
# Direct memory store
weft remember "Prefers concise code without explanation" --type preference

# Memory curation
weft memory list                          # List stored semantic & episode memory
weft memory suggest [--llm] [--limit 10]  # Propose new memory items from notes
weft memory pending                       # Show unreviewed memory proposals
weft memory accept <id>                   # Accept a memory proposal
weft memory reject <id>                   # Reject a memory proposal
weft memory forget <id>                   # Delete a memory item
weft memory compact                       # Compact/deduplicate active memory
weft memory mirror --vault /path/to/vault # Write memory items to vault Markdown file
```

#### `weft serve`
*Run local loopback HTTP REST API server.*
```bash
weft serve --host 127.0.0.1 --port 8765 --store .weft/index
```

---

## 4. Performance Tuning Execution Recipes

### Recipe 1: Maximum Performance Vault Query (`weft ask`)
Combines dense vector + BM25 RRF hybrid retrieval, CrossEncoder reranking, wikilink graph traversal, and memory query fusion:

```bash
weft ask "What is the security architecture of Weft?" \
  --k 8 \
  --rerank \
  --memory-query
```

### Recipe 2: Contextual High-Precision Indexing (`weft index`)
Uses an LLM to generate contextual chunk headers for superior vector search precision:

```bash
WEFT_PROVIDER=ollama WEFT_MODEL=qwen3.5:9b \
weft index /path/to/vault \
  --chunking heading \
  --contextual \
  --store .weft/index
```

### Recipe 3: Optimized CRAG Benchmark Run (`weft benchmark crag`)
Enables all score-improving add-ons for CRAG benchmark evaluation:

```bash
WEFT_MODEL=qwen3.5:9b WEFT_ENDPOINT=http://localhost:11434 \
uv run weft benchmark crag /path/to/crag_task1_dev.jsonl.bz2 \
  --reranker \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rewrite-llm \
  --rerank-pool 30 \
  --k 8 \
  --output .weft/crag-max-performance.jsonl
```
