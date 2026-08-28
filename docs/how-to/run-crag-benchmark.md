# Run CRAG Task 1 with local Ollama

Weft exposes the model class expected by the
[CRAG benchmark](https://github.com/facebookresearch/CRAG). This integration is
for **Task 1**, the general RAG task over cached web search results. It does not
call CRAG's Task 2/3 mock knowledge-graph APIs.

## Data flow

For every CRAG interaction, `weft.benchmarks.crag.CRAGModel`:

1. receives `interaction_id`, `query`, `search_results`, and `query_time`;
2. removes active HTML content and bounds the amount of each page processed;
3. chunks and embeds the cached pages locally;
4. builds an ephemeral NumPy index isolated to that interaction;
5. retrieves the top evidence and asks a loopback Ollama model for a short answer;
6. returns one plain-text response, bounded conservatively for CRAG's 75-token limit.

The CRAG path sends Ollama's top-level `think=false` control so reasoning models
spend the short output budget on the final answer rather than hidden thinking.

The adapter does not read the Obsidian index, write benchmark pages to disk,
record agent memory, use the wikilink graph, or permit a remote fallback.

## Configure Ollama

Install and start Ollama, then select a pulled model explicitly:

```bash
ollama serve
ollama pull qwen2.5:14b
export WEFT_MODEL=qwen2.5:14b
export WEFT_ENDPOINT=http://localhost:11434
```

`WEFT_CRAG_BATCH_SIZE` defaults to `1`, which is safest for a single local model.
It may be set from `1` through `16`. The adapter uses `.weft/config.toml` when
the corresponding environment override is absent, but it always forces the
provider to local Ollama and disables fallback.

## Score optimization modes

To maximize performance on the CRAG benchmark, several retrieval and generation modes can be enabled:

- **BM25 Hybrid Fusion** (`--no-hybrid` to disable, `WEFT_CRAG_HYBRID=1` default): Fuses sparse BM25 keyword matching with dense vector embeddings via Reciprocal Rank Fusion (RRF).
- **CrossEncoder Reranking** (`--reranker`, `--reranker-model`, `WEFT_CRAG_RERANKER=1`, `WEFT_CRAG_RERANKER_MODEL`): Re-scores candidate chunks with a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to improve evidence precision.
- **LLM Query Rewriting** (`--rewrite-llm`, `WEFT_CRAG_REWRITE=1`): Generates a keyword-focused search query with the local LLM and fuses results across raw and rewritten queries.
- **Rerank Candidate Pool Size** (`--rerank-pool`, `WEFT_CRAG_RERANK_POOL`): Controls the number of candidate hits fetched before cross-encoder reranking (default 20).
- **Chunking Control** (`--chunk-chars`, `--overlap`, `WEFT_CRAG_CHUNK_CHARS`, `WEFT_CRAG_CHUNK_OVERLAP`): Fine-tunes web page text chunk size (default 900 chars) and overlap (default 120 chars).
- **Evidence Count** (`--k`, `WEFT_CRAG_K`): Adjusts the number of top evidence chunks included in the LLM prompt (default 8).
- **System Prompt Override** (`--system-prompt`, `WEFT_CRAG_SYSTEM_PROMPT`): Overrides the default CRAG prompt instruction.

Example maximum-performance run:

```bash
uv run weft benchmark crag /path/to/crag.jsonl.bz2 \
  --reranker \
  --rewrite-llm \
  --k 8 \
  --rerank-pool 30 \
  --output .weft/crag-optimized.jsonl
```

## Connect CRAG

Install Weft into the environment used by a CRAG checkout:

```bash
cd /path/to/CRAG
python -m pip install -e /Users/kane/Dev/AgentDevelopment
```

Set CRAG's `models/user_config.py` to:

```python
from weft.benchmarks.crag import CRAGModel

UserModel = CRAGModel
```

Then run CRAG's generation/evaluation entry point from its repository root:

```bash
python local_evaluation.py
```

## Run locally without CRAG's hosted judge

The Weft CLI can read CRAG's `.jsonl.bz2` data directly, avoiding the baseline's
vLLM/AutoGPTQ dependencies and hosted evaluation call:

```bash
uv run weft benchmark crag /path/to/crag_task_1_and_2_dev_v4.jsonl.bz2 \
  --limit 20 \
  --output .weft/crag-task1-smoke.jsonl
```

Start with a small `--limit`; every non-exact prediction can require one local
generation call and one local judging call. Add `--generation-only` to skip the
judge. The result file is written with mode `0600` and contains IDs, questions,
accepted answers, predictions, and local statuses, but not cached HTML pages.

The local judge reports the CRAG-style score `(correct - incorrect) / total` and
the accuracy, hallucination, and missing rates. Because Ollama replaces CRAG's
official judging model, this is an **approximate local development score**, not a
leaderboard-comparable result.

## Cost boundary

Weft answer generation and the Weft-side approximate judge are local and do not
consume hosted-model tokens. CRAG's upstream `local_evaluation.py` separately
uses an OpenAI model as its automatic judge. That upstream judging step is
outside Weft and can still incur API token cost. Run generation on a small slice
first, and do not interpret a generation-only smoke test as a scored CRAG result.

The current upstream script should also be reviewed before a full run: its local
Llama judge branch is not implemented, so selecting a Llama evaluator does not
currently provide a drop-in offline scoring path.
