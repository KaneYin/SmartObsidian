# How to Change Weft's Model Provider

Weft runs its reasoning step (`ask`, `suggest --rationale`, `memory suggest --llm`)
through a **pluggable provider**. This guide shows how to see the current provider,
switch between a local open-weight model and a remote API, override the choice for a
single command, and diagnose problems.

Retrieval, embeddings, and the wikilink graph are always local; the provider only
governs the LLM reasoning call.

## Providers at a glance

| Provider | What it is | Runs where | Needs |
|----------|------------|-----------|-------|
| `ollama` | Local open-weight models via [Ollama](https://ollama.com) | Your machine's GPU | Ollama installed + a pulled model |
| `anthropic` | Claude API (default model `claude-opus-4-8`) | Anthropic's servers | `ANTHROPIC_API_KEY` in the environment |
| `openai` | Any OpenAI-compatible endpoint | Remote API **or** local server | `endpoint` + explicit `model`; key for remote |
| `fake` | Deterministic stub | In-process | Nothing (testing/offline demos) |

> `openai` targets any OpenAI-compatible endpoint — remote (OpenAI, Groq,
> OpenRouter) or a local server (LM Studio, vLLM, llama.cpp-server). Point
> `endpoint` at the base URL (including `/v1`) and set an explicit `model`.

## Where the choice comes from (precedence)

Weft resolves the provider and model from four layers, highest wins:

```
--provider / --model flags   (this command only)
  > WEFT_PROVIDER / WEFT_MODEL / WEFT_ENDPOINT   (this shell)
    > .weft/config.toml        (persistent default)
      > built-in GPU default   (model = "auto" picks by GPU)
```

API keys are **never** part of this resolution or written to `config.toml`; they are
read only from the environment (e.g. `ANTHROPIC_API_KEY`).

## See what is configured now

```bash
uv run weft config show      # provider / model / endpoint from .weft/config.toml
uv run weft config path      # where that file lives
uv run weft models show      # detected GPU budget + the model auto-pick
```

`model = auto` means "let Weft choose a model sized for this GPU" (see
[Run fully offline](configure-env-and-use-cli.md#run-fully-offline-with-a-local-model)).

## Switch the persistent default

`weft config set` writes the choice to `.weft/config.toml` so every later command
uses it.

### To a local open-weight model (offline)

```bash
uv run weft config set provider ollama
uv run weft models list        # see tiers; the row for your GPU is marked [recommended]
uv run weft models pull        # pull the recommended model (confirms before downloading)
uv run weft ask "what did I decide about X?"   # answered locally, no network
```

Pin a specific model instead of the GPU auto-pick:

```bash
uv run weft config set model qwen2.5:14b
uv run weft models pull qwen2.5:14b
```

Reset to the automatic GPU-sized pick:

```bash
uv run weft config set model auto
```

### To the Claude API

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # keep this in your shell/.env, never in config
uv run weft config set provider anthropic
uv run weft config set model auto          # 'auto' uses claude-opus-4-8
uv run weft ask "summarize my decisions about the release"
```

### To an OpenAI-compatible endpoint

```bash
uv run weft config set provider openai
uv run weft config set endpoint https://api.openai.com/v1   # or a local server URL
uv run weft config set model gpt-4o-mini                    # explicit; `auto` is rejected
export OPENAI_API_KEY=sk-...                                # not needed for local hosts
uv run weft ask "..."
```

A local server (LM Studio / vLLM on `localhost`) needs no key and logs as
`left_machine: false`; a remote endpoint logs as `left_machine: true`. Only
`temperature`, `top_p`, and `max_tokens` from `[params]` are forwarded.

## Configure an opt-in fallback

By default Weft **fails fast** when the chosen provider is unavailable. You can opt
in to a fallback chain (tried in order); it stays off unless you set it:

```bash
uv run weft config set fallback anthropic          # or: anthropic,openai
uv run weft config show                             # shows the fallback list
```

If the primary is a local provider and a fallback sends content to a remote API,
Weft prints a one-line notice before the call and records `left_machine: true` in
the audit log. Fallback providers use their own default model (not the primary's).
Clear it with `weft config set fallback ""`.

## Override for a single command

The `--provider` / `--model` flags change the backend for just that invocation
without touching `config.toml`. They are available on the commands that call an LLM:

```bash
# Ask this one question with Claude, even if the default is ollama
uv run weft ask "..." --provider anthropic --model claude-opus-4-8

# A/B a different local model for one question
uv run weft ask "..." --provider ollama --model llama3.1:8b

# Use a provider for the opt-in suggestion rationale
uv run weft suggest "/path/to/Vault" --rationale --provider ollama

# Use a provider for opt-in inferred-memory extraction
uv run weft memory suggest --llm --provider anthropic
```

## Override for a shell session (env vars)

Handy in scripts or CI without editing files:

```bash
export WEFT_PROVIDER=ollama
export WEFT_MODEL=qwen2.5:14b
uv run weft ask "..."          # uses the env values (unless a --flag overrides)
```

`WEFT_ENDPOINT` points at a non-default host (see below).

## Point at a different endpoint

`endpoint` is the base URL of the local provider. Change it to reach Ollama on
another machine or a non-standard port:

```bash
uv run weft config set endpoint http://192.168.1.20:11434
# or, for one session:
export WEFT_ENDPOINT=http://localhost:11500
```

The default is `http://localhost:11434`.

## What gets logged

Every LLM call — local or remote — appends one record to `.weft/api-log.jsonl`
(mode `0600`) with the exact payload plus `provider`, `model`, and a `left_machine`
flag: `false` for local Ollama, `true` for a remote API. This is an honest record of
when your note content crossed the machine boundary; treat the log as sensitive.

## Troubleshooting

Weft **fails fast with an actionable message** and never silently switches providers
(a silent local→remote switch would send note content off your machine).

| Message | Fix |
|---------|-----|
| `Ollama not reachable at http://localhost:11434 …` | Start Ollama: `ollama serve`, or switch: `weft config set provider anthropic`. |
| `Model 'llama3.1:8b' is not installed — run 'weft models pull llama3.1:8b'` | Pull it: `uv run weft models pull` (or the exact tag). |
| `ANTHROPIC_API_KEY is not set …` | `export ANTHROPIC_API_KEY=sk-ant-...`, or switch to `ollama`. |
| `openai needs an explicit model …` | Set one: `weft config set model gpt-4o-mini`. |
| `OPENAI_API_KEY is not set for remote endpoint …` | `export OPENAI_API_KEY=…`, or use a local endpoint. |
| `provider must be one of: anthropic, fake, ollama, openai` | Typo in `weft config set provider <x>`. |

For `--rationale` and `memory suggest --llm`, an unavailable provider does **not**
fail the command — it prints a warning and falls back to the local heuristic.

## Quick reference

```bash
uv run weft config show                       # current provider/model/endpoint
uv run weft config set provider ollama        # persist a provider
uv run weft config set model auto             # GPU-sized default model
uv run weft config set endpoint <url>         # custom host
uv run weft models list                       # tiers + recommended + installed
uv run weft models pull [tag] [--yes]         # download a model (confirms first)
uv run weft ask "..." --provider <p> --model <m>   # one-off override
WEFT_PROVIDER=<p> WEFT_MODEL=<m> uv run weft ask "..."   # session override
```
