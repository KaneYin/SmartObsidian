# Weft Local LLM & Pluggable Provider Layer — Design (M5)

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-19.
**Scope:** Run the full Weft pipeline offline on the local GPU via open-weight
models, behind a pluggable provider abstraction that also supports remote model
APIs. Chooses a sensible default open-weight model by GPU performance.

## 1. Problem & current state

Weft's reasoning step currently only targets the Anthropic API. `llm.py` defines
an `LLMClient` Protocol (`complete(system, prompt) -> str`) with two
implementations: `ClaudeClient` (Anthropic SDK) and `FakeLLM` (tests).
`cli.make_llm()` hard-codes `ClaudeClient()`. Embeddings are already local
(`sentence-transformers`), so the *only* thing forcing network egress during
`ask`/`suggest --rationale` is the Claude call.

`CLAUDE.md` already lists "a local LLM backend" as explicitly not yet implemented.

**Goal:** make the whole pipeline runnable fully offline on the machine's GPU
using open-weight models, while abstracting the provider layer so users can pick
their own provider — a local open-weight runtime **or** another model's API.

## 2. Decisions (locked during brainstorming)

| Axis | Decision |
|---|---|
| Runtime | Ollama as the default local runtime, behind a pluggable provider layer |
| Providers | `ollama` (local open-weight), `anthropic` (existing), `openai`-compatible (remote APIs **or** local OpenAI-schema servers), `fake` (tests) |
| Selection | `.weft/config.toml` + CLI/env overrides; API keys stay in env, never on disk |
| Model choice | GPU-tier registry → auto-pick a known-good default; confirm-before-pull (`--yes` to skip) |
| Unavailable provider | Fail fast with actionable remedy; automatic fallback is opt-in only, default off |
| Auditing | Every provider logged to the `0600` audit log with `provider`, `model`, `left_machine` |

Rationale for Ollama default: it already solves GPU detection, quantization,
model download/caching, and multi-platform backends (Metal/CUDA/ROCm), letting
Weft focus on *policy* (selection, config, prompt plumbing) behind the existing
`LLMClient` seam. It is a local service — nothing leaves the machine.

Automatic local↔remote fallback is rejected as a default: a local→remote fallback
would send private note chunks off-machine, and crossing the machine boundary must
be a deliberate, explicit choice.

## 3. Reference hardware

Design validated against the developer machine: **Apple M3 Pro, 18-core GPU,
~18 GB unified memory, Metal 3**. On Apple Silicon the GPU shares unified memory,
so the usable weight budget is roughly `total × 0.7 ≈ 12.9 GB` → `medium` tier
(ceiling 13 GB) →
default `llama3.1:8b`. NVIDIA machines are supported via `nvidia-smi` VRAM
detection; unknown/CPU-only machines fall back to the smallest tier.

## 4. Architecture & modules

Approach: a registry of provider factories behind the unchanged `LLMClient`
Protocol (mirrors the existing `make_llm()`/`make_embedder()` factory style and
the project's "small, well-bounded modules" ethos). Rejected alternatives:
adopting LangChain's chat-model abstraction (too heavy, dilutes the audited
`complete()` boundary) and a single OpenAI-compatible client for everything
(loses Ollama-native model list/pull/introspection).

```
src/weft/
  llm.py           KEEP  — LLMClient Protocol + ClaudeClient + FakeLLM (interface unchanged)
  providers.py     NEW   — provider registry; resolve config -> LLMClient
  ollama_client.py NEW   — LLMClient over Ollama HTTP (chat + tags/pull/ps introspection)
  openai_client.py NEW   — LLMClient over OpenAI-compatible /v1/chat/completions
  hardware.py      NEW   — detect GPU memory budget (Apple Metal / NVIDIA / CPU fallback)
  models.py        NEW   — curated tier registry (data) + tier<-budget + default pick
  config.py        NEW   — load/merge .weft/config.toml + env + CLI overrides
  cli.py           TOUCH — make_llm() -> providers.resolve_llm(); new `weft config`, `weft models`
  security.py      REUSE — 0600 audit log, now provider/model/left_machine aware
```

`agent.py` and `suggest.py` are **untouched**: they receive an `LLMClient` and
call `complete()`. Adding a provider = registering one class; no changes to the
retrieve/reason or suggestion code.

## 5. Provider abstraction & config resolution

```python
# providers.py
class Provider(Protocol):
    name: str
    def build(self, cfg: ResolvedConfig) -> LLMClient: ...
    def available(self, cfg: ResolvedConfig) -> Availability: ...   # fail-fast diagnostics

REGISTRY: dict[str, Provider]   # "ollama", "anthropic", "openai", "fake"

def resolve_llm(overrides: dict) -> LLMClient:
    cfg = load_config()                    # .weft/config.toml
    cfg = cfg.merge(env, overrides)        # precedence: CLI > env > file > auto-default
    provider = REGISTRY[cfg.provider]
    avail = provider.available(cfg)
    if not avail.ok:
        raise ProviderUnavailable(avail.remedy)   # fail fast, actionable
    return provider.build(cfg)
```

`.weft/config.toml` (non-secret; keys via env):

```toml
provider = "ollama"          # ollama | anthropic | openai
model    = "auto"            # "auto" => GPU-tier pick; or a pinned tag
endpoint = "http://localhost:11434"

[params]
temperature = 0.2
num_ctx = 8192
```

**Resolution precedence (the one rule everything obeys):**
`--provider/--model` flag > env (`WEFT_PROVIDER`, `WEFT_MODEL`, `ANTHROPIC_API_KEY`, …)
> `config.toml` > built-in auto-default. `model = "auto"` defers to the GPU-tier
pick (Section 6).

## 6. GPU detection, model registry & provisioning

`hardware.py` — always returns a number so callers never crash:

```python
@dataclass
class GpuInfo:
    backend: str      # "metal" | "cuda" | "cpu"
    total_mb: int
    budget_mb: int    # usable for weights, after headroom

def detect_gpu() -> GpuInfo:
    # Apple Silicon: hw.memsize * ~0.7 (unified mem, macOS headroom) -> metal
    # NVIDIA: nvidia-smi --query-gpu=memory.total -> cuda
    # else: cpu, conservative budget -> smallest tier
```

`models.py` — curated registry as data (pinned, known-good tags; bump as better
open-weight models ship):

```python
TIERS = [
    Tier("small",  max_budget_mb=6000,  default="qwen2.5:3b",  min_mb=3000),
    Tier("medium", max_budget_mb=13000, default="llama3.1:8b", min_mb=6000),
    Tier("large",  max_budget_mb=24000, default="qwen2.5:14b", min_mb=10000),
    Tier("xl",     max_budget_mb=10**9, default="qwen2.5:32b", min_mb=20000),
]
def pick_default(gpu: GpuInfo) -> str    # budget -> tier -> default tag
def alternatives(tier) -> list[str]      # opt-up/down choices for `weft models`
```

**Provisioning flow** (`weft models`, and lazily in `resolve_llm` when
`model="auto"`):

1. `detect_gpu()` → `pick_default()` → target tag.
2. Query Ollama `/api/tags`; if already pulled → use it, fully offline.
3. If missing → prompt: *"Selected `llama3.1:8b` (~4.7 GB) for your GPU. Pull now?
   [y/N]"*. `--yes` skips; declining prints the manual `ollama pull` command and
   exits cleanly.
4. The pull is the only network action and is always user-gated (never silent).

`weft models` subcommands: `list` (tiers + installed + recommended),
`pull <tag>`, `show` (current selection + detected GPU).

## 7. Offline pipeline, failure handling, security

**Full offline path:** embeddings are already local; with `provider=ollama` the
Claude reasoning call becomes local, so `weft index`, `weft ask`, and
`weft suggest` (including `--rationale`, which also routes through `resolve_llm`)
run with **zero network** once the model is pulled.

**Failure = fail fast, actionable.** `ProviderUnavailable` carries a specific
remedy: Ollama not running → `ollama serve`; model missing → the pull
prompt/command; API provider missing key → the exact env var to set. No automatic
local↔remote fallback; an explicit config change is required to cross the machine
boundary. Optional fallback chains are opt-in and default off.

**Security / invariants preserved:**

- `config.toml` is non-secret (provider/model/endpoint); API keys stay in env,
  never written to disk.
- Audit log stays `0600` and gains `provider`, `model`, `left_machine`
  (`false` for Ollama/local, `true` for remote APIs) — an honest record of every
  boundary crossing.
- Endpoint URLs validated (default `localhost`); model tags and model output
  treated as untrusted data with the existing escaping discipline.
- Numeric params (`num_ctx`, `temperature`, `--k`) bounds-checked in code, not
  just argparse.
- Ollama is a local service; nothing about it sends vault data off-machine.

## 8. Testing

Offline, no daemon required (HTTP mocked; `FakeLLM` for agent tests):

- `test_providers.py` — registry lookup; config precedence (CLI > env > file >
  auto); `ProviderUnavailable` remedies; keys never persisted.
- `test_hardware.py` — Metal/NVIDIA/CPU branches via mocked `sysctl`/`nvidia-smi`;
  budget math; graceful CPU fallback.
- `test_models.py` — budget→tier→default mapping at boundaries; `alternatives`.
- `test_ollama_client.py` / `test_openai_client.py` — request shape, `complete()`
  contract, error mapping (mocked HTTP).
- `test_config.py` — load/merge/override; missing-file defaults.
- `test_cli_config_models.py` — `weft config`, `weft models`, confirm-before-pull
  (+ `--yes`), audit record carries `left_machine`.

## 9. Milestones

This is **M5** (after the M3 memory work).

- **M5.0:** provider layer + `ollama` + `anthropic` behind the registry;
  `.weft/config.toml`; `hardware.py` + `models.py` tiering; `weft config` /
  `weft models`; confirm-before-pull; boundary-aware audit. → full offline
  pipeline on Apple Silicon.
- **M5.1:** `openai`-compatible provider (remote APIs + local OpenAI-schema
  servers such as LM Studio / vLLM / llama.cpp-server); opt-in fallback chain.
- **M5.2 (optional):** in-process fast paths (`llama.cpp` via `llama-cpp-python`,
  or MLX on Apple Silicon) as additional providers — same seam, no daemon.

## 10. Dependencies

M5.0 adds only an HTTP client for Ollama (`httpx`, already transitively present)
and a TOML reader (`tomllib`, stdlib in Python ≥ 3.11). No torch-for-generation.
Ollama itself is an external local install, documented as a prerequisite for the
local provider. M5.2 in-process paths would add `llama-cpp-python` / `mlx-lm`
behind their own providers, kept optional.
