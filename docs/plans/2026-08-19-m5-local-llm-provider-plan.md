# M5.0 Local LLM & Pluggable Provider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Weft run its reasoning step through a user-selectable provider — a local open-weight model via Ollama (default, fully offline on the GPU) or a remote API — chosen by GPU capability, behind the existing `LLMClient` seam.

**Architecture:** A provider registry (`providers.py`) resolves `.weft/config.toml` + env + CLI overrides into a concrete `LLMClient`. `hardware.py` detects the GPU-memory budget; `models.py` maps that budget to a curated default open-weight model. `ollama_client.py` speaks the Ollama HTTP API. The `AuditedLLM` wrapper gains `provider`/`model`/`left_machine` so the audit log records every machine-boundary crossing. `agent.py` and `suggest.py` are untouched.

**Tech Stack:** Python ≥3.11, `httpx` (transitive via `anthropic`), `tomllib` (stdlib), `argparse`, `pytest`. Ollama is an external local prerequisite for the local provider.

**Scope:** M5.0 only — providers `ollama`, `anthropic`, `fake`; `.weft/config.toml`; GPU tiering; `weft config` and `weft models`; confirm-before-pull; boundary-aware audit. The `openai`-compatible provider and opt-in fallback are M5.1; in-process runtimes are M5.2.

**Design deviation (intentional):** the spec sketched pulling a missing model lazily inside `resolve_llm`. This plan keeps `resolve_llm` pure (no prompts/network side effects) and does provisioning explicitly via `weft models pull` (confirm-before-pull, `--yes`). `ask`/`suggest` fail fast with an actionable remedy when the model is absent. This is more testable and matches the locked "fail fast, actionable" decision.

**Run tests with:** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Create `src/weft/hardware.py` — `GpuInfo` + `detect_gpu()` (Metal/CUDA/CPU).
- Create `src/weft/models.py` — `Tier`, `TIERS`, `tier_for()`, `pick_default()`.
- Create `src/weft/config.py` — `ResolvedConfig`, load/merge/save `.weft/config.toml`.
- Create `src/weft/ollama_client.py` — `OllamaClient` + `ping()`/`list_models()`/`pull()`.
- Create `src/weft/providers.py` — `Provider`, `Availability`, `ProviderUnavailable`, `REGISTRY`, `resolve_llm()`.
- Modify `src/weft/llm.py` — `AuditedLLM` records `provider/model/left_machine`; `ClaudeClient` gains metadata attrs.
- Modify `src/weft/cli.py` — `make_llm` → `resolve_llm`; add `--provider/--model` to `ask`/`suggest`; add `config` and `models` subcommands.
- Create tests: `test_hardware.py`, `test_models.py`, `test_config.py`, `test_ollama_client.py`, `test_providers.py`, `test_llm_audit.py`, `test_cli_config.py`, `test_cli_models.py`.

---

## Task 1: Hardware detection (`hardware.py`)

**Files:**
- Create: `src/weft/hardware.py`
- Test: `tests/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hardware.py
from weft.hardware import GpuInfo, detect_gpu


def _runner(mapping):
    def run(cmd):
        return mapping.get(tuple(cmd))
    return run


def test_apple_silicon_budget_is_70_percent():
    run = _runner({
        ("sysctl", "-n", "hw.memsize"): str(18 * 1024**3),
        ("sysctl", "-n", "hw.optional.arm64"): "1",
    })
    gpu = detect_gpu(system="Darwin", run=run)
    assert gpu.backend == "metal"
    assert gpu.total_mb == 18 * 1024
    assert gpu.budget_mb == int(18 * 1024 * 0.7)


def test_nvidia_reports_vram_as_budget():
    run = _runner({
        ("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"): "24564",
    })
    gpu = detect_gpu(system="Linux", run=run)
    assert gpu.backend == "cuda"
    assert gpu.budget_mb == 24564


def test_unknown_machine_falls_back_to_cpu():
    gpu = detect_gpu(system="Linux", run=lambda cmd: None)
    assert gpu.backend == "cpu"
    assert gpu.budget_mb == 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_hardware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weft.hardware'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/hardware.py
"""Detect the usable GPU-memory budget for model selection. Always returns a
number so callers never crash: unknown hardware falls back to the CPU tier."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

_MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuInfo:
    backend: str      # "metal" | "cuda" | "cpu"
    total_mb: int
    budget_mb: int    # usable for weights, after headroom


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def detect_gpu(
    system: str | None = None,
    run: Callable[[list[str]], str | None] = _run,
) -> GpuInfo:
    system = system or platform.system()
    if system == "Darwin":
        mem = run(["sysctl", "-n", "hw.memsize"])
        arm = run(["sysctl", "-n", "hw.optional.arm64"])
        if mem and mem.isdigit():
            total_mb = int(mem) // _MIB
            if (arm or "").strip() == "1":
                return GpuInfo("metal", total_mb, int(total_mb * 0.7))
            return GpuInfo("cpu", total_mb, min(total_mb // 2, 4000))
    else:
        vram = run(["nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits"])
        if vram:
            first = vram.splitlines()[0].strip()
            if first.isdigit():
                total_mb = int(first)
                return GpuInfo("cuda", total_mb, total_mb)
    return GpuInfo("cpu", 0, 2000)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_hardware.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/hardware.py tests/test_hardware.py
git commit -m "feat(weft): M5 GPU-memory budget detection"
```

---

## Task 2: Model tier registry (`models.py`)

**Files:**
- Create: `src/weft/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from weft.hardware import GpuInfo
from weft.models import TIERS, pick_default, tier_for


def test_tiers_ascending_and_named():
    assert [t.name for t in TIERS] == ["small", "medium", "large", "xl"]
    budgets = [t.max_budget_mb for t in TIERS]
    assert budgets == sorted(budgets)


def test_tier_for_boundaries():
    assert tier_for(5000).name == "small"
    assert tier_for(6000).name == "small"     # inclusive upper edge
    assert tier_for(9000).name == "medium"
    assert tier_for(20000).name == "large"
    assert tier_for(99999).name == "xl"


def test_pick_default_for_m3_pro():
    gpu = GpuInfo("metal", 18 * 1024, int(18 * 1024 * 0.7))  # ~12.6 GB budget
    assert pick_default(gpu) == "llama3.1:8b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weft.models'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/models.py
"""Curated open-weight model registry, keyed by GPU-memory tier. Pinned,
known-good Ollama tags; bump these as better open-weight models ship."""

from __future__ import annotations

from dataclasses import dataclass

from weft.hardware import GpuInfo


@dataclass(frozen=True)
class Tier:
    name: str
    max_budget_mb: int   # inclusive upper bound of this tier's budget
    default: str         # Ollama model tag
    min_mb: int          # advisory floor to run the default comfortably


TIERS: list[Tier] = [
    Tier("small",  6000,        "qwen2.5:3b",  3000),
    Tier("medium", 12000,       "llama3.1:8b", 6000),
    Tier("large",  24000,       "qwen2.5:14b", 10000),
    Tier("xl",     10**9,       "qwen2.5:32b", 20000),
]


def tier_for(budget_mb: int) -> Tier:
    for tier in TIERS:
        if budget_mb <= tier.max_budget_mb:
            return tier
    return TIERS[-1]


def pick_default(gpu: GpuInfo) -> str:
    return tier_for(gpu.budget_mb).default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_models.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/models.py tests/test_models.py
git commit -m "feat(weft): M5 GPU-tier model registry"
```

---

## Task 3: Config load/merge/save (`config.py`)

**Files:**
- Create: `src/weft/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from weft.config import (
    ResolvedConfig,
    config_path_for,
    env_overrides,
    load_config,
    merge,
    save_config,
)


def test_missing_file_yields_defaults(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.provider == "ollama"
    assert cfg.model == "auto"
    assert cfg.endpoint == "http://localhost:11434"


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    save_config(path, ResolvedConfig(provider="anthropic", model="claude-opus-4-8"))
    cfg = load_config(path)
    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-opus-4-8"


def test_precedence_cli_over_env_over_file():
    base = ResolvedConfig(provider="ollama", model="llama3.1:8b")
    env = env_overrides({"WEFT_PROVIDER": "anthropic", "WEFT_MODEL": "env-model"})
    merged = merge(base, env, {"model": "cli-model"})
    assert merged.provider == "anthropic"   # from env (no CLI override)
    assert merged.model == "cli-model"      # CLI wins over env


def test_config_path_is_beside_store():
    assert config_path_for("/x/.weft/index").name == "config.toml"
    assert config_path_for("/x/.weft/index").parent.name == ".weft"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weft.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/config.py
"""Provider/model selection config: a non-secret `.weft/config.toml` merged with
env and CLI overrides. API keys never live here — they stay in the environment.

Precedence (lowest to highest): built-in defaults < file < env < CLI overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from weft.security import secure_write_text

VALID_PROVIDERS = {"ollama", "anthropic", "openai", "fake"}
DEFAULT_ENDPOINT = "http://localhost:11434"


@dataclass
class ResolvedConfig:
    provider: str = "ollama"
    model: str = "auto"
    endpoint: str = DEFAULT_ENDPOINT
    params: dict = field(default_factory=lambda: {"temperature": 0.2, "num_ctx": 8192})


def config_path_for(store_path: Path) -> Path:
    """`.weft/index` -> `.weft/config.toml` (config sits beside the index)."""
    return Path(store_path).parent / "config.toml"


def load_config(path: Path) -> ResolvedConfig:
    path = Path(path)
    if not path.exists():
        return ResolvedConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    default = ResolvedConfig()
    return ResolvedConfig(
        provider=str(data.get("provider", default.provider)),
        model=str(data.get("model", default.model)),
        endpoint=str(data.get("endpoint", default.endpoint)),
        params=dict(data.get("params", default.params)),
    )


def env_overrides(env: dict) -> dict:
    """Map WEFT_* environment variables to config keys (only those present)."""
    out: dict = {}
    if env.get("WEFT_PROVIDER"):
        out["provider"] = env["WEFT_PROVIDER"]
    if env.get("WEFT_MODEL"):
        out["model"] = env["WEFT_MODEL"]
    if env.get("WEFT_ENDPOINT"):
        out["endpoint"] = env["WEFT_ENDPOINT"]
    return out


def merge(cfg: ResolvedConfig, *overrides: dict) -> ResolvedConfig:
    """Apply override dicts in order (later wins), skipping falsy values."""
    provider, model, endpoint, params = cfg.provider, cfg.model, cfg.endpoint, dict(cfg.params)
    for layer in overrides:
        if layer.get("provider"):
            provider = layer["provider"]
        if layer.get("model"):
            model = layer["model"]
        if layer.get("endpoint"):
            endpoint = layer["endpoint"]
        if layer.get("params"):
            params.update(layer["params"])
    return ResolvedConfig(provider=provider, model=model, endpoint=endpoint, params=params)


def _to_toml(cfg: ResolvedConfig) -> str:
    lines = [
        f'provider = "{cfg.provider}"',
        f'model = "{cfg.model}"',
        f'endpoint = "{cfg.endpoint}"',
        "",
        "[params]",
    ]
    for key, value in cfg.params.items():
        rendered = value if isinstance(value, (int, float)) else f'"{value}"'
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def save_config(path: Path, cfg: ResolvedConfig) -> Path:
    return secure_write_text(Path(path), _to_toml(cfg), overwrite=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/config.py tests/test_config.py
git commit -m "feat(weft): M5 provider/model config with override precedence"
```

---

## Task 4: Ollama HTTP client (`ollama_client.py`)

**Files:**
- Create: `src/weft/ollama_client.py`
- Test: `tests/test_ollama_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ollama_client.py
import httpx

from weft.ollama_client import OllamaClient, list_models, ping


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_complete_sends_chat_and_returns_content():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["json"] = httpx.Response(200).is_success and __import__("json").loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hello from llama"}})

    oc = OllamaClient("http://localhost:11434", "llama3.1:8b", client=_client(handler))
    out = oc.complete(system="S", prompt="P")
    assert out == "hello from llama"
    assert seen["url"].endswith("/api/chat")
    assert seen["json"]["stream"] is False
    assert seen["json"]["messages"][0]["role"] == "system"
    assert oc.provider == "ollama" and oc.left_machine is False and oc.model == "llama3.1:8b"


def test_list_models_parses_tags():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:3b"}]})

    assert list_models("http://localhost:11434", client=_client(handler)) == [
        "llama3.1:8b", "qwen2.5:3b",
    ]


def test_ping_false_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    assert ping("http://localhost:11434", client=_client(handler)) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weft.ollama_client'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/ollama_client.py
"""LLMClient over the local Ollama HTTP API, plus introspection/provisioning
helpers. Talks only to a local endpoint; data never leaves the machine."""

from __future__ import annotations

from collections.abc import Callable

import httpx


class OllamaClient:
    """Chat completion via Ollama. Metadata attrs feed the audit log."""

    provider = "ollama"
    left_machine = False

    def __init__(self, endpoint: str, model: str, params: dict | None = None,
                 client: httpx.Client | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._params = params or {}
        self._http = client or httpx.Client(timeout=120)

    def complete(self, system: str, prompt: str) -> str:
        resp = self._http.post(
            f"{self.endpoint}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "options": self._params,
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def ping(endpoint: str, client: httpx.Client | None = None) -> bool:
    http = client or httpx.Client(timeout=5)
    try:
        return http.get(f"{endpoint.rstrip('/')}/api/tags").status_code == 200
    except httpx.HTTPError:
        return False


def list_models(endpoint: str, client: httpx.Client | None = None) -> list[str]:
    http = client or httpx.Client(timeout=10)
    resp = http.get(f"{endpoint.rstrip('/')}/api/tags")
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]


def pull(endpoint: str, tag: str, client: httpx.Client | None = None,
         on_line: Callable[[dict], None] | None = None) -> None:
    """Stream a model pull. Raises httpx.HTTPError on failure."""
    http = client or httpx.Client(timeout=None)
    with http.stream("POST", f"{endpoint.rstrip('/')}/api/pull",
                     json={"name": tag}) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and on_line is not None:
                import json as _json
                on_line(_json.loads(line))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_ollama_client.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/ollama_client.py tests/test_ollama_client.py
git commit -m "feat(weft): M5 Ollama HTTP client + introspection helpers"
```

---

## Task 5: Boundary-aware audit (`llm.py`)

**Files:**
- Modify: `src/weft/llm.py:38-56` (AuditedLLM) and `src/weft/llm.py:59-68` (ClaudeClient `__init__`)
- Test: `tests/test_llm_audit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_audit.py
import json

from weft.llm import AuditedLLM, ClaudeClient


class _Stub:
    provider = "ollama"
    model = "llama3.1:8b"
    left_machine = False

    def complete(self, system, prompt):
        return "ok"


def test_audit_record_includes_boundary_fields(tmp_path):
    log = tmp_path / "api-log.jsonl"
    AuditedLLM(_Stub(), log, "ask").complete(system="S", prompt="P")
    rec = json.loads(log.read_text().strip())
    assert rec["provider"] == "ollama"
    assert rec["model"] == "llama3.1:8b"
    assert rec["left_machine"] is False
    assert rec["purpose"] == "ask" and rec["system"] == "S" and rec["prompt"] == "P"


def test_claude_client_declares_remote_metadata():
    # Construct without network: anthropic reads the key lazily on call, not init,
    # but __init__ instantiates the SDK client; skip if the SDK/key is absent.
    import pytest
    try:
        client = ClaudeClient()
    except Exception:
        pytest.skip("anthropic SDK/key unavailable in this environment")
    assert client.provider == "anthropic"
    assert client.left_machine is True
    assert client.model == "claude-opus-4-8"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_llm_audit.py -v`
Expected: FAIL — audit record has no `provider`/`left_machine` keys.

- [ ] **Step 3: Write minimal implementation**

Replace `AuditedLLM.complete` (`src/weft/llm.py:46-56`) with:

```python
    def complete(self, system: str, prompt: str) -> str:
        secure_append_json(
            self._log_path,
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "purpose": self._purpose,
                "provider": getattr(self._delegate, "provider", "unknown"),
                "model": getattr(self._delegate, "model", "unknown"),
                "left_machine": bool(getattr(self._delegate, "left_machine", True)),
                "system": system,
                "prompt": prompt,
            },
        )
        return self._delegate.complete(system=system, prompt=prompt)
```

In `ClaudeClient` (`src/weft/llm.py:59`), add class-level metadata and set the model attr. Change the class body so the top reads:

```python
class ClaudeClient:
    """Real backend. Uses adaptive thinking per Anthropic guidance for 4.8;
    non-streaming is fine at this max_tokens for a single CLI answer."""

    provider = "anthropic"
    left_machine = True

    def __init__(self, model: str = MODEL, max_tokens: int = 4000):
        import anthropic

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._model = model
        self.model = model  # public metadata for the audit log
        self._max_tokens = max_tokens
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_llm_audit.py -v`
Expected: PASS (1 passed, 1 skipped if no anthropic key).

- [ ] **Step 5: Commit**

```bash
git add src/weft/llm.py tests/test_llm_audit.py
git commit -m "feat(weft): M5 boundary-aware audit (provider/model/left_machine)"
```

---

## Task 6: Provider registry & resolution (`providers.py`)

**Files:**
- Create: `src/weft/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers.py
import pytest

from weft.config import ResolvedConfig, save_config
from weft.providers import ProviderUnavailable, resolve_llm


def test_resolve_ollama_when_available(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.providers.ollama_ping", lambda endpoint: True)
    monkeypatch.setattr("weft.providers.ollama_installed", lambda endpoint: ["llama3.1:8b"])
    save_config(tmp_path / "config.toml",
                ResolvedConfig(provider="ollama", model="llama3.1:8b"))
    llm = resolve_llm({}, store_path=tmp_path / "index", env={})
    assert llm.provider == "ollama" and llm.model == "llama3.1:8b"


def test_ollama_daemon_down_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.providers.ollama_ping", lambda endpoint: False)
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="ollama"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "ollama serve" in str(exc.value)


def test_missing_model_fails_with_pull_remedy(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.providers.ollama_ping", lambda endpoint: True)
    monkeypatch.setattr("weft.providers.ollama_installed", lambda endpoint: [])
    save_config(tmp_path / "config.toml",
                ResolvedConfig(provider="ollama", model="llama3.1:8b"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "weft models pull" in str(exc.value)


def test_anthropic_requires_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="anthropic"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_cli_override_selects_provider(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="anthropic"))
    llm = resolve_llm({"provider": "fake"}, store_path=tmp_path / "index", env={})
    assert llm.provider == "fake"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'weft.providers'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/providers.py
"""Provider registry: resolve config (file + env + CLI) into a concrete
LLMClient, or fail fast with an actionable remedy. resolve_llm is pure — it never
prompts or downloads; provisioning is an explicit `weft models pull`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from weft.config import (
    ResolvedConfig,
    config_path_for,
    env_overrides,
    load_config,
    merge,
)
from weft.hardware import detect_gpu
from weft.llm import ClaudeClient, FakeLLM, LLMClient
from weft.models import pick_default
from weft.ollama_client import OllamaClient
from weft.ollama_client import list_models as ollama_installed
from weft.ollama_client import ping as ollama_ping


class ProviderUnavailable(RuntimeError):
    """The selected provider cannot serve; the message is the user's remedy."""


@dataclass
class Availability:
    ok: bool
    remedy: str = ""


class Provider(Protocol):
    name: str

    def available(self, cfg: ResolvedConfig) -> Availability: ...
    def build(self, cfg: ResolvedConfig) -> LLMClient: ...


def _resolve_model(cfg: ResolvedConfig) -> str:
    if cfg.model and cfg.model != "auto":
        return cfg.model
    return pick_default(detect_gpu())


class OllamaProvider:
    name = "ollama"

    def available(self, cfg: ResolvedConfig) -> Availability:
        if not ollama_ping(cfg.endpoint):
            return Availability(False,
                f"Ollama not reachable at {cfg.endpoint} — start it with "
                f"`ollama serve`, or `weft config set provider anthropic`.")
        model = _resolve_model(cfg)
        if model not in ollama_installed(cfg.endpoint):
            return Availability(False,
                f"Model {model!r} is not installed — run `weft models pull {model}`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig) -> LLMClient:
        return OllamaClient(cfg.endpoint, _resolve_model(cfg), params=cfg.params)


class AnthropicProvider:
    name = "anthropic"

    def available(self, cfg: ResolvedConfig, env: dict | None = None) -> Availability:
        import os
        key = (env or os.environ).get("ANTHROPIC_API_KEY")
        if not key:
            return Availability(False,
                "ANTHROPIC_API_KEY is not set — export it, or switch provider with "
                "`weft config set provider ollama`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig) -> LLMClient:
        if cfg.model and cfg.model != "auto":
            return ClaudeClient(model=cfg.model)
        return ClaudeClient()


class FakeProvider:
    name = "fake"

    def available(self, cfg: ResolvedConfig) -> Availability:
        return Availability(True)

    def build(self, cfg: ResolvedConfig) -> LLMClient:
        client = FakeLLM(response="[fake]")
        client.provider = "fake"          # metadata for the audit log
        client.model = cfg.model
        client.left_machine = False
        return client


REGISTRY: dict[str, Provider] = {
    "ollama": OllamaProvider(),
    "anthropic": AnthropicProvider(),
    "fake": FakeProvider(),
}


def resolve_llm(overrides: dict, *, store_path: Path, env: dict) -> LLMClient:
    cfg = load_config(config_path_for(store_path))
    cfg = merge(cfg, env_overrides(env), overrides)
    provider = REGISTRY.get(cfg.provider)
    if provider is None:
        raise ProviderUnavailable(
            f"Unknown provider {cfg.provider!r}. Choose one of: "
            f"{', '.join(sorted(REGISTRY))}.")
    # AnthropicProvider.available needs env; pass it when supported.
    if isinstance(provider, AnthropicProvider):
        avail = provider.available(cfg, env=env)
    else:
        avail = provider.available(cfg)
    if not avail.ok:
        raise ProviderUnavailable(avail.remedy)
    return provider.build(cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_providers.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/providers.py tests/test_providers.py
git commit -m "feat(weft): M5 provider registry with fail-fast resolution"
```

---

## Task 7: CLI wiring — `config`/`models`, overrides, resolve_llm (`cli.py`)

**Files:**
- Modify: `src/weft/cli.py` (`make_llm`, `_cmd_ask`, `_cmd_suggest`, `build_parser`); add `_cmd_config`, `_cmd_models`.
- Test: `tests/test_cli_config.py`, `tests/test_cli_models.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_config.py
import tomllib

from weft.cli import main


def test_config_set_and_show(tmp_path, capsys):
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "provider", "ollama", "--store", str(store)]) == 0
    data = tomllib.loads((tmp_path / ".weft" / "config.toml").read_text())
    assert data["provider"] == "ollama"
    assert main(["config", "show", "--store", str(store)]) == 0
    assert "ollama" in capsys.readouterr().out


def test_config_set_rejects_unknown_provider(tmp_path):
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "provider", "bogus", "--store", str(store)]) == 2
```

```python
# tests/test_cli_models.py
from weft.cli import main


def test_models_list_shows_tiers_and_recommendation(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("weft.cli.ollama_installed", lambda endpoint: ["llama3.1:8b"])
    store = tmp_path / ".weft" / "index"
    assert main(["models", "list", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "llama3.1:8b" in out and "recommended" in out.lower()


def test_models_pull_requires_confirmation_declined(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("weft.cli.input", lambda prompt="": "n", raising=False)
    called = {"pulled": False}
    def fake_pull(endpoint, tag, on_line=None):
        called["pulled"] = True
    monkeypatch.setattr("weft.cli.ollama_pull", fake_pull)
    store = tmp_path / ".weft" / "index"
    rc = main(["models", "pull", "qwen2.5:3b", "--store", str(store)])
    assert rc == 0 and called["pulled"] is False
    assert "ollama pull" in capsys.readouterr().out  # printed the manual command


def test_models_pull_yes_skips_prompt(tmp_path, monkeypatch):
    called = {"tag": None}
    def fake_pull(endpoint, tag, on_line=None):
        called["tag"] = tag
    monkeypatch.setattr("weft.cli.ollama_pull", fake_pull)
    store = tmp_path / ".weft" / "index"
    rc = main(["models", "pull", "qwen2.5:3b", "--yes", "--store", str(store)])
    assert rc == 0 and called["tag"] == "qwen2.5:3b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_cli_config.py tests/test_cli_models.py -v`
Expected: FAIL — `config`/`models` subcommands do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add imports near the top of `cli.py` (after existing imports):

```python
from weft.config import (
    ResolvedConfig,
    VALID_PROVIDERS,
    config_path_for,
    load_config,
    save_config,
)
from weft.hardware import detect_gpu
from weft.models import TIERS, pick_default
from weft.ollama_client import list_models as ollama_installed
from weft.ollama_client import pull as ollama_pull
from weft.providers import ProviderUnavailable, resolve_llm
```

Replace `make_llm` (`cli.py:38-39`) and update the two call sites. New helper + call sites:

```python
def make_llm(overrides: dict, store_path: Path, purpose: str) -> LLMClient:
    """Resolve the configured provider into an audited LLM client."""
    import os
    client = resolve_llm(overrides, store_path=store_path, env=dict(os.environ))
    return AuditedLLM(client, store_path.parent / "api-log.jsonl", purpose)


def _llm_overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    if getattr(args, "provider", None):
        out["provider"] = args.provider
    if getattr(args, "model", None):
        out["model"] = args.model
    return out
```

In `_cmd_ask`, replace the line that builds the llm
(`cli.py:115` — `llm = AuditedLLM(make_llm(), ...)`) with:

```python
    try:
        llm = make_llm(_llm_overrides(args), store_path, "ask")
    except ProviderUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 1
```

In `_cmd_suggest`, replace the `AuditedLLM(make_llm(), ...)` construction
(`cli.py:222-223`) with:

```python
        try:
            llm = make_llm(_llm_overrides(args), store_path, "suggest_rationale")
        except ProviderUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 1
```

Add the two new command handlers:

```python
def _cmd_config(args: argparse.Namespace) -> int:
    path = config_path_for(Path(args.store))
    if args.action == "show":
        cfg = load_config(path)
        print(f"provider = {cfg.provider}\nmodel = {cfg.model}\nendpoint = {cfg.endpoint}")
        return 0
    if args.action == "path":
        print(path)
        return 0
    # set
    if args.key not in {"provider", "model", "endpoint"}:
        print(f"Unknown config key: {args.key}", file=sys.stderr)
        return 2
    if args.key == "provider" and args.value not in VALID_PROVIDERS:
        print(f"provider must be one of: {', '.join(sorted(VALID_PROVIDERS))}",
              file=sys.stderr)
        return 2
    cfg = load_config(path)
    setattr(cfg, args.key, args.value)
    save_config(path, cfg)
    print(f"{args.key} = {args.value}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    cfg = load_config(config_path_for(Path(args.store)))
    if args.action == "list":
        recommended = pick_default(detect_gpu())
        try:
            installed = set(ollama_installed(cfg.endpoint))
        except Exception:
            installed = set()
        for tier in TIERS:
            flags = []
            if tier.default == recommended:
                flags.append("recommended")
            if tier.default in installed:
                flags.append("installed")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(f"{tier.name:6} {tier.default}{suffix}")
        return 0
    if args.action == "show":
        gpu = detect_gpu()
        print(f"gpu: {gpu.backend} budget={gpu.budget_mb}MB")
        print(f"recommended: {pick_default(gpu)}")
        print(f"configured model: {cfg.model}")
        return 0
    # pull
    tag = args.tag or pick_default(detect_gpu())
    if not args.yes:
        answer = input(f"Pull {tag!r} via Ollama? This downloads several GB. [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print(f"Skipped. To pull manually: ollama pull {tag}")
            return 0
    ollama_pull(cfg.endpoint, tag)
    print(f"Pulled {tag}")
    return 0
```

In `build_parser`, add `--provider`/`--model` to the `ask` and `suggest`
subparsers, and register the two new subcommands:

```python
    for sub_p in (p_ask, p_suggest):
        sub_p.add_argument("--provider", help="Override the configured provider.")
        sub_p.add_argument("--model", help="Override the configured model tag.")

    p_config = sub.add_parser("config", help="Show or set provider/model config.")
    p_config.add_argument("action", choices=["show", "set", "path"])
    p_config.add_argument("key", nargs="?", help="Config key for `set`.")
    p_config.add_argument("value", nargs="?", help="Config value for `set`.")
    p_config.add_argument("--store", default=DEFAULT_STORE)
    p_config.set_defaults(func=_cmd_config)

    p_models = sub.add_parser("models", help="List/inspect/pull local models.")
    p_models.add_argument("action", choices=["list", "show", "pull"])
    p_models.add_argument("tag", nargs="?", help="Model tag for `pull`.")
    p_models.add_argument("--yes", action="store_true", help="Skip the pull confirmation.")
    p_models.add_argument("--store", default=DEFAULT_STORE)
    p_models.set_defaults(func=_cmd_models)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_cli_config.py tests/test_cli_models.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/cli.py tests/test_cli_config.py tests/test_cli_models.py
git commit -m "feat(weft): M5 weft config/models commands + provider-aware ask/suggest"
```

---

## Task 8: Full suite + docs

**Files:**
- Modify: `docs/how-to/configure-env-and-use-cli.md`
- Modify: `README.md`

- [ ] **Step 1: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS — all prior tests plus the new modules. Investigate any regression in `test_cli.py`/`test_cli_suggest.py` (the `make_llm` signature changed; those tests monkeypatch `weft.cli.make_llm` or `resolve_llm` — update them to the new signature if they construct the LLM directly).

- [ ] **Step 2: Document the offline workflow**

Add a section to `docs/how-to/configure-env-and-use-cli.md`:

```markdown
## Running fully offline with a local model

Weft can run its reasoning step on a local open-weight model via Ollama — no
network, no API key. Install Ollama, then:

    weft config set provider ollama        # persists to .weft/config.toml
    weft models list                       # see tiers + the pick for your GPU
    weft models pull                        # pull the recommended model (confirms first)
    weft ask "what did I decide about X?"  # now answered locally

Override per run without changing config:

    weft ask "..." --provider anthropic --model claude-opus-4-8

Selection precedence: CLI flags > WEFT_* env vars > .weft/config.toml > GPU default.
API keys are read from the environment and never written to config.
```

- [ ] **Step 3: Note the prerequisite in README**

Add under Setup in `README.md`:

```markdown
For fully-offline answers, install [Ollama](https://ollama.com) and run
`weft config set provider ollama` then `weft models pull`. Otherwise set
`ANTHROPIC_API_KEY` to use Claude.
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/how-to/configure-env-and-use-cli.md
git commit -m "docs(weft): M5 offline local-model workflow"
```

---

## Self-Review Notes (author)

- **Spec coverage:** provider abstraction (Task 6) · Ollama runtime (Task 4) · `.weft/config.toml` + precedence (Task 3) · GPU detection (Task 1) · tier registry + default pick (Task 2) · confirm-before-pull (Task 7) · fail-fast remedies (Task 6) · boundary-aware audit (Task 5) · offline docs (Task 8). `openai` provider + fallback = M5.1 (out of scope, noted). in-process runtimes = M5.2.
- **Type consistency:** `GpuInfo(backend,total_mb,budget_mb)`, `Tier(name,max_budget_mb,default,min_mb)`, `ResolvedConfig(provider,model,endpoint,params)`, `Availability(ok,remedy)`, `resolve_llm(overrides,*,store_path,env)`, `OllamaClient(endpoint,model,params,client)` and its `provider/model/left_machine` attrs are used identically across tasks.
- **Known coupling to fix during Task 8:** existing `test_cli.py`/`test_cli_suggest.py` may monkeypatch `weft.cli.make_llm`; its signature changed to `make_llm(overrides, store_path, purpose)`. Update those doubles to patch `weft.cli.resolve_llm` (return a `FakeLLM`) instead.
```
