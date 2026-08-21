# M5.1 OpenAI Provider & Opt-in Fallback — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a working `openai`-compatible provider (remote APIs and local OpenAI-schema servers) and an opt-in, default-off provider fallback chain with a machine-boundary notice.

**Architecture:** A new `OpenAIClient` (POST `{endpoint}/chat/completions`) plugs into the M5.0 registry via `OpenAIProvider`. `resolve_llm` gains a fallback loop over `config.fallback` with an `on_fallback` callback; the CLI prints a loud stderr notice when a fallback crosses the machine boundary. `config.fallback: list[str]` (default `[]`) preserves today's strict fail-fast.

**Tech Stack:** Python ≥3.11, `httpx`, `urllib.parse`, `pytest` with `httpx.MockTransport`.

**Scope:** M5.1 only. Builds on the M5.0 provider layer. `chat` REPL is M4.

**Run tests with:** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Create `src/weft/openai_client.py` — `OpenAIClient`, `is_local_endpoint()`.
- Modify `src/weft/config.py` — `ResolvedConfig.fallback`; load/merge/_to_toml.
- Modify `src/weft/providers.py` — standardize `available(cfg, env)`/`build(cfg, env)`; add `OpenAIProvider`; fallback loop + helpers in `resolve_llm`.
- Modify `src/weft/cli.py` — `config set fallback a,b`; `on_fallback` notice via `make_llm`.
- Modify `docs/how-to/change-model-providers.md` — openai now real; fallback section.
- Tests: `test_openai_client.py`, `test_config.py` (+), `test_providers.py` (+), `test_cli_config.py` (+).

---

## Task 1: `fallback` config field (`config.py`)

**Files:**
- Modify: `src/weft/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test (append to tests/test_config.py)**

```python
def test_fallback_defaults_empty(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.fallback == []


def test_fallback_roundtrips(tmp_path):
    from weft.config import ResolvedConfig
    path = tmp_path / "config.toml"
    save_config(path, ResolvedConfig(provider="ollama", fallback=["anthropic", "openai"]))
    cfg = load_config(path)
    assert cfg.fallback == ["anthropic", "openai"]


def test_merge_preserves_fallback():
    from weft.config import ResolvedConfig
    base = ResolvedConfig(provider="ollama", fallback=["anthropic"])
    merged = merge(base, {"model": "x"})
    assert merged.fallback == ["anthropic"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_config.py -k fallback -v`
Expected: FAIL — `ResolvedConfig` has no `fallback`.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/config.py`, add `fallback` to the dataclass (after `params`):

```python
@dataclass
class ResolvedConfig:
    provider: str = "ollama"
    model: str = "auto"
    endpoint: str = DEFAULT_ENDPOINT
    params: dict = field(default_factory=lambda: {"temperature": 0.2, "num_ctx": 8192})
    fallback: list[str] = field(default_factory=list)
```

In `load_config`, add `fallback` to the returned config:

```python
    return ResolvedConfig(
        provider=str(data.get("provider", default.provider)),
        model=str(data.get("model", default.model)),
        endpoint=str(data.get("endpoint", default.endpoint)),
        params=dict(data.get("params", default.params)),
        fallback=list(data.get("fallback", [])),
    )
```

In `merge`, carry `fallback` through:

```python
def merge(cfg: ResolvedConfig, *overrides: dict) -> ResolvedConfig:
    """Apply override dicts in order (later wins), skipping falsy values."""
    provider, model, endpoint = cfg.provider, cfg.model, cfg.endpoint
    params = dict(cfg.params)
    fallback = list(cfg.fallback)
    for layer in overrides:
        if layer.get("provider"):
            provider = layer["provider"]
        if layer.get("model"):
            model = layer["model"]
        if layer.get("endpoint"):
            endpoint = layer["endpoint"]
        if layer.get("params"):
            params.update(layer["params"])
        if layer.get("fallback"):
            fallback = list(layer["fallback"])
    return ResolvedConfig(provider=provider, model=model, endpoint=endpoint,
                          params=params, fallback=fallback)
```

In `_to_toml`, render `fallback` as a top-level array **before** the `[params]`
table (TOML requires tables after top-level keys):

```python
def _to_toml(cfg: ResolvedConfig) -> str:
    fallback_rendered = ", ".join(f'"{p}"' for p in cfg.fallback)
    lines = [
        f'provider = "{cfg.provider}"',
        f'model = "{cfg.model}"',
        f'endpoint = "{cfg.endpoint}"',
        f"fallback = [{fallback_rendered}]",
        "",
        "[params]",
    ]
    for key, value in cfg.params.items():
        rendered = value if isinstance(value, (int, float)) else f'"{value}"'
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: PASS (existing config tests + 3 new).

- [ ] **Step 5: Commit**

```bash
git add src/weft/config.py tests/test_config.py
git commit -m "feat(weft): M5.1 fallback list in provider config"
```

---

## Task 2: OpenAIClient (`openai_client.py`)

**Files:**
- Create: `src/weft/openai_client.py`
- Test: `tests/test_openai_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openai_client.py
import json

import httpx

from weft.openai_client import OpenAIClient, is_local_endpoint


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_is_local_endpoint():
    assert is_local_endpoint("http://localhost:1234/v1")
    assert is_local_endpoint("http://127.0.0.1:8000/v1")
    assert is_local_endpoint("http://my-box.local/v1")
    assert not is_local_endpoint("https://api.openai.com/v1")


def test_complete_posts_chat_completions_with_auth_and_whitelist():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    oc = OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", api_key="sk-x",
                      params={"temperature": 0.3, "num_ctx": 8192}, client=_client(handler))
    out = oc.complete(system="S", prompt="P")
    assert out == "hi"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["auth"] == "Bearer sk-x"
    assert seen["body"]["temperature"] == 0.3
    assert "num_ctx" not in seen["body"]           # ollama-only key dropped
    assert seen["body"]["messages"][0]["role"] == "system"
    assert oc.provider == "openai" and oc.left_machine is True and oc.model == "gpt-4o-mini"


def test_local_endpoint_needs_no_auth_and_stays_on_machine():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "local"}}]})

    oc = OpenAIClient("http://localhost:1234/v1", "local-model", client=_client(handler))
    assert oc.complete(system="S", prompt="P") == "local"
    assert oc.left_machine is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_openai_client.py -v`
Expected: FAIL — `weft.openai_client` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/openai_client.py
"""LLMClient over an OpenAI-compatible chat-completions endpoint. Works with remote
APIs (OpenAI, Groq, OpenRouter, ...) and local servers (LM Studio, vLLM,
llama.cpp-server). `left_machine` is derived from the endpoint host, so a local
server correctly logs as on-machine."""

from __future__ import annotations

from urllib.parse import urlparse

import httpx

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_OPENAI_PARAM_KEYS = ("temperature", "top_p", "max_tokens")


def is_local_endpoint(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".local")


class OpenAIClient:
    provider = "openai"

    def __init__(self, endpoint: str, model: str, api_key: str | None = None,
                 params: dict | None = None, client: httpx.Client | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._params = params or {}
        self._http = client or httpx.Client(timeout=120)
        self.left_machine = not is_local_endpoint(self.endpoint)

    def complete(self, system: str, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        for key in _OPENAI_PARAM_KEYS:
            if key in self._params:
                body[key] = self._params[key]
        resp = self._http.post(f"{self.endpoint}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_openai_client.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/openai_client.py tests/test_openai_client.py
git commit -m "feat(weft): M5.1 OpenAI-compatible chat client"
```

---

## Task 3: OpenAIProvider + standardized availability (`providers.py`)

**Files:**
- Modify: `src/weft/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test (append to tests/test_providers.py)**

```python
def test_openai_available_with_model_and_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="gpt-4o-mini", endpoint="https://api.openai.com/v1"))
    llm = resolve_llm({}, store_path=tmp_path / "index", env={"OPENAI_API_KEY": "sk-x"})
    assert llm.provider == "openai" and llm.model == "gpt-4o-mini"


def test_openai_auto_model_fails_fast(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="auto", endpoint="https://api.openai.com/v1"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={"OPENAI_API_KEY": "sk-x"})
    assert "explicit model" in str(exc.value)


def test_openai_remote_requires_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="gpt-4o-mini", endpoint="https://api.openai.com/v1"))
    with pytest.raises(ProviderUnavailable) as exc:
        resolve_llm({}, store_path=tmp_path / "index", env={})
    assert "OPENAI_API_KEY" in str(exc.value)


def test_openai_local_needs_no_key(tmp_path):
    save_config(tmp_path / "config.toml", ResolvedConfig(
        provider="openai", model="local-model", endpoint="http://localhost:1234/v1"))
    llm = resolve_llm({}, store_path=tmp_path / "index", env={})
    assert llm.provider == "openai" and llm.left_machine is False
```

(The existing `test_providers.py` already imports `pytest`, `ResolvedConfig`,
`save_config`, `resolve_llm`, `ProviderUnavailable`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_providers.py -k openai -v`
Expected: FAIL — `openai` not in `REGISTRY`.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/providers.py`, import the OpenAI client near the other imports:

```python
from weft.openai_client import OpenAIClient, is_local_endpoint
```

Standardize the `Provider` protocol and every provider to `available(cfg, env)` /
`build(cfg, env)`. Replace the protocol (lines 37-41):

```python
class Provider(Protocol):
    name: str

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability: ...
    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient: ...
```

Update `OllamaProvider` methods to accept `env` (unused):

```python
    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        if not ollama_ping(cfg.endpoint):
            return Availability(False,
                f"Ollama not reachable at {cfg.endpoint} — start it with "
                f"`ollama serve`, or `weft config set provider anthropic`.")
        model = _resolve_model(cfg)
        if model not in ollama_installed(cfg.endpoint):
            return Availability(False,
                f"Model {model!r} is not installed — run `weft models pull {model}`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        return OllamaClient(cfg.endpoint, _resolve_model(cfg), params=cfg.params)
```

Update `AnthropicProvider` to the uniform signature (drop the `env=None` special form):

```python
    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        if not env.get("ANTHROPIC_API_KEY"):
            return Availability(False,
                "ANTHROPIC_API_KEY is not set — export it, or switch provider with "
                "`weft config set provider ollama`.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        if cfg.model and cfg.model != "auto":
            return ClaudeClient(model=cfg.model)
        return ClaudeClient()
```

Update `FakeProvider`:

```python
    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        client = FakeLLM(response="[fake]")
        client.provider = "fake"
        client.model = cfg.model
        client.left_machine = False
        return client
```

Add the new provider before `REGISTRY`:

```python
class OpenAIProvider:
    name = "openai"

    def available(self, cfg: ResolvedConfig, env: dict) -> Availability:
        if not cfg.model or cfg.model == "auto":
            return Availability(False,
                "openai needs an explicit model — run `weft config set model <name>`.")
        if not is_local_endpoint(cfg.endpoint) and not env.get("OPENAI_API_KEY"):
            return Availability(False,
                f"OPENAI_API_KEY is not set for remote endpoint {cfg.endpoint} — export "
                "it, use a local endpoint, or switch provider.")
        return Availability(True)

    def build(self, cfg: ResolvedConfig, env: dict) -> LLMClient:
        return OpenAIClient(cfg.endpoint, cfg.model,
                            api_key=env.get("OPENAI_API_KEY"), params=cfg.params)
```

Register it:

```python
REGISTRY: dict[str, Provider] = {
    "ollama": OllamaProvider(),
    "anthropic": AnthropicProvider(),
    "openai": OpenAIProvider(),
    "fake": FakeProvider(),
}
```

Update `resolve_llm` to call the uniform signature (remove the isinstance
special-case; the fallback loop comes in Task 4):

```python
def resolve_llm(overrides: dict, *, store_path: Path, env: dict) -> LLMClient:
    cfg = load_config(config_path_for(store_path))
    cfg = merge(cfg, env_overrides(env), overrides)
    provider = REGISTRY.get(cfg.provider)
    if provider is None:
        raise ProviderUnavailable(
            f"Unknown provider {cfg.provider!r}. Choose one of: "
            f"{', '.join(sorted(REGISTRY))}.")
    avail = provider.available(cfg, env)
    if not avail.ok:
        raise ProviderUnavailable(avail.remedy)
    return provider.build(cfg, env)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_providers.py -v`
Expected: PASS (existing provider tests + 4 new openai tests).

- [ ] **Step 5: Commit**

```bash
git add src/weft/providers.py tests/test_providers.py
git commit -m "feat(weft): M5.1 OpenAIProvider + standardized availability(cfg, env)"
```

---

## Task 4: Fallback chain in `resolve_llm` (`providers.py`)

**Files:**
- Modify: `src/weft/providers.py`
- Test: `tests/test_providers.py`

- [ ] **Step 1: Write the failing test (append)**

```python
class _StubProvider:
    def __init__(self, name, ok, left_machine):
        self.name = name
        self._ok = ok
        self._left = left_machine

    def available(self, cfg, env):
        from weft.providers import Availability
        return Availability(self._ok, "" if self._ok else f"{self.name} down")

    def build(self, cfg, env):
        c = __import__("weft.llm", fromlist=["FakeLLM"]).FakeLLM(response="x")
        c.provider = self.name
        c.model = "m"
        c.left_machine = self._left
        return c


def test_fallback_used_when_primary_down(tmp_path, monkeypatch):
    import weft.providers as P
    monkeypatch.setitem(P.REGISTRY, "remote_stub", _StubProvider("remote_stub", True, True))
    monkeypatch.setattr(P, "ollama_ping", lambda endpoint: False)  # primary down
    save_config(tmp_path / "config.toml",
                ResolvedConfig(provider="ollama", fallback=["remote_stub"]))
    events = []
    llm = resolve_llm({}, store_path=tmp_path / "index", env={},
                      on_fallback=lambda a, b, crossed: events.append((a, b, crossed)))
    assert llm.provider == "remote_stub"
    assert events == [("ollama", "remote_stub", True)]  # local -> remote crossed


def test_fallback_local_target_does_not_cross(tmp_path, monkeypatch):
    import weft.providers as P
    monkeypatch.setattr(P, "ollama_ping", lambda endpoint: False)
    save_config(tmp_path / "config.toml",
                ResolvedConfig(provider="ollama", fallback=["fake"]))
    events = []
    llm = resolve_llm({}, store_path=tmp_path / "index", env={},
                      on_fallback=lambda a, b, crossed: events.append((a, b, crossed)))
    assert llm.provider == "fake"
    assert events == [("ollama", "fake", False)]


def test_empty_fallback_still_fails_fast(tmp_path, monkeypatch):
    import weft.providers as P
    monkeypatch.setattr(P, "ollama_ping", lambda endpoint: False)
    save_config(tmp_path / "config.toml", ResolvedConfig(provider="ollama", fallback=[]))
    with pytest.raises(ProviderUnavailable):
        resolve_llm({}, store_path=tmp_path / "index", env={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_providers.py -k fallback -v`
Expected: FAIL — `resolve_llm` has no `on_fallback` and no fallback loop.

- [ ] **Step 3: Write minimal implementation**

Add helpers and rewrite `resolve_llm` in `src/weft/providers.py`:

```python
_LOCAL_PROVIDERS = {"ollama", "fake"}


def _is_local_provider(name: str, cfg: ResolvedConfig) -> bool:
    if name == "openai":
        return is_local_endpoint(cfg.endpoint)
    return name in _LOCAL_PROVIDERS


def _fallback_cfg(cfg: ResolvedConfig, name: str) -> ResolvedConfig:
    """A fallback provider uses its own defaults, not the primary's model/endpoint."""
    from weft.config import DEFAULT_ENDPOINT
    return ResolvedConfig(provider=name, model="auto", endpoint=DEFAULT_ENDPOINT,
                          params=dict(cfg.params))


def resolve_llm(overrides: dict, *, store_path: Path, env: dict,
                on_fallback=None) -> LLMClient:
    cfg = load_config(config_path_for(store_path))
    cfg = merge(cfg, env_overrides(env), overrides)
    order = [cfg.provider] + [p for p in cfg.fallback if p != cfg.provider]

    remedies: list[str] = []
    for i, name in enumerate(order):
        provider = REGISTRY.get(name)
        if provider is None:
            remedies.append(f"unknown provider {name!r}")
            continue
        use_cfg = cfg if i == 0 else _fallback_cfg(cfg, name)
        avail = provider.available(use_cfg, env)
        if not avail.ok:
            remedies.append(avail.remedy)
            continue
        client = provider.build(use_cfg, env)
        if i > 0 and on_fallback is not None:
            crossed = _is_local_provider(order[0], cfg) and bool(
                getattr(client, "left_machine", True))
            on_fallback(order[0], name, crossed)
        return client
    raise ProviderUnavailable(" ; ".join(remedies))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_providers.py -v`
Expected: PASS (all provider tests + 3 fallback tests).

- [ ] **Step 5: Commit**

```bash
git add src/weft/providers.py tests/test_providers.py
git commit -m "feat(weft): M5.1 opt-in provider fallback with boundary callback"
```

---

## Task 5: CLI — `config set fallback` + boundary notice (`cli.py`)

**Files:**
- Modify: `src/weft/cli.py`
- Test: `tests/test_cli_config.py`

- [ ] **Step 1: Write the failing test (append to tests/test_cli_config.py)**

```python
def test_config_set_fallback_list(tmp_path):
    import tomllib
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "fallback", "anthropic,openai", "--store", str(store)]) == 0
    data = tomllib.loads((tmp_path / ".weft" / "config.toml").read_text())
    assert data["fallback"] == ["anthropic", "openai"]


def test_config_set_fallback_rejects_unknown(tmp_path):
    store = tmp_path / ".weft" / "index"
    assert main(["config", "set", "fallback", "anthropic,bogus", "--store", str(store)]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli_config.py -k fallback -v`
Expected: FAIL — `config set` only handles provider/model/endpoint.

- [ ] **Step 3: Write minimal implementation**

In `src/weft/cli.py`, add a fallback-notice helper near `make_llm`:

```python
def _fallback_notice(primary: str, chosen: str, crossed: bool) -> None:
    if crossed:
        print(
            f"falling back to {chosen} — this sends note content off your machine "
            f"(configured in fallback).",
            file=sys.stderr,
        )
    else:
        print(f"primary {primary} unavailable; using fallback {chosen}.", file=sys.stderr)
```

Change `make_llm` to pass the callback:

```python
def make_llm(overrides: dict, store_path: Path) -> LLMClient:
    """Resolve the configured provider into a raw LLM client. The caller wraps
    it in AuditedLLM. Raises ProviderUnavailable with an actionable remedy."""
    return resolve_llm(overrides, store_path=store_path, env=dict(os.environ),
                       on_fallback=_fallback_notice)
```

In `_cmd_config`, extend the `set` branch to accept `fallback` (a comma-separated
list). Replace the key-validation and set block:

```python
    # set
    if args.key not in {"provider", "model", "endpoint", "fallback"}:
        print(f"Unknown config key: {terminal_safe(str(args.key))}", file=sys.stderr)
        return 2
    cfg = load_config(path)
    if args.key == "fallback":
        entries = [v.strip() for v in (args.value or "").split(",") if v.strip()]
        bad = [e for e in entries if e not in VALID_PROVIDERS]
        if bad:
            print(f"unknown provider(s) in fallback: {', '.join(bad)}", file=sys.stderr)
            return 2
        cfg.fallback = entries
        save_config(path, cfg)
        print(f"fallback = {', '.join(entries) or '(none)'}")
        return 0
    if args.key == "provider" and args.value not in VALID_PROVIDERS:
        print(
            f"provider must be one of: {', '.join(sorted(VALID_PROVIDERS))}",
            file=sys.stderr,
        )
        return 2
    setattr(cfg, args.key, args.value)
    save_config(path, cfg)
    print(f"{args.key} = {terminal_safe(str(args.value))}")
    return 0
```

Extend `_cmd_config`'s `show` to print the fallback list. Replace its `show` block:

```python
    if args.action == "show":
        cfg = load_config(path)
        print(f"provider = {cfg.provider}\nmodel = {cfg.model}\nendpoint = {cfg.endpoint}")
        print(f"fallback = {', '.join(cfg.fallback) or '(none)'}")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --extra dev pytest tests/test_cli_config.py -v`
Expected: PASS (existing config CLI tests + 2 new).

- [ ] **Step 5: Commit**

```bash
git add src/weft/cli.py tests/test_cli_config.py
git commit -m "feat(weft): M5.1 config set fallback + machine-boundary notice"
```

---

## Task 6: Full suite + docs

**Files:**
- Modify: `docs/how-to/change-model-providers.md`

- [ ] **Step 1: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS across all modules.

- [ ] **Step 2: Update the providers how-to**

In `docs/how-to/change-model-providers.md`, change the `openai` row note from
"planned for M5.1 / not wired" to available, and add usage + a fallback section:

Replace the reserved-note block with:

```markdown
> `openai` targets any OpenAI-compatible endpoint — remote (OpenAI, Groq,
> OpenRouter) or a local server (LM Studio, vLLM, llama.cpp-server). Point
> `endpoint` at the base URL (including `/v1`) and set an explicit `model`.
```

Add a "Use an OpenAI-compatible endpoint" section:

```markdown
## Use an OpenAI-compatible endpoint

    weft config set provider openai
    weft config set endpoint https://api.openai.com/v1     # or a local server URL
    weft config set model gpt-4o-mini                      # explicit; `auto` is rejected
    export OPENAI_API_KEY=sk-...                           # not needed for local hosts
    weft ask "..."

A local server (LM Studio / vLLM on localhost) needs no key and logs as
`left_machine: false`; a remote endpoint logs as `left_machine: true`.

## Configure an opt-in fallback

By default Weft fails fast when the chosen provider is unavailable. You can opt in
to a fallback chain (tried in order) — it stays off unless you set it:

    weft config set fallback anthropic          # or: anthropic,openai

If the primary is a local provider and a fallback sends content to a remote API,
Weft prints a one-line notice before the call and records `left_machine: true` in
the audit log. Fallback providers use their own default model.
```

- [ ] **Step 3: Commit**

```bash
git add docs/how-to/change-model-providers.md
git commit -m "docs(weft): M5.1 openai endpoint + fallback usage"
```

---

## Self-Review Notes (author)

- **Spec coverage:** openai client + `/chat/completions` + auth + param whitelist +
  host-derived `left_machine` (Task 2) · OpenAIProvider availability/remedies
  (Task 3) · standardized `available(cfg, env)` (Task 3) · `fallback` config
  round-trip (Task 1) · fallback loop + `on_fallback` + boundary flag (Task 4) ·
  `config set fallback` + notice (Task 5) · docs (Task 6). Default-off fallback
  preserves strict fail-fast (Task 4 empty-fallback test).
- **Type consistency:** `is_local_endpoint(url)`, `OpenAIClient(endpoint, model,
  api_key, params, client)` with `provider/model/left_machine`; every provider now
  `available(cfg, env)`/`build(cfg, env)`; `resolve_llm(overrides, *, store_path,
  env, on_fallback=None)`; `_fallback_cfg`, `_is_local_provider`. `ResolvedConfig`
  gains `fallback: list[str]`.
- **Migration:** `resolve_llm`'s new `on_fallback` is keyword-only with default
  `None`, so the existing `make_llm` call remains valid until Task 5 wires the notice.
```
