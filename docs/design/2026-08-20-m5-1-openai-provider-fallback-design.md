# Weft M5.1 — OpenAI-Compatible Provider & Opt-in Fallback — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-20.
**Scope:** Add an `openai` provider (remote OpenAI-compatible APIs and local
OpenAI-schema servers) to the M5.0 registry, plus an opt-in provider fallback chain.

## 1. Problem & current state

M5.0 shipped a provider registry behind `LLMClient` with `ollama`, `anthropic`, and
`fake`; `openai` is a reserved name in `VALID_PROVIDERS` but not in `REGISTRY`, so
selecting it fails fast. `resolve_llm` resolves exactly one provider and fails fast
if it is unavailable — there is no fallback.

M5.1 makes `openai` real (covering OpenAI, Groq, OpenRouter, Together, and local
servers like LM Studio / vLLM / llama.cpp-server, all of which speak
`POST /v1/chat/completions`) and adds a configurable, default-off fallback chain.

## 2. Decisions (locked during brainstorming)

- **openai identity:** reuse the existing `config.endpoint` as the base URL and read
  the key from the standard `OPENAI_API_KEY`. The key is optional when the endpoint
  host is local (local servers usually need none). `left_machine` is derived from the
  endpoint host — `false` for local, `true` for remote — so a local OpenAI-schema
  server correctly logs as on-machine.
- **openai model:** requires an explicit model (`weft config set model <name>`);
  `auto` is rejected because there is no GPU tiering for a remote endpoint.
- **params:** only a whitelisted subset is forwarded (`temperature`, `top_p`,
  `max_tokens`); Ollama-specific keys such as `num_ctx` are dropped, not sent.
- **fallback:** a `fallback: list[str]` in `config.toml`, default empty (which
  preserves today's strict fail-fast). On the primary being unavailable, Weft tries
  each fallback in order and uses the first available. A fallback that crosses the
  machine boundary emits a loud stderr notice — no blocking prompt (scripts keep
  working); configuring the list is the deliberate opt-in.
- **fallback defaults:** a fallback provider is resolved with `model="auto"` and the
  default endpoint, using its own sensible default, not the primary's model/endpoint.

## 3. Components

```
src/weft/
  openai_client.py  NEW   — OpenAIClient over POST {endpoint}/chat/completions
  providers.py      TOUCH — register OpenAIProvider; fallback in resolve_llm;
                            standardize available(cfg, env) across providers
  config.py         TOUCH — add fallback: list[str] (default []); parse/render it
  cli.py            TOUCH — `config set fallback a,b`; on_fallback boundary notice
```

## 4. OpenAIClient

```python
class OpenAIClient:
    provider = "openai"

    def __init__(self, endpoint, model, api_key=None, params=None, client=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self._api_key = api_key
        self._params = params or {}
        self._http = client or httpx.Client(timeout=120)
        self.left_machine = not is_local_endpoint(self.endpoint)

    def complete(self, system, prompt) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        for key in ("temperature", "top_p", "max_tokens"):
            if key in self._params:
                body[key] = self._params[key]
        resp = self._http.post(f"{self.endpoint}/chat/completions", headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

`is_local_endpoint(url)` — host in `{localhost, 127.0.0.1, ::1, 0.0.0.0}` or ending
in `.local`. Lives in `openai_client.py` and is imported by `providers.py`.

The user sets `endpoint` to include the API version prefix, e.g.
`https://api.openai.com/v1` or `http://localhost:1234/v1`; the client appends
`/chat/completions`.

## 5. OpenAIProvider & standardized availability

All providers adopt `available(self, cfg, env)` (env accepted everywhere, ignored
where unused), letting `resolve_llm` drop its `isinstance(AnthropicProvider)`
special-case.

```python
class OpenAIProvider:
    name = "openai"

    def available(self, cfg, env):
        if cfg.model == "auto" or not cfg.model:
            return Availability(False,
                "openai needs an explicit model — run `weft config set model <name>`.")
        if not is_local_endpoint(cfg.endpoint) and not env.get("OPENAI_API_KEY"):
            return Availability(False,
                f"OPENAI_API_KEY is not set for remote endpoint {cfg.endpoint} — export "
                "it, use a local endpoint, or switch provider.")
        return Availability(True)

    def build(self, cfg, env):
        return OpenAIClient(cfg.endpoint, cfg.model,
                            api_key=env.get("OPENAI_API_KEY"), params=cfg.params)
```

## 6. Fallback chain

`ResolvedConfig` gains `fallback: list[str] = []`. `load_config` reads a `fallback`
array from TOML; `_to_toml` renders it; `merge` preserves it (CLI/env do not override
the list in M5.1).

```python
def resolve_llm(overrides, *, store_path, env, on_fallback=None) -> LLMClient:
    cfg = merge(load_config(config_path_for(store_path)), env_overrides(env), overrides)
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

- `_fallback_cfg(cfg, name)` = a copy of `cfg` with `provider=name`, `model="auto"`,
  `endpoint=DEFAULT_ENDPOINT`, same `params` — so each fallback uses its own defaults.
- `_is_local_provider(name, cfg)` — `ollama`/`fake` local; `anthropic` remote;
  `openai` by `is_local_endpoint(cfg.endpoint)`. Used only to decide the notice.
- `crossed` is true only when the primary is local and the chosen fallback reports
  `left_machine=True` (a genuine machine-boundary crossing).

## 7. CLI wiring

- `make_llm(overrides, store_path)` passes `on_fallback=_fallback_notice`, which
  prints to stderr. When `crossed` is true it prints the loud line:
  `"falling back to <p> — this sends note content off your machine (configured in fallback)."`
  Otherwise a quiet informational line: `"primary <a> unavailable; using fallback <b>."`
- `weft config set fallback anthropic` — comma-separated for multiple
  (`anthropic,openai`); each entry validated against `VALID_PROVIDERS`; stored as a
  list. `weft config show` prints the list.

## 8. Security & invariants

- Keys stay in the environment (`OPENAI_API_KEY`), never written to `config.toml`.
- Default-off fallback preserves the existing strict fail-fast behavior exactly.
- Any boundary crossing is (a) opt-in via the config list, (b) announced per run on
  stderr, and (c) permanently recorded in `api-log.jsonl` as `left_machine: true`.
- Endpoint URLs validated by reuse; model output treated as untrusted, same as other
  providers.

## 9. Testing (offline, mocked httpx / recorder callbacks)

- `test_openai_client.py` — request shape (`/chat/completions`, messages); auth header
  present with key / absent without; param whitelist drops `num_ctx`; `left_machine`
  true for remote host, false for `localhost`; response parsing; error mapping.
- `test_providers.py` additions — openai available with explicit model + key; remedy
  when `model=auto`; remedy when remote + no key; local endpoint needs no key;
  fallback picks the first available; `on_fallback` fires with correct `crossed`;
  `fallback=[]` still fails fast; standardized `available(cfg, env)` for all providers.
- `test_config.py` addition — `fallback` round-trips through save/load; default `[]`.
- `test_cli_config.py` addition — `config set fallback anthropic` persists a list;
  invalid entry rejected (exit 2).

## 10. Milestone

This is **M5.1**, completing the provider layer. The remaining roadmap item is
**M4** — the `weft chat` REPL on the MemoryStore.
