# Weft M6 — Local HTTP API Layer — Design

**Status:** Approved design, pre-implementation.
**Date:** 2026-08-21.
**Scope:** A small, secure, local HTTP API over existing Weft functionality, to back
a future GUI for non-technical users. Read + core-interaction endpoints only.

## 1. Problem & goal

Weft is CLI-only today. A future GUI needs a stable local API so non-technical users
can ask their notes, review what Weft remembers, and see suggestions without opening
a terminal. This milestone extracts a shared service layer and exposes it over a
loopback HTTP server. The GUI front-end itself is out of scope.

## 2. Decisions (locked during brainstorming)

- **Framework:** stdlib `http.server` (`ThreadingHTTPServer`), zero new dependencies.
  FastAPI is the documented escalation path if the GUI later needs OpenAPI schema or
  streaming.
- **Security baseline (non-negotiable):** bind loopback only; validate the `Host`
  header; require a `0600` bearer token; validate `Origin` for browser requests.
- **Scope:** read + core-interaction endpoints (health, ask, memory read/curate,
  config/models read). Long-running/reconfiguring operations (`index`, `suggest`,
  `memory suggest`, `models pull`, `config set`) stay CLI-only for now.
- **Architecture:** a shared `service.py` core that both the CLI and the API call, so
  the API re-exposes existing behavior rather than duplicating it.

## 3. Components

```
src/weft/
  service.py   NEW  — backend factories (make_embedder / make_llm / make_memory /
                      make_proposals, moved from cli) + service_* functions returning
                      plain JSON-able dicts. Single source of truth.
  api.py       NEW  — ThreadingHTTPServer + WeftHandler: routing, auth, Host/Origin
                      checks, JSON I/O, body-size limits. Calls service_*.
  cli.py       TOUCH — import factories from service; refactor the exposed handlers
                      (ask, memory list/pending/remember/accept/reject, config show,
                      models) to call service_* and format for the terminal; add
                      `weft serve`.
```

Moving the `make_*` factories into `service.py` avoids a CLI↔API circular import and
gives tests one monkeypatch point.

### service.py functions

Each takes `store_path: Path` plus arguments and returns a dict:

- `service_health(store_path)` → `{"index": bool, "chunks": int, "provider": str}`
- `service_ask(store_path, question, k)` → `{"answer": str, "sources": [str]}`
- `service_memory_list(store_path)` → `{"items": [{"id","type","text"}]}`
- `service_memory_pending(store_path)` → `{"proposals": [{"id","type","text"}]}`
- `service_remember(store_path, type, text)` → `{"id","type","text"}`
- `service_memory_accept(store_path, id)` → `{"accepted": id}`
- `service_memory_reject(store_path, id)` → `{"rejected": id}`
- `service_config(store_path)` → `{"provider","model","endpoint","fallback"}`
- `service_models(store_path)` → `{"gpu": {...}, "recommended": str, "tiers": [...]}`

`service_ask` runs the identical retrieve→reason→record path as `weft ask` (loads the
store + graph + memory, resolves the provider, wraps in `AuditedLLM`, calls
`agent.ask`), so an episode is logged and the LLM call is audited exactly as today.
Validation lives in the service (k in 1–50, type in the memory types, text ≤ 2000)
so both callers enforce it; on bad input it raises `ValueError`, which the API maps to
`400` and the CLI maps to its existing exit code.

## 4. Endpoints

All JSON. Errors are `{"error": "<message>"}` with the status below.

| Method + path | Body | Success response |
|---|---|---|
| `GET /health` | — | `{"index": true, "chunks": 128, "provider": "ollama"}` |
| `POST /ask` | `{"question": "...", "k": 5}` | `{"answer": "...", "sources": ["a.md"]}` |
| `GET /memory` | — | `{"items": [{"id","type","text"}]}` |
| `GET /memory/pending` | — | `{"proposals": [{"id","type","text"}]}` |
| `POST /memory/remember` | `{"type":"fact","text":"..."}` | `{"id":"mem_…","type","text"}` |
| `POST /memory/accept` | `{"id":"prop_…"}` | `{"accepted":"prop_…"}` |
| `POST /memory/reject` | `{"id":"prop_…"}` | `{"rejected":"prop_…"}` |
| `GET /config` | — | `{"provider","model","endpoint","fallback"}` |
| `GET /models` | — | `{"gpu":{...},"recommended":"…","tiers":[...]}` |

Status codes: `400` malformed/oversized/invalid body; `401` missing/bad token; `403`
bad `Host` or cross-site `Origin`; `404` unknown path; `405` wrong method; `500`
unexpected (message generic, details not leaked). `k` defaults to 5 when omitted.

## 5. Security model

Applied to **every** request before any work, in `WeftHandler`:

1. **Loopback bind.** `ThreadingHTTPServer(("127.0.0.1", port), WeftHandler)`.
   `weft serve --host` is validated to be a loopback address; non-loopback is refused
   so the server can't be accidentally exposed.
2. **Host-header check.** Host (minus port) in `{localhost, 127.0.0.1, ::1}` → else
   `403`. Blocks DNS-rebinding.
3. **Origin check + minimal CORS.** If `Origin` is present it must be a loopback
   origin → else `403`. For an allowed origin, echo it in
   `Access-Control-Allow-Origin`, allow the `Authorization` header and the used
   method, and answer `OPTIONS` preflight. Cross-site pages are refused.
4. **Bearer token.** On first `serve`, generate `secrets.token_urlsafe(32)` and write
   `.weft/api-token` at `0600` via `secure_write_text`. Every request must send
   `Authorization: Bearer <token>`, compared with `hmac.compare_digest`
   (constant-time) → else `401`.
5. **Body limits.** Reject `Content-Length` > 1 MB and missing/invalid JSON with
   `400`; never read an unbounded body.
6. **No secrets on the wire.** `GET /config` returns provider/model/endpoint/fallback
   only, never API keys. Responses set `X-Content-Type-Options: nosniff`.

Privacy posture is unchanged: `service_ask` uses the same `AuditedLLM`, so every call
is logged with `provider`/`model`/`left_machine`.

## 6. CLI: `weft serve`

```
weft serve [--host 127.0.0.1] [--port 8765] [--store .weft/index]
```

Prints the base URL and the token file path (not the token value), then serves until
interrupted. `--host` is validated loopback-only. The command wires `WeftHTTPServer`
with the resolved `store_path` and the loaded-or-created token.

## 7. Testing (offline, no network)

- `test_service.py` — each `service_*` against a temp `.weft/` with a `provider=fake`
  config and a `FakeEmbedder`-built store: `service_ask` returns `{answer, sources}`
  and logs an episode; memory list/pending/remember/accept/reject; health reflects
  index presence; validation errors raise `ValueError`.
- `test_api.py` — start a real `ThreadingHTTPServer` on `127.0.0.1:0` in a thread and
  drive it with `http.client`: `401` without token; `403` on a non-loopback `Host`;
  `403` on a cross-site `Origin`; `200 /health` with token; `POST /ask` round-trip; a
  memory accept/reject flow; `400` on oversized/malformed body; `404`/`405` for
  unknown route/method; `.weft/api-token` created at `0600`.

## 8. Milestone & deferred

This is **M6**. Deferred to a later increment (explicitly out of scope): `index`,
`suggest`, `memory suggest`, `models pull`, `config set` — long-running or
reconfiguring operations that want streaming/async — plus the GUI front-end itself.
The remaining memory-roadmap item, **M4** (`weft chat` REPL), is independent.
