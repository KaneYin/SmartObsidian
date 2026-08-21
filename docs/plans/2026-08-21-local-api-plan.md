# M6 Local HTTP API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose Weft's ask/memory/config/models operations over a secure loopback HTTP server backing a future GUI, sharing one `service.py` core with the CLI.

**Architecture:** `service.py` holds the shared, JSON-returning `service_*` functions plus backend factories; the CLI's exposed handlers delegate to them, and a stdlib `ThreadingHTTPServer` (`api.py`) routes requests to the same functions behind loopback-bind + Host/Origin checks + a `0600` bearer token.

**Tech Stack:** Python ≥3.11 stdlib (`http.server`, `hmac`, `secrets`, `json`), `pytest` with `http.client`, `FakeEmbedder`/`FakeLLM`.

**Scope:** M6 — health, ask, memory read/curate, config/models read. `index`/`suggest`/`models pull`/`config set` stay CLI-only.

**Run tests with:** `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`

---

## File Structure

- Create `src/weft/service.py` — factories + `service_*` functions (single source of truth).
- Create `src/weft/api.py` — server, handler, auth, routing.
- Modify `src/weft/cli.py` — route exposed handlers through `service`; add `weft serve`.
- Tests: `test_service.py`, `test_api.py`; update `tests/test_cli.py`, `tests/test_cli_graph.py` (ask doubles).

---

## Task 1: service.py core (health/config/models) + read-handler refactor

**Files:**
- Create: `src/weft/service.py`
- Modify: `src/weft/cli.py` (`_cmd_config` show, `_cmd_models`)
- Test: `tests/test_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_service.py
from weft.config import ResolvedConfig, config_path_for, save_config
from weft.embeddings import FakeEmbedder
from weft.service import service_config, service_health, service_models
from weft.store import VectorStore


def _index(store_path):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed(["hello"])[0],
              {"rel_path": "n.md", "heading": "H", "text": "hello",
               "ordinal": 0, "tags": [], "wikilinks": []})
    store.save(store_path)


def test_service_health_reports_index(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    assert service_health(store)["index"] is False
    _index(store)
    h = service_health(store)
    assert h["index"] is True and h["chunks"] == 1 and h["provider"] == "fake"


def test_service_config_has_no_secrets(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="anthropic", model="claude-opus-4-8"))
    cfg = service_config(store)
    assert cfg == {"provider": "anthropic", "model": "claude-opus-4-8",
                   "endpoint": "http://localhost:11434", "fallback": []}


def test_service_models_shape(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="ollama"))
    m = service_models(store)
    assert "recommended" in m and isinstance(m["tiers"], list)
    assert {"name", "default", "installed"} <= set(m["tiers"][0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_service.py -v`
Expected: FAIL — `weft.service` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/service.py
"""Shared service core: the operations behind both the CLI and the HTTP API.
Functions take a store path plus arguments and return plain JSON-able dicts. This
is the single source of truth; the API never reimplements CLI logic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from weft.config import config_path_for, load_config
from weft.embeddings import Embedder, SentenceTransformerEmbedder
from weft.graph import LinkGraph
from weft.hardware import detect_gpu
from weft.index import graph_path_for
from weft.llm import AuditedLLM, LLMClient
from weft.memory import SEMANTIC_TYPES, MemoryStore
from weft.models import TIERS, pick_default
from weft.ollama_client import list_models as ollama_installed
from weft.proposals import ProposalStore
from weft.providers import resolve_llm
from weft.store import VectorStore

MAX_K = 50
MAX_MEMORY_TEXT = 2000


def make_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


def _fallback_notice(primary: str, chosen: str, crossed: bool) -> None:
    if crossed:
        print(f"falling back to {chosen} — this sends note content off your machine "
              f"(configured in fallback).", file=sys.stderr)
    else:
        print(f"primary {primary} unavailable; using fallback {chosen}.", file=sys.stderr)


def make_llm(overrides: dict, store_path: Path) -> LLMClient:
    return resolve_llm(overrides, store_path=store_path, env=dict(os.environ),
                       on_fallback=_fallback_notice)


def make_memory(store_path: Path) -> MemoryStore:
    return MemoryStore(store_path.parent / "memory.jsonl",
                       store_path.parent / "episodes.jsonl")


def make_proposals(store_path: Path) -> ProposalStore:
    return ProposalStore(store_path.parent / "memory-proposals.jsonl")


def service_health(store_path: Path) -> dict:
    store_path = Path(store_path)
    exists = store_path.with_suffix(".npz").exists()
    chunks = len(VectorStore.load(store_path)) if exists else 0
    cfg = load_config(config_path_for(store_path))
    return {"index": exists, "chunks": chunks, "provider": cfg.provider}


def service_config(store_path: Path) -> dict:
    cfg = load_config(config_path_for(Path(store_path)))
    return {"provider": cfg.provider, "model": cfg.model,
            "endpoint": cfg.endpoint, "fallback": list(cfg.fallback)}


def service_models(store_path: Path) -> dict:
    cfg = load_config(config_path_for(Path(store_path)))
    gpu = detect_gpu()
    try:
        installed = set(ollama_installed(cfg.endpoint))
    except Exception:
        installed = set()
    tiers = [{"name": t.name, "default": t.default, "installed": t.default in installed}
             for t in TIERS]
    return {"gpu": {"backend": gpu.backend, "budget_mb": gpu.budget_mb},
            "recommended": pick_default(gpu), "tiers": tiers}
```

Refactor `cli.py` to delegate. Add `from weft import service` to the imports.
Replace the `_cmd_config` `show` block:

```python
    if args.action == "show":
        cfg = service.service_config(Path(args.store))
        print(f"provider = {cfg['provider']}\nmodel = {cfg['model']}\n"
              f"endpoint = {cfg['endpoint']}")
        print(f"fallback = {', '.join(cfg['fallback']) or '(none)'}")
        return 0
```

Replace the `_cmd_models` `list` and `show` blocks:

```python
    data = service.service_models(Path(args.store))
    if args.action == "list":
        installed = {t["default"] for t in data["tiers"] if t["installed"]}
        for tier in data["tiers"]:
            flags = []
            if tier["default"] == data["recommended"]:
                flags.append("recommended")
            if tier["default"] in installed:
                flags.append("installed")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(f"{tier['name']:6} {tier['default']}{suffix}")
        return 0
    if args.action == "show":
        print(f"gpu: {data['gpu']['backend']} budget={data['gpu']['budget_mb']}MB")
        print(f"recommended: {data['recommended']}")
        cfg = load_config(config_path_for(Path(args.store)))
        print(f"configured model: {cfg.model}")
        return 0
```

(Leave the `pull` branch of `_cmd_models` unchanged — it stays CLI-only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_service.py tests/test_cli_config.py tests/test_cli_models.py -v`
Expected: PASS (new service tests + existing config/models CLI tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/weft/service.py src/weft/cli.py tests/test_service.py
git commit -m "feat(weft): M6 service core (health/config/models) + CLI delegation"
```

---

## Task 2: service memory functions + memory-handler refactor

**Files:**
- Modify: `src/weft/service.py`, `src/weft/cli.py`
- Test: `tests/test_service.py`

- [ ] **Step 1: Write the failing test (append to tests/test_service.py)**

```python
def test_service_memory_flow(tmp_path):
    from weft.service import (service_memory_accept, service_memory_list,
                              service_memory_pending, service_memory_reject,
                              service_remember, make_proposals)
    from weft.proposals import Candidate
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True, exist_ok=True)

    item = service_remember(store, "preference", "Answer concisely")
    assert item["id"].startswith("mem_")
    assert [i["text"] for i in service_memory_list(store)["items"]] == ["Answer concisely"]

    prop = make_proposals(store).add([Candidate("fact", "Recurring interest: x", "heuristic")])[0]
    assert [p["id"] for p in service_memory_pending(store)["proposals"]] == [prop.id]
    assert service_memory_accept(store, prop.id) == {"accepted": prop.id}
    assert any(i["text"] == "Recurring interest: x" for i in service_memory_list(store)["items"])


def test_service_remember_validates(tmp_path):
    import pytest
    from weft.service import service_remember
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        service_remember(store, "bogus", "x")
    with pytest.raises(ValueError):
        service_remember(store, "fact", "x" * 3000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_service.py -k memory -v`
Expected: FAIL — service memory functions missing.

- [ ] **Step 3: Write minimal implementation**

Append to `src/weft/service.py`:

```python
def service_memory_list(store_path: Path) -> dict:
    ms = make_memory(Path(store_path))
    return {"items": [{"id": i.id, "type": i.type, "text": i.text}
                      for i in ms.active_semantic()]}


def service_memory_pending(store_path: Path) -> dict:
    ps = make_proposals(Path(store_path))
    return {"proposals": [{"id": p.id, "type": p.type, "text": p.text}
                          for p in ps.pending()]}


def service_remember(store_path: Path, type: str, text: str) -> dict:
    if type not in SEMANTIC_TYPES:
        raise ValueError(f"type must be one of: {', '.join(sorted(SEMANTIC_TYPES))}")
    if not text or len(text) > MAX_MEMORY_TEXT:
        raise ValueError(f"text must be 1-{MAX_MEMORY_TEXT} characters")
    item = make_memory(Path(store_path)).remember(type, text)
    return {"id": item.id, "type": item.type, "text": item.text}


def service_memory_accept(store_path: Path, prop_id: str) -> dict:
    sp = Path(store_path)
    proposals, memory = make_proposals(sp), make_memory(sp)
    prop = proposals.get(prop_id)   # raises KeyError if absent
    memory.remember(prop.type, prop.text, provenance="inferred", source=prop.source)
    proposals.mark(prop.id, "accepted")
    return {"accepted": prop.id}


def service_memory_reject(store_path: Path, prop_id: str) -> dict:
    make_proposals(Path(store_path)).mark(prop_id, "rejected")  # raises KeyError if absent
    return {"rejected": prop_id}
```

Refactor `cli.py` `_cmd_memory` and `_cmd_remember` to delegate. Replace the
`_cmd_remember` body after validation-free path:

```python
def _cmd_remember(args: argparse.Namespace) -> int:
    try:
        item = service.service_remember(Path(args.store), args.type, args.text)
    except ValueError as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 2
    print(f"remembered [{item['type']}] {item['id']}: {terminal_safe(item['text'])}")
    return 0
```

In `_cmd_memory`, replace the `list`, `pending`, `accept`, `reject` branches:

```python
    if args.action == "list":
        for item in service.service_memory_list(store_path)["items"]:
            print(f"{item['id']}  [{item['type']}]  {terminal_safe(item['text'])}")
        return 0
    ...
    if args.action == "pending":
        for prop in service.service_memory_pending(store_path)["proposals"]:
            print(f"{prop['id']}  [{prop['type']}]  {terminal_safe(prop['text'])}")
        return 0
    if args.action == "accept":
        try:
            service.service_memory_accept(store_path, args.id)
        except KeyError:
            print(f"no proposal with id {terminal_safe(str(args.id))}", file=sys.stderr)
            return 1
        print(f"accepted {args.id} -> memory")
        return 0
    if args.action == "reject":
        try:
            service.service_memory_reject(store_path, args.id)
        except KeyError:
            print(f"no proposal with id {terminal_safe(str(args.id))}", file=sys.stderr)
            return 1
        print(f"rejected {args.id}")
        return 0
```

(Leave `show`, `forget`, `compact`, `suggest`, `mirror` branches as they are.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_service.py tests/test_cli_memory.py tests/test_cli_memory_infer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/weft/service.py src/weft/cli.py tests/test_service.py
git commit -m "feat(weft): M6 service memory functions + CLI delegation"
```

---

## Task 3: service_ask + `_cmd_ask` refactor

**Files:**
- Modify: `src/weft/service.py`, `src/weft/cli.py`
- Test: `tests/test_service.py`, `tests/test_cli.py`, `tests/test_cli_graph.py`

- [ ] **Step 1: Write the failing test (append to tests/test_service.py)**

```python
def test_service_ask_returns_answer_and_logs_episode(tmp_path, monkeypatch):
    import weft.service as S
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    from weft.config import ResolvedConfig, config_path_for, save_config
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    _index(store)
    monkeypatch.setattr(S, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(S, "make_llm", lambda *a, **k: FakeLLM(response="answer [1]"))
    data = S.service_ask(store, "hello?", k=3)
    assert data["answer"] == "answer [1]"
    assert data["sources"] == ["n.md"]
    assert len(S.make_memory(store).episodes()) == 1


def test_service_ask_validates_k(tmp_path):
    import pytest
    from weft.service import service_ask
    store = tmp_path / ".weft" / "index"
    _index(store)
    with pytest.raises(ValueError):
        service_ask(store, "q", k=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_service.py -k ask -v`
Expected: FAIL — `service_ask` missing.

- [ ] **Step 3: Write minimal implementation**

Append to `src/weft/service.py`:

```python
def service_ask(store_path: Path, question: str, k: int = 5, *,
                use_graph: bool = True, use_memory: bool = True,
                overrides: dict | None = None) -> dict:
    from weft.agent import ask  # local import keeps langgraph off the read path
    if not isinstance(k, int) or not (1 <= k <= MAX_K):
        raise ValueError(f"k must be between 1 and {MAX_K}")
    if not question or not str(question).strip():
        raise ValueError("question must not be empty")
    sp = Path(store_path)
    if not sp.with_suffix(".npz").exists():
        raise ValueError("no index; run `weft index <vault>` first")
    store = VectorStore.load(sp)
    graph = None
    if use_graph:
        gpath = graph_path_for(sp)
        graph = LinkGraph.load(gpath) if gpath.exists() else None
    raw = make_llm(overrides or {}, sp)         # may raise ProviderUnavailable
    llm = AuditedLLM(raw, sp.parent / "api-log.jsonl", "ask")
    memory = make_memory(sp) if use_memory else None
    result = ask(question, make_embedder(), store, llm, k=k, graph=graph, memory=memory)
    return {"answer": result.answer, "sources": list(result.sources)}
```

Refactor `_cmd_ask` in `cli.py`. Keep its no-index guard and error messages; replace
the resolve/ask core:

```python
def _cmd_ask(args: argparse.Namespace) -> int:
    store_path = Path(args.store)
    if not store_path.with_suffix(".npz").exists():
        print(
            f"No index at {terminal_safe(args.store)}. Run `weft index <vault>` first.",
            file=sys.stderr,
        )
        return 1
    try:
        data = service.service_ask(
            store_path, args.question, k=args.k,
            use_graph=not args.no_graph, use_memory=not args.no_memory,
            overrides=_llm_overrides(args),
        )
    except ProviderUnavailable as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(terminal_safe(exc), file=sys.stderr)
        return 1
    print(terminal_safe(data["answer"]))
    if data["sources"]:
        print("\nSources:")
        for s in data["sources"]:
            print(f"  - {terminal_safe(s)}")
    return 0
```

Update the two `ask` test doubles to also patch the service backends. In
`tests/test_cli.py` `test_cli_index_then_ask`, add after the existing two setattrs:

```python
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm",
                        lambda *a, **k: FakeLLM(response="Coffee is brewed [1]."))
```

In `tests/test_cli.py` `test_cli_ask_missing_index_errors`, no change is needed (the
no-index guard returns before calling the service).

In `tests/test_cli_graph.py`, both tests patch `cli.make_llm`; add alongside each the
service patches (matching each test's expected response text):

```python
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="ans [1]"))
```
(use `"pure vector answer"` in the `--no-graph` test to match its assertion; import
`FakeEmbedder` in `test_cli_graph.py` if not already imported.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_service.py tests/test_cli.py tests/test_cli_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/weft/service.py src/weft/cli.py tests/test_service.py tests/test_cli.py tests/test_cli_graph.py
git commit -m "feat(weft): M6 service_ask + CLI ask delegation"
```

---

## Task 4: API server security core + `/health` (`api.py`)

**Files:**
- Create: `src/weft/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import http.client
import json
import os
import stat
import threading

import pytest

from weft.api import WeftHTTPServer, load_or_create_token


@pytest.fixture
def server(tmp_path):
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True, exist_ok=True)
    from weft.config import ResolvedConfig, config_path_for, save_config
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    token = load_or_create_token(store)
    srv = WeftHTTPServer(("127.0.0.1", 0), store, token)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, token, srv.server_address[1], store
    srv.shutdown()


def _conn(port):
    return http.client.HTTPConnection("127.0.0.1", port, timeout=5)


def _get(port, path, headers):
    c = _conn(port); c.request("GET", path, headers=headers)
    r = c.getresponse(); body = r.read(); c.close()
    return r.status, body


def test_token_file_is_private(server):
    _srv, _token, _port, store = server
    mode = stat.S_IMODE(os.stat(store.parent / "api-token").st_mode)
    assert mode == 0o600


def test_missing_token_is_401(server):
    _srv, _token, port, _store = server
    status, _ = _get(port, "/health", {"Host": "127.0.0.1"})
    assert status == 401


def test_bad_host_is_403(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/health",
                     {"Host": "evil.com", "Authorization": f"Bearer {token}"})
    assert status == 403


def test_cross_site_origin_is_403(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/health",
                     {"Host": "127.0.0.1", "Origin": "https://evil.com",
                      "Authorization": f"Bearer {token}"})
    assert status == 403


def test_health_ok_with_token(server):
    _srv, token, port, _store = server
    status, body = _get(port, "/health",
                        {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"})
    assert status == 200
    assert json.loads(body)["index"] is False


def test_unknown_path_404(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/nope",
                     {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"})
    assert status == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --extra dev pytest tests/test_api.py -v`
Expected: FAIL — `weft.api` missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/weft/api.py
"""A small, loopback-only HTTP API over the shared service core. Every request is
gated by a Host-header check, an Origin check (for browser CORS), and a 0600 bearer
token. Bind address is always 127.0.0.1. No dependency beyond the stdlib."""

from __future__ import annotations

import hmac
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from weft import service
from weft.providers import ProviderUnavailable
from weft.security import UnsafeWriteError, secure_write_text

MAX_BODY = 1_048_576
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def load_or_create_token(store_path: Path) -> str:
    path = Path(store_path).parent / "api-token"
    if path.exists():
        if path.is_symlink():
            raise UnsafeWriteError(f"Refusing to read token symlink: {path}")
        return path.read_text(encoding="utf-8").strip()
    token = secrets.token_urlsafe(32)
    secure_write_text(path, token + "\n", overwrite=True)
    return token


def _host_ok(host_header: str | None) -> bool:
    host = (host_header or "").rsplit(":", 1)[0].strip("[]").lower()
    return host in _LOOPBACK_HOSTS


def _origin_ok(origin: str | None) -> bool:
    if not origin:
        return True
    return (urlparse(origin).hostname or "").lower() in _LOOPBACK_HOSTS


# route table: (method, path) -> callable(store_path, body) -> dict
def _r_health(sp, body):
    return service.service_health(sp)


def _r_memory(sp, body):
    return service.service_memory_list(sp)


def _r_pending(sp, body):
    return service.service_memory_pending(sp)


def _r_config(sp, body):
    return service.service_config(sp)


def _r_models(sp, body):
    return service.service_models(sp)


def _r_ask(sp, body):
    return service.service_ask(sp, body.get("question", ""), k=body.get("k", 5))


def _r_remember(sp, body):
    return service.service_remember(sp, body.get("type", ""), body.get("text", ""))


def _r_accept(sp, body):
    return service.service_memory_accept(sp, body.get("id", ""))


def _r_reject(sp, body):
    return service.service_memory_reject(sp, body.get("id", ""))


ROUTES = {
    ("GET", "/health"): _r_health,
    ("GET", "/memory"): _r_memory,
    ("GET", "/memory/pending"): _r_pending,
    ("GET", "/config"): _r_config,
    ("GET", "/models"): _r_models,
    ("POST", "/ask"): _r_ask,
    ("POST", "/memory/remember"): _r_remember,
    ("POST", "/memory/accept"): _r_accept,
    ("POST", "/memory/reject"): _r_reject,
}
_PATHS = {path for _m, path in ROUTES}


class WeftHTTPServer(ThreadingHTTPServer):
    def __init__(self, address, store_path, token):
        if address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise ValueError("Weft API binds loopback only")
        super().__init__(address, WeftHandler)
        self.store_path = Path(store_path)
        self.token = token


class WeftHandler(BaseHTTPRequestHandler):
    server_version = "Weft/1"

    def log_message(self, *args):  # silence default stderr logging
        pass

    def _cors(self):
        origin = self.headers.get("Origin")
        if origin and _origin_ok(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _guard(self) -> bool:
        if not _host_ok(self.headers.get("Host")):
            self._send(403, {"error": "bad host"})
            return False
        if not _origin_ok(self.headers.get("Origin")):
            self._send(403, {"error": "bad origin"})
            return False
        want = f"Bearer {self.server.token}"
        if not hmac.compare_digest(self.headers.get("Authorization", ""), want):
            self._send(401, {"error": "unauthorized"})
            return False
        return True

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _dispatch(self, method: str, body: dict):
        route = ROUTES.get((method, self.path))
        if route is None:
            self._send(405 if self.path in _PATHS else 404, {"error": "no such route"})
            return
        try:
            self._send(200, route(self.server.store_path, body))
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except KeyError as exc:
            self._send(404, {"error": f"not found: {exc}"})
        except ProviderUnavailable as exc:
            self._send(503, {"error": str(exc)})
        except Exception:
            self._send(500, {"error": "internal error"})

    def do_GET(self):
        if not self._guard():
            return
        self._dispatch("GET", {})

    def do_POST(self):
        if not self._guard():
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "bad content-length"})
            return
        if length > MAX_BODY:
            self._send(400, {"error": "request body too large"})
            return
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        self._dispatch("POST", body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_api.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/weft/api.py tests/test_api.py
git commit -m "feat(weft): M6 loopback API server with token/Host/Origin auth"
```

---

## Task 5: API endpoints — ask & memory round-trips (`test_api.py`)

**Files:**
- Test: `tests/test_api.py` (the routes already exist from Task 4; this verifies them)

- [ ] **Step 1: Write the failing test (append)**

```python
def _post(port, path, headers, obj):
    c = _conn(port)
    payload = json.dumps(obj)
    c.request("POST", path, body=payload,
              headers={**headers, "Content-Type": "application/json"})
    r = c.getresponse(); body = r.read(); c.close()
    return r.status, body


def test_remember_then_list(server):
    _srv, token, port, _store = server
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"}
    status, body = _post(port, "/memory/remember", h, {"type": "fact", "text": "likes tea"})
    assert status == 200 and json.loads(body)["id"].startswith("mem_")
    status, body = _get(port, "/memory", h)
    assert status == 200
    assert any(i["text"] == "likes tea" for i in json.loads(body)["items"])


def test_remember_bad_type_is_400(server):
    _srv, token, port, _store = server
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"}
    status, _ = _post(port, "/memory/remember", h, {"type": "bogus", "text": "x"})
    assert status == 400


def test_oversized_body_is_400(server):
    _srv, token, port, _store = server
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}",
         "Content-Type": "application/json", "Content-Length": str(2_000_000)}
    c = _conn(port)
    c.request("POST", "/ask", body=b"{}", headers=h)
    r = c.getresponse(); r.read(); c.close()
    assert r.status == 400


def test_wrong_method_is_405(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/ask", {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"})
    assert status == 405


def test_ask_round_trip(server, monkeypatch):
    import weft.service as S
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    from weft.store import VectorStore
    _srv, token, port, store = server
    emb = FakeEmbedder(dim=16)
    vs = VectorStore(dim=emb.dim)
    vs.add(emb.embed(["hello world"])[0],
           {"rel_path": "n.md", "heading": "H", "text": "hello world",
            "ordinal": 0, "tags": [], "wikilinks": []})
    vs.save(store)
    monkeypatch.setattr(S, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(S, "make_llm", lambda *a, **k: FakeLLM(response="hi [1]"))
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"}
    status, body = _post(port, "/ask", h, {"question": "hello?", "k": 3})
    assert status == 200
    data = json.loads(body)
    assert data["answer"] == "hi [1]" and data["sources"] == ["n.md"]
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `uv run --extra dev pytest tests/test_api.py -v`
Expected: PASS (routes exist; these assert behavior). If `test_wrong_method_is_405`
fails because `/ask` isn't in `_PATHS`, confirm Task 4's `_PATHS` includes every
route path (it does).

- [ ] **Step 3: (implementation already present — no code change)**

- [ ] **Step 4: Re-run**

Run: `uv run --extra dev pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py
git commit -m "test(weft): M6 API ask + memory round-trip coverage"
```

---

## Task 6: `weft serve` command + docs + full suite

**Files:**
- Modify: `src/weft/cli.py`, `docs/how-to/configure-env-and-use-cli.md`

- [ ] **Step 1: Write the failing test (append to tests/test_api.py)**

```python
def test_serve_rejects_non_loopback_host():
    from weft.api import WeftHTTPServer
    with pytest.raises(ValueError):
        WeftHTTPServer(("0.0.0.0", 0), "/tmp/x/.weft/index", "tok")
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `uv run --extra dev pytest tests/test_api.py -k loopback -v`
Expected: PASS (guard already in `WeftHTTPServer.__init__`).

- [ ] **Step 3: Add the `weft serve` command**

In `cli.py`, add the handler:

```python
def _cmd_serve(args: argparse.Namespace) -> int:
    from weft.api import WeftHTTPServer, load_or_create_token
    store_path = Path(args.store)
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        print("--host must be a loopback address (127.0.0.1, ::1, localhost)",
              file=sys.stderr)
        return 2
    token = load_or_create_token(store_path)
    server = WeftHTTPServer((args.host, args.port), store_path, token)
    host, port = server.server_address[0], server.server_address[1]
    print(f"Weft API on http://{host}:{port}  (token in "
          f"{terminal_safe(store_path.parent / 'api-token')})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", file=sys.stderr)
    finally:
        server.server_close()
    return 0
```

Register the subparser in `build_parser` (before `return parser`):

```python
    p_serve = sub.add_parser("serve", help="Run the local HTTP API for a GUI.")
    p_serve.add_argument("--host", default="127.0.0.1", help="Loopback host (default 127.0.0.1).")
    p_serve.add_argument("--port", type=_bounded_int("port", 1, 65535), default=8765,
                         help="Port (default 8765).")
    p_serve.add_argument("--store", default=DEFAULT_STORE)
    p_serve.set_defaults(func=_cmd_serve)
```

- [ ] **Step 4: Run the whole suite**

Run: `UV_CACHE_DIR=/tmp/weft-uv-cache uv run --extra dev pytest`
Expected: PASS across all modules.

- [ ] **Step 5: Document + commit**

Add to `docs/how-to/configure-env-and-use-cli.md` a short section:

```markdown
## Serve the local API (for a GUI)

Weft can expose its read + memory operations over a loopback HTTP API that a local
GUI can call:

    weft serve                      # http://127.0.0.1:8765

It binds `127.0.0.1` only, validates the `Host`/`Origin` headers, and requires a
bearer token stored at `.weft/api-token` (mode `0600`) — send it as
`Authorization: Bearer <token>`. Endpoints: `GET /health`, `POST /ask`,
`GET /memory`, `GET /memory/pending`, `POST /memory/remember|accept|reject`,
`GET /config`, `GET /models`. Indexing and configuration stay in the CLI.
```

```bash
git add src/weft/cli.py tests/test_api.py docs/how-to/configure-env-and-use-cli.md
git commit -m "feat(weft): M6 weft serve command + API docs"
```

---

## Self-Review Notes (author)

- **Spec coverage:** shared `service.py` (Tasks 1-3) · endpoints health/ask/memory/config/models (Tasks 4-5) · loopback bind + Host + Origin/CORS + token + body limit (Task 4) · `weft serve` loopback-only (Task 6) · audit unchanged via `service_ask`'s `AuditedLLM` (Task 3) · testing service + api (all tasks). Deferred (index/suggest/pull/config-set) untouched.
- **Type consistency:** `service_*` all take `store_path` first and return dicts with the documented keys; `service_ask(..., k, *, use_graph, use_memory, overrides)`; `WeftHTTPServer(address, store_path, token)`; `load_or_create_token(store_path)`; `ROUTES`/`_PATHS`.
- **Regression control:** only the two `ask` CLI tests change (added `weft.service.make_*` patches); config/models/memory CLI tests are unaffected because those handlers delegate to pure (no-backend) service functions. `_cmd_suggest` and `_cmd_index` keep the CLI's own factories, so `test_cli_suggest`/`test_cli` index paths are unchanged.
```
