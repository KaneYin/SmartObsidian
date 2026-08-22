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
