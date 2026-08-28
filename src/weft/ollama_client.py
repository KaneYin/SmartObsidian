"""LLMClient over the local Ollama HTTP API, plus introspection/provisioning
helpers. Talks only to a local endpoint; data never leaves the machine."""

from __future__ import annotations

import json
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
        options = dict(self._params)
        think = options.pop("think", None)
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": options,
        }
        # Ollama treats thinking as a top-level chat control, not a model option.
        if "think" in self._params:
            body["think"] = think
        resp = self._http.post(
            f"{self.endpoint}/api/chat",
            json=body,
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
                on_line(json.loads(line))
