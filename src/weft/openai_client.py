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
