"""LLM backends plus the exact-payload audit wrapper used by the CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from weft.security import secure_append_json

MODEL = "claude-opus-4-8"


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str:
        """Return the model's text answer to `prompt` under `system`."""
        ...


class FakeLLM:
    """Records the last call; returns a canned response, or echoes the prompt
    if none was given (handy for asserting the prompt was built correctly)."""

    def __init__(self, response: str | None = None):
        self._response = response
        self.last_system: str | None = None
        self.last_prompt: str | None = None

    def complete(self, system: str, prompt: str) -> str:
        self.last_system = system
        self.last_prompt = prompt
        if self._response is not None:
            return self._response
        return f"[echo] {prompt}"


class AuditedLLM:
    """Log the exact outbound payload before delegating to an LLM backend."""

    def __init__(self, delegate: LLMClient, log_path: Path, purpose: str):
        self._delegate = delegate
        self._log_path = Path(log_path)
        self._purpose = purpose

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

    def complete(self, system: str, prompt: str) -> str:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
