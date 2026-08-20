import json

import pytest

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
    try:
        client = ClaudeClient()
    except Exception:
        pytest.skip("anthropic SDK/key unavailable in this environment")
    assert client.provider == "anthropic"
    assert client.left_machine is True
    assert client.model == "claude-opus-4-8"
