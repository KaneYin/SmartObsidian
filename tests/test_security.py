import json
import os

import pytest

from weft.llm import AuditedLLM, FakeLLM
from weft.parser import parse_vault
from weft.security import terminal_safe


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_parse_vault_refuses_markdown_symlink_that_points_outside(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("sensitive content")
    (vault / "leak.md").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        parse_vault(vault)


def test_audited_llm_logs_exact_payload_with_private_permissions(tmp_path):
    log_path = tmp_path / ".weft" / "api-log.jsonl"
    llm = AuditedLLM(FakeLLM(response="answer"), log_path, "ask")

    assert llm.complete(system="system text", prompt="private prompt") == "answer"

    record = json.loads(log_path.read_text().strip())
    assert record["purpose"] == "ask"
    assert record["system"] == "system text"
    assert record["prompt"] == "private prompt"
    assert (log_path.stat().st_mode & 0o777) == 0o600
    assert (log_path.parent.stat().st_mode & 0o777) == 0o700


def test_audit_log_refuses_symlink(tmp_path):
    target = tmp_path / "outside.jsonl"
    target.write_text("keep")
    log_path = tmp_path / "api-log.jsonl"
    log_path.symlink_to(target)
    llm = AuditedLLM(FakeLLM(response="answer"), log_path, "ask")

    with pytest.raises(ValueError, match="symlink"):
        llm.complete(system="system", prompt="prompt")
    assert target.read_text() == "keep"


def test_terminal_safe_removes_escape_controls_but_preserves_lines():
    assert terminal_safe("safe\n\x1b[31mred") == "safe\n[31mred"
