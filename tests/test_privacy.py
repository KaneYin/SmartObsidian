import re

import pytest

from weft.privacy import PrivacyPolicy


def test_default_policy_excludes_private_and_internal_folders():
    policy = PrivacyPolicy()

    assert policy.allows("Projects/plan.md")
    assert not policy.allows("Private/health.md")
    assert not policy.allows("private/lowercase.md")
    assert not policy.allows(".obsidian/plugins/readme.md")
    assert not policy.allows(".weft/debug.md")


def test_include_paths_form_an_allowlist():
    policy = PrivacyPolicy(includes=("Projects",), excludes=())

    assert policy.allows("Projects/plan.md")
    assert not policy.allows("Journal/today.md")


def test_redaction_runs_before_indexing():
    policy = PrivacyPolicy(redaction_patterns=(r"sk-ant-[A-Za-z0-9-]+", r"(?i)secret:.*$"))

    redacted = policy.redact("token sk-ant-example\nSecret: keep this private")

    assert "sk-ant-example" not in redacted
    assert "keep this private" not in redacted
    assert redacted.count("[REDACTED]") == 2


def test_privacy_paths_must_be_relative_and_cannot_traverse():
    with pytest.raises(ValueError, match="relative vault paths"):
        PrivacyPolicy(excludes=("../outside",))
    with pytest.raises(ValueError, match="relative vault paths"):
        PrivacyPolicy(includes=("/absolute",))


def test_invalid_redaction_regex_is_rejected():
    with pytest.raises(re.error):
        PrivacyPolicy(redaction_patterns=("[",))
