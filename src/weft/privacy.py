"""Index-time privacy policy for selecting and redacting vault notes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath


DEFAULT_EXCLUDES = ("Private", ".obsidian", ".trash", ".weft")
REDACTION_MARKER = "[REDACTED]"


def _normalize_prefix(value: str) -> str:
    value = value.replace("\\", "/")
    if PurePosixPath(value).is_absolute():
        raise ValueError(f"Privacy paths must be relative vault paths: {value!r}")
    value = value.strip("/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Privacy paths must be relative vault paths: {value!r}")
    return path.as_posix()


def _under_prefix(rel_path: str, prefix: str) -> bool:
    rel = rel_path.casefold()
    base = prefix.casefold()
    return rel == base or rel.startswith(base + "/")


@dataclass(frozen=True)
class PrivacyPolicy:
    """Allow, exclude, and redact content before it reaches an embedding model."""

    includes: tuple[str, ...] = ()
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES
    redaction_patterns: tuple[str, ...] = ()
    _compiled: tuple[re.Pattern[str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "includes", tuple(_normalize_prefix(p) for p in self.includes)
        )
        object.__setattr__(
            self, "excludes", tuple(_normalize_prefix(p) for p in self.excludes)
        )
        object.__setattr__(
            self,
            "_compiled",
            tuple(re.compile(pattern, re.MULTILINE) for pattern in self.redaction_patterns),
        )

    def allows(self, rel_path: str) -> bool:
        if self.includes and not any(_under_prefix(rel_path, p) for p in self.includes):
            return False
        return not any(_under_prefix(rel_path, p) for p in self.excludes)

    def redact(self, text: str) -> str:
        for pattern in self._compiled:
            text = pattern.sub(REDACTION_MARKER, text)
        return text

    def as_dict(self) -> dict:
        return {
            "includes": list(self.includes),
            "excludes": list(self.excludes),
            "redaction_patterns": list(self.redaction_patterns),
        }
