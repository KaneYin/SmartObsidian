"""The display-only inbox: render inferred-link suggestions into `_inbox.md`.

This is the dogfooded "never silent writes" surface. Existing inboxes are kept
unless replacement is explicit, and symlinks are always refused. The checkboxes
are cosmetic in M2; accept/reject feedback is M3. Pure rendering takes
`LinkSuggestion`s and returns/writes Markdown."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from weft.security import (
    UnsafeWriteError,
    secure_write_text,
    terminal_safe,
    vault_root,
)
from weft.suggest import LinkSuggestion

INBOX_NAME = "_inbox.md"
_TITLE = "# Weft Inbox — inferred links"
_MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_[\]<>])")


def _markdown_text(value: str) -> str:
    """Render dynamic text as one inert Markdown line."""
    value = " ".join(terminal_safe(value).splitlines())
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", value)


def _code_span(value: str) -> str:
    """Choose a delimiter longer than any backtick run in dynamic content."""
    value = " ".join(terminal_safe(value).splitlines())
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * (longest + 1)
    return f"{fence}{value}{fence}"


def render_inbox(suggestions: list[LinkSuggestion], generated_at: datetime) -> str:
    """Deterministic markdown for the suggestion list. Suggestions render in the
    order given (the caller ranks them); one block per suggestion."""
    ts = generated_at.strftime("%Y-%m-%d %H:%M")
    n = len(suggestions)
    plural = "suggestion" if n == 1 else "suggestions"
    lines = [
        _TITLE,
        f"_Generated {ts} · {n} {plural} · review-only, nothing was changed_",
        "",
    ]

    if not suggestions:
        lines.append(
            "No new inferred links to review. Your notes are well-connected — "
            "or run `weft index` to refresh, then `weft suggest` again."
        )
        return "\n".join(lines) + "\n"

    for s in suggestions:
        target = Path(s.note_b).with_suffix("").as_posix()
        lines.append(
            f"- [ ] **{_markdown_text(s.note_a)} ↔ {_markdown_text(s.note_b)}**  "
            f"·  score {s.score:.2f}  ·  {_code_span(f'id {s.id}')}"
        )
        lines.append(f"      {_markdown_text(s.rationale)}")
        lines.append(
            f"      Suggested: add {_code_span(f'[[{target}]]')} to "
            f"{_markdown_text(s.note_a)} (and/or the reverse)"
        )
    return "\n".join(lines) + "\n"


def validate_inbox_target(vault_path: Path, *, overwrite: bool = False) -> Path:
    """Validate an inbox destination before optional network work begins."""
    root = vault_root(vault_path)
    path = root / INBOX_NAME
    if path.is_symlink():
        raise UnsafeWriteError(f"Refusing to replace symlink: {path}")
    if path.exists() and not path.is_file():
        raise UnsafeWriteError(f"Refusing to replace non-regular file: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    return path


def write_inbox(vault_path: Path, text: str, *, overwrite: bool = False) -> Path:
    """Safely write ``_inbox.md`` without following or silently replacing links."""
    path = validate_inbox_target(vault_path, overwrite=overwrite)
    return secure_write_text(path, text, overwrite=overwrite)
