"""Render Weft's active memory (and pending inferred proposals) into a read-only
`_memory.md` note for visibility inside Obsidian. Write reuses the `_inbox.md`
security guarantees: refuse symlink / non-regular targets, atomic 0600 write."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from weft.inbox import _markdown_text  # reuse inert-Markdown escaping
from weft.security import UnsafeWriteError, secure_write_text, vault_root

MEMORY_MIRROR_NAME = "_memory.md"
_TITLE = "# Weft Memory — what I remember"


def render_mirror(active_items: list, pending: list, generated_at: datetime) -> str:
    ts = generated_at.strftime("%Y-%m-%d %H:%M")
    lines = [_TITLE, f"_Generated {ts} · read-only; edit memory with the `weft` CLI_", ""]

    lines.append("## Remembered")
    if active_items:
        for item in active_items:
            lines.append(f"- **[{item.type}]** {_markdown_text(item.text)}")
    else:
        lines.append("_Nothing yet. Add with_ `weft remember \"...\"`.")
    lines.append("")

    if pending:
        lines.append("## Pending — run `weft memory accept <id>`")
        for prop in pending:
            lines.append(
                f"- `{prop.id}` **[{prop.type}]** {_markdown_text(prop.text)}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _validate_mirror_target(vault_path: Path) -> Path:
    root = vault_root(vault_path)
    path = root / MEMORY_MIRROR_NAME
    if path.is_symlink():
        raise UnsafeWriteError(f"Refusing to replace symlink: {path}")
    if path.exists() and not path.is_file():
        raise UnsafeWriteError(f"Refusing to replace non-regular file: {path}")
    return path


def write_mirror(vault_path: Path, text: str) -> Path:
    path = _validate_mirror_target(vault_path)
    return secure_write_text(path, text, overwrite=True)
