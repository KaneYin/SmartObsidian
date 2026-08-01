"""The display-only inbox: render inferred-link suggestions into `_inbox.md`.

This is the dogfooded "never silent writes" surface — `weft suggest` writes a
generated `_inbox.md` into the vault and never touches an existing note. The
checkboxes are cosmetic in M2; accept/reject feedback is M3. Pure rendering:
takes `LinkSuggestion`s, returns/writes markdown."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from weft.suggest import LinkSuggestion

INBOX_NAME = "_inbox.md"
_TITLE = "# Weft Inbox — inferred links"


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
        b_stem = Path(s.note_b).stem
        lines.append(
            f"- [ ] **{s.note_a} ↔ {s.note_b}**  ·  score {s.score:.2f}  ·  `id {s.id}`"
        )
        lines.append(f"      {s.rationale}")
        lines.append(
            f"      Suggested: add `[[{b_stem}]]` to {s.note_a} (and/or the reverse)"
        )
    return "\n".join(lines) + "\n"


def write_inbox(vault_path: Path, text: str) -> Path:
    """Write the rendered inbox to `<vault>/_inbox.md` and return its path."""
    path = Path(vault_path) / INBOX_NAME
    path.write_text(text, encoding="utf-8")
    return path
