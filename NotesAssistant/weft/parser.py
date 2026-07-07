"""Parse an Obsidian markdown vault into Note and Chunk objects.

Frontmatter parsing is intentionally minimal (no external YAML dep for M0):
it handles the `key: value` and `tags: [a, b]` shapes an Obsidian vault
commonly uses. Swap in a real YAML parser in a later milestone if needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")  # [[target]], [[target|alias]], [[target#h]]
INLINE_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Note:
    rel_path: str
    title: str
    frontmatter: dict
    tags: list[str]
    wikilinks: list[str]
    body: str  # markdown after the frontmatter block


@dataclass
class Chunk:
    rel_path: str
    heading: str
    text: str
    ordinal: int = field(default=0)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    block = raw[3:end].strip("\n")
    body = raw[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("[") and value.endswith("]"):
            fm[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            fm[key] = value
    return fm, body


def parse_note(path: Path) -> Note:
    # errors="replace": one non-UTF-8 note must not abort indexing the vault.
    raw = path.read_text(encoding="utf-8", errors="replace")
    fm, body = _split_frontmatter(raw)

    tags: list[str] = []
    fm_tags = fm.get("tags")
    if isinstance(fm_tags, list):
        tags.extend(fm_tags)
    elif isinstance(fm_tags, str) and fm_tags:
        tags.append(fm_tags)
    for m in INLINE_TAG_RE.finditer(body):
        tags.append(m.group(1))

    wikilinks = [m.group(1).strip() for m in WIKILINK_RE.finditer(body)]

    # Take the first real heading as the title, ignoring `#` lines inside
    # fenced code blocks (shell comments, etc.).
    title = path.stem
    in_fence = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and (h := HEADING_RE.match(line)):
            title = h.group(2).strip()
            break

    return Note(
        rel_path=path.name,  # replaced with true rel_path by parse_vault
        title=title,
        frontmatter=fm,
        tags=list(dict.fromkeys(tags)),
        wikilinks=list(dict.fromkeys(wikilinks)),
        body=body,
    )


def parse_vault(vault_path: Path) -> list[Note]:
    vault_path = Path(vault_path)
    notes: list[Note] = []
    for md in sorted(vault_path.rglob("*.md")):
        note = parse_note(md)
        note.rel_path = md.relative_to(vault_path).as_posix()
        notes.append(note)
    return notes


def chunk_note(note: Note) -> list[Chunk]:
    """Split a note into chunks at heading boundaries. Text under a heading
    (until the next heading) becomes one chunk, tagged with that heading."""
    chunks: list[Chunk] = []
    heading = note.title
    buf: list[str] = []
    ordinal = 0
    in_fence = False

    def flush():
        nonlocal ordinal, buf
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(rel_path=note.rel_path, heading=heading, text=text, ordinal=ordinal))
            ordinal += 1
        buf = []

    for line in note.body.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        h = None if in_fence else HEADING_RE.match(line)
        if h:
            flush()
            heading = h.group(2).strip()
        else:
            buf.append(line)
    flush()
    return chunks
