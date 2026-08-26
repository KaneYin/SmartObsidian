"""Pluggable chunking strategies. `heading` reproduces the original heading-boundary
chunks; `parent_child` embeds paragraph children but carries the parent heading
section for context. Contextual Retrieval and sliding-window register here later."""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from weft.parser import Chunk, Note, chunk_note


def _parent_id(rel_path: str, section_ordinal: int) -> str:
    digest = hashlib.sha1(f"{rel_path}#{section_ordinal}".encode("utf-8")).hexdigest()
    return "par_" + digest[:8]


def _split_paragraphs(text: str) -> list[str]:
    """Split a section into paragraphs on blank lines, keeping fenced code blocks
    whole. A section with no blank line yields a single paragraph."""
    paras: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal buf
        joined = "\n".join(buf).strip()
        if joined:
            paras.append(joined)
        buf = []

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not in_fence and line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return paras


def _parent_child_chunks(note: Note) -> list[Chunk]:
    out: list[Chunk] = []
    ordinal = 0
    for section in chunk_note(note):
        pid = _parent_id(note.rel_path, section.ordinal)
        for para in _split_paragraphs(section.text):
            out.append(Chunk(
                rel_path=note.rel_path, heading=section.heading, text=para,
                ordinal=ordinal, parent_id=pid, parent_text=section.text,
            ))
            ordinal += 1
    return out


def chunk_strategy(name: str) -> Callable[[Note], list[Chunk]]:
    if name == "heading":
        return chunk_note
    if name == "parent_child":
        return _parent_child_chunks
    raise ValueError(f"unknown chunking strategy: {name}")
