"""Durable agent memory: an episodic interaction log plus curated semantic items
(preference / fact / decision / task) with a lifecycle status. Append-only with
latest-record-wins on load; rejected/superseded records are kept as tombstones.
Private to `.weft/` — memory is the most sensitive surface Weft has."""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from weft.security import UnsafeWriteError, secure_append_json, secure_write_text

SEMANTIC_TYPES = {"preference", "fact", "decision", "task"}
_ALWAYS_INJECT = {"preference", "fact"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryItem:
    id: str
    type: str
    text: str
    status: str = "active"          # active | superseded | rejected
    provenance: str = "explicit"    # explicit | inferred
    confidence: float | None = None
    source: str = "weft remember"
    supersedes: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass
class Episode:
    id: str
    ts: str
    question: str
    answer: str
    sources: list[str]
    vault_root: str | None = None


@dataclass
class MemoryHit:
    score: float
    kind: str    # "decision" | "task" | "log"
    text: str


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    if path.is_symlink():
        raise UnsafeWriteError(f"Refusing to read memory symlink: {path}")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class MemoryStore:
    def __init__(self, memory_path: Path, episodes_path: Path, vault_root: str = ""):
        self._memory_path = Path(memory_path)
        self._episodes_path = Path(episodes_path)
        self._vault_root = vault_root

    # --- semantic items ---------------------------------------------------
    def _items(self) -> dict[str, MemoryItem]:
        """Latest record per id wins (append-only collapse)."""
        latest: dict[str, MemoryItem] = {}
        for rec in _read_jsonl(self._memory_path):
            latest[rec["id"]] = MemoryItem(**rec)
        return latest

    def remember(self, type: str, text: str, *, provenance: str = "explicit",
                 confidence: float | None = None, source: str = "weft remember") -> MemoryItem:
        if type not in SEMANTIC_TYPES:
            raise ValueError(f"unknown memory type: {type}")
        item = MemoryItem(
            id="mem_" + secrets.token_hex(4),
            type=type,
            text=text,
            provenance=provenance,
            confidence=confidence,
            source=source,
        )
        secure_append_json(self._memory_path, asdict(item))
        return item

    def active_semantic(self, type: str | None = None) -> list[MemoryItem]:
        items = [i for i in self._items().values() if i.status == "active"]
        if type is not None:
            items = [i for i in items if i.type == type]
        return sorted(items, key=lambda i: i.created_at)

    def get(self, item_id: str) -> MemoryItem:
        item = self._items().get(item_id)
        if item is None:
            raise KeyError(item_id)
        return item

    def supersede(self, item_id: str, new_text: str) -> MemoryItem:
        old = self.get(item_id)
        new = MemoryItem(
            id="mem_" + secrets.token_hex(4),
            type=old.type,
            text=new_text,
            provenance=old.provenance,
            source=old.source,
            supersedes=old.id,
        )
        secure_append_json(self._memory_path, asdict(new))
        tomb = MemoryItem(**{**asdict(old), "status": "superseded", "updated_at": _now()})
        secure_append_json(self._memory_path, asdict(tomb))
        return new

    def reject(self, item_id: str) -> None:
        old = self.get(item_id)
        tomb = MemoryItem(**{**asdict(old), "status": "rejected", "updated_at": _now()})
        secure_append_json(self._memory_path, asdict(tomb))

    def compact(self) -> None:
        collapsed = self._items()
        lines = []
        for item in collapsed.values():
            if item.status == "rejected":
                item = MemoryItem(**{**asdict(item), "text": ""})
            lines.append(json.dumps(asdict(item), ensure_ascii=False))
        secure_write_text(self._memory_path, "\n".join(lines) + "\n", overwrite=True)

    # --- episodic log -----------------------------------------------------
    def log_episode(self, question: str, answer: str, sources: list[str]) -> Episode:
        ep = Episode(
            id="ep_" + secrets.token_hex(4),
            ts=_now(),
            question=question,
            answer=answer,
            sources=list(sources),
            vault_root=self._vault_root,
        )
        secure_append_json(self._episodes_path, asdict(ep))
        return ep

    def episodes(self) -> list[Episode]:
        """Only episodes logged against this exact vault_root. A missing or
        mismatched vault_root (including every pre-scoping record) is
        excluded -- fail closed rather than assume it matches."""
        return [
            ep for ep in (Episode(**rec) for rec in _read_jsonl(self._episodes_path))
            if ep.vault_root == self._vault_root
        ]

    # --- recall -----------------------------------------------------------
    def _episode_line(self, ep: Episode) -> str:
        return f'on {ep.ts[:10]} you asked "{ep.question}" — answered "{ep.answer[:200]}"'

    def recall(self, embedder, query: str, k: int, *, kinds: set[str]) -> list[MemoryHit]:
        if k <= 0:
            return []
        candidates: list[tuple[str, str]] = []  # (kind, text)
        for item in self.active_semantic():
            if item.type in kinds:
                candidates.append((item.type, item.text))
        if "log" in kinds:
            for ep in self.episodes():
                candidates.append(("log", self._episode_line(ep)))
        if not candidates:
            return []
        texts = [t for _, t in candidates]
        vecs = np.asarray(embedder.embed(texts), dtype=np.float32)
        qv = np.asarray(embedder.embed([query])[0], dtype=np.float32)
        scores = vecs @ qv   # embedder returns unit-norm vectors -> cosine
        order = np.argsort(-scores)[:k]
        return [MemoryHit(float(scores[i]), candidates[i][0], candidates[i][1]) for i in order]
