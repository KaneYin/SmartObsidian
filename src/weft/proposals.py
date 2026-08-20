"""Inferred-memory proposals: candidates mined from the episodic log, awaiting
explicit accept/reject. Append-only `.weft/memory-proposals.jsonl` with
latest-record-per-id and an idempotent id from the candidate text, so re-running
`suggest` never repeats and rejected candidates stay dead (tombstone dedup)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from weft.security import UnsafeWriteError, secure_append_json

PROPOSAL_TYPES = {"preference", "fact"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def proposal_id(text: str) -> str:
    return "prop_" + hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()[:12]


@dataclass
class Candidate:
    type: str
    text: str
    source: str   # "heuristic" | "inferred:llm"


@dataclass
class Proposal:
    id: str
    type: str
    text: str
    status: str          # pending | accepted | rejected
    signature: str
    source: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


def _read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    if path.is_symlink():
        raise UnsafeWriteError(f"Refusing to read proposals symlink: {path}")
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


class ProposalStore:
    def __init__(self, path: Path):
        self._path = Path(path)

    def _items(self) -> dict[str, Proposal]:
        latest: dict[str, Proposal] = {}
        for rec in _read_jsonl(self._path):
            latest[rec["id"]] = Proposal(**rec)
        return latest

    def known_ids(self) -> set[str]:
        return set(self._items().keys())

    def add(self, candidates: list[Candidate]) -> list[Proposal]:
        known = self.known_ids()
        added: list[Proposal] = []
        for cand in candidates:
            pid = proposal_id(cand.text)
            if pid in known:
                continue
            known.add(pid)
            prop = Proposal(
                id=pid,
                type=cand.type,
                text=cand.text,
                status="pending",
                signature=pid[len("prop_"):],
                source=cand.source,
            )
            secure_append_json(self._path, asdict(prop))
            added.append(prop)
        return added

    def pending(self) -> list[Proposal]:
        return sorted(
            (p for p in self._items().values() if p.status == "pending"),
            key=lambda p: p.created_at,
        )

    def get(self, prop_id: str) -> Proposal:
        prop = self._items().get(prop_id)
        if prop is None:
            raise KeyError(prop_id)
        return prop

    def mark(self, prop_id: str, status: str) -> Proposal:
        prop = self.get(prop_id)
        updated = Proposal(**{**asdict(prop), "status": status, "updated_at": _now()})
        secure_append_json(self._path, asdict(updated))
        return updated
