"""Proposed-pair memory: an append-only `.weft/suggestions.jsonl` that records
which note pairs `weft suggest` has already proposed, so re-runs don't repeat
themselves. For M2 the ledger only tracks `status: "proposed"` — real
accept/reject feedback is deferred to M3. Pure file I/O over canonical pairs;
knows nothing about embeddings or Claude."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from weft.suggest import LinkSuggestion


def _canonical(a: str, b: str) -> tuple[str, str]:
    lo, hi = sorted((a, b))
    return (lo, hi)


def load_seen(path: Path) -> set[tuple[str, str]]:
    """The set of canonical `(a, b)` pairs already proposed. Missing file → empty."""
    path = Path(path)
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        seen.add(_canonical(rec["a"], rec["b"]))
    return seen


def record(path: Path, suggestions: list[LinkSuggestion]) -> None:
    """Append newly proposed pairs, one JSON record per line. Pairs already in
    the ledger are skipped so re-recording is idempotent."""
    path = Path(path)
    seen = load_seen(path)
    now = datetime.now(timezone.utc).isoformat()

    new_lines: list[str] = []
    for s in suggestions:
        pair = _canonical(s.note_a, s.note_b)
        if pair in seen:
            continue
        seen.add(pair)
        new_lines.append(
            json.dumps(
                {
                    "id": s.id,
                    "a": pair[0],
                    "b": pair[1],
                    "score": s.score,
                    "status": "proposed",
                    "first_seen": now,
                }
            )
        )

    if not new_lines:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
