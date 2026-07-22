"""The explicit link graph: an undirected networkx graph whose nodes are notes
(keyed by rel_path) and whose edges are resolved [[wikilinks]]. Obsidian-style
resolution: link targets match a note's filename stem (case-insensitive) or its
path without extension. Persists as adjacency JSON beside the vector index."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from weft.parser import Note


def _resolve_link(
    link: str, by_path: dict[str, str], by_stem: dict[str, list[str]]
) -> str | None:
    """Resolve a wikilink target to a note's rel_path, Obsidian-style. A
    path-qualified link (`[[notes/tea]]`) matches the note's path-without-ext.
    A bare `[[tea]]` matches by filename stem, but only when that stem is
    unambiguous across the vault — if two notes share a stem in different
    folders, we refuse to guess and drop the link rather than link the wrong
    note. (`[[a/tea]]` still resolves precisely via the path key.)"""
    key = link.strip().lower()
    if not key:
        return None
    if key in by_path:
        return by_path[key]
    candidates = by_stem.get(key)
    if candidates is not None and len(candidates) == 1:
        return candidates[0]
    return None


class LinkGraph:
    def __init__(self, graph: nx.Graph | None = None):
        self._g = graph if graph is not None else nx.Graph()

    @classmethod
    def from_notes(cls, notes: list[Note]) -> "LinkGraph":
        g = nx.Graph()
        by_path: dict[str, str] = {}
        by_stem: dict[str, list[str]] = {}
        for note in notes:
            g.add_node(note.rel_path)
            p = Path(note.rel_path)
            by_path[p.with_suffix("").as_posix().lower()] = note.rel_path
            by_stem.setdefault(p.stem.lower(), []).append(note.rel_path)
        for note in notes:
            for link in note.wikilinks:
                target = _resolve_link(link, by_path, by_stem)
                if target is not None and target != note.rel_path:
                    g.add_edge(note.rel_path, target)
        return cls(g)

    def nodes(self) -> list[str]:
        return list(self._g.nodes())

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def has_edge(self, a: str, b: str) -> bool:
        return self._g.has_edge(a, b)

    def neighbors(self, rel_path: str) -> set[str]:
        if rel_path not in self._g:
            return set()
        return set(self._g.neighbors(rel_path))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": list(self._g.nodes()),
            "edges": [list(e) for e in self._g.edges()],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LinkGraph":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        g = nx.Graph()
        g.add_nodes_from(data["nodes"])
        g.add_edges_from(tuple(e) for e in data["edges"])
        return cls(g)
