"""Inferred-link suggestions: find notes that are semantically close but not
already `[[linked]]`. This is the local, no-LLM heart of M2 — pure embedding
math over the stored chunk vectors, fused with M1's explicit link graph.

Everything here is deterministic and offline: candidate generation is an N×N
cosine over mean-pooled note vectors, and the default rationale is a local
template. The optional rationale pass is orchestrated by the application service."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np

from weft.graph import LinkGraph
from weft.index import OVERVIEW_REL_PATH
from weft.store import VectorStore

# A small stopword set so shared "salient terms" aren't dominated by glue words.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "with", "as", "at", "by", "from", "it", "this", "that", "be", "into", "its",
}
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")


@dataclass
class LinkSuggestion:
    id: str
    note_a: str
    note_b: str
    score: float
    shared_tags: list[str] = field(default_factory=list)
    rationale: str = ""


def pair_id(a: str, b: str) -> str:
    """Stable short id for a note pair, order-independent. Canonicalizes the
    pair first so A↔B and B↔A share one id (used by the ledger for dedup)."""
    lo, hi = sorted((a, b))
    return hashlib.sha1(f"{lo}|{hi}".encode("utf-8")).hexdigest()[:12]


def note_vectors(store: VectorStore) -> dict[str, np.ndarray]:
    """Mean-pool each note's chunk vectors into one L2-normalized note vector.
    Notes with no chunks never appear (they aren't in the store). The
    synthetic vault-overview chunk isn't a real note, so it's excluded here
    -- it must never become a link-suggestion candidate."""
    out: dict[str, np.ndarray] = {}
    for rel_path, chunk_vecs in store.vectors_by_note().items():
        if rel_path == OVERVIEW_REL_PATH:
            continue
        mean = chunk_vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        out[rel_path] = mean.astype(np.float32)
    return out


def _shared_tags(meta_by_note: dict[str, list[dict]], a: str, b: str) -> list[str]:
    """Tags common to both notes, in note-a's order. Tags are note-level, so
    the first chunk's metadata carries them."""
    tags_a = meta_by_note.get(a, [{}])[0].get("tags", []) or []
    tags_b = set(meta_by_note.get(b, [{}])[0].get("tags", []) or [])
    return [t for t in dict.fromkeys(tags_a) if t in tags_b]


def _salient_terms(meta_by_note: dict[str, list[dict]], rel_path: str) -> set[str]:
    """Lowercased salient terms from a note's headings — cheap, deterministic,
    no vault re-parse (headings live in the stored chunk metadata)."""
    terms: set[str] = set()
    for chunk in meta_by_note.get(rel_path, []):
        for m in _TERM_RE.finditer(chunk.get("heading", "")):
            w = m.group(0).lower()
            if w not in _STOPWORDS:
                terms.add(w)
    return terms


def _local_rationale(
    meta_by_note: dict[str, list[dict]], a: str, b: str, score: float, shared_tags: list[str]
) -> str:
    """cosine score + shared tags + a few shared salient terms. Zero network."""
    shared_terms = sorted(_salient_terms(meta_by_note, a) & _salient_terms(meta_by_note, b))
    parts = [f"cosine {score:.2f}"]
    if shared_tags:
        parts.append("shared tags: " + ", ".join(f"#{t}" for t in shared_tags))
    if shared_terms:
        parts.append("shared terms: " + ", ".join(shared_terms[:4]))
    return " · ".join(parts)


def infer_links(
    store: VectorStore,
    graph: LinkGraph,
    *,
    threshold: float,
    limit: int,
    seen_pairs: set[tuple[str, str]],
) -> list[LinkSuggestion]:
    """Propose note↔note links: pairs above `threshold` cosine that aren't
    already 1-hop-linked in `graph` and haven't been proposed before.

    O(N²) note-pair cosine — trivial at current vault scale; revisit with an
    ANN index if vaults grow large (consistent with M1's full-scan note)."""
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite cosine value between -1 and 1")
    if limit < 0:
        raise ValueError("limit must not be negative")
    if limit == 0:
        return []
    vecs = note_vectors(store)
    names = sorted(vecs)  # sorted → deterministic; pairs emitted in canonical order
    if len(names) < 2:
        return []

    matrix = np.vstack([vecs[n] for n in names])  # N × dim, each row unit-norm
    sims = matrix @ matrix.T  # N × N cosine

    meta_by_note = store.metadata_by_note()
    suggestions: list[LinkSuggestion] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]  # i < j and names sorted → a < b (canonical)
            score = float(sims[i, j])
            if score < threshold:
                continue
            if graph.has_edge(a, b):
                continue
            if (a, b) in seen_pairs:
                continue
            shared_tags = _shared_tags(meta_by_note, a, b)
            suggestions.append(
                LinkSuggestion(
                    id=pair_id(a, b),
                    note_a=a,
                    note_b=b,
                    score=score,
                    shared_tags=shared_tags,
                    rationale=_local_rationale(meta_by_note, a, b, score, shared_tags),
                )
            )

    # Rank by score desc; break ties on the canonical pair for stable output.
    suggestions.sort(key=lambda s: (-s.score, s.note_a, s.note_b))
    return suggestions[:limit]
