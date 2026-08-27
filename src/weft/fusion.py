"""Reciprocal Rank Fusion: merge several ranked hit lists into one ranking by summing
each item's reciprocal rank across lists. The shared fusion primitive for dual-query
retrieval (M8) and vector+BM25 hybrid search (M9)."""

from __future__ import annotations

from collections.abc import Callable


def reciprocal_rank_fusion(rankings: list[list], *, k: int = 60, key: Callable) -> list:
    """Fuse ranked lists by summed reciprocal rank (score += 1/(k + rank)). `key(hit)`
    identifies a hit for dedup across lists. Returns unique hits, fused score desc."""
    scores: dict = {}
    representative: dict = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            kk = key(hit)
            scores[kk] = scores.get(kk, 0.0) + 1.0 / (k + rank + 1)
            representative.setdefault(kk, hit)
    ordered = sorted(scores, key=lambda kk: -scores[kk])
    return [representative[kk] for kk in ordered]
