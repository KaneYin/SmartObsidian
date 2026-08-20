"""Curated open-weight model registry, keyed by GPU-memory tier. Pinned,
known-good Ollama tags; bump these as better open-weight models ship.

The `medium` ceiling (13 GB) is deliberately just above an 18 GB Apple Silicon
machine's ~12.9 GB budget, so unified-memory laptops get the safe 8B default and
must opt up to the 14B `large` tier explicitly."""

from __future__ import annotations

from dataclasses import dataclass

from weft.hardware import GpuInfo


@dataclass(frozen=True)
class Tier:
    name: str
    max_budget_mb: int   # inclusive upper bound of this tier's budget
    default: str         # Ollama model tag
    min_mb: int          # advisory floor to run the default comfortably


TIERS: list[Tier] = [
    Tier("small",  6000,   "qwen2.5:3b",  3000),
    Tier("medium", 13000,  "llama3.1:8b", 6000),
    Tier("large",  24000,  "qwen2.5:14b", 10000),
    Tier("xl",     10**9,  "qwen2.5:32b", 20000),
]


def tier_for(budget_mb: int) -> Tier:
    for tier in TIERS:
        if budget_mb <= tier.max_budget_mb:
            return tier
    return TIERS[-1]


def pick_default(gpu: GpuInfo) -> str:
    return tier_for(gpu.budget_mb).default
