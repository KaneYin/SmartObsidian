"""Detect the usable GPU-memory budget for model selection. Always returns a
number so callers never crash: unknown hardware falls back to the CPU tier."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

_MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuInfo:
    backend: str      # "metal" | "cuda" | "cpu"
    total_mb: int
    budget_mb: int    # usable for weights, after headroom


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def detect_gpu(
    system: str | None = None,
    run: Callable[[list[str]], str | None] = _run,
) -> GpuInfo:
    system = system or platform.system()
    if system == "Darwin":
        mem = run(["sysctl", "-n", "hw.memsize"])
        arm = run(["sysctl", "-n", "hw.optional.arm64"])
        if mem and mem.isdigit():
            total_mb = int(mem) // _MIB
            if (arm or "").strip() == "1":
                return GpuInfo("metal", total_mb, int(total_mb * 0.7))
            return GpuInfo("cpu", total_mb, min(total_mb // 2, 4000))
    else:
        vram = run(["nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits"])
        if vram:
            first = vram.splitlines()[0].strip()
            if first.isdigit():
                total_mb = int(first)
                return GpuInfo("cuda", total_mb, total_mb)
    return GpuInfo("cpu", 0, 2000)
