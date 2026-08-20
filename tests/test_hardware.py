from weft.hardware import GpuInfo, detect_gpu


def _runner(mapping):
    def run(cmd):
        return mapping.get(tuple(cmd))
    return run


def test_apple_silicon_budget_is_70_percent():
    run = _runner({
        ("sysctl", "-n", "hw.memsize"): str(18 * 1024**3),
        ("sysctl", "-n", "hw.optional.arm64"): "1",
    })
    gpu = detect_gpu(system="Darwin", run=run)
    assert gpu.backend == "metal"
    assert gpu.total_mb == 18 * 1024
    assert gpu.budget_mb == int(18 * 1024 * 0.7)


def test_nvidia_reports_vram_as_budget():
    run = _runner({
        ("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"): "24564",
    })
    gpu = detect_gpu(system="Linux", run=run)
    assert gpu.backend == "cuda"
    assert gpu.budget_mb == 24564


def test_unknown_machine_falls_back_to_cpu():
    gpu = detect_gpu(system="Linux", run=lambda cmd: None)
    assert gpu.backend == "cpu"
    assert gpu.budget_mb == 2000
