from weft.hardware import GpuInfo
from weft.models import TIERS, pick_default, tier_for


def test_tiers_ascending_and_named():
    assert [t.name for t in TIERS] == ["small", "medium", "large", "xl"]
    budgets = [t.max_budget_mb for t in TIERS]
    assert budgets == sorted(budgets)


def test_tier_for_boundaries():
    assert tier_for(5000).name == "small"
    assert tier_for(6000).name == "small"     # inclusive upper edge
    assert tier_for(9000).name == "medium"
    assert tier_for(20000).name == "large"
    assert tier_for(99999).name == "xl"


def test_pick_default_for_m3_pro():
    gpu = GpuInfo("metal", 18 * 1024, int(18 * 1024 * 0.7))  # ~12.6 GB budget
    assert pick_default(gpu) == "llama3.1:8b"
