import pytest


@pytest.fixture
def sample_vault(tmp_path):
    """A tiny Obsidian-style vault with frontmatter, tags, wikilinks, headings."""
    (tmp_path / "coffee.md").write_text(
        "---\ntags: [drinks, morning]\n---\n"
        "# Coffee\n\n"
        "Coffee is brewed from roasted beans. See [[water]] for the other half.\n\n"
        "## Espresso\n\n"
        "Espresso is a concentrated form. #strong\n"
    )
    (tmp_path / "water.md").write_text(
        "# Water\n\nWater is essential for [[coffee]] and life.\n"
    )
    sub = tmp_path / "notes"
    sub.mkdir()
    (sub / "tea.md").write_text("# Tea\n\nTea is steeped, not brewed.\n")
    (tmp_path / "ignore.txt").write_text("not markdown")
    return tmp_path
