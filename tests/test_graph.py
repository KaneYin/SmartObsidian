from weft.graph import LinkGraph
from weft.parser import parse_vault


def test_graph_edges_from_wikilinks(sample_vault):
    notes = parse_vault(sample_vault)
    g = LinkGraph.from_notes(notes)
    # coffee <-> water are mutually linked; edge is undirected
    assert g.has_edge("coffee.md", "water.md")
    # neighbors are by rel_path, undirected (includes backlinks)
    assert "water.md" in g.neighbors("coffee.md")
    assert "coffee.md" in g.neighbors("water.md")


def test_graph_resolves_link_by_stem_case_insensitively(tmp_path):
    (tmp_path / "Water.md").write_text("# Water\n\ntext\n")
    (tmp_path / "note.md").write_text("# Note\n\nSee [[water]] and [[WATER]].\n")
    notes = parse_vault(tmp_path)
    g = LinkGraph.from_notes(notes)
    assert g.has_edge("note.md", "Water.md")


def test_graph_resolves_link_by_path(tmp_path):
    sub = tmp_path / "notes"
    sub.mkdir()
    (sub / "tea.md").write_text("# Tea\n\ntext\n")
    (tmp_path / "root.md").write_text("# Root\n\nSee [[notes/tea]].\n")
    notes = parse_vault(tmp_path)
    g = LinkGraph.from_notes(notes)
    assert g.has_edge("root.md", "notes/tea.md")


def test_graph_drops_unresolvable_links(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\nSee [[does-not-exist]].\n")
    notes = parse_vault(tmp_path)
    g = LinkGraph.from_notes(notes)
    assert "a.md" in g.nodes()
    assert g.neighbors("a.md") == set()


def test_graph_ambiguous_bare_stem_dropped_but_path_resolves(tmp_path):
    # Two notes share the stem "tea" in different folders (a common Obsidian case).
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "tea.md").write_text("# Tea A\n\ntext\n")
    (tmp_path / "b" / "tea.md").write_text("# Tea B\n\ntext\n")
    (tmp_path / "bare.md").write_text("# Bare\n\nSee [[tea]].\n")
    (tmp_path / "qualified.md").write_text("# Q\n\nSee [[b/tea]].\n")
    notes = parse_vault(tmp_path)
    g = LinkGraph.from_notes(notes)
    # Bare [[tea]] is ambiguous -> dropped, not mis-linked to whichever note came last.
    assert g.neighbors("bare.md") == set()
    # Path-qualified [[b/tea]] still resolves precisely.
    assert g.has_edge("qualified.md", "b/tea.md")
    assert not g.has_edge("qualified.md", "a/tea.md")


def test_graph_neighbors_of_unknown_node_is_empty(sample_vault):
    g = LinkGraph.from_notes(parse_vault(sample_vault))
    assert g.neighbors("missing.md") == set()


def test_graph_save_and_load_roundtrip(sample_vault, tmp_path):
    g = LinkGraph.from_notes(parse_vault(sample_vault))
    path = tmp_path / "index.graph.json"
    g.save(path)
    loaded = LinkGraph.load(path)
    assert loaded.neighbors("coffee.md") == g.neighbors("coffee.md")
    assert set(loaded.nodes()) == set(g.nodes())
    assert loaded.edge_count() == g.edge_count()
