from weft.parser import Note, Chunk, parse_note, parse_vault, chunk_note


def test_parse_note_extracts_frontmatter_tags_links(sample_vault):
    note = parse_note(sample_vault / "coffee.md")
    assert note.title == "Coffee"
    assert note.frontmatter["tags"] == ["drinks", "morning"]
    assert "drinks" in note.tags and "strong" in note.tags  # frontmatter + inline
    assert "water" in note.wikilinks
    assert note.body.startswith("# Coffee")


def test_parse_note_without_frontmatter(sample_vault):
    note = parse_note(sample_vault / "water.md")
    assert note.title == "Water"
    assert note.frontmatter == {}
    assert "coffee" in note.wikilinks


def test_parse_vault_finds_all_markdown_recursively(sample_vault):
    notes = parse_vault(sample_vault)
    rel = sorted(n.rel_path for n in notes)
    assert rel == ["coffee.md", "notes/tea.md", "water.md"]


def test_parse_vault_excludes_generated_inbox(sample_vault):
    # The suggest command writes _inbox.md into the vault; it must never
    # round-trip back into the index (else it self-suggests on re-index).
    (sample_vault / "_inbox.md").write_text("# Weft Inbox\n\n- [[coffee]]\n")
    rel = {n.rel_path for n in parse_vault(sample_vault)}
    assert "_inbox.md" not in rel


def test_chunk_note_splits_by_heading_and_carries_source(sample_vault):
    note = parse_note(sample_vault / "coffee.md")
    chunks = chunk_note(note)
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.rel_path == "coffee.md" for c in chunks)
    headings = [c.heading for c in chunks]
    assert "Coffee" in headings and "Espresso" in headings
    espresso = next(c for c in chunks if c.heading == "Espresso")
    assert "concentrated" in espresso.text
