from weft.parser import parse_note, parse_vault, chunk_note


def test_code_fence_hash_not_treated_as_heading(tmp_path):
    p = tmp_path / "code.md"
    p.write_text(
        "# Real Title\n\nIntro.\n\n"
        "```bash\n# not a real heading\necho hi\n```\n\n"
        "More text under the real title.\n"
    )
    note = parse_note(p)
    assert note.title == "Real Title"
    chunks = chunk_note(note)
    headings = [c.heading for c in chunks]
    assert headings == ["Real Title"]  # the shell comment did NOT split
    assert "echo hi" in chunks[0].text


def test_note_with_no_headings_falls_back_to_stem(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("Just a paragraph, no heading at all.\n")
    note = parse_note(p)
    assert note.title == "plain"
    chunks = chunk_note(note)
    assert len(chunks) == 1
    assert chunks[0].heading == "plain"


def test_scalar_frontmatter_tag(tmp_path):
    p = tmp_path / "s.md"
    p.write_text("---\ntags: solo\n---\n# S\n\nbody #inline\n")
    note = parse_note(p)
    assert "solo" in note.tags and "inline" in note.tags


def test_wikilink_alias_and_heading_forms(tmp_path):
    p = tmp_path / "links.md"
    p.write_text("# L\n\nSee [[target|an alias]] and [[other#section]].\n")
    note = parse_note(p)
    assert note.wikilinks == ["target", "other"]


def test_non_utf8_file_does_not_abort_indexing(tmp_path):
    (tmp_path / "good.md").write_text("# Good\n\nfine text\n")
    (tmp_path / "bad.md").write_bytes(b"# Bad\n\n\xff\xfe not utf-8 \x80\n")
    notes = parse_vault(tmp_path)
    titles = sorted(n.title for n in notes)
    assert titles == ["Bad", "Good"]  # both parsed, no crash
