from datetime import datetime
import os

import pytest

from weft.inbox import render_inbox, write_inbox
from weft.suggest import LinkSuggestion

GEN = datetime(2026, 7, 22, 14, 30)


def _sugg(a, b, score=0.86, id="3f2a1b", tags=None, rationale="cosine 0.86"):
    return LinkSuggestion(
        id=id, note_a=a, note_b=b, score=score,
        shared_tags=tags or [], rationale=rationale,
    )


def test_render_contains_header_and_count():
    text = render_inbox([_sugg("coffee.md", "espresso.md")], GEN)
    assert "# Weft Inbox" in text
    assert "2026-07-22 14:30" in text
    assert "1 suggestion" in text


def test_render_contains_checkbox_pair_id_and_wikilink_suggestion():
    text = render_inbox([_sugg("coffee.md", "espresso.md", id="3f2a1b")], GEN)
    assert "- [ ]" in text  # cosmetic checkbox
    assert "coffee.md" in text and "espresso.md" in text
    assert "3f2a1b" in text
    assert "score 0.86" in text
    assert "[[espresso]]" in text  # wikilink suggestion uses note_b's stem


def test_render_includes_rationale_line():
    text = render_inbox(
        [_sugg("a.md", "b.md", rationale="cosine 0.91 · shared tags: #x")], GEN
    )
    assert "cosine 0.91 · shared tags: #x" in text


def test_render_is_deterministic_and_ordered():
    suggs = [_sugg("a.md", "b.md", id="111"), _sugg("c.md", "d.md", id="222")]
    text = render_inbox(suggs, GEN)
    assert text.index("111") < text.index("222")  # order preserved
    assert render_inbox(suggs, GEN) == text  # deterministic


def test_render_escapes_untrusted_markdown_and_uses_qualified_link_path():
    text = render_inbox(
        [
            _sugg(
                "folder/evil[link].md",
                "other/note`name.md",
                rationale="<script>*not formatting*</script>\nsecond line",
            )
        ],
        GEN,
    )

    assert r"evil\[link\].md" in text
    assert r"\<script\>\*not formatting\*\</script\> second line" in text
    assert "[[other/note`name]]" in text


def test_render_empty_state():
    text = render_inbox([], GEN)
    assert "# Weft Inbox" in text
    assert "0 suggestions" in text
    assert "- [ ]" not in text  # no checkboxes when empty
    assert "No new inferred links" in text


def test_write_inbox_writes_file_and_returns_path(tmp_path):
    path = write_inbox(tmp_path, "hello inbox")
    assert path == tmp_path / "_inbox.md"
    assert path.read_text(encoding="utf-8") == "hello inbox"
    assert (path.stat().st_mode & 0o777) == 0o600


def test_write_inbox_refuses_existing_file_without_explicit_overwrite(tmp_path):
    path = tmp_path / "_inbox.md"
    path.write_text("keep me")
    with pytest.raises(FileExistsError):
        write_inbox(tmp_path, "replacement")
    assert path.read_text() == "keep me"


def test_write_inbox_explicitly_overwrites_regular_file(tmp_path):
    path = tmp_path / "_inbox.md"
    path.write_text("old")
    write_inbox(tmp_path, "new", overwrite=True)
    assert path.read_text() == "new"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_write_inbox_never_follows_symlink(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    target = tmp_path / "outside.md"
    target.write_text("do not replace")
    (vault / "_inbox.md").symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        write_inbox(vault, "malicious replacement", overwrite=True)
    assert target.read_text() == "do not replace"
