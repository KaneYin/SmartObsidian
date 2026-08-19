import pytest

import weft.cli as cli
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM


@pytest.fixture
def linkable_vault(tmp_path):
    """Two unlinked notes with identical chunk text -> cosine 1.0 under
    FakeEmbedder, so `suggest` proposes linking them."""
    (tmp_path / "alpha.md").write_text("# Alpha\n\nshared body text here\n")
    (tmp_path / "beta.md").write_text("# Beta\n\nshared body text here\n")
    return tmp_path


def _index(vault, store, monkeypatch):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    cli.main(["index", str(vault), "--store", str(store)])


def test_suggest_writes_inbox_prints_summary_and_updates_ledger(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    capsys.readouterr()

    rc = cli.main(["suggest", str(linkable_vault), "--store", str(store)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "_inbox.md" in out
    assert "1 suggestion" in out

    inbox = linkable_vault / "_inbox.md"
    assert inbox.exists()
    body = inbox.read_text(encoding="utf-8")
    assert "alpha.md ↔ beta.md" in body

    ledger = store.parent / "suggestions.jsonl"
    assert ledger.exists()
    assert "alpha.md" in ledger.read_text(encoding="utf-8")


def test_suggest_second_run_proposes_nothing_new(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    capsys.readouterr()

    cli.main(["suggest", str(linkable_vault), "--store", str(store)])
    inbox_after_first = (linkable_vault / "_inbox.md").read_text(encoding="utf-8")
    ledger = store.parent / "suggestions.jsonl"
    lines_after_first = ledger.read_text(encoding="utf-8").count("\n")
    capsys.readouterr()

    rc = cli.main(["suggest", str(linkable_vault), "--store", str(store)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "0 suggestions" in out
    assert (linkable_vault / "_inbox.md").read_text(encoding="utf-8") == inbox_after_first
    # ledger unchanged: the already-proposed pair is not recorded again
    assert ledger.read_text(encoding="utf-8").count("\n") == lines_after_first


def test_suggest_rationale_uses_llm_and_logs_payload(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    monkeypatch.setattr(
        cli, "make_llm", lambda: FakeLLM(response="these notes cover the same topic")
    )
    capsys.readouterr()

    rc = cli.main(
        ["suggest", str(linkable_vault), "--store", str(store), "--rationale"]
    )
    assert rc == 0

    body = (linkable_vault / "_inbox.md").read_text(encoding="utf-8")
    assert "these notes cover the same topic" in body

    api_log = store.parent / "api-log.jsonl"
    assert api_log.exists()
    assert api_log.read_text(encoding="utf-8").strip()  # a payload was logged


def test_reindex_after_suggest_ignores_generated_inbox(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    cli.main(["suggest", str(linkable_vault), "--store", str(store)])
    assert (linkable_vault / "_inbox.md").exists()

    # Re-index the vault (now containing _inbox.md), then suggest fresh with a
    # clean ledger: _inbox.md must not appear as a note or a suggestion.
    _index(linkable_vault, store, monkeypatch)
    (store.parent / "suggestions.jsonl").unlink()
    capsys.readouterr()

    cli.main(
        [
            "suggest", str(linkable_vault), "--store", str(store),
            "--overwrite-inbox",
        ]
    )
    body = (linkable_vault / "_inbox.md").read_text(encoding="utf-8")
    assert "_inbox.md" not in body  # never self-suggests


def test_suggest_empty_state_leaves_ledger_untouched(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    # A zero attention budget produces no suggestions and must not touch inbox.
    rc = cli.main(
        ["suggest", str(linkable_vault), "--store", str(store), "--limit", "0"]
    )
    assert rc == 0
    assert "0 suggestions" in capsys.readouterr().out
    assert not (linkable_vault / "_inbox.md").exists()
    assert not (store.parent / "suggestions.jsonl").exists()  # ledger untouched


def test_suggest_no_index_errors(tmp_path, capsys):
    rc = cli.main(["suggest", str(tmp_path), "--store", str(tmp_path / "missing")])
    assert rc == 1
    assert "No index" in capsys.readouterr().err


def test_suggest_preserves_existing_inbox_and_does_not_record(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    existing = linkable_vault / "_inbox.md"
    existing.write_text("unchecked user review")
    capsys.readouterr()

    rc = cli.main(["suggest", str(linkable_vault), "--store", str(store)])

    assert rc == 1
    assert "already exists" in capsys.readouterr().err
    assert existing.read_text() == "unchecked user review"
    assert not (store.parent / "suggestions.jsonl").exists()


def test_suggest_rejects_index_bound_to_another_vault(
    linkable_vault, tmp_path, monkeypatch, capsys
):
    store = tmp_path / "idx"
    _index(linkable_vault, store, monkeypatch)
    other_vault = tmp_path / "other"
    other_vault.mkdir()
    capsys.readouterr()

    rc = cli.main(["suggest", str(other_vault), "--store", str(store)])

    assert rc == 1
    assert "different vault" in capsys.readouterr().err
