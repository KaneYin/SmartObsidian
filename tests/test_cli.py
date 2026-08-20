import weft.cli as cli
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM


def test_cli_index_then_ask(sample_vault, tmp_path, monkeypatch, capsys):
    # Force the fake backends so the test needs no model download / API key.
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM(response="Coffee is brewed [1]."))

    idx = tmp_path / "weft_index"

    rc = cli.main(["index", str(sample_vault), "--store", str(idx)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Indexed" in out

    rc = cli.main(["ask", "what is coffee?", "--store", str(idx)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Coffee is brewed [1]." in out
    assert "coffee.md" in out  # sources printed
    api_log = idx.parent / "api-log.jsonl"
    assert '"purpose": "ask"' in api_log.read_text()
    assert "what is coffee?" in api_log.read_text()


def test_cli_ask_missing_index_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM(response="x"))
    rc = cli.main(["ask", "q", "--store", str(tmp_path / "nope")])
    assert rc == 1
    assert "No index" in capsys.readouterr().err


def test_cli_rejects_unsafe_numeric_arguments():
    parser = cli.build_parser()
    for argv in (
        ["ask", "question", "--k", "-1"],
        ["ask", "question", "--k", "51"],
        ["suggest", ".", "--limit", "-1"],
        ["suggest", ".", "--threshold", "nan"],
        ["suggest", ".", "--threshold", "1.1"],
    ):
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"unsafe arguments unexpectedly accepted: {argv}")


def test_cli_rejects_privacy_path_traversal_before_indexing(tmp_path, capsys):
    rc = cli.main(["index", str(tmp_path), "--exclude", "../outside"])

    assert rc == 2
    assert "relative vault paths" in capsys.readouterr().err
