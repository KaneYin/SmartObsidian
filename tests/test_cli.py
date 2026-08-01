import weft.cli as cli
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM


def test_cli_index_then_ask(sample_vault, tmp_path, monkeypatch, capsys):
    # Force the fake backends so the test needs no model download / API key.
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda: FakeLLM(response="Coffee is brewed [1]."))

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


def test_cli_ask_missing_index_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda: FakeLLM(response="x"))
    rc = cli.main(["ask", "q", "--store", str(tmp_path / "nope")])
    assert rc == 1
    assert "No index" in capsys.readouterr().err
