import weft.cli as cli
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM


def test_cli_index_reports_edges(sample_vault, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    rc = cli.main(["index", str(sample_vault), "--store", str(tmp_path / "idx")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Indexed" in out
    assert "edges" in out  # graph stats reported


def test_cli_ask_uses_graph_by_default(sample_vault, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM(response="ans [1]"))
    idx = tmp_path / "idx"
    cli.main(["index", str(sample_vault), "--store", str(idx)])
    capsys.readouterr()

    rc = cli.main(["ask", "coffee", "--store", str(idx)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ans [1]" in out
    assert "Sources:" in out


def test_cli_ask_no_graph_flag_runs(sample_vault, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM(response="pure vector answer"))
    idx = tmp_path / "idx"
    cli.main(["index", str(sample_vault), "--store", str(idx)])
    capsys.readouterr()

    rc = cli.main(["ask", "coffee", "--store", str(idx), "--no-graph"])
    assert rc == 0
    assert "pure vector answer" in capsys.readouterr().out
