import weft.cli as cli
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM


def test_cli_index_then_ask(sample_vault, tmp_path, monkeypatch, capsys):
    # Force the fake backends so the test needs no model download / API key.
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM(response="Coffee is brewed [1]."))
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm",
                        lambda *a, **k: FakeLLM(response="Coffee is brewed [1]."))

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


def test_cli_index_parent_child(sample_vault, tmp_path, monkeypatch):
    from weft.embeddings import FakeEmbedder
    from weft.store import VectorStore
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    idx = tmp_path / "idx"
    rc = cli.main(["index", str(sample_vault), "--store", str(idx), "--chunking", "parent-child"])
    assert rc == 0
    metas = next(iter(VectorStore.load(idx).metadata_by_note().values()))
    assert "parent_id" in metas[0]


def test_cli_ask_no_hybrid_runs(sample_vault, tmp_path, monkeypatch, capsys):
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="ans [1]"))
    idx = tmp_path / "idx"
    cli.main(["index", str(sample_vault), "--store", str(idx)])
    capsys.readouterr()
    rc = cli.main(["ask", "coffee", "--store", str(idx), "--no-hybrid"])
    assert rc == 0
    assert "ans [1]" in capsys.readouterr().out


def test_cli_index_sliding_and_contextual(sample_vault, tmp_path, monkeypatch):
    import json
    from weft.index import manifest_path_for
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="ctx."))
    idx = tmp_path / "idx"
    rc = cli.main(["index", str(sample_vault), "--store", str(idx),
                   "--chunking", "sliding", "--contextual"])
    assert rc == 0
    manifest = json.loads(manifest_path_for(idx).read_text())
    assert manifest["chunking"] == "sliding" and manifest["contextual"] is True


def test_cli_ask_and_chat_memory_query(sample_vault, tmp_path, monkeypatch):
    from weft.embeddings import FakeEmbedder
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="A [1]"))
    idx = tmp_path / "idx"
    cli.main(["index", str(sample_vault), "--store", str(idx)])
    rc = cli.main(["ask", "hello", "--store", str(idx), "--memory-query"])
    assert rc == 0
    monkeypatch.setattr("weft.chat.run_repl", lambda session: 0)
    monkeypatch.setattr("weft.service.make_llm", lambda *a, **k: FakeLLM(response="A [1]"))
    rc2 = cli.main(["chat", "--store", str(idx), "--memory-query"])
    assert rc2 == 0


def test_cli_mode_and_explicit_options_reach_service(monkeypatch, capsys):
    seen = {}

    def fake_ask(store, question, **kwargs):
        seen.update(question=question, **kwargs)
        return {"answer": "ok", "sources": [], "retrieval": {}}

    monkeypatch.setattr("weft.service.service_ask", fake_ask)
    rc = cli.main([
        "ask", "question", "--mode", "best", "--k", "12",
        "--no-hybrid", "--no-rerank", "--no-memory-query",
    ])
    assert rc == 0
    assert seen["mode"] == "best"
    overrides = seen["retrieval_overrides"]
    assert overrides.k == 12
    assert overrides.hybrid is False
    assert overrides.rerank is False
    assert overrides.memory_query is False
    assert "ok" in capsys.readouterr().out


def test_cli_model_timeout_is_rendered_without_traceback(monkeypatch, capsys):
    from weft.llm import LLMRequestError

    def timeout(*args, **kwargs):
        raise LLMRequestError("model timed out")

    monkeypatch.setattr("weft.service.service_ask", timeout)
    assert cli.main(["ask", "question"]) == 1
    assert "Model request error: model timed out" in capsys.readouterr().err
