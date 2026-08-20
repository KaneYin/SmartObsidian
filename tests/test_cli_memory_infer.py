from weft import cli
from weft.cli import main
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM


def _seed_episodes(store, monkeypatch):
    monkeypatch.setattr(cli, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: FakeLLM(response="ok"))
    ms = cli.make_memory(store)
    for q in ("how does retrieval work", "retrieval tuning", "improve retrieval"):
        ms.log_episode(q, "a", [])


def test_suggest_pending_accept_flow(tmp_path, monkeypatch, capsys):
    store = tmp_path / ".weft" / "index"
    _seed_episodes(store, monkeypatch)

    assert main(["memory", "suggest", "--store", str(store)]) == 0
    assert "proposal" in capsys.readouterr().out  # count printed; clears buffer

    assert main(["memory", "pending", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "Recurring interest: retrieval" in out
    pid = [tok for tok in out.split() if tok.startswith("prop_")][0]

    assert main(["memory", "accept", pid, "--store", str(store)]) == 0
    ms = cli.make_memory(store)
    items = ms.active_semantic()
    assert any(i.provenance == "inferred" and "retrieval" in i.text for i in items)


def test_reject_prevents_reproposal(tmp_path, monkeypatch, capsys):
    store = tmp_path / ".weft" / "index"
    _seed_episodes(store, monkeypatch)
    main(["memory", "suggest", "--store", str(store)])
    capsys.readouterr()  # clear buffer
    main(["memory", "pending", "--store", str(store)])
    pid = [tok for tok in capsys.readouterr().out.split() if tok.startswith("prop_")][0]
    assert main(["memory", "reject", pid, "--store", str(store)]) == 0
    assert main(["memory", "suggest", "--store", str(store)]) == 0
    assert main(["memory", "pending", "--store", str(store)]) == 0
    assert "Recurring interest: retrieval" not in capsys.readouterr().out


def test_mirror_writes_note(tmp_path, monkeypatch):
    store = tmp_path / ".weft" / "index"
    vault = tmp_path / "vault"
    vault.mkdir()
    _seed_episodes(store, monkeypatch)
    main(["memory", "suggest", "--store", str(store)])
    assert main(["memory", "mirror", "--vault", str(vault), "--store", str(store)]) == 0
    assert (vault / "_memory.md").exists()
