from weft.cli import main


def test_models_list_shows_tiers_and_recommendation(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("weft.cli.ollama_installed", lambda endpoint: ["llama3.1:8b"])
    store = tmp_path / ".weft" / "index"
    assert main(["models", "list", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "llama3.1:8b" in out and "recommended" in out.lower()


def test_models_pull_requires_confirmation_declined(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("weft.cli.input", lambda prompt="": "n", raising=False)
    called = {"pulled": False}

    def fake_pull(endpoint, tag, on_line=None):
        called["pulled"] = True

    monkeypatch.setattr("weft.cli.ollama_pull", fake_pull)
    store = tmp_path / ".weft" / "index"
    rc = main(["models", "pull", "qwen2.5:3b", "--store", str(store)])
    assert rc == 0 and called["pulled"] is False
    assert "ollama pull" in capsys.readouterr().out  # printed the manual command


def test_models_pull_yes_skips_prompt(tmp_path, monkeypatch):
    called = {"tag": None}

    def fake_pull(endpoint, tag, on_line=None):
        called["tag"] = tag

    monkeypatch.setattr("weft.cli.ollama_pull", fake_pull)
    store = tmp_path / ".weft" / "index"
    rc = main(["models", "pull", "qwen2.5:3b", "--yes", "--store", str(store)])
    assert rc == 0 and called["tag"] == "qwen2.5:3b"
