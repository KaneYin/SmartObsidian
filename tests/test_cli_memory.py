from weft.cli import main


def test_remember_writes_item(tmp_path, capsys):
    store = tmp_path / ".weft" / "index"
    rc = main(["remember", "Prefer LanceDB", "--type", "decision", "--store", str(store)])
    assert rc == 0
    assert (tmp_path / ".weft" / "memory.jsonl").exists()
    out = capsys.readouterr().out
    assert "decision" in out and "mem_" in out


def test_remember_rejects_bad_type(tmp_path):
    store = tmp_path / ".weft" / "index"
    rc = main(["remember", "x", "--type", "bogus", "--store", str(store)])
    assert rc == 2


def test_remember_rejects_overlong_text(tmp_path):
    store = tmp_path / ".weft" / "index"
    rc = main(["remember", "x" * 3000, "--store", str(store)])
    assert rc == 2


def test_memory_list_and_forget(tmp_path, capsys):
    store = tmp_path / ".weft" / "index"
    main(["remember", "Answer concisely", "--type", "preference", "--store", str(store)])
    capsys.readouterr()  # discard the remember output so we parse only `list`
    assert main(["memory", "list", "--store", str(store)]) == 0
    out = capsys.readouterr().out
    assert "Answer concisely" in out
    ident = [tok for tok in out.split() if tok.startswith("mem_")][0]
    assert main(["memory", "forget", ident, "--store", str(store)]) == 0
    assert main(["memory", "list", "--store", str(store)]) == 0
    assert "Answer concisely" not in capsys.readouterr().out
