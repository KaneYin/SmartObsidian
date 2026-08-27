import builtins

import weft.cli as cli
import weft.service as service
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.store import VectorStore


def _index(store_path):
    store_path.parent.mkdir(parents=True, exist_ok=True)
    emb = FakeEmbedder(dim=16)
    vs = VectorStore(dim=emb.dim)
    vs.add(emb.embed(["coffee"])[0],
           {"rel_path": "c.md", "heading": "H", "text": "coffee",
            "ordinal": 0, "tags": [], "wikilinks": []})
    vs.save(store_path)


def _script(lines):
    it = iter(lines)

    def read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return read


def test_chat_missing_index_errors(tmp_path, capsys):
    rc = cli.main(["chat", "--store", str(tmp_path / "nope")])
    assert rc == 1
    assert "No index" in capsys.readouterr().err


def test_chat_happy_path(tmp_path, monkeypatch, capsys):
    store = tmp_path / ".weft" / "index"
    _index(store)
    monkeypatch.setattr(service, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(service, "make_llm", lambda *a, **k: FakeLLM(response="chat answer"))
    monkeypatch.setattr(builtins, "input", _script(["hello", "/exit"]))
    rc = cli.main(["chat", "--store", str(store)])
    assert rc == 0
    assert "chat answer" in capsys.readouterr().out


def test_chat_rewrite_llm_flag(tmp_path, monkeypatch, capsys):
    store = tmp_path / ".weft" / "index"
    _index(store)
    monkeypatch.setattr(service, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(service, "make_llm", lambda *a, **k: FakeLLM(response="chat answer"))
    monkeypatch.setattr(builtins, "input", _script(["hello", "why?", "/exit"]))
    rc = cli.main(["chat", "--store", str(store), "--rewrite-llm"])
    assert rc == 0
    assert "chat answer" in capsys.readouterr().out
