from weft.chat import ChatSession
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.memory import MemoryStore
from weft.store import VectorStore


def _store(dim=16):
    emb = FakeEmbedder(dim=dim)
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed(["coffee drip"])[0],
              {"rel_path": "c.md", "heading": "H", "text": "coffee drip",
               "ordinal": 0, "tags": [], "wikilinks": []})
    return emb, store


def _memory(tmp_path):
    return MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl")


def test_history_carried_to_next_turn():
    emb, store = _store()
    llm = FakeLLM()  # echoes the prompt
    s = ChatSession(emb, store, llm)
    s.send("first question about x")
    s.send("second")
    assert "first question about x" in llm.last_prompt


def test_window_trims():
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="a"), window=2)
    for i in range(5):
        s.send(f"q{i}")
    assert len(s.history) <= 4  # 2 turn-pairs


def test_episode_logged_per_turn(tmp_path):
    emb, store = _store()
    memory = _memory(tmp_path)
    s = ChatSession(emb, store, FakeLLM(response="a"), memory=memory)
    s.send("q1")
    s.send("q2")
    assert len(memory.episodes()) == 2
    assert s.last_sources == ["c.md"]


def test_remember_writes_memory(tmp_path):
    emb, store = _store()
    memory = _memory(tmp_path)
    s = ChatSession(emb, store, FakeLLM(response="a"), memory=memory)
    item = s.remember("preference", "concise")
    assert item.id.startswith("mem_")
    assert any(i.text == "concise" for i in memory.active_semantic())


def test_empty_store_still_answers():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    s = ChatSession(emb, store, FakeLLM(response="hi"))
    assert s.send("hello").answer == "hi"


def test_reset_clears_history():
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="a"))
    s.send("q1")
    s.reset()
    assert s.history == []


def _reader(lines):
    it = iter(lines)

    def read(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return read


def test_run_repl_full_flow(tmp_path):
    from weft.chat import run_repl
    emb, store = _store()
    memory = _memory(tmp_path)
    s = ChatSession(emb, store, FakeLLM(response="an answer"), memory=memory)
    out = []
    rc = run_repl(
        s,
        read=_reader(["hello", "/remember --type preference concise", "/reset", "/exit"]),
        emit=out.append,
    )
    assert rc == 0
    assert "an answer" in "\n".join(out)
    assert any(i.text == "concise" for i in memory.active_semantic())
    assert s.history == []


def test_run_repl_unknown_command(tmp_path):
    from weft.chat import run_repl
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="a"))
    out = []
    run_repl(s, read=_reader(["/bogus", "/exit"]), emit=out.append)
    assert any("unknown command" in line for line in out)
