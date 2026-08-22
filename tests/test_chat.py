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
