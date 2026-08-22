from weft.agent import ask, collect_memory
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.memory import MemoryStore
from weft.store import VectorStore


def _memory(tmp_path):
    return MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl")


def _store_with_chunk(emb):
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed(["coffee is brewed by dripping"])[0],
              {"rel_path": "brew.md", "heading": "Brewing", "text": "drip coffee",
               "ordinal": 0, "tags": [], "wikilinks": []})
    return store


def test_collect_memory_returns_durable_and_recalled(tmp_path):
    ms = _memory(tmp_path)
    ms.remember("preference", "Answer concisely")
    ms.remember("decision", "coffee: prefer pour-over method")
    emb = FakeEmbedder(dim=16)
    mem = collect_memory(ms, emb, "coffee: prefer pour-over method", k=3)
    assert {"type": "preference", "text": "Answer concisely"} in mem["durable"]
    assert any(r["kind"] == "decision" for r in mem["recalled"])


def test_ask_injects_memory_and_logs_episode(tmp_path):
    ms = _memory(tmp_path)
    ms.remember("preference", "Answer concisely")
    emb = FakeEmbedder(dim=16)
    store = _store_with_chunk(emb)
    llm = FakeLLM()  # echoes the prompt so we can assert injection
    result = ask("how is coffee made?", emb, store, llm, k=1, memory=ms)
    assert "Answer concisely" in llm.last_prompt
    assert '"memory"' in llm.last_prompt
    eps = ms.episodes()
    assert len(eps) == 1 and eps[0].question == "how is coffee made?"


def test_ask_without_memory_is_unchanged(tmp_path):
    emb = FakeEmbedder(dim=16)
    store = _store_with_chunk(emb)
    llm = FakeLLM(response="ans")
    result = ask("how is coffee made?", emb, store, llm, k=1)
    assert result.answer == "ans"


def test_build_prompt_includes_conversation():
    import json
    from weft.agent import build_prompt
    convo = [{"role": "user", "text": "first"}, {"role": "assistant", "text": "reply"}]
    prompt = build_prompt("why?", [], memory=None, conversation=convo)
    payload = json.loads(prompt)
    assert payload["conversation"] == convo
    assert payload["question"] == "why?"


def test_build_prompt_omits_conversation_when_empty():
    import json
    from weft.agent import build_prompt
    payload = json.loads(build_prompt("q", []))
    assert "conversation" not in payload
