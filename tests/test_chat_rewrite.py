from weft.chat import ChatSession
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.store import VectorStore


def _store():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    store.add(emb.embed(["coffee drip"])[0],
              {"rel_path": "c.md", "heading": "H", "text": "coffee drip",
               "ordinal": 0, "tags": [], "wikilinks": []})
    return emb, store


def test_context_query_none_without_history():
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="a"))
    assert s._context_query("why?") is None


def test_context_query_heuristic_is_prior_user_turns():
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="a"))
    s.send("how is coffee made?")
    assert s._context_query("why?") == "how is coffee made?"


def _history():
    return [{"role": "user", "text": "how is coffee made?"},
            {"role": "assistant", "text": "a"}]


def test_context_query_llm_rewrite_and_fallback():
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="standalone coffee query"), rewrite_llm=True)
    s.history = _history()
    assert s._context_query("why?") == "standalone coffee query"

    class Boom(FakeLLM):
        def complete(self, system, prompt):
            raise RuntimeError("no provider")

    s2 = ChatSession(emb, store, Boom(), rewrite_llm=True)
    s2.history = _history()
    assert s2._context_query("why?") == "how is coffee made?"  # heuristic fallback


def test_send_with_history_still_answers():
    emb, store = _store()
    s = ChatSession(emb, store, FakeLLM(response="ans"))
    s.send("first")
    turn = s.send("why?")
    assert turn.answer == "ans"
    assert turn.sources == ["c.md"]


def test_chat_reranks_when_set():
    from weft.rerank import FakeReranker
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    for i, t in enumerate(["zebra stripes", "coffee drip", "zebra herd"]):
        store.add(emb.embed([t])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": t, "ordinal": i,
                   "tags": [], "wikilinks": []})
    s = ChatSession(emb, store, FakeLLM(response="a"), reranker=FakeReranker(), k=2)
    turn = s.send("zebra")
    assert set(turn.sources) == {"n0.md", "n2.md"}
