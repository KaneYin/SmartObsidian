from weft.agent import ask, retrieve
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.rerank import FakeReranker
from weft.store import VectorStore


def _store():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    for i, t in enumerate(["zebra stripes here", "coffee beans", "random text", "zebra pattern"]):
        store.add(emb.embed([t])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": t, "ordinal": i,
                   "tags": [], "wikilinks": []})
    return emb, store


def test_ask_reranker_selects_overlap_docs():
    emb, store = _store()
    res = ask("zebra", emb, store, FakeLLM(response="a"), k=2, reranker=FakeReranker())
    assert set(res.sources) == {"n0.md", "n3.md"}


def test_ask_no_reranker_unchanged():
    emb, store = _store()
    res = ask("coffee", emb, store, FakeLLM(response="a"), k=2)
    vec = retrieve("coffee", emb, store, k=2)
    assert res.sources == list(dict.fromkeys(h.metadata["rel_path"] for h in vec))
