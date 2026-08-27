from weft.agent import _hit_key, dual_query_retrieve, retrieve
from weft.embeddings import FakeEmbedder
from weft.fusion import reciprocal_rank_fusion
from weft.store import VectorStore


def _store():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    for i, text in enumerate(["alpha coffee", "beta tea", "gamma water", "delta juice"]):
        store.add(emb.embed([text])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": text, "ordinal": i,
                   "tags": [], "wikilinks": []})
    return emb, store


def test_dual_query_none_context_is_single():
    emb, store = _store()
    assert dual_query_retrieve("coffee", None, emb, store, k=3) == \
        retrieve("coffee", emb, store, k=3)


def test_dual_query_fuses_both():
    emb, store = _store()
    got = dual_query_retrieve("coffee", "tea", emb, store, k=3)
    expect = reciprocal_rank_fusion(
        [retrieve("coffee", emb, store, k=3), retrieve("tea", emb, store, k=3)],
        key=_hit_key,
    )[:3]
    assert [_hit_key(h) for h in got] == [_hit_key(h) for h in expect]
