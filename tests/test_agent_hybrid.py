from weft.agent import _hit_key, fused_retrieve, retrieve
from weft.bm25 import BM25Index
from weft.embeddings import FakeEmbedder
from weft.fusion import reciprocal_rank_fusion
from weft.store import SearchHit, VectorStore


def _store():
    emb = FakeEmbedder(dim=16)
    store = VectorStore(dim=emb.dim)
    texts = ["alpha coffee", "beta tea", "gamma water", "zebra delta"]
    for i, text in enumerate(texts):
        store.add(emb.embed([text])[0],
                  {"rel_path": f"n{i}.md", "heading": "H", "text": text, "ordinal": i,
                   "tags": [], "wikilinks": []})
    return emb, store, texts


def test_fused_none_bm25_is_vector():
    emb, store, _ = _store()
    assert fused_retrieve(["coffee"], emb, store, bm25=None, k=3) == \
        retrieve("coffee", emb, store, k=3)


def test_fused_with_bm25_fuses_both():
    emb, store, texts = _store()
    bm25 = BM25Index.build(texts)
    got = fused_retrieve(["coffee"], emb, store, bm25=bm25, k=3)
    vec = retrieve("coffee", emb, store, k=3)
    lex = [SearchHit(score=s, metadata=store.metadata_rows()[i])
           for i, s in bm25.search("coffee", 3)]
    expect = reciprocal_rank_fusion([vec, lex], key=_hit_key)[:3]
    assert [_hit_key(h) for h in got] == [_hit_key(h) for h in expect]


def test_bm25_surfaces_exact_term():
    emb, store, texts = _store()
    bm25 = BM25Index.build(texts)
    got = fused_retrieve(["zebra"], emb, store, bm25=bm25, k=4)
    assert any(h.metadata["rel_path"] == "n3.md" for h in got)
