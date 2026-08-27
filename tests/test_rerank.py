from weft.rerank import FakeReranker
from weft.store import SearchHit


def _h(text):
    return SearchHit(score=0.0, metadata={"rel_path": "n.md", "text": text, "ordinal": 0})


def test_fake_reranker_orders_by_overlap():
    hits = [_h("cats and dogs"), _h("quantum physics"), _h("zebra stripes pattern")]
    out = FakeReranker().rerank("zebra stripes", hits, 2)
    assert out[0].metadata["text"] == "zebra stripes pattern"
    assert len(out) == 2


def test_fake_reranker_empty():
    assert FakeReranker().rerank("q", [], 5) == []
