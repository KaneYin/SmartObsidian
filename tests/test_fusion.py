from weft.fusion import reciprocal_rank_fusion
from weft.store import SearchHit


def _h(rel, ordinal):
    return SearchHit(score=0.0, metadata={"rel_path": rel, "ordinal": ordinal})


def _key(h):
    return (h.metadata["rel_path"], h.metadata["ordinal"])


def test_rrf_agreement_wins():
    a, b, c = _h("a", 0), _h("b", 0), _h("c", 0)
    fused = reciprocal_rank_fusion([[a, b], [a, c]], key=_key)
    assert _key(fused[0]) == ("a", 0)
    assert {_key(h) for h in fused} == {("a", 0), ("b", 0), ("c", 0)}


def test_rrf_single_passthrough():
    a, b = _h("a", 0), _h("b", 0)
    fused = reciprocal_rank_fusion([[a, b]], key=_key)
    assert [_key(h) for h in fused] == [("a", 0), ("b", 0)]


def test_rrf_dedups_by_key():
    a1, a2, b = _h("a", 0), _h("a", 0), _h("b", 0)
    fused = reciprocal_rank_fusion([[a1], [a2, b]], key=_key)
    assert [_key(h) for h in fused].count(("a", 0)) == 1


def test_rrf_three_rankings():
    a, b = _h("a", 0), _h("b", 0)
    fused = reciprocal_rank_fusion([[a], [a], [b]], key=_key)
    assert _key(fused[0]) == ("a", 0)
