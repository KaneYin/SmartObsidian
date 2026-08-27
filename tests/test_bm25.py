from weft.bm25 import BM25Index, _tokenize


def test_tokenize():
    assert _tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_bm25_ranks_exact_term_first():
    idx = BM25Index.build(["the quick brown fox", "lazy dog sleeps", "zebra stripes here"])
    ranked = idx.search("zebra", k=3)
    assert ranked[0][0] == 2  # only doc 2 contains 'zebra'


def test_bm25_empty_index_and_query():
    assert BM25Index.build([]).search("x", 3) == []
    assert BM25Index.build(["a b c"]).search("", 3) == []
    assert BM25Index.build(["a b c"]).search("nomatch", 3) == []


def test_bm25_roundtrip(tmp_path):
    idx = BM25Index.build(["alpha beta", "gamma delta", "gamma gamma"])
    p = tmp_path / "index.bm25.json"
    idx.save(p)
    loaded = BM25Index.load(p)
    assert loaded.search("gamma", 3) == idx.search("gamma", 3)
