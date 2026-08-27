from weft.bm25 import BM25Index
from weft.embeddings import FakeEmbedder
from weft.index import bm25_path_for, build_index
from weft.store import VectorStore


def _vault(tmp_path):
    (tmp_path / "n.md").write_text("# A\nzebra alpha\n\nbeta gamma\n", encoding="utf-8")
    return tmp_path


def test_build_writes_bm25_by_default(tmp_path):
    store_path = tmp_path / "idx"
    build_index(_vault(tmp_path), FakeEmbedder(dim=16), store_path)
    path = bm25_path_for(store_path)
    assert path.exists()
    bm25 = BM25Index.load(path)
    store = VectorStore.load(store_path)
    assert len(bm25.docs) == len(store)
    assert bm25.search("zebra", 3)[0][0] == 0


def test_no_bm25_skips_file(tmp_path):
    store_path = tmp_path / "idx"
    build_index(_vault(tmp_path), FakeEmbedder(dim=16), store_path, no_bm25=True)
    assert not bm25_path_for(store_path).exists()
