from weft.embeddings import FakeEmbedder
from weft.index import build_index, graph_path_for
from weft.graph import LinkGraph


def test_build_index_writes_graph_file(sample_vault, tmp_path):
    store_path = tmp_path / "index"
    n_chunks, n_edges = build_index(sample_vault, FakeEmbedder(dim=16), store_path)
    assert n_chunks > 0
    assert n_edges >= 1  # coffee<->water

    gpath = graph_path_for(store_path)
    assert gpath.exists()
    g = LinkGraph.load(gpath)
    assert g.has_edge("coffee.md", "water.md")
    assert g.edge_count() == n_edges


def test_graph_path_for_derives_sibling(tmp_path):
    assert graph_path_for(tmp_path / "index").name == "index.graph.json"
