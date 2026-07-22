import networkx as nx
import numpy as np

from weft.agent import graph_aware_retrieve, ask
from weft.graph import LinkGraph
from weft.llm import FakeLLM
from weft.store import VectorStore


class StubEmbedder:
    """Query embeds to a fixed direction so we control which chunk scores top."""
    dim = 2

    def __init__(self, vec):
        self._vec = np.asarray(vec, dtype=np.float32)

    def embed(self, texts):
        return np.tile(self._vec, (len(texts), 1)).astype(np.float32)


def _store_two_notes():
    # a.md points along x; b.md and c.md along y. Query along x -> a.md is top.
    store = VectorStore(dim=2)
    store.add(np.array([1.0, 0.0]), {"rel_path": "a.md", "heading": "A", "text": "alpha", "ordinal": 0})
    store.add(np.array([0.0, 1.0]), {"rel_path": "b.md", "heading": "B", "text": "beta", "ordinal": 0})
    store.add(np.array([0.0, 1.0]), {"rel_path": "c.md", "heading": "C", "text": "gamma", "ordinal": 0})
    return store


def _graph(edges, nodes=()):
    gx = nx.Graph()
    gx.add_nodes_from(nodes)
    gx.add_edges_from(edges)
    return LinkGraph(gx)


def test_graph_aware_pulls_in_linked_note_below_topk():
    store = _store_two_notes()
    g = _graph([("a.md", "b.md")], nodes=["a.md", "b.md", "c.md"])

    emb = StubEmbedder([1.0, 0.0])  # query aligns with a.md
    hits = graph_aware_retrieve("q", emb, store, g, k=1, neighbor_budget=5)
    paths = [h.metadata["rel_path"] for h in hits]
    assert paths[0] == "a.md"          # seed preserved, first
    assert "b.md" in paths             # linked neighbor pulled in
    assert "c.md" not in paths         # unlinked, equally-scored note excluded


def test_graph_aware_one_chunk_per_neighbor_note():
    store = VectorStore(dim=2)
    store.add(np.array([1.0, 0.0]), {"rel_path": "a.md", "heading": "A", "text": "seed", "ordinal": 0})
    store.add(np.array([0.0, 1.0]), {"rel_path": "b.md", "heading": "B1", "text": "b-one", "ordinal": 0})
    store.add(np.array([0.0, 1.0]), {"rel_path": "b.md", "heading": "B2", "text": "b-two", "ordinal": 1})
    g = _graph([("a.md", "b.md")])

    emb = StubEmbedder([1.0, 0.0])
    hits = graph_aware_retrieve("q", emb, store, g, k=1, neighbor_budget=5)
    b_hits = [h for h in hits if h.metadata["rel_path"] == "b.md"]
    assert len(b_hits) == 1  # only the best chunk from the neighbor note


def test_graph_aware_respects_neighbor_budget():
    store = VectorStore(dim=2)
    store.add(np.array([1.0, 0.0]), {"rel_path": "seed.md", "heading": "S", "text": "s", "ordinal": 0})
    edges = []
    for name in ["n1.md", "n2.md", "n3.md"]:
        store.add(np.array([0.0, 1.0]), {"rel_path": name, "heading": name, "text": name, "ordinal": 0})
        edges.append(("seed.md", name))
    g = _graph(edges)

    emb = StubEmbedder([1.0, 0.0])
    hits = graph_aware_retrieve("q", emb, store, g, k=1, neighbor_budget=2)
    neighbor_hits = [h for h in hits if h.metadata["rel_path"] != "seed.md"]
    assert len(neighbor_hits) == 2  # budget caps expansion


def test_ask_with_graph_lists_expanded_sources():
    store = _store_two_notes()
    g = _graph([("a.md", "b.md")], nodes=["a.md", "b.md", "c.md"])

    emb = StubEmbedder([1.0, 0.0])
    result = ask("q", emb, store, FakeLLM(response="ans [1][2]"), k=1, graph=g)
    assert "a.md" in result.sources
    assert "b.md" in result.sources  # graph expansion surfaced in citations
