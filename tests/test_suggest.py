import numpy as np

from weft.graph import LinkGraph
from weft.index import OVERVIEW_REL_PATH, build_index, graph_path_for
from weft.parser import Note
from weft.store import VectorStore
from weft.suggest import (
    LinkSuggestion,
    infer_links,
    note_vectors,
    pair_id,
)


class _ConstantEmbedder:
    """Every chunk -- real or synthetic -- gets the same unit vector, so every
    note pair (including the synthetic overview chunk) is cosine 1.0. This
    forces the vault-overview leak to manifest deterministically if it isn't
    filtered, unlike FakeEmbedder's hash-based vectors which only correlate
    by chance."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed(self, texts):
        vec = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
        return np.tile(vec, (len(texts), 1))


def _note(rel_path, wikilinks=None):
    return Note(
        rel_path=rel_path,
        title=rel_path,
        frontmatter={},
        tags=[],
        wikilinks=wikilinks or [],
        body="",
    )


def _store_with(vectors_by_note, tags=None, heading="H", dim=2):
    """Build a store: {rel_path: [chunk vectors]} plus optional per-note tags."""
    tags = tags or {}
    store = VectorStore(dim=dim)
    for rel_path, vecs in vectors_by_note.items():
        for i, v in enumerate(vecs):
            store.add(
                np.asarray(v, dtype=np.float32),
                {
                    "rel_path": rel_path,
                    "heading": heading,
                    "text": f"{rel_path} chunk {i}",
                    "ordinal": i,
                    "tags": tags.get(rel_path, []),
                    "wikilinks": [],
                },
            )
    return store


def test_note_vectors_mean_pools_and_normalizes():
    store = _store_with({"a.md": [[1.0, 0.0], [0.0, 1.0]]})
    vecs = note_vectors(store)
    assert set(vecs) == {"a.md"}
    # mean of [1,0] and [0,1] is [0.5,0.5] -> normalized [0.7071, 0.7071]
    np.testing.assert_allclose(vecs["a.md"], [0.70710677, 0.70710677], rtol=1e-5)
    assert np.isclose(np.linalg.norm(vecs["a.md"]), 1.0)


def test_infer_links_proposes_high_cosine_unlinked_pairs():
    store = _store_with(
        {"a.md": [[1.0, 0.0]], "b.md": [[1.0, 0.0]], "c.md": [[0.0, 1.0]]}
    )
    graph = LinkGraph.from_notes([_note("a.md"), _note("b.md"), _note("c.md")])

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert [(s.note_a, s.note_b) for s in out] == [("a.md", "b.md")]
    assert out[0].score > 0.99
    assert isinstance(out[0], LinkSuggestion)


def test_infer_links_canonical_order_and_single_suggestion_per_pair():
    store = _store_with({"z.md": [[1.0, 0.0]], "a.md": [[1.0, 0.0]]})
    graph = LinkGraph.from_notes([_note("z.md"), _note("a.md")])

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert len(out) == 1
    # canonical (sorted) order regardless of iteration order
    assert (out[0].note_a, out[0].note_b) == ("a.md", "z.md")


def test_infer_links_excludes_already_linked_pairs():
    store = _store_with({"a.md": [[1.0, 0.0]], "b.md": [[1.0, 0.0]]})
    # a.md links to b.md -> already a 1-hop edge, so no suggestion
    graph = LinkGraph.from_notes([_note("a.md", ["b"]), _note("b.md")])

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert out == []


def test_infer_links_excludes_below_threshold():
    store = _store_with({"a.md": [[1.0, 0.0]], "b.md": [[0.0, 1.0]]})
    graph = LinkGraph.from_notes([_note("a.md"), _note("b.md")])

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert out == []


def test_infer_links_excludes_seen_pairs():
    store = _store_with({"a.md": [[1.0, 0.0]], "b.md": [[1.0, 0.0]]})
    graph = LinkGraph.from_notes([_note("a.md"), _note("b.md")])

    out = infer_links(
        store, graph, threshold=0.8, limit=10, seen_pairs={("a.md", "b.md")}
    )

    assert out == []


def test_infer_links_ranks_by_score_and_caps_at_limit():
    store = _store_with(
        {
            "a.md": [[1.0, 0.0]],
            "b.md": [[1.0, 0.0]],       # cos(a,b) = 1.0
            "c.md": [[0.8, 0.6]],       # cos(a,c) = 0.8, cos(b,c) = 0.8
        }
    )
    graph = LinkGraph.from_notes([_note("a.md"), _note("b.md"), _note("c.md")])

    out = infer_links(store, graph, threshold=0.8, limit=2, seen_pairs=set())

    assert len(out) == 2
    assert (out[0].note_a, out[0].note_b) == ("a.md", "b.md")  # highest score first
    scores = [s.score for s in out]
    assert scores == sorted(scores, reverse=True)


def test_infer_links_reports_shared_tags():
    store = _store_with(
        {"a.md": [[1.0, 0.0]], "b.md": [[1.0, 0.0]]},
        tags={"a.md": ["coffee", "chem"], "b.md": ["coffee", "drinks"]},
    )
    graph = LinkGraph.from_notes([_note("a.md"), _note("b.md")])

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert out[0].shared_tags == ["coffee"]


def test_infer_links_fills_local_rationale():
    store = _store_with({"a.md": [[1.0, 0.0]], "b.md": [[1.0, 0.0]]})
    graph = LinkGraph.from_notes([_note("a.md"), _note("b.md")])

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert out[0].rationale  # non-empty
    assert "0.9" in out[0].rationale or "1.0" in out[0].rationale  # cosine mentioned


def test_pair_id_is_stable_and_canonical():
    assert pair_id("a.md", "b.md") == pair_id("b.md", "a.md")
    assert pair_id("a.md", "b.md") != pair_id("a.md", "c.md")
    assert len(pair_id("a.md", "b.md")) <= 12


def test_note_vectors_skips_notes_with_no_chunks():
    store = _store_with({"a.md": [[1.0, 0.0]]})
    vecs = note_vectors(store)
    assert "ghost.md" not in vecs


def test_note_vectors_excludes_synthetic_vault_overview_chunk():
    store = _store_with(
        {"a.md": [[1.0, 0.0]], "b.md": [[0.0, 1.0]], OVERVIEW_REL_PATH: [[1.0, 1.0]]}
    )
    vecs = note_vectors(store)
    assert set(vecs) == {"a.md", "b.md"}
    assert OVERVIEW_REL_PATH not in vecs


def test_infer_links_end_to_end_never_suggests_vault_overview(tmp_path):
    """Real pipeline: build_index() adds the synthetic overview chunk
    automatically. With every chunk forced to the same vector, the overview
    chunk is cosine 1.0 with every real note -- if it weren't filtered out
    of note_vectors(), it would appear in every suggestion."""
    (tmp_path / "alpha.md").write_text("# Alpha\n\nbody text\n")
    (tmp_path / "beta.md").write_text("# Beta\n\nbody text\n")
    store_path = tmp_path / "idx"
    build_index(tmp_path, _ConstantEmbedder(), store_path)

    store = VectorStore.load(store_path)
    rel_paths = {m["rel_path"] for m in store._metadata}
    assert OVERVIEW_REL_PATH in rel_paths  # sanity: overview was indexed

    graph = LinkGraph.load(graph_path_for(store_path))

    out = infer_links(store, graph, threshold=0.8, limit=10, seen_pairs=set())

    assert out  # the real alpha<->beta suggestion still fires
    for suggestion in out:
        assert OVERVIEW_REL_PATH not in (suggestion.note_a, suggestion.note_b)
