import json

from weft.embeddings import FakeEmbedder
from weft.index import build_index, manifest_path_for
from weft.privacy import PrivacyPolicy
from weft.store import VectorStore


def test_build_index_populates_and_persists(sample_vault, tmp_path):
    store_path = tmp_path / "weft_index"
    n_chunks, n_edges = build_index(sample_vault, FakeEmbedder(dim=16), store_path)
    assert n_chunks > 0

    store = VectorStore.load(store_path)
    assert len(store) == n_chunks
    rel_paths = {m["rel_path"] for m in store._metadata}
    assert {"coffee.md", "water.md", "notes/tea.md"} <= rel_paths
    # every chunk carries the fields the agent needs to cite
    for m in store._metadata:
        assert {"rel_path", "heading", "text"} <= set(m)


def test_build_index_dim_matches_embedder(sample_vault, tmp_path):
    build_index(sample_vault, FakeEmbedder(dim=16), tmp_path / "idx")
    assert VectorStore.load(tmp_path / "idx").dim == 16


def test_build_index_applies_privacy_before_persisting(tmp_path):
    (tmp_path / "public.md").write_text("# Public\n\nkey sk-ant-secret")
    private = tmp_path / "Private"
    private.mkdir()
    (private / "health.md").write_text("# Health\n\nprivate diagnosis")
    store_path = tmp_path / "store" / "index"

    build_index(
        tmp_path,
        FakeEmbedder(dim=16),
        store_path,
        policy=PrivacyPolicy(redaction_patterns=(r"sk-ant-[A-Za-z-]+",)),
    )

    store = VectorStore.load(store_path)
    serialized = json.dumps(store._metadata)
    assert "health.md" not in serialized
    assert "private diagnosis" not in serialized
    assert "sk-ant-secret" not in serialized
    assert "[REDACTED]" in serialized

    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["vault_root"] == str(tmp_path.resolve())
    assert "Private" in manifest["privacy"]["excludes"]
    assert (manifest_path_for(store_path).stat().st_mode & 0o777) == 0o600
