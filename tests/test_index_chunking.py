import json

from weft.embeddings import FakeEmbedder
from weft.index import build_index, manifest_path_for
from weft.store import VectorStore


def _vault(tmp_path):
    (tmp_path / "n.md").write_text("# A\npara one\n\npara two\n", encoding="utf-8")
    return tmp_path


def test_parent_child_persists_parent_fields(tmp_path):
    vault = _vault(tmp_path)
    store_path = tmp_path / "idx"
    build_index(vault, FakeEmbedder(dim=16), store_path, chunking="parent_child")
    store = VectorStore.load(store_path)
    metas = store.metadata_by_note()["n.md"]
    assert [m["text"] for m in metas] == ["para one", "para two"]
    assert all(m["parent_id"] == metas[0]["parent_id"] for m in metas)
    assert all("para one" in m["parent_text"] for m in metas)
    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["chunking"] == "parent_child"


def test_default_heading_omits_parent_fields(tmp_path):
    vault = _vault(tmp_path)
    store_path = tmp_path / "idx"
    build_index(vault, FakeEmbedder(dim=16), store_path)
    metas = VectorStore.load(store_path).metadata_by_note()["n.md"]
    assert "parent_id" not in metas[0]
    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["chunking"] == "heading"


def test_build_index_contextual_embeds_context_but_stores_raw(tmp_path):
    from weft.llm import FakeLLM
    vault = _vault(tmp_path)
    store_path = tmp_path / "idx"
    build_index(vault, FakeEmbedder(dim=16), store_path,
                contextual_llm=FakeLLM(response="Section A context."))
    metas = VectorStore.load(store_path).metadata_by_note()["n.md"]
    assert metas[0]["text"] == "para one\n\npara two"
    assert "embed_text" not in metas[0]
    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["contextual"] is True


def test_build_index_default_not_contextual(tmp_path):
    vault = _vault(tmp_path)
    store_path = tmp_path / "idx"
    build_index(vault, FakeEmbedder(dim=16), store_path)
    manifest = json.loads(manifest_path_for(store_path).read_text())
    assert manifest["contextual"] is False
