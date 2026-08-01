from weft.agent import retrieve, reason, ask, build_prompt
from weft.embeddings import FakeEmbedder
from weft.index import build_index
from weft.llm import FakeLLM
from weft.store import VectorStore


def _indexed_store(sample_vault, tmp_path):
    build_index(sample_vault, FakeEmbedder(dim=16), tmp_path / "idx")
    return VectorStore.load(tmp_path / "idx")


def test_retrieve_returns_k_hits(sample_vault, tmp_path):
    store = _indexed_store(sample_vault, tmp_path)
    hits = retrieve("coffee", FakeEmbedder(dim=16), store, k=3)
    assert 1 <= len(hits) <= 3
    assert all({"rel_path", "heading", "text"} <= set(h.metadata) for h in hits)


def test_build_prompt_includes_numbered_sources_and_question():
    class H:
        def __init__(self, rp, hd, tx):
            self.metadata = {"rel_path": rp, "heading": hd, "text": tx}
    hits = [H("coffee.md", "Coffee", "Coffee is brewed."), H("water.md", "Water", "Water is essential.")]
    prompt = build_prompt("what is coffee?", hits)
    assert "[1]" in prompt and "[2]" in prompt
    assert "coffee.md" in prompt and "water.md" in prompt
    assert "what is coffee?" in prompt


def test_reason_passes_sources_to_llm_and_returns_answer():
    class H:
        def __init__(self, rp, hd, tx):
            self.metadata = {"rel_path": rp, "heading": hd, "text": tx}
    hits = [H("coffee.md", "Coffee", "Coffee is brewed from beans.")]
    llm = FakeLLM(response="Coffee is a brewed drink [1].")
    answer = reason("what is coffee?", hits, llm)
    assert answer == "Coffee is a brewed drink [1]."
    assert "coffee.md" in llm.last_prompt


def test_ask_end_to_end_with_fakes(sample_vault, tmp_path):
    store = _indexed_store(sample_vault, tmp_path)
    llm = FakeLLM(response="Answer citing [1].")
    result = ask("tell me about coffee", FakeEmbedder(dim=16), store, llm, k=3)
    assert result.answer == "Answer citing [1]."
    assert len(result.sources) >= 1
    assert result.sources[0].endswith(".md")
