from weft.embeddings import FakeEmbedder
from weft.memory import MemoryStore


def _store(tmp_path):
    return MemoryStore(tmp_path / "memory.jsonl", tmp_path / "episodes.jsonl")


def test_recall_ranks_relevant_decision_first(tmp_path):
    ms = _store(tmp_path)
    ms.remember("decision", "database choice: use LanceDB for vectors")
    ms.remember("task", "buy milk on the way home")
    emb = FakeEmbedder(dim=16)
    hits = ms.recall(emb, "database choice: use LanceDB for vectors", k=1,
                     kinds={"decision", "task"})
    assert len(hits) == 1
    assert hits[0].kind == "decision"
    assert "LanceDB" in hits[0].text


def test_recall_includes_episodes_when_log_requested(tmp_path):
    ms = _store(tmp_path)
    ms.log_episode("how do I configure X?", "set it in config.toml", ["x.md"])
    emb = FakeEmbedder(dim=16)
    hits = ms.recall(emb, "how do I configure X?", k=3, kinds={"log"})
    assert any(h.kind == "log" for h in hits)


def test_recall_empty_when_no_candidates(tmp_path):
    ms = _store(tmp_path)
    emb = FakeEmbedder(dim=16)
    assert ms.recall(emb, "anything", k=5, kinds={"decision"}) == []
