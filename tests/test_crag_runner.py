import bz2
import json
import stat

import pytest

from weft.benchmarks.crag import CRAGModel
from weft.benchmarks.crag_runner import local_judge, run_crag
from weft.embeddings import FakeEmbedder


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        return self.responses.pop(0)


def _write_dataset(path, items):
    with bz2.open(path, "wt", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


def _item(identifier, answer):
    return {
        "interaction_id": identifier,
        "query": "What is the answer?",
        "search_results": [{"page_name": "Fact", "page_snippet": f"The answer is {answer}."}],
        "query_time": "2024-01-01",
        "answer": answer,
        "alt_ans": [],
    }


def test_run_crag_scores_locally_and_writes_private_results(tmp_path):
    dataset = tmp_path / "crag.jsonl.bz2"
    output = tmp_path / "results.jsonl"
    _write_dataset(dataset, [_item("one", "42"), _item("two", "blue")])
    llm = QueueLLM(["42", "I don't know"])
    model = CRAGModel(embedder=FakeEmbedder(), llm=llm, batch_size=1)

    result = run_crag(dataset, model, output_path=output)

    assert result["score"] == 0.5
    assert result["n_correct"] == 1 and result["n_missing"] == 1
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["status"] for record in records] == ["correct", "missing"]
    assert "search_results" not in records[0]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_local_judge_accepts_json_wrapped_by_model_text():
    llm = QueueLLM(['Result: {"score": 1}'])
    assert local_judge("q", ["ground truth"], "equivalent", llm) is True


def test_run_crag_generation_only_omits_score(tmp_path):
    dataset = tmp_path / "crag.jsonl"
    output = tmp_path / "results.jsonl"
    dataset.write_text(json.dumps(_item("one", "42")) + "\n")
    model = CRAGModel(embedder=FakeEmbedder(), llm=QueueLLM(["answer"]))

    result = run_crag(dataset, model, output_path=output, judge=False)

    assert result["judge"] is None and "score" not in result
    assert json.loads(output.read_text())["status"] is None


def test_run_crag_rejects_empty_dataset(tmp_path):
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("")
    model = CRAGModel(embedder=FakeEmbedder(), llm=QueueLLM([]))
    with pytest.raises(ValueError, match="empty"):
        run_crag(dataset, model, output_path=tmp_path / "results.jsonl")
