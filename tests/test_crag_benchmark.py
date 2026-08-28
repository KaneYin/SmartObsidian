import json

import pytest

from weft.benchmarks.crag import (
    CRAGModel,
    UserModel,
    _local_ollama,
    chunk_text,
    html_to_text,
    search_result_chunks,
)
from weft.embeddings import FakeEmbedder
from weft.llm import FakeLLM
from weft.providers import ProviderUnavailable


def _batch(search_results):
    return {
        "interaction_id": ["interaction-1"],
        "query": ["Who won?"],
        "search_results": [search_results],
        "query_time": ["2024-06-01 12:00:00"],
    }


def test_html_to_text_ignores_active_content():
    text = html_to_text(
        "<h1>Result</h1><script>steal()</script><style>.x{}</style><p>The answer is 42.</p>"
    )
    assert "Result" in text and "The answer is 42." in text
    assert "steal" not in text and ".x" not in text


def test_chunk_text_enforces_overlap_and_limit():
    chunks = chunk_text(
        " ".join(f"word{i}" for i in range(200)),
        chunk_chars=80,
        overlap=10,
        limit=3,
    )
    assert len(chunks) == 3
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_search_result_chunks_uses_snippet_when_html_is_empty():
    chunks = search_result_chunks(
        [{
            "page_name": "Page",
            "page_url": "https://example.test",
            "page_snippet": "Useful fact",
        }]
    )
    assert len(chunks) == 1
    assert "Useful fact" in chunks[0]["text"]
    assert chunks[0]["rel_path"] == "https://example.test"


def test_crag_model_matches_official_interface_and_builds_local_prompt():
    llm = FakeLLM(response="The home team won")
    model = CRAGModel(embedder=FakeEmbedder(), llm=llm, batch_size=1, k=3)
    answers = model.batch_generate_answer(_batch([
        {
            "page_name": "Match report",
            "page_url": "https://example.test/report",
            "page_snippet": "The home team won 3-1.",
            "page_result": "<p>The home team won the final.</p>",
            "page_last_modified": "2024-06-01",
        }
    ]))
    payload = json.loads(llm.last_prompt)
    assert UserModel is CRAGModel
    assert model.get_batch_size() == 1
    assert answers == ["The home team won"]
    assert payload["query_time"] == "2024-06-01 12:00:00"
    assert payload["question"] == "Who won?"
    assert payload["references"][0]["url"] == "https://example.test/report"
    assert "untrusted" in llm.last_system.lower()


def test_crag_model_skips_generation_without_evidence():
    llm = FakeLLM(response="hallucination")
    model = CRAGModel(embedder=FakeEmbedder(), llm=llm)
    assert model.batch_generate_answer(_batch([])) == ["I don't know"]
    assert llm.last_prompt is None


def test_crag_model_bounds_output():
    llm = FakeLLM(response=" ".join(["token"] * 100))
    model = CRAGModel(embedder=FakeEmbedder(), llm=llm)
    answer = model.batch_generate_answer(_batch([
        {"page_name": "Page", "page_snippet": "Evidence"}
    ]))[0]
    assert len(answer.split()) == 50
    assert len(answer) <= 300


def test_crag_model_validates_batch_shape_and_batch_size():
    with pytest.raises(ValueError, match="between 1 and 16"):
        CRAGModel(embedder=FakeEmbedder(), llm=FakeLLM(), batch_size=17)
    model = CRAGModel(embedder=FakeEmbedder(), llm=FakeLLM())
    with pytest.raises(ValueError, match="missing keys"):
        model.batch_generate_answer({"query": []})
    with pytest.raises(ValueError, match="equally sized"):
        model.batch_generate_answer({
            "interaction_id": ["x"], "query": ["q"],
            "search_results": [], "query_time": ["now"],
        })


def test_crag_default_provider_rejects_non_loopback_endpoint(tmp_path):
    with pytest.raises(ProviderUnavailable, match="loopback Ollama"):
        _local_ollama(
            tmp_path / "index",
            {"WEFT_ENDPOINT": "https://remote.example/api"},
        )


def test_crag_model_accepts_tuning_parameters():
    llm = FakeLLM(response="answer")
    model = CRAGModel(
        embedder=FakeEmbedder(),
        llm=llm,
        chunk_chars=400,
        overlap=50,
        rerank_pool=30,
        system_prompt="Custom system prompt",
    )
    assert model.chunk_chars == 400
    assert model.overlap == 50
    assert model.rerank_pool == 30
    assert model.system_prompt == "Custom system prompt"
    answer = model.batch_generate_answer(_batch([
        {"page_name": "Page", "page_snippet": "Sample text for chunking and testing"}
    ]))[0]
    assert answer == "answer"
    assert llm.last_system == "Custom system prompt"
