from weft.memory import Episode
from weft.memory_infer import infer_candidates
from weft.proposals import proposal_id


def _ep(q):
    return Episode(id="ep_x", ts="2026-08-19T00:00:00Z", question=q, answer="a", sources=[])


def test_heuristic_proposes_recurring_terms(tmp_path):
    eps = [_ep("how does retrieval work"), _ep("retrieval tuning"), _ep("improve retrieval")]
    cands = infer_candidates(eps, existing_texts=set(), seen_ids=set(), limit=10)
    texts = [c.text for c in cands]
    assert "Recurring interest: retrieval" in texts
    assert all(c.type == "fact" and c.source == "heuristic" for c in cands)


def test_heuristic_ignores_rare_terms(tmp_path):
    eps = [_ep("retrieval one"), _ep("retrieval two"), _ep("retrieval three")]
    cands = infer_candidates(eps, existing_texts=set(), seen_ids=set(), limit=10)
    assert [c.text for c in cands] == ["Recurring interest: retrieval"]


def test_dedup_against_existing_and_seen(tmp_path):
    eps = [_ep("retrieval a"), _ep("retrieval b"), _ep("retrieval c")]
    out = infer_candidates(eps, existing_texts={"Recurring interest: retrieval"},
                           seen_ids=set(), limit=10)
    assert out == []
    out2 = infer_candidates(eps, existing_texts=set(),
                            seen_ids={proposal_id("Recurring interest: retrieval")}, limit=10)
    assert out2 == []


def test_limit_caps_candidates(tmp_path):
    eps = []
    for term in ("alpha", "bravo", "charlie"):
        eps += [_ep(f"{term} q1"), _ep(f"{term} q2"), _ep(f"{term} q3")]
    cands = infer_candidates(eps, existing_texts=set(), seen_ids=set(), limit=2)
    assert len(cands) == 2


def test_llm_path_used_and_falls_back(tmp_path):
    class OKLLM:
        def complete(self, system, prompt):
            return "preference: Answer concisely\nfact: Works on Weft"

    eps = [_ep("anything")]
    cands = infer_candidates(eps, existing_texts=set(), seen_ids=set(), limit=10, llm=OKLLM())
    assert {"Answer concisely", "Works on Weft"} <= {c.text for c in cands}

    class BoomLLM:
        def complete(self, system, prompt):
            raise RuntimeError("no provider")

    eps2 = [_ep("retrieval a"), _ep("retrieval b"), _ep("retrieval c")]
    fb = infer_candidates(eps2, existing_texts=set(), seen_ids=set(), limit=10, llm=BoomLLM())
    assert "Recurring interest: retrieval" in {c.text for c in fb}
