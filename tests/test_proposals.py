import os
import stat

from weft.proposals import Candidate, ProposalStore, proposal_id


def _store(tmp_path):
    return ProposalStore(tmp_path / "memory-proposals.jsonl")


def test_proposal_id_is_idempotent_over_normalized_text():
    assert proposal_id("Recurring interest: X") == proposal_id("recurring   interest: x")
    assert proposal_id("a").startswith("prop_")


def test_add_skips_known_signatures(tmp_path):
    ps = _store(tmp_path)
    added = ps.add([Candidate("fact", "Recurring interest: retrieval", "heuristic")])
    assert len(added) == 1 and added[0].status == "pending"
    again = ps.add([Candidate("fact", "Recurring interest: retrieval", "heuristic")])
    assert again == []
    assert [p.text for p in ps.pending()] == ["Recurring interest: retrieval"]


def test_mark_accepted_removes_from_pending_and_is_private(tmp_path):
    ps = _store(tmp_path)
    p = ps.add([Candidate("preference", "Answer concisely", "heuristic")])[0]
    ps.mark(p.id, "accepted")
    assert ps.pending() == []
    assert ps.get(p.id).status == "accepted"
    assert p.id in ps.known_ids()
    mode = stat.S_IMODE(os.stat(tmp_path / "memory-proposals.jsonl").st_mode)
    assert mode == 0o600


def test_rejected_signature_stays_known(tmp_path):
    ps = _store(tmp_path)
    p = ps.add([Candidate("fact", "Recurring interest: X", "heuristic")])[0]
    ps.mark(p.id, "rejected")
    assert ps.add([Candidate("fact", "Recurring interest: X", "heuristic")]) == []
