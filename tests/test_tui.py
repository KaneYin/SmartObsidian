from pathlib import Path

import weft.tui as tui
from weft.llm import LLMRequestError


def _reader(lines):
    iterator = iter(lines)

    def read(_prompt=""):
        try:
            return next(iterator)
        except StopIteration as exc:
            raise EOFError from exc

    return read


def test_tui_exit_is_dependency_free(tmp_path):
    output = []
    rc = tui.run_tui(
        store_path=tmp_path / ".weft" / "index",
        read=_reader(["6"]),
        emit=output.append,
    )
    assert rc == 0
    assert "Ask vault" in "\n".join(output)
    assert output[-1] == "Goodbye."


def test_bare_weft_launches_tui(monkeypatch):
    import weft.cli as cli
    monkeypatch.setattr(tui, "run_tui", lambda **kwargs: 17)
    assert cli.main([]) == 17


def test_tui_ask_calls_shared_service_with_current_mode(tmp_path, monkeypatch):
    seen = {}

    def fake_ask(store_path, question, *, mode):
        seen.update(store=Path(store_path), question=question, mode=mode)
        return {"answer": "Shared answer", "sources": ["decisions.md"]}

    monkeypatch.setattr(tui.service, "service_ask", fake_ask)
    output = []
    store = tmp_path / ".weft" / "index"
    rc = tui.run_tui(
        store_path=store,
        read=_reader(["1", "What did I decide?", "6"]),
        emit=output.append,
    )
    assert rc == 0
    assert seen == {"store": store, "question": "What did I decide?", "mode": "balanced"}
    assert "Shared answer" in output
    assert any("decisions.md" in line for line in output)


def test_tui_model_timeout_is_recoverable(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise LLMRequestError("model timed out")

    monkeypatch.setattr(tui.service, "service_ask", timeout)
    output = []
    rc = tui.run_tui(
        store_path=tmp_path / ".weft" / "index",
        read=_reader(["1", "question", "6"]),
        emit=output.append,
    )
    assert rc == 0
    assert any("Error: model timed out" in line for line in output)
    assert output[-1] == "Goodbye."


def test_tui_settings_persist_mode(tmp_path):
    output = []
    store = tmp_path / ".weft" / "index"
    rc = tui.run_tui(
        store_path=store,
        read=_reader(["5", "1", "best", "5", "6"]),
        emit=output.append,
    )
    assert rc == 0
    assert tui.service.service_config(store)["mode"] == "best"
    assert any("Mode: best" in block for block in output)
