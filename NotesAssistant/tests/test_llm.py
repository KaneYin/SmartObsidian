from weft.llm import LLMClient, FakeLLM


def test_fake_llm_records_prompt_and_returns_canned():
    llm = FakeLLM(response="ANSWER")
    out = llm.complete(system="sys", prompt="hello?")
    assert out == "ANSWER"
    assert llm.last_system == "sys"
    assert llm.last_prompt == "hello?"


def test_fake_llm_satisfies_protocol():
    assert isinstance(FakeLLM(response="x"), LLMClient)


def test_fake_llm_echo_mode_includes_prompt():
    llm = FakeLLM()  # no canned response -> echoes prompt for assertion-friendliness
    out = llm.complete(system="s", prompt="the coffee question")
    assert "the coffee question" in out
