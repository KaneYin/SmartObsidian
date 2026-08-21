import json

import httpx

from weft.openai_client import OpenAIClient, is_local_endpoint


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_is_local_endpoint():
    assert is_local_endpoint("http://localhost:1234/v1")
    assert is_local_endpoint("http://127.0.0.1:8000/v1")
    assert is_local_endpoint("http://my-box.local/v1")
    assert not is_local_endpoint("https://api.openai.com/v1")


def test_complete_posts_chat_completions_with_auth_and_whitelist():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    oc = OpenAIClient("https://api.openai.com/v1", "gpt-4o-mini", api_key="sk-x",
                      params={"temperature": 0.3, "num_ctx": 8192}, client=_client(handler))
    out = oc.complete(system="S", prompt="P")
    assert out == "hi"
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["auth"] == "Bearer sk-x"
    assert seen["body"]["temperature"] == 0.3
    assert "num_ctx" not in seen["body"]           # ollama-only key dropped
    assert seen["body"]["messages"][0]["role"] == "system"
    assert oc.provider == "openai" and oc.left_machine is True and oc.model == "gpt-4o-mini"


def test_local_endpoint_needs_no_auth_and_stays_on_machine():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "local"}}]})

    oc = OpenAIClient("http://localhost:1234/v1", "local-model", client=_client(handler))
    assert oc.complete(system="S", prompt="P") == "local"
    assert oc.left_machine is False
