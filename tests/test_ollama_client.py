import json

import httpx

from weft.ollama_client import OllamaClient, list_models, ping


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_complete_sends_chat_and_returns_content():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hello from llama"}})

    oc = OllamaClient("http://localhost:11434", "llama3.1:8b", client=_client(handler))
    out = oc.complete(system="S", prompt="P")
    assert out == "hello from llama"
    assert seen["url"].endswith("/api/chat")
    assert seen["json"]["stream"] is False
    assert seen["json"]["messages"][0]["role"] == "system"
    assert oc.provider == "ollama" and oc.left_machine is False and oc.model == "llama3.1:8b"


def test_list_models_parses_tags():
    def handler(request):
        return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:3b"}]})

    assert list_models("http://localhost:11434", client=_client(handler)) == [
        "llama3.1:8b", "qwen2.5:3b",
    ]


def test_ping_false_on_connection_error():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    assert ping("http://localhost:11434", client=_client(handler)) is False
