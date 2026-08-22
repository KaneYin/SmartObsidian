import http.client
import json
import os
import stat
import threading

import pytest

from weft.api import WeftHTTPServer, load_or_create_token


@pytest.fixture
def server(tmp_path):
    store = tmp_path / ".weft" / "index"
    store.parent.mkdir(parents=True, exist_ok=True)
    from weft.config import ResolvedConfig, config_path_for, save_config
    save_config(config_path_for(store), ResolvedConfig(provider="fake"))
    token = load_or_create_token(store)
    srv = WeftHTTPServer(("127.0.0.1", 0), store, token)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, token, srv.server_address[1], store
    srv.shutdown()
    srv.server_close()


def _conn(port):
    return http.client.HTTPConnection("127.0.0.1", port, timeout=5)


def _get(port, path, headers):
    c = _conn(port)
    c.request("GET", path, headers=headers)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, body


def test_token_file_is_private(server):
    _srv, _token, _port, store = server
    mode = stat.S_IMODE(os.stat(store.parent / "api-token").st_mode)
    assert mode == 0o600


def test_missing_token_is_401(server):
    _srv, _token, port, _store = server
    status, _ = _get(port, "/health", {"Host": "127.0.0.1"})
    assert status == 401


def test_bad_host_is_403(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/health",
                     {"Host": "evil.com", "Authorization": f"Bearer {token}"})
    assert status == 403


def test_cross_site_origin_is_403(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/health",
                     {"Host": "127.0.0.1", "Origin": "https://evil.com",
                      "Authorization": f"Bearer {token}"})
    assert status == 403


def test_health_ok_with_token(server):
    _srv, token, port, _store = server
    status, body = _get(port, "/health",
                        {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"})
    assert status == 200
    assert json.loads(body)["index"] is False


def test_unknown_path_404(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/nope",
                     {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"})
    assert status == 404


def _post(port, path, headers, obj):
    c = _conn(port)
    payload = json.dumps(obj)
    c.request("POST", path, body=payload,
              headers={**headers, "Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, body


def test_remember_then_list(server):
    _srv, token, port, _store = server
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"}
    status, body = _post(port, "/memory/remember", h, {"type": "fact", "text": "likes tea"})
    assert status == 200 and json.loads(body)["id"].startswith("mem_")
    status, body = _get(port, "/memory", h)
    assert status == 200
    assert any(i["text"] == "likes tea" for i in json.loads(body)["items"])


def test_remember_bad_type_is_400(server):
    _srv, token, port, _store = server
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"}
    status, _ = _post(port, "/memory/remember", h, {"type": "bogus", "text": "x"})
    assert status == 400


def test_oversized_body_is_400(server):
    _srv, token, port, _store = server
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}",
         "Content-Type": "application/json", "Content-Length": str(2_000_000)}
    c = _conn(port)
    c.request("POST", "/ask", body=b"{}", headers=h)
    r = c.getresponse()
    r.read()
    c.close()
    assert r.status == 400


def test_wrong_method_is_405(server):
    _srv, token, port, _store = server
    status, _ = _get(port, "/ask", {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"})
    assert status == 405


def test_ask_round_trip(server, monkeypatch):
    import weft.service as S
    from weft.embeddings import FakeEmbedder
    from weft.llm import FakeLLM
    from weft.store import VectorStore
    _srv, token, port, store = server
    emb = FakeEmbedder(dim=16)
    vs = VectorStore(dim=emb.dim)
    vs.add(emb.embed(["hello world"])[0],
           {"rel_path": "n.md", "heading": "H", "text": "hello world",
            "ordinal": 0, "tags": [], "wikilinks": []})
    vs.save(store)
    monkeypatch.setattr(S, "make_embedder", lambda: FakeEmbedder(dim=16))
    monkeypatch.setattr(S, "make_llm", lambda *a, **k: FakeLLM(response="hi [1]"))
    h = {"Host": "127.0.0.1", "Authorization": f"Bearer {token}"}
    status, body = _post(port, "/ask", h, {"question": "hello?", "k": 3})
    assert status == 200
    data = json.loads(body)
    assert data["answer"] == "hi [1]" and data["sources"] == ["n.md"]
