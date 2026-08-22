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
