"""Model calls and the outbound proxy: [models] proxy, a system model's own, direct; a proxy on a
user's model is ignored (docs/configuration.md, System models)."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def proxy_server():
    seen = []

    class Proxy(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.path)            # a proxy gets the absolute URL
            self.rfile.read(int(self.headers.get("content-length", 0)))
            data = json.dumps({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", seen
    server.shutdown()


def test_a_model_call_goes_through_its_proxy_not_the_environments(proxy_server, monkeypatch):
    from agenticwf.kb.runtime.models import clients

    proxy, seen = proxy_server
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")      # the environment's: must not be used
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    spec = clients.ModelSpec(id="m", name="m", type="Embedding", base_url="http://brain.test/v1",
                             proxy=clients.outbound_proxy(proxy))
    data = asyncio.run(clients._send(spec, "/embeddings", {"model": "m", "input": ["x"]}, 5.0))
    assert data["data"][0]["embedding"] == [1.0, 0.0] and seen == ["http://brain.test/v1/embeddings"]


def test_which_proxy_a_model_gets(monkeypatch):
    from agenticwf.kb.runtime.models.clients import httpx_proxy, outbound_proxy
    from agenticwf.kb.runtime.services import models

    configured = {"proxy": "proxy.example.org:3128"}
    monkeypatch.setattr(models, "config_has_option", lambda s, o: s == "models" and o in configured)
    monkeypatch.setattr(models, "config_get", lambda s, o: configured[o])
    assert models.model_proxy(True, {}) == "http://proxy.example.org:3128"           # [models] proxy
    assert models.model_proxy(True, {"proxy": "direct"}) == "direct"                 # a system model's own
    assert models.model_proxy(False, {"proxy": "http://evil:8080"}) == "http://proxy.example.org:3128"
    configured.clear()
    assert models.model_proxy(True, {}) is None                                      # unset: the environment's
    assert outbound_proxy("off") == "direct" and outbound_proxy("  ") is None

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    assert httpx_proxy(None) == {} and httpx_proxy("direct") == {"trust_env": False}
    assert httpx_proxy("http://p:1") == {"proxy": "http://p:1", "trust_env": False}
    assert models._parameters_from_section({"base_url": "u", "proxy": "direct"})["proxy"] == "direct"
