"""Tests for codex_oauth.bridge — the FastAPI proxy.

All upstream traffic is mocked via httpx.MockTransport (installed by the
``patch_async_upstream`` fixture BEFORE build_app constructs its client), so no
real ChatGPT/Codex subscription token is ever spent.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from codex_oauth import bridge
from codex_oauth import translate as x
from codex_oauth.credentials import load_account_pool

from _helpers import auth_json_blob, sse_bytes


class FakeProvider:
    """Minimal single-account provider satisfying the bridge's contract."""

    def __init__(self, token: str = "tok-abcdef123456", account: str = "acct-xyz789abc"):
        self._token = token
        self.account_id = account

    def get_access_token(self) -> str:
        return self._token

    def get_token_and_account(self):
        return self._token, self.account_id


def _completed_sse(text: str = "hello", with_items: bool = True) -> bytes:
    """A valid Responses SSE stream: item.done(s) + completed (empty output)."""
    events = []
    if with_items:
        events.append({"type": "response.output_item.done", "item": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}})
    events.append({"type": "response.completed", "response": {
        "id": "resp_1", "status": "completed", "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}})
    return sse_bytes(*events)


def _make_client(patch_async_upstream, monkeypatch, *, secret: str | None = "s3cret",
                 handler=None, capture: dict | None = None):
    """Install a mock upstream + build the app + return a TestClient."""
    if secret is not None:
        monkeypatch.setenv("NEXUS_CODEX_BRIDGE_SECRET", secret)
    else:
        monkeypatch.delenv("NEXUS_CODEX_BRIDGE_SECRET", raising=False)

    def default_handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["body"] = request.content
            capture["headers"] = dict(request.headers)
            capture["url"] = str(request.url)
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_completed_sse())

    patch_async_upstream(handler or default_handler)
    app = bridge.build_app(provider=FakeProvider())
    return TestClient(app)


def _auth(secret: str = "s3cret") -> dict:
    return {"Authorization": f"Bearer {secret}"}


# --- health / quota ----------------------------------------------------------
def test_healthz_reports_token_and_account_prefix(patch_async_upstream, monkeypatch):
    client = _make_client(patch_async_upstream, monkeypatch)
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["token_prefix"].startswith("tok-abcdef12")
    assert body["account_prefix"].startswith("acct-xyz")


# --- auth gate ---------------------------------------------------------------
def test_responses_rejects_missing_or_wrong_secret(patch_async_upstream, monkeypatch):
    client = _make_client(patch_async_upstream, monkeypatch)
    # no key
    assert client.post("/responses", json={"input": "hi"}).status_code == 401
    # wrong key
    assert client.post("/responses", json={"input": "hi"},
                       headers={"Authorization": "Bearer nope"}).status_code == 401


def test_responses_accepts_secret_via_bearer_and_x_api_key(patch_async_upstream, monkeypatch):
    client = _make_client(patch_async_upstream, monkeypatch)
    assert client.post("/responses", json={"input": "hi"}, headers=_auth()).status_code == 200
    assert client.post("/responses", json={"input": "hi"},
                       headers={"x-api-key": "s3cret"}).status_code == 200


def test_unauthenticated_mode_allows_when_no_secret_set(patch_async_upstream, monkeypatch):
    client = _make_client(patch_async_upstream, monkeypatch, secret=None)
    assert client.post("/responses", json={"input": "hi"}).status_code == 200


# --- request normalization forwarded upstream --------------------------------
def test_strips_unsupported_params_before_forwarding(patch_async_upstream, monkeypatch):
    cap: dict = {}
    client = _make_client(patch_async_upstream, monkeypatch, capture=cap)
    client.post("/responses", headers=_auth(), json={
        "input": "hi", "temperature": 0.7, "top_p": 0.9, "seed": 5, "max_output_tokens": 10})
    fwd = json.loads(cap["body"])
    for banned in ("temperature", "top_p", "seed", "max_output_tokens"):
        assert banned not in fwd


def test_normalizes_dated_model_suffix(patch_async_upstream, monkeypatch):
    cap: dict = {}
    client = _make_client(patch_async_upstream, monkeypatch, capture=cap)
    client.post("/responses", headers=_auth(),
                json={"input": "hi", "model": "gpt-5.5-2026-04-23"})
    assert json.loads(cap["body"])["model"] == "gpt-5.5"


def test_coerces_string_input_to_list(patch_async_upstream, monkeypatch):
    cap: dict = {}
    client = _make_client(patch_async_upstream, monkeypatch, capture=cap)
    client.post("/responses", headers=_auth(), json={"input": "hello there"})
    fwd = json.loads(cap["body"])
    assert isinstance(fwd["input"], list)
    assert fwd["input"][0]["content"][0]["text"] == "hello there"


def test_forces_stream_true_and_store_false_upstream(patch_async_upstream, monkeypatch):
    cap: dict = {}
    client = _make_client(patch_async_upstream, monkeypatch, capture=cap)
    client.post("/responses", headers=_auth(), json={"input": "hi", "stream": False})
    fwd = json.loads(cap["body"])
    assert fwd["stream"] is True
    assert fwd["store"] is False


def test_injects_oauth_and_codex_headers_upstream(patch_async_upstream, monkeypatch):
    cap: dict = {}
    client = _make_client(patch_async_upstream, monkeypatch, capture=cap)
    client.post("/responses", headers=_auth(), json={"input": "hi"})
    h = cap["headers"]
    assert h["authorization"] == "Bearer tok-abcdef123456"
    assert h["chatgpt-account-id"] == "acct-xyz789abc"
    assert h["openai-beta"] == bridge.OAUTH_BETA
    assert h["originator"] == bridge.ORIGINATOR
    assert "session_id" in h
    assert h["accept"] == "text/event-stream"


# --- unary aggregation (output spliced from output_item.done) ----------------
def test_unary_responses_aggregates_output_from_item_done_events(patch_async_upstream, monkeypatch):
    client = _make_client(patch_async_upstream, monkeypatch)
    r = client.post("/responses", headers=_auth(), json={"input": "hi", "stream": False})
    assert r.status_code == 200
    final = r.json()
    # The completed event carried output=[]; it must be spliced from item.done.
    assert final["output"]
    assert final["output"][0]["content"][0]["text"] == "hello"


def test_unary_returns_502_when_no_terminal_event(patch_async_upstream, monkeypatch):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=sse_bytes({"type": "response.output_text.delta",
                                                 "delta": "partial"}))
    client = _make_client(patch_async_upstream, monkeypatch, handler=handler)
    r = client.post("/responses", headers=_auth(), json={"input": "hi", "stream": False})
    assert r.status_code == 502


# --- chat/completions translation --------------------------------------------
def test_chat_completions_unary_translates_both_ways(patch_async_upstream, monkeypatch):
    client = _make_client(patch_async_upstream, monkeypatch)
    r = client.post("/v1/chat/completions", headers=_auth(), json={
        "model": "openai/gpt-6.1-sol",
        "messages": [{"role": "user", "content": "hi"}], "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"


def _delta_sse() -> bytes:
    return sse_bytes(
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {"type": "response.completed", "response": {
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}},
    )


def test_chat_completions_streaming_translates_through_testclient(patch_async_upstream, monkeypatch):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_delta_sse())
    client = _make_client(patch_async_upstream, monkeypatch, handler=handler)
    with client.stream("POST", "/v1/chat/completions", headers=_auth(), json={
            "model": "openai/gpt-6.1-sol",
            "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())
    assert b"chat.completion.chunk" in body
    assert b"Hel" in body and b"lo" in body
    assert body.rstrip().endswith(b"[DONE]")


# --- multi-account in-request failover ---------------------------------------
def _two_account_pool(tmp_path):
    import json
    a = tmp_path / "a.json"
    a.write_text(json.dumps(auth_json_blob(account="acctA", refresh="rt.a")))
    b = tmp_path / "b.json"
    b.write_text(json.dumps(auth_json_blob(account="acctB", refresh="rt.b")))
    return load_account_pool(f"{a}:{b}")


def test_multi_account_failover_rotates_on_429_then_succeeds(
        patch_async_upstream, monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEXUS_CODEX_BRIDGE_SECRET", "s3cret")
    pool = _two_account_pool(tmp_path)

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        acct = request.headers.get("chatgpt-account-id")
        seen.append(acct)
        if acct == "acctA":
            return httpx.Response(429, json={"error": {"message": "Too many requests"}})
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              content=_completed_sse())

    patch_async_upstream(handler)
    client = TestClient(bridge.build_app(provider=pool))
    r = client.post("/responses", headers=_auth(), json={"input": "hi", "stream": False})
    assert r.status_code == 200
    assert "acctA" in seen and "acctB" in seen
    assert r.json()["output"][0]["content"][0]["text"] == "hello"


def test_multi_account_failover_surfaces_error_when_all_capped(
        patch_async_upstream, monkeypatch, tmp_path):
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEXUS_CODEX_BRIDGE_SECRET", "s3cret")
    pool = _two_account_pool(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "Too many requests"}})

    patch_async_upstream(handler)
    client = TestClient(bridge.build_app(provider=pool))
    r = client.post("/responses", headers=_auth(), json={"input": "hi", "stream": False})
    assert r.status_code == 429


# --- upstream error passthrough ----------------------------------------------
def test_upstream_400_is_returned_to_client(patch_async_upstream, monkeypatch):
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter"}})
    client = _make_client(patch_async_upstream, monkeypatch, handler=handler)
    r = client.post("/responses", headers=_auth(), json={"input": "hi", "stream": False})
    assert r.status_code == 400


# --- _stream_with_keepalive internals (direct async drive) -------------------
def _drive(agen):
    async def _collect():
        return [chunk async for chunk in agen]
    return asyncio.run(_collect())


def test_stream_with_keepalive_injects_failed_event_on_close_without_terminal():
    async def chunks():
        yield b'data: {"type":"response.created"}\n\n'  # no terminal event
    closed = {"v": False}

    async def aclose():
        closed["v"] = True

    out = b"".join(_drive(bridge._stream_with_keepalive(chunks(), aclose, interval=0.0)))
    assert b"response.failed" in out            # truncation guard fired
    assert closed["v"] is True                   # upstream always closed


def test_stream_with_keepalive_passes_terminal_stream_through_cleanly():
    async def chunks():
        yield b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'
        yield b'data: {"type":"response.completed","response":{}}\n\n'

    async def aclose():
        return None

    out = b"".join(_drive(bridge._stream_with_keepalive(chunks(), aclose, interval=0.0)))
    assert b"response.failed" not in out
    assert b"response.completed" in out


def test_stream_with_keepalive_emits_keepalive_when_upstream_idles():
    async def chunks():
        await asyncio.sleep(0.08)               # idle past the interval
        yield b'data: {"type":"response.completed","response":{}}\n\n'

    async def aclose():
        return None

    out = b"".join(_drive(bridge._stream_with_keepalive(chunks(), aclose, interval=0.02)))
    assert bridge._KEEPALIVE_LINE in out        # kept the client connection warm
    assert b"response.completed" in out
