"""Shared fixtures for the codex_oauth bridge test suite.

This package lives at ``harness/scripts/codex_oauth`` and is NOT part of the
installed ``harbor`` package, so the sys.path shim below makes ``import
codex_oauth`` (and ``import _helpers``) resolve regardless of how pytest is
invoked. No real network / no real ChatGPT subscription is ever touched:
refreshes go through a monkeypatched ``httpx.post`` and the upstream Codex
backend through an ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import pytest

# --- sys.path shim (must run before any _helpers/codex_oauth import) ---------
_TESTS_DIR = Path(__file__).resolve().parent          # .../codex_oauth/tests
_SCRIPTS_DIR = _TESTS_DIR.parents[1]                  # .../scripts
for _p in (str(_SCRIPTS_DIR), str(_TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _helpers import auth_json_blob, make_jwt  # noqa: E402


# --- fixtures ----------------------------------------------------------------
@pytest.fixture
def jwt_factory():
    return make_jwt


@pytest.fixture
def fake_auth_path(tmp_path, monkeypatch) -> Path:
    """Write a valid chatgpt-mode auth.json and point the loader at it via
    NEXUS_CODEX_AUTH_PATH; clear any inline override so file precedence applies."""
    p = tmp_path / "auth.json"
    p.write_text(json.dumps(auth_json_blob()))
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEXUS_CODEX_AUTH_PATH", str(p))
    return p


@pytest.fixture
def inline_creds(monkeypatch):
    """Set CODEX_CREDENTIALS inline. Returns a setter choosing the expiry."""
    def _set(access_exp_delta: float = 600.0, **kw) -> dict:
        blob = auth_json_blob(access_exp_delta, **kw)
        monkeypatch.setenv("CODEX_CREDENTIALS", json.dumps(blob))
        monkeypatch.delenv("NEXUS_CODEX_AUTH_PATH", raising=False)
        return blob
    return _set


@pytest.fixture
def patch_refresh(monkeypatch):
    """Monkeypatch credentials.refresh_credentials with a call-counting stub."""
    from codex_oauth import credentials as _creds

    class _Stub:
        def __init__(self) -> None:
            self.calls = 0
            self.new_access = make_jwt(600)
            self.delay = 0.0

        def __call__(self, creds):
            self.calls += 1
            if self.delay:
                time.sleep(self.delay)
            return _creds.CodexCredentials(
                access_token=self.new_access,
                account_id=creds.account_id,
                refresh_token=creds.refresh_token,
                expires_at=_creds._decode_jwt_exp(self.new_access),
            )

    stub = _Stub()
    monkeypatch.setattr(_creds, "refresh_credentials", stub)
    return stub


@pytest.fixture
def patch_oauth_post(monkeypatch):
    """Monkeypatch the module-level httpx.post used by refresh_credentials."""
    from codex_oauth import credentials as _creds

    captured: dict = {}

    def _install(handler):
        def _post(url, json=None, timeout=None):  # noqa: A002 - mirror signature
            captured["url"] = url
            captured["json"] = json
            return handler(url, json)
        monkeypatch.setattr(_creds.httpx, "post", _post)
        return captured

    return _install


@pytest.fixture
def patch_async_upstream(monkeypatch):
    """Patch httpx.AsyncClient so bridge.build_app's client routes to a
    MockTransport handler. ``handler(request) -> httpx.Response``."""
    real_async_client = httpx.AsyncClient

    def _install(handler):
        def _factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(*args, **kwargs)
        monkeypatch.setattr(httpx, "AsyncClient", _factory)

    return _install
