"""Tests for codex_oauth.credentials — loading, JWT exp decode, refresh (mocked),
atomic 0600 write-back, and the two-lock refresh-race / multi-account rotation."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest

from codex_oauth import credentials as creds
from codex_oauth.credentials import (
    CodexCredentials,
    CredentialProvider,
    CredentialsError,
    MultiAccountCredentialProvider,
    load_account_pool,
    load_credentials,
    refresh_credentials,
)

from _helpers import auth_json_blob, make_jwt


# --- load + JWT exp ----------------------------------------------------------
def test_load_credentials_decodes_jwt_exp(fake_auth_path):
    c = load_credentials()
    assert c.account_id == "acct-fake-0001"
    assert c.refresh_token == "rt.fake"
    assert c.expires_at is not None
    rem = c.seconds_remaining()
    assert 590 < rem <= 600
    assert c.is_expired() is False


def test_load_credentials_rejects_api_key_auth_mode(tmp_path, monkeypatch):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps(auth_json_blob(auth_mode="api_key")))
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEXUS_CODEX_AUTH_PATH", str(p))
    with pytest.raises(CredentialsError, match="ChatGPT-auth mode"):
        load_credentials()


def test_load_credentials_missing_tokens_raises(tmp_path, monkeypatch):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {}}))
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEXUS_CODEX_AUTH_PATH", str(p))
    with pytest.raises(CredentialsError, match="access_token"):
        load_credentials()


def test_inline_credentials_take_precedence_over_file(tmp_path, monkeypatch):
    # File says account A; inline env says account B -> inline wins.
    p = tmp_path / "auth.json"
    p.write_text(json.dumps(auth_json_blob(account="file-account")))
    monkeypatch.setenv("NEXUS_CODEX_AUTH_PATH", str(p))
    monkeypatch.setenv("CODEX_CREDENTIALS",
                       json.dumps(auth_json_blob(account="inline-account")))
    assert load_credentials().account_id == "inline-account"


def test_missing_everything_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    monkeypatch.setenv("NEXUS_CODEX_AUTH_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.codex/auth.json either
    with pytest.raises(CredentialsError, match="No Codex credentials"):
        load_credentials()


# --- refresh (mocked httpx) --------------------------------------------------
def test_refresh_credentials_posts_to_oauth_url(patch_oauth_post):
    new_access = make_jwt(900)

    def handler(url, payload):
        assert url == creds.OAUTH_TOKEN_URL
        assert payload["grant_type"] == "refresh_token"
        assert payload["refresh_token"] == "rt.old"
        assert payload["client_id"] == creds.DEFAULT_CLIENT_ID
        return httpx.Response(200, json={"access_token": new_access, "refresh_token": "rt.new"},
                              request=httpx.Request("POST", url))

    captured = patch_oauth_post(handler)
    old = CodexCredentials(access_token=make_jwt(-10), account_id="a", refresh_token="rt.old")
    out = refresh_credentials(old)
    assert out.access_token == new_access
    assert out.refresh_token == "rt.new"      # rotated refresh token kept
    assert out.account_id == "a"
    assert captured["url"] == creds.OAUTH_TOKEN_URL


def test_refresh_credentials_raises_on_non_200(patch_oauth_post):
    patch_oauth_post(lambda url, payload: httpx.Response(
        400, json={"error": "bad refresh"}, request=httpx.Request("POST", url)))
    old = CodexCredentials(access_token=make_jwt(-10), account_id="a", refresh_token="rt.x")
    with pytest.raises(CredentialsError, match="token refresh returned 400"):
        refresh_credentials(old)


def test_refresh_without_refresh_token_raises():
    old = CodexCredentials(access_token=make_jwt(-10), account_id="a", refresh_token=None)
    with pytest.raises(CredentialsError, match="no refresh_token"):
        refresh_credentials(old)


# --- atomic 0600 write-back --------------------------------------------------
def test_persist_writes_atomic_0600(fake_auth_path):
    provider = CredentialProvider()
    fresh = CodexCredentials(access_token=make_jwt(1200), account_id="acct-fake-0001",
                             refresh_token="rt.new2")
    provider._persist(fresh)
    # File exists, mode is 0600, no leftover .tmp, content round-trips.
    assert fake_auth_path.is_file()
    assert oct(fake_auth_path.stat().st_mode)[-3:] == "600"
    assert not fake_auth_path.with_suffix(".json.tmp").exists()
    data = json.loads(fake_auth_path.read_text())
    assert data["tokens"]["access_token"] == fresh.access_token
    assert data["tokens"]["refresh_token"] == "rt.new2"
    assert data["auth_mode"] == "chatgpt"


# --- two-lock refresh race ---------------------------------------------------
def test_get_access_token_dedupes_concurrent_refresh(inline_creds, patch_refresh):
    # Start with an already-expired token so every caller wants a refresh.
    inline_creds(access_exp_delta=-100)
    provider = CredentialProvider()
    patch_refresh.delay = 0.05  # widen the window so threads overlap

    results: list[str] = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        results.append(provider.get_access_token())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # _refresh_lock must dedupe: a single refresh serves all 20 racers.
    assert patch_refresh.calls == 1
    assert len(results) == 20
    assert set(results) == {patch_refresh.new_access}


def test_valid_token_is_not_refreshed(inline_creds, patch_refresh):
    inline_creds(access_exp_delta=600)  # still valid
    provider = CredentialProvider()
    tok = provider.get_access_token()
    assert patch_refresh.calls == 0
    assert tok  # returns the cached token unchanged


# --- multi-account pool ------------------------------------------------------
def _file_provider(tmp_path, name, exp=600) -> Path:
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(auth_json_blob(exp, account=name, refresh=f"rt.{name}")))
    return p


def test_load_account_pool_builds_multi_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    a = _file_provider(tmp_path, "acctA")
    b = _file_provider(tmp_path, "acctB")
    pool = load_account_pool(f"{a}:{b}")
    assert isinstance(pool, MultiAccountCredentialProvider)
    assert pool.account_id in ("acctA", "acctB")


def test_multi_account_rotates_on_penalize(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    a = _file_provider(tmp_path, "acctA")
    b = _file_provider(tmp_path, "acctB")
    pool = load_account_pool(f"{a}:{b}")
    first = pool.account_id
    pool.penalize(60.0)          # cool down the active slot + advance
    assert pool.account_id != first


def test_multi_account_mark_invalid_not_reselected_until_reload(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_CREDENTIALS", raising=False)
    a = _file_provider(tmp_path, "acctA")
    b = _file_provider(tmp_path, "acctB")
    pool = load_account_pool(f"{a}:{b}")
    tok, acct = pool.get_token_and_account()
    pool.mark_invalid(tok)
    # The invalidated account must not be handed out again.
    for _ in range(5):
        _, acct2 = pool.get_token_and_account()
        assert acct2 != acct
    # reload() clears the invalid flag (swapped-in fresh creds on disk).
    pool.reload()
    assert pool.next_reset_at() is None  # a healthy slot is usable now
