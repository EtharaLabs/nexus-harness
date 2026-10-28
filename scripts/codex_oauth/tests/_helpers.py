"""Importable test helpers for the codex_oauth suite (no fixtures here).

Kept separate from conftest.py so test modules can ``from _helpers import ...``
without relative-import/package gymnastics (the tests dir is put on sys.path by
conftest.py's shim)."""

from __future__ import annotations

import base64
import json
import time


def make_jwt(exp_delta_seconds: float, *, sub: str | None = None) -> str:
    """Build an unsigned JWT whose ``exp`` is now + ``exp_delta_seconds``.

    Only the payload's ``exp`` claim is read by credentials._decode_jwt_exp (it
    base64url-decodes the middle segment and never verifies the signature). A
    ``sub`` claim makes the token string unique per subject, mirroring how real
    accounts carry distinct access tokens (the bridge attributes upstream errors
    to a slot by exact token match)."""
    def _b64(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    claims = {"exp": int(time.time() + exp_delta_seconds), "client_id": "test"}
    if sub is not None:
        claims["sub"] = sub
    header = _b64({"alg": "none", "typ": "JWT"})
    payload = _b64(claims)
    return f"{header}.{payload}.sig"


def auth_json_blob(access_exp_delta: float = 600.0, *, refresh: str | None = "rt.fake",
                   account: str = "acct-fake-0001", auth_mode: str = "chatgpt") -> dict:
    """A well-formed ~/.codex/auth.json structure for tests."""
    tokens: dict = {"access_token": make_jwt(access_exp_delta, sub=account),
                    "account_id": account}
    if refresh is not None:
        tokens["refresh_token"] = refresh
    return {"auth_mode": auth_mode, "OPENAI_API_KEY": None, "tokens": tokens,
            "last_refresh": "2026-01-01T00:00:00Z"}


def sse_bytes(*events: dict) -> bytes:
    """Serialize a sequence of Responses-API events as an SSE byte stream."""
    out = b""
    for e in events:
        out += b"data: " + json.dumps(e).encode() + b"\n\n"
    return out
