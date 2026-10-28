"""Tests for codex_oauth.errors — upstream status/body -> ErrorKind taxonomy."""

from __future__ import annotations

from codex_oauth.errors import (
    ErrorKind,
    classify_openai_error,
    extract_retry_after,
)


def test_classify_2xx_is_ok():
    c = classify_openai_error(200, b"{}")
    assert c.kind is ErrorKind.OK
    assert not c.retryable


def test_classify_401_as_auth():
    c = classify_openai_error(401, b'{"error":{"message":"invalid token"}}')
    assert c.kind is ErrorKind.AUTH
    assert c.should_refresh_token
    assert not c.should_rotate_account


def test_classify_403_with_cap_signal_is_cap():
    c = classify_openai_error(403, b'{"error":{"message":"You have hit your plan limit"}}')
    assert c.kind is ErrorKind.CAP
    assert c.should_rotate_account


def test_classify_403_without_cap_signal_is_auth():
    c = classify_openai_error(403, b'{"error":{"message":"forbidden"}}')
    assert c.kind is ErrorKind.AUTH


def test_classify_429_rate_limit_vs_cap():
    rl = classify_openai_error(429, b'{"error":{"message":"Too many requests, slow down"}}')
    assert rl.kind is ErrorKind.RATE_LIMIT
    assert rl.should_rotate_account and rl.retryable

    cap = classify_openai_error(429, b'{"error":{"message":"You exceeded your monthly limit"}}')
    assert cap.kind is ErrorKind.CAP


def test_classify_404_is_not_found_and_fatal():
    c = classify_openai_error(404, b"not found")
    assert c.kind is ErrorKind.NOT_FOUND
    assert not c.retryable


def test_classify_400_422_bad_request_non_retryable():
    for code in (400, 422):
        c = classify_openai_error(code, b'{"error":{"message":"Unsupported parameter"}}')
        assert c.kind is ErrorKind.BAD_REQUEST
        assert not c.retryable


def test_classify_5xx_is_transient_retryable():
    c = classify_openai_error(503, b"service unavailable")
    assert c.kind is ErrorKind.TRANSIENT
    assert c.retryable


def test_classify_none_status_is_transient():
    c = classify_openai_error(None, b"")
    assert c.kind is ErrorKind.TRANSIENT


def test_extract_retry_after_from_header():
    assert extract_retry_after({"retry-after": "12"}) == 12.0
    assert extract_retry_after({"Retry-After": "2m"}) == 120.0


def test_extract_retry_after_from_body_phrase():
    assert extract_retry_after(None, "please try again in 5s") == 5.0
    assert extract_retry_after(None, "try again in 500ms") == 0.5


def test_extract_retry_after_absent_returns_none():
    assert extract_retry_after({}, "no hint here") is None
