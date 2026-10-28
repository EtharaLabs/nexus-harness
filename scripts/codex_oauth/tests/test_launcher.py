"""Tests for codex_oauth.launcher — the generic host-side bridge lifecycle.

No uvicorn is ever spawned and no port is really served: subprocess + health
polling are monkeypatched so we assert the preflight/health/teardown contract and
the container-host resolution deterministically.
"""

from __future__ import annotations

import urllib.error

import pytest

from codex_oauth import launcher
from codex_oauth.launcher import BridgeStartupError, CodexBridge, run_with_bridge


class _FakeProc:
    def __init__(self, poll_value):
        self._poll = poll_value
        self.returncode = poll_value if poll_value is not None else 0
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


def _bridge(**kw) -> CodexBridge:
    kw.setdefault("port", 12399)       # avoid _find_free_port socket bind
    kw.setdefault("startup_timeout", 0.3)
    return CodexBridge(**kw)


# --- preflight fail-fast -----------------------------------------------------
def test_preflight_fails_fast_with_codex_login_hint(monkeypatch):
    class _R:
        returncode = 2
        stdout = ""
        stderr = "credentials error: no login"

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _R())
    b = _bridge()
    with pytest.raises(BridgeStartupError, match="codex login"):
        b.preflight()


def test_preflight_succeeds_on_zero_exit(monkeypatch):
    class _R:
        returncode = 0
        stdout = "[codex-bridge] credentials OK"
        stderr = ""

    monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _R())
    _bridge().preflight()  # must not raise


# --- health wait -------------------------------------------------------------
def test_await_health_raises_if_child_exited_early(monkeypatch):
    b = _bridge()
    b._proc = _FakeProc(poll_value=1)   # already dead
    with pytest.raises(BridgeStartupError, match="exited with code 1"):
        b._await_health()


def test_await_health_raises_on_timeout_and_tears_down(monkeypatch):
    b = _bridge(startup_timeout=0.2)
    b._proc = _FakeProc(poll_value=None)  # still "running"

    def _always_fail(url, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(launcher.urllib.request, "urlopen", _always_fail)
    with pytest.raises(BridgeStartupError, match="did not answer"):
        b._await_health()
    assert b._proc is None or b._proc.terminated  # stop() ran during the failure


# --- teardown on every path --------------------------------------------------
def test_run_with_bridge_tears_down_even_if_start_raises(monkeypatch):
    stopped = {"v": False}
    monkeypatch.setattr(CodexBridge, "start",
                        lambda self: (_ for _ in ()).throw(RuntimeError("no start")))
    monkeypatch.setattr(CodexBridge, "stop",
                        lambda self: stopped.__setitem__("v", True))
    with pytest.raises(RuntimeError, match="no start"):
        run_with_bridge(["echo", "hi"], port=12399)
    assert stopped["v"] is True


def test_stop_is_noop_without_proc():
    b = _bridge()
    b._proc = None
    b.stop()  # must not raise


def test_stop_terminates_running_proc():
    b = _bridge()
    proc = _FakeProc(poll_value=None)
    b._proc = proc
    b.stop()
    assert proc.terminated is True


# --- container-host resolution -----------------------------------------------
def test_default_container_host_desktop(monkeypatch):
    monkeypatch.delenv("NEXUS_CODEX_BRIDGE_HOST_ALIAS", raising=False)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    assert launcher.default_container_host() == "host.docker.internal"


def test_default_container_host_linux_uses_gateway(monkeypatch):
    monkeypatch.delenv("NEXUS_CODEX_BRIDGE_HOST_ALIAS", raising=False)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "_docker_bridge_gateway", lambda: "172.17.0.1")
    assert launcher.default_container_host() == "172.17.0.1"


def test_default_container_host_linux_falls_back_when_no_gateway(monkeypatch):
    monkeypatch.delenv("NEXUS_CODEX_BRIDGE_HOST_ALIAS", raising=False)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    monkeypatch.setattr(launcher, "_docker_bridge_gateway", lambda: None)
    assert launcher.default_container_host() == "host.docker.internal"


def test_default_container_host_env_override_wins(monkeypatch):
    monkeypatch.setenv("NEXUS_CODEX_BRIDGE_HOST_ALIAS", "10.0.0.5")
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    assert launcher.default_container_host() == "10.0.0.5"


def test_default_bind_host_linux_vs_desktop(monkeypatch):
    monkeypatch.delenv("NEXUS_CODEX_BRIDGE_BIND", raising=False)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Linux")
    assert launcher.default_bind_host() == "0.0.0.0"
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    assert launcher.default_bind_host() == "127.0.0.1"


def test_client_env_and_base_url_shapes():
    b = _bridge(port=8788, host="127.0.0.1", container_host="host.docker.internal",
                bridge_secret="sek")
    assert b.base_url == "http://127.0.0.1:8788"
    assert b.container_base_url == "http://host.docker.internal:8788"
    env = b.client_env()
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:8788"
    assert env["OPENAI_API_KEY"] == "sek"
    assert b.stub_api_key == "sek"
