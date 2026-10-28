"""Tests for codex_oauth.harbor_launcher — the Harbor-specific wrapper.

Unlike the generic launcher (which exports OPENAI_BASE_URL/OPENAI_API_KEY into a
child's env), the Harbor wrapper injects --base-url + --ae OPENAI_API_KEY=<secret>
into the `harbor trial start` ARGV, using the CONTAINER-reachable bridge address
(the agent runs inside a Docker sandbox, several process hops from the CLI).
"""

from __future__ import annotations

import pytest

from codex_oauth import harbor_launcher as hl


# --- pure command-building logic ---------------------------------------------
def test_build_harbor_command_injects_base_url_and_api_key():
    cmd = hl.build_harbor_command(
        ["python", "-m", "harbor", "trial", "start", "-p", "task", "-a", "openhands",
         "-m", "openai/gpt-6.1-sol"],
        container_base_url="http://host.docker.internal:8788",
        bridge_secret="sekret",
    )
    assert "--base-url" in cmd
    assert "http://host.docker.internal:8788/v1" in cmd
    # the secret is passed as an agent-env, not an ambient var
    assert "--ae" in cmd
    assert "OPENAI_API_KEY=sekret" in cmd


def test_build_harbor_command_appends_v1_suffix_once():
    cmd = hl.build_harbor_command(
        ["harbor", "trial", "start"],
        container_base_url="http://host.docker.internal:9999/",  # trailing slash
        bridge_secret="s",
    )
    idx = cmd.index("--base-url")
    assert cmd[idx + 1] == "http://host.docker.internal:9999/v1"


def test_build_harbor_command_respects_user_supplied_base_url():
    # If the caller already pinned --base-url, do NOT override it.
    cmd = hl.build_harbor_command(
        ["harbor", "trial", "start", "--base-url", "http://custom:1/v1"],
        container_base_url="http://host.docker.internal:8788",
        bridge_secret="s",
    )
    assert cmd.count("--base-url") == 1
    assert "http://host.docker.internal:8788/v1" not in cmd
    assert "http://custom:1/v1" in cmd


def test_build_harbor_command_respects_user_supplied_openai_api_key():
    cmd = hl.build_harbor_command(
        ["harbor", "trial", "start", "--ae", "OPENAI_API_KEY=mine"],
        container_base_url="http://host.docker.internal:8788",
        bridge_secret="s",
    )
    # exactly one OPENAI_API_KEY agent-env, the caller's
    assert sum(1 for a in cmd if a.startswith("OPENAI_API_KEY=")) == 1
    assert "OPENAI_API_KEY=mine" in cmd
    assert "OPENAI_API_KEY=s" not in cmd


def test_build_harbor_command_preserves_original_args_order():
    base = ["harbor", "trial", "start", "-p", "t", "-a", "openhands", "-m", "openai/gpt-6.1-sol"]
    cmd = hl.build_harbor_command(base, container_base_url="http://h:1", bridge_secret="s")
    assert cmd[: len(base)] == base  # originals untouched, injections appended


# --- orchestration (fake bridge + fake runner) -------------------------------
class _FakeBridge:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.container_base_url = "http://host.docker.internal:8788"
        self.bridge_secret = "fake-secret"

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_run_starts_bridge_injects_command_and_tears_down():
    seen = {}
    fake = _FakeBridge()

    def runner(cmd):
        seen["cmd"] = cmd
        assert fake.started and not fake.stopped  # bridge up while command runs
        return 0

    rc = hl.run(["harbor", "trial", "start", "-a", "openhands"], bridge=fake, runner=runner)
    assert rc == 0
    assert fake.started and fake.stopped
    assert "--base-url" in seen["cmd"]
    assert "http://host.docker.internal:8788/v1" in seen["cmd"]
    assert "OPENAI_API_KEY=fake-secret" in seen["cmd"]


def test_run_tears_down_even_if_runner_raises():
    fake = _FakeBridge()

    def runner(cmd):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        hl.run(["harbor", "trial", "start"], bridge=fake, runner=runner)
    assert fake.stopped is True  # teardown on every exit path


def test_run_tears_down_even_if_start_raises():
    class _BadBridge(_FakeBridge):
        def start(self):
            raise RuntimeError("startup failed")

    fake = _BadBridge()
    with pytest.raises(RuntimeError, match="startup failed"):
        hl.run(["harbor", "trial", "start"], bridge=fake, runner=lambda c: 0)
    assert fake.stopped is True


def test_split_wrapped_command_strips_leading_double_dash():
    assert hl.split_wrapped_command(["--", "harbor", "run"]) == ["harbor", "run"]
    assert hl.split_wrapped_command(["harbor", "run"]) == ["harbor", "run"]
