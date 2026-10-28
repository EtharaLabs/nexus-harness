"""Host-side launcher for the Codex (ChatGPT subscription) bridge.

The sibling of ``claude_oauth.launcher``. Where that one fronts api.anthropic.com
with a Claude Max OAuth token, this fronts ``chatgpt.com/backend-api/codex`` with
a ChatGPT OAuth token read from ``~/.codex/auth.json``, and exposes it as an
OpenAI-compatible ``/v1/chat/completions`` endpoint (``codex_oauth.translate``
converts to the Responses API the Codex backend actually speaks).

This module:
  * pre-flights the credential so a missing/expired login fails in a second
    rather than after a full transcript has been built,
  * starts ``python -m codex_oauth`` on a free port,
  * blocks until ``/healthz`` answers,
  * runs the wrapped command with ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``
    pointed at the bridge,
  * tears the bridge down on every exit path.

Usage::

    python -m codex_oauth.launcher -- python judge_trajectory.py --provider openai ...

NOTE ON SCOPE: this exports the OPENAI_* pair only. It deliberately does not
touch ANTHROPIC_*, so running a Codex-judged finalize inside a Claude-bridge
session leaves the agent's Anthropic routing untouched.
"""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

ENV_PORT = "NEXUS_CODEX_BRIDGE_PORT"
ENV_SECRET = "NEXUS_CODEX_BRIDGE_SECRET"

#: Docker Desktop's host alias, for a container reaching the bridge on the host.
DESKTOP_HOST_ALIAS = "host.docker.internal"


class BridgeStartupError(RuntimeError):
    """The bridge process died or never became healthy."""


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _package_parent() -> str:
    """Directory holding the ``codex_oauth`` package, for PYTHONPATH."""
    return str(Path(__file__).resolve().parent.parent)


def _docker_bridge_gateway() -> str | None:
    """Gateway IP of Docker's default bridge, for native-Linux containers."""
    try:
        out = subprocess.run(
            ["docker", "network", "inspect", "bridge",
             "-f", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    gw = (out.stdout or "").strip()
    return gw or None


def default_container_host() -> str:
    """Hostname a container uses to reach the bridge on the host.

    Mirrors ``claude_oauth.launcher.default_container_host``: Docker Desktop gets
    ``host.docker.internal``; native Linux gets the default-bridge gateway IP,
    because the Desktop alias does not resolve there.
    """
    override = os.environ.get("NEXUS_CODEX_BRIDGE_HOST_ALIAS", "").strip()
    if override:
        return override
    if platform.system() == "Linux":
        gateway = _docker_bridge_gateway()
        if gateway:
            return gateway
        logger.warning(
            "Could not determine the Docker bridge gateway on this Linux host; "
            "falling back to %s. Set NEXUS_CODEX_BRIDGE_HOST_ALIAS if the "
            "container cannot resolve it.",
            DESKTOP_HOST_ALIAS,
        )
    return DESKTOP_HOST_ALIAS


def default_bind_host() -> str:
    """Address the bridge binds on.

    Loopback on Docker Desktop (which proxies host.docker.internal to the host
    loopback); all interfaces on native Linux, where containers arrive over the
    docker gateway and loopback-only would refuse them. The shared secret, not
    the bind address, is what keeps the subscription private.
    """
    override = os.environ.get("NEXUS_CODEX_BRIDGE_BIND", "").strip()
    if override:
        return override
    return "0.0.0.0" if platform.system() == "Linux" else "127.0.0.1"


class CodexBridge:
    """Owns the bridge subprocess lifetime for one run."""

    def __init__(
        self,
        *,
        port: int | None = None,
        host: str | None = None,
        container_host: str | None = None,
        startup_timeout: float = 45.0,
        log_level: str = "info",
        bridge_secret: str | None = None,
    ) -> None:
        self.host = host or default_bind_host()
        env_port = os.environ.get(ENV_PORT, "").strip()
        self.port = port or (int(env_port) if env_port else _find_free_port())
        self.container_host = container_host or default_container_host()
        self.startup_timeout = startup_timeout
        self.log_level = log_level
        # The shared secret is the only thing between a co-tenant process and the
        # ChatGPT subscription. Generated per run unless pinned.
        self.bridge_secret = (
            bridge_secret
            or os.environ.get(ENV_SECRET, "").strip()
            or secrets.token_urlsafe(24)
        )
        self._proc: subprocess.Popen[str] | None = None

    @property
    def stub_api_key(self) -> str:
        """What a client presents as ``OPENAI_API_KEY``.

        Equal to the bridge secret: the bridge authenticates the caller with it,
        strips it, and forwards the real OAuth bearer upstream.
        """
        return self.bridge_secret

    @property
    def base_url(self) -> str:
        loopback = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{loopback}:{self.port}"

    @property
    def container_base_url(self) -> str:
        return f"http://{self.container_host}:{self.port}"

    def client_env(self) -> dict[str, str]:
        return {
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_KEY": self.stub_api_key,
        }

    def preflight(self) -> None:
        """Fail fast on a missing/expired Codex login.

        Without this the first symptom of a bad credential is a 401 after the
        judge has already built a full transcript for every trial.
        """
        env = self._child_env()
        proc = subprocess.run(
            [sys.executable, "-m", "codex_oauth", "--check"],
            env=env, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            raise BridgeStartupError(
                f"Codex credentials unusable: {msg}\n"
                "Run `codex login` (ChatGPT Pro/Team/Enterprise required), then "
                "verify with: PYTHONPATH=scripts python3 -m codex_oauth --check"
            )
        logger.info("%s", (proc.stdout or "").strip())

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env[ENV_SECRET] = self.bridge_secret
        parent = _package_parent()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{parent}{os.pathsep}{existing}" if existing else parent
        return env

    def start(self) -> None:
        self.preflight()
        env = self._child_env()
        self._proc = subprocess.Popen(
            [
                sys.executable, "-m", "codex_oauth",
                "--host", self.host,
                "--port", str(self.port),
                "--log-level", self.log_level,
            ],
            env=env,
        )
        self._await_health()
        logger.info("Codex bridge ready at %s", self.base_url)

    def _await_health(self) -> None:
        health_url = f"{self.base_url}/healthz"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise BridgeStartupError(
                    f"codex bridge exited with code {self._proc.returncode} before "
                    "becoming healthy"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as r:
                    if 200 <= r.status < 300:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.25)
        self.stop()
        raise BridgeStartupError(
            f"codex bridge did not answer {health_url} within {self.startup_timeout}s"
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logger.info("Codex bridge stopped")


def run_with_bridge(command: Sequence[str], **bridge_kwargs: object) -> int:
    """Start the bridge, run ``command`` against it, always tear down."""
    bridge = CodexBridge(**bridge_kwargs)  # type: ignore[arg-type]
    child: subprocess.Popen[bytes] | None = None
    try:
        bridge.start()
        env = dict(os.environ)
        env.update(bridge.client_env())
        logger.info("OPENAI_BASE_URL=%s", env["OPENAI_BASE_URL"])
        child = subprocess.Popen(list(command), env=env)
        return child.wait()
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
        return 130
    finally:
        bridge.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m codex_oauth.launcher",
        description=(
            "Start the Codex subscription bridge, run a command with "
            "OPENAI_BASE_URL/OPENAI_API_KEY pointed at it, then tear it down."
        ),
    )
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument(
        "--container-host",
        default=None,
        help="Hostname a container uses to reach the bridge (default host.docker.internal).",
    )
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--check", action="store_true",
        help="Pre-flight the Codex credential and exit (starts no server).",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    bridge_kwargs = dict(
        port=args.port,
        host=args.host,
        container_host=args.container_host,
        log_level=args.log_level,
    )

    if args.check:
        try:
            CodexBridge(**bridge_kwargs).preflight()  # type: ignore[arg-type]
        except BridgeStartupError as e:
            print(f"[codex-launcher] {e}", file=sys.stderr)
            return 2
        return 0

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given (use: ... -- <command>)")
    try:
        return run_with_bridge(command, **bridge_kwargs)  # type: ignore[arg-type]
    except BridgeStartupError as e:
        # A missing login is an operator condition, not a crash. A traceback here
        # buries the one line that says what to do about it.
        print(f"[codex-launcher] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
